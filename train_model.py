"""
train_model.py
--------------
Trains and compares several machine-learning models on the scikit-learn
handwritten digits dataset (1797 images, 8x8 pixels, classes 0-9) and saves
the best one for the Streamlit app.

Run:  python train_model.py
"""

import json
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import preprocess_gray

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "digit_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CM_PATH = MODEL_DIR / "confusion_matrix.png"
SAMPLES_PATH = MODEL_DIR / "sample_digits.png"
SEED = 42


def augment(X, y, copies=6, seed=SEED):
    """
    Data augmentation: for every training image we create several distorted
    copies (rotated, sheared, stretched, thicker or thinner strokes).
    Real drawings differ a lot from the few dozen writers in the dataset, so
    this makes the model more forgiving.

    Each copy goes through the SAME preprocessing the app uses, so the model
    is trained on exactly the kind of input it will see later.
    """
    rng = np.random.default_rng(seed)
    X_out, y_out = [X], [y]
    c = 64  # centre of the 128x128 working canvas
    for _ in range(copies):
        feats, labels = [], []
        for img, label in zip(X.reshape(-1, 8, 8), y):
            # enlarge the 8x8 image to 96x96 and give it a 16px border
            big = cv2.resize(img.astype(np.float32) / 16 * 255, (96, 96),
                             interpolation=cv2.INTER_CUBIC)
            big = np.pad(np.clip(big, 0, 255), 16)

            # random rotation + shear + stretch + shift as one matrix
            rot = np.vstack([cv2.getRotationMatrix2D(
                (c, c), rng.uniform(-15, 15), 1.0), [0, 0, 1]])
            scale = np.array([[rng.uniform(0.8, 1.4), rng.uniform(-0.25, 0.25), 0],
                              [0, rng.uniform(0.9, 1.1), 0],
                              [0, 0, 1]])
            to_origin = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
            back = np.array([[1, 0, c + rng.uniform(-8, 8)],
                             [0, 1, c + rng.uniform(-8, 8)], [0, 0, 1]])
            M = (back @ scale @ to_origin @ rot)[:2]
            warped = cv2.warpAffine(big, M, (128, 128), flags=cv2.INTER_LINEAR)

            # thicker / thinner pen strokes
            style = rng.random()
            kernel = np.ones((3, 3), np.uint8)
            if style < 0.25:
                warped = cv2.dilate(warped, kernel, iterations=int(rng.integers(1, 3)))
            elif style < 0.40:
                warped = cv2.erode(warped, kernel, iterations=1)

            result = preprocess_gray(warped, invert=False)
            if result is not None:
                feats.append(result[0][0])
                labels.append(label)
        X_out.append(np.array(feats))
        y_out.append(np.array(labels))
    return np.vstack(X_out), np.concatenate(y_out)


def get_candidates():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "K-Nearest Neighbours": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=SEED),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                          random_state=SEED)),
        # SVC does not give probabilities by itself, so we wrap it in
        # CalibratedClassifierCV (the recommended way in recent scikit-learn)
        "Support Vector Machine": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(kernel="rbf", C=10), ensemble=False,
                                   cv=5)),
    }


def save_sample_image(digits):
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for ax, img, label in zip(axes.ravel(), digits.images, digits.target):
        ax.imshow(img, cmap="gray_r")
        ax.set_title(str(label))
        ax.axis("off")
    fig.suptitle("Sample images from the dataset (8x8 pixels)")
    fig.tight_layout()
    fig.savefig(SAMPLES_PATH, dpi=100)
    plt.close(fig)


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    digits = load_digits()
    X, y = digits.data, digits.target
    save_sample_image(digits)

    # 64% train / 16% validation / 20% test
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.20, stratify=y_rest, random_state=SEED)

    # Compare models on the validation set
    X_train_aug, y_train_aug = augment(X_train, y_train)
    results = {}
    for name, model in get_candidates().items():
        model.fit(X_train_aug, y_train_aug)
        results[name] = round(accuracy_score(y_val, model.predict(X_val)), 4)
        print(f"{name:26s} validation accuracy = {results[name]:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name}")

    # Refit the winner on train + validation, then test once on unseen data
    X_full_aug, y_full_aug = augment(X_rest, y_rest)
    best_model = get_candidates()[best_name]
    best_model.fit(X_full_aug, y_full_aug)

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (unseen data): {test_acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=100)
    plt.close(fig)

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "best_model": best_name,
        "test_accuracy": round(test_acc, 4),
        "validation_accuracies": results,
        "per_digit_f1": {str(d): round(report[str(d)]["f1-score"], 3)
                         for d in range(10)},
        "dataset": {
            "name": "scikit-learn digits (UCI optical recognition)",
            "total_images": int(len(X)),
            "train_images": int(len(X_train)),
            "validation_images": int(len(X_val)),
            "test_images": int(len(X_test)),
            "image_size": "8x8 pixels",
            "augmented_training_images": int(len(X_full_aug)),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train()
