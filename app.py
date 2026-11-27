"""
app.py - Handwritten Digit Predictor (Streamlit web app)

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from preprocess import image_to_gray, preprocess_gray

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "digit_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
CM_PATH = BASE_DIR / "models" / "confusion_matrix.png"
SAMPLES_PATH = BASE_DIR / "models" / "sample_digits.png"

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:  # package missing or incompatible
    CANVAS_AVAILABLE = False

st.set_page_config(page_title="Handwritten Digit Predictor",
                   page_icon="✍️", layout="centered")


@st.cache_resource
def load_model():
    """Load the trained model (trains it first if the file is missing)."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        # file missing, or saved with a different scikit-learn version
        from train_model import train
        with st.spinner("Training the model (takes about a minute)..."):
            train()
        return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def show_prediction(gray, invert=None):
    """Pre-process, predict and display the result."""
    result = preprocess_gray(gray, invert)
    if result is None:
        st.warning("No digit found. Draw or upload a clear digit.")
        return
    features, small = result

    model = load_model()
    proba = model.predict_proba(features)[0]
    digit = int(model.classes_[np.argmax(proba)])
    confidence = float(proba.max())

    col1, col2 = st.columns(2)
    col1.metric("Predicted digit", digit)
    col1.metric("Confidence", f"{confidence * 100:.1f}%")
    with col2:
        st.caption("What the model sees (8x8 pixels)")
        preview = Image.fromarray((small / 16 * 255).astype(np.uint8))
        st.image(preview.resize((160, 160), Image.NEAREST))

    if confidence < 0.5:
        st.info("The model is not very sure. Try writing the digit bigger, "
                "thicker and centred.")

    st.caption("Probability for each digit")
    st.bar_chart(pd.DataFrame({"probability": proba},
                              index=[str(c) for c in model.classes_]))


INVERT_OPTIONS = {
    "Auto-detect": None,
    "Dark digit on light paper": True,
    "Light digit on dark background": False,
}


def image_tab(file, key):
    if file is None:
        return
    img = Image.open(file)
    st.image(img, caption="Your image", width=250)
    choice = st.radio("Colours in the image", list(INVERT_OPTIONS),
                      horizontal=True, key=key)
    show_prediction(image_to_gray(img), INVERT_OPTIONS[choice])


# ----------------------------------------------------------------- UI ----
st.title("✍️ Handwritten Digit Predictor")
st.write("An AI project that recognises handwritten digits (0-9) using "
         "machine learning.")

tab_draw, tab_upload, tab_camera, tab_perf, tab_about = st.tabs(
    ["🖊️ Draw", "📁 Upload", "📷 Camera", "📊 Model performance", "ℹ️ About"])

with tab_draw:
    if not CANVAS_AVAILABLE:
        st.error("The drawing canvas package is not installed. Run "
                 "`pip install streamlit-drawable-canvas`, or use the "
                 "Upload / Camera tabs.")
    else:
        st.write("Draw **one digit**, big and in the middle of the box.")
        stroke = st.slider("Pen thickness", 10, 30, 18)
        try:
            canvas = st_canvas(stroke_width=stroke, stroke_color="#FFFFFF",
                               background_color="#000000", height=280,
                               width=280, drawing_mode="freedraw",
                               key="canvas")
            data = canvas.image_data
            if data is not None:
                # white ink on black, alpha-aware
                rgb = data[:, :, :3].astype(np.float32).mean(axis=2)
                gray = rgb * data[:, :, 3].astype(np.float32) / 255.0
                if gray.max() > 20:
                    show_prediction(gray, invert=False)
                else:
                    st.caption("Start drawing to see a prediction.")
        except Exception as exc:
            st.error("The drawing canvas does not work with this Streamlit "
                     f"version ({exc}). Please use the Upload or Camera tab.")

with tab_upload:
    st.write("Upload a photo or screenshot of a single handwritten digit.")
    image_tab(st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"]),
              key="upload_invert")

with tab_camera:
    st.write("Write a digit on white paper with a dark pen and take a photo.")
    image_tab(st.camera_input("Take a picture"), key="camera_invert")

with tab_perf:
    metrics = load_metrics()
    if metrics is None:
        load_model()
        metrics = load_metrics()
    st.subheader("Results")
    st.metric(f"Test accuracy ({metrics['best_model']})",
              f"{metrics['test_accuracy'] * 100:.2f}%")
    st.write("Validation accuracy of every model that was compared:")
    st.dataframe(
        pd.DataFrame({"Model": list(metrics["validation_accuracies"]),
                      "Validation accuracy": [
                          f"{v * 100:.2f}%"
                          for v in metrics["validation_accuracies"].values()]}),
        hide_index=True)
    st.write("F1-score for each digit:")
    st.bar_chart(pd.Series(metrics["per_digit_f1"], name="F1-score"))
    if CM_PATH.exists():
        st.image(str(CM_PATH), caption="Confusion matrix on unseen test data")
    if SAMPLES_PATH.exists():
        st.image(str(SAMPLES_PATH))

with tab_about:
    metrics = load_metrics()
    st.markdown("""
**How it works**

1. The image is converted to grayscale and cleaned (noise removed).
2. The digit is cropped, centred and shrunk to **8 x 8 pixels**.
3. The 64 pixel values go into a trained **machine-learning model**.
4. The model gives a probability for each digit 0-9; the highest one is shown.

**Dataset:** scikit-learn digits dataset (1,797 labelled images, 8x8 pixels).
Training images were also rotated and shifted slightly (*data augmentation*).

**Tech used:** Python, scikit-learn, NumPy, OpenCV, Streamlit.
""")
    if metrics:
        st.json(metrics["dataset"])
