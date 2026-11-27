# HANDWRITTEN DIGIT PREDICTOR
### Artificial Intelligence Project - CBSE Class XII

**Name:** [Your Name]  **Class / Section:** XII - [  ]  **Roll No.:** [  ]
**School:** [School Name]  **Session:** 2026-27
**Teacher:** [Teacher's Name]

---

## Certificate

This is to certify that [Your Name] of Class XII, [School Name], has successfully completed the project "Handwritten Digit Predictor" in Artificial Intelligence during the academic session 2026-27 under my guidance.

Teacher's signature: __________  Examiner's signature: __________

## Acknowledgement

I thank my Artificial Intelligence teacher [Teacher's Name] for guidance, my principal for the opportunity, and my parents and friends for their support.

---

## 1. Abstract

This project builds an AI system that recognises handwritten digits (0-9). Several machine-learning models were trained and compared on the scikit-learn digits dataset, and the best one - a Support Vector Machine - reached **98.61% accuracy on unseen test images**. The model is used in a web app built with Streamlit, where the user can draw, upload or photograph a digit and instantly see the prediction with its confidence.

## 2. Introduction

Reading handwriting is easy for humans but hard for computers because every person writes differently. Digit recognition is used in real life for reading postal PIN codes, bank cheques, exam OMR/answer sheets and forms. This project uses **supervised learning** (classification) to teach a computer to do this.

## 3. Objectives

1. Build a machine-learning model that classifies handwritten digits.
2. Compare different algorithms and choose the best.
3. Make the model easy to use through a web app.
4. Publish the code on GitHub.

## 4. Requirements

**Hardware:** any computer with 4 GB RAM or more, internet connection.
**Software:** Python 3.10+, scikit-learn, NumPy, Pandas, OpenCV, Matplotlib, Streamlit, Git/GitHub.

## 5. AI Project Cycle

### 5.1 Problem Scoping (4Ws canvas)

| | |
|---|---|
| **Who** | Anyone who needs digits from handwritten images read automatically (schools, post offices, banks) |
| **What** | Computers cannot easily read different handwriting styles |
| **Where** | Document scanning, forms, exam sheets, postal sorting |
| **Why** | Automation saves time and reduces human error |

**Problem statement:** Our users need a quick way to convert a handwritten digit into a typed digit because manual reading is slow and error-prone.

### 5.2 Data Acquisition

Dataset: scikit-learn *digits* (UCI Optical Recognition of Handwritten Digits) - 1,797 images, 8x8 pixels, each pixel 0 (white) to 16 (black), labels 0-9.

### 5.3 Data Exploration

- 64 features per image (one per pixel), 10 classes, roughly 180 images per digit (balanced).
- Sample images are saved in `models/sample_digits.png`.
- Every digit fills the full height of the 8x8 box and is centred, so user drawings are cropped and scaled the same way.

### 5.4 Modelling

The data was split into **train 64% / validation 16% / test 20%** (stratified, so every split has all digits in equal proportion).

**Data augmentation:** each training image was rotated, sheared, stretched and made thicker/thinner to create extra training examples (1,437 -> 10,059 images). This helps the model handle real handwriting.

Five algorithms were compared:

| Algorithm | Idea |
|---|---|
| Logistic Regression | draws linear boundaries between digits |
| K-Nearest Neighbours | looks at the 5 most similar training images and votes |
| Random Forest | many decision trees vote together |
| Neural Network (MLP) | layers of connected neurons learn patterns |
| Support Vector Machine | finds the widest-margin boundary between classes |

### 5.5 Evaluation

| Model | Validation accuracy |
|---|---|
| Logistic Regression | 96.18% |
| K-Nearest Neighbours | 97.57% |
| Neural Network (MLP) | 97.92% |
| Random Forest | 98.61% |
| Support Vector Machine | **98.96%** |

The SVM was chosen, retrained on train + validation data, and tested **once** on the unseen test set: **98.61% accuracy**.

The **confusion matrix** (`models/confusion_matrix.png`) shows which digits get confused with each other. F1-scores per digit are in `models/metrics.json`; the lowest are digits 1 and 7 (about 0.96), which look alike.

**Metrics used**

- Accuracy = correct predictions / total predictions
- Precision = of the images predicted as a digit, how many were correct
- Recall = of the images that really are that digit, how many were found
- F1-score = balance of precision and recall

### 5.6 Deployment

A Streamlit web app (`app.py`) loads the saved model. The user's image is pre-processed (grayscale, noise removal, crop, resize to 8x8) and the model outputs a probability for each digit; the highest is shown.

## 6. Technologies Used

| Tool | Use |
|---|---|
| Python | programming language |
| scikit-learn | datasets, models, evaluation |
| NumPy / Pandas | arrays and tables |
| OpenCV | image processing |
| Matplotlib | confusion matrix and sample images |
| Streamlit | web app |
| GitHub | storing and sharing code |

## 7. Ethical Considerations

- The model can be wrong; it should not be the only check for important documents (cheques, marks).
- Accuracy can be lower for handwriting styles not seen in the training data (**bias in data**).
- No user images are stored by the app.

## 8. Limitations

- Only 8x8 pixels of detail, so untidy writing can be misread.
- Recognises one digit at a time.
- Small training dataset.

## 9. Future Scope

- Use the MNIST dataset (70,000 images) with a Convolutional Neural Network (CNN).
- Read multi-digit numbers and full words.
- Let users correct wrong predictions and learn from them.

## 10. Conclusion

The project shows how supervised machine learning can solve a real problem. Out of five algorithms the Support Vector Machine performed best, with 98.61% test accuracy, and the Streamlit app makes it usable by anyone.

## 11. Bibliography

- scikit-learn documentation - https://scikit-learn.org
- Streamlit documentation - https://docs.streamlit.io
- UCI Machine Learning Repository: Optical Recognition of Handwritten Digits dataset
- CBSE Artificial Intelligence (Code 843) curriculum

---

## Possible viva questions

1. **What type of machine learning is this?** Supervised learning - classification, because the data has labels (the correct digit).
2. **Why split data into train, validation and test?** Train to learn, validation to choose the model, test to check on completely unseen data; otherwise accuracy would be overstated.
3. **What is overfitting?** When a model memorises the training data and does badly on new data.
4. **Why data augmentation?** More varied examples make the model handle different handwriting.
5. **Why resize to 8x8?** The model was trained on 8x8 images, so the input must be in the same format.
6. **What is a confusion matrix?** A table of true vs predicted classes that shows which digits get mixed up.
7. **What is confidence?** The probability the model gives to its predicted digit.
8. **How can accuracy be improved?** More data (MNIST), a CNN, better pre-processing.
