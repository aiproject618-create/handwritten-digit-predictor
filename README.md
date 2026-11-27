# ✍️ Handwritten Digit Predictor

An AI / Machine Learning project that recognises handwritten digits (0-9) and runs as a web app using **Streamlit**.
Made for the CBSE Class 12 Artificial Intelligence project.

You can **draw** a digit with your mouse or finger, **upload** an image, or take a **camera** photo. The app shows the predicted digit, the confidence, and a probability bar chart.

## Features

- 🖊️ Draw a digit on a canvas and get a live prediction
- 📁 Upload a photo / screenshot of a digit
- 📷 Take a picture with the camera
- 📊 "Model performance" tab: accuracy, model comparison, confusion matrix
- Shows what the model actually "sees" (the 8x8 pixel version of your digit)

## How it works

```
Image  ->  grayscale  ->  clean noise  ->  crop digit  ->  scale to 8x8 pixels  ->  ML model  ->  digit 0-9
```

1. **Pre-processing** (`preprocess.py`): the picture is converted to grayscale, background noise is removed, the digit is cropped, scaled to the full height of the box and shrunk to 8x8 pixels (the same format as the training data).
2. **Model** (`train_model.py`): five models are trained and compared - Logistic Regression, K-Nearest Neighbours, Random Forest, Neural Network (MLP) and Support Vector Machine. The best one on the validation set (SVM) is saved.
3. **Web app** (`app.py`): Streamlit loads the saved model and predicts.

## Dataset

scikit-learn **digits** dataset (from the UCI "Optical Recognition of Handwritten Digits" collection): 1,797 labelled images, 8x8 pixels, 10 classes (0-9). It comes built into scikit-learn, so no download is needed.

| Split | Images | Purpose |
|---|---|---|
| Train | 1,149 | learn from |
| Validation | 288 | choose the best model |
| Test | 360 | final check on unseen data |

Training images were also rotated, sheared, stretched and made thicker/thinner (**data augmentation**, 10,059 training images in total) so the model copes better with real drawings.

## Results

| Model | Validation accuracy |
|---|---|
| Logistic Regression | 96.18% |
| K-Nearest Neighbours | 97.57% |
| Neural Network (MLP) | 97.92% |
| Random Forest | 98.61% |
| **Support Vector Machine** | **98.96%** |

**Final test accuracy (SVM, unseen test images): 98.61%**

> Note: this is accuracy on the dataset's own images. Digits you draw yourself can be recognised less accurately because handwriting styles vary. Draw the digit **big, thick and in the middle** for the best result.

## Project structure

```
handwritten-digit-predictor/
├── app.py              # Streamlit web app
├── train_model.py      # trains, compares and saves the model
├── preprocess.py       # turns a drawing/photo into 8x8 model input
├── requirements.txt    # Python libraries
├── PROJECT_REPORT.md   # report for school submission
└── models/
    ├── digit_model.joblib      # trained model
    ├── metrics.json            # accuracy numbers
    ├── confusion_matrix.png
    └── sample_digits.png
```

## Run on your computer

You need Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/handwritten-digit-predictor.git
cd handwritten-digit-predictor

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

To train the model again yourself: `python train_model.py` (takes about 30 seconds).

## Put it on GitHub

```bash
git init
git add .
git commit -m "Handwritten digit predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/handwritten-digit-predictor.git
git push -u origin main
```

(Create an empty repository named `handwritten-digit-predictor` on github.com first.)

## Deploy online (free) on Streamlit Community Cloud

1. Push the project to GitHub (above).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository, branch `main`, and main file `app.py`.
4. Click **Deploy**. You get a public link you can share with your teacher.

## Limitations

- The model sees only 8x8 pixels, so very fancy or very untidy handwriting can be misread.
- It recognises **one digit at a time** (not whole numbers).
- Trained on a small dataset (1,797 images).

## Future scope

- Train on the larger MNIST dataset (70,000 images, 28x28) with a CNN
- Recognise multi-digit numbers
- Learn from the corrections users make

## Tech stack

Python · scikit-learn · NumPy · OpenCV · Pandas · Matplotlib · Streamlit

## Author

[Your Name] · Class XII · [School Name] · [Year]
