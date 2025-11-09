# Handwritten Digit Recognition (KNN) — Streamlit App

Small Streamlit app that lets you draw a digit (0–9) and predicts it using a KNN model.

## Project layout
- app.py — Streamlit frontend
- knn_digit_model.joblib — trained KNN model (expected in project root)
- scaler.joblib — feature scaler used before prediction (expected in project root)
- requirements.txt — Python dependencies

## Requirements
Install dependencies (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```
If `opencv-python` fails to install, ensure you are using a supported Python version and run:
```powershell
pip install opencv-python
```

## Run the app
From project root:
```powershell
.\.venv\Scripts\Activate
streamlit run app.py
```
Open the displayed local URL in your browser.

## How to use
1. Draw a digit (0–9) on the canvas.
2. The app preprocesses the canvas image to an 8x8 feature vector, scales it with `scaler.joblib`, and predicts with `knn_digit_model.joblib`.
3. The predicted digit is shown below the canvas.

## Notes & troubleshooting
- Make sure `knn_digit_model.joblib` and `scaler.joblib` are in the same folder as `app.py`. If you used different filenames or a subfolder, update the paths in `app.py`.
- The canvas returns an RGBA image; the app converts it to grayscale before resizing. If predictions seem wrong, you may need to adjust preprocessing (resize method, inversion/scaling) to match how the model was trained.
- If you get errors loading the joblib files, confirm they were saved with the same scikit-learn / joblib versions used here.
- Use `opencv-python` (not `cv2-python`) in requirements if installation issues occur.

## Re-training a model (brief)
If you want to retrain using scikit-learn's digits dataset:
- Load dataset from `sklearn.datasets.load_digits()`
- Preprocess (flatten 8x8 images), fit a `StandardScaler`, then a `KNeighborsClassifier`
- Save with joblib:
```python
import joblib
joblib.dump(model, "knn_digit_model.joblib")
joblib.dump(scaler, "scaler.joblib")
```

## License
MIT

