import streamlit as st
import numpy as np
import joblib
import cv2
from streamlit_drawable_canvas import st_canvas

# Load model and scaler
model = joblib.load("knn_digit_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Digit Recognition", page_icon="✏️")
st.title("✏️ Handwritten Digit Recognition (KNN)")
st.write("Draw a digit (0–9) below and let the model predict it!")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict if something is drawn
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    inverted = 16 - (resized / 16)  # scale similar to dataset
    flat = inverted.reshape(1, -1)
    flat = scaler.transform(flat)

    pred = model.predict(flat)
    st.subheader(f"✅ Predicted Digit: {pred[0]}")

