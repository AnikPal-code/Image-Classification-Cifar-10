# Import the Package
import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Styling ---
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon=":camera:")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #f5f5f5;
    }
    .st-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4c4c;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .st-subheader {
        font-size: 1.5rem;
        color: #ff4c4c;
        text-align: center;
        margin-bottom: 1rem;
    }
    .st-text {
        color: #cccccc;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .st-success {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 0.75rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-top: 1rem;
    }
    .st-error {
        background-color: #f44336 !important;
        color: white !important;
        padding: 0.75rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-top: 1rem;
    }
    .css-1cpxqw2 edgvbvh3 { 
        background-color: #2e2e2e; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- End Styling ---

# Class names for CIFAR-10 dataset
class_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load the model
@st.cache_data
def load_my_model():
    return tf.keras.models.load_model("model.h5")

model = load_my_model()

# App Title & Subtitle
st.markdown("<div class='st-header'>Image Classification with CIFAR-10 Dataset</div>", unsafe_allow_html=True)
st.markdown("<div class='st-subheader'>Upload images from one of the following categories:</div>", unsafe_allow_html=True)
st.markdown(f"<div class='st-text'>{', '.join(class_name)}</div>", unsafe_allow_html=True)

# File uploader
file = st.file_uploader("Upload the image", type=["jpg", "png"])

# Prediction function
def import_and_predict(image_data, model):
    image = image_data.resize((32, 32), resample=Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Prediction
if st.button("Predict"):
    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        result = f"<div class='st-success'>Image mostly matches: <strong>{class_name[np.argmax(predictions)]}</strong></div>"
        st.markdown(result, unsafe_allow_html=True)
    else:
        st.markdown("<div class='st-error'>Please upload an image to classify.</div>", unsafe_allow_html=True)
