# Import the Package
import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Styling ---
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon=":camera:")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-header {
        background-color: #1c83e1;
        color: white;
        padding: 1rem 0;
        text-align: center;
    }
    .st-subheader {
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }
    .st-text {
        color: #555;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .st-file-uploader {
        background-color: #e6e6e6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .st-button {
        background-color: #1c83e1 !important;
        color: white !important;
        border-radius: 5px;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }
    .st-button:hover {
        background-color: #0e5caa !important;
    }
    .st-image {
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .st-success {
        background-color: #4CAF50 !important;
        color: white !important; /* Make text white */
        padding: 0.75rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold !important;
        margin-top: 1rem;
    }
    .st-error {
        background-color: #f44336 !important;
        color: white !important;
        padding: 0.75rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold !important;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# --- End Styling ---

# Define the class names for the CIFAR-10 dataset
class_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Create a function to load the saved model (updated with st.cache_data)
@st.cache_data
def load_my_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_my_model()

# Create a title for the web app
st.markdown("<h1 class='st-header'>Image Classification with CIFAR-10 Dataset</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='st-subheader'>Please Upload images related to the following categories:</h2>", unsafe_allow_html=True)
st.markdown(f"<p class='st-text'>{', '.join(class_name)}</p>", unsafe_allow_html=True)

# Create a file uploader for the user to upload an image (jpg or png)
file = st.file_uploader("Upload the image", type=["jpg", "png"])

# Create a function to process the image and predict the class
def import_and_predict(image_data, model):
    size = (32, 32)
    # Resize the image and use resampling (LANCZOS)
    image = image_data.resize(size, resample=Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Predict the class when the user clicks the "Predict" button
if st.button("Predict"):
    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)

        result = f"<p class='st-success'>Image mostly matches: <strong>{class_name[np.argmax(predictions)]}</strong></p>"
        st.markdown(result, unsafe_allow_html=True)
    else:
        st.markdown("<p class='st-error'>Please upload an image to classify.</p>", unsafe_allow_html=True)
