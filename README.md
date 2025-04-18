# Image-Classification-Cifar-10

This is a simple web application built with Streamlit that allows users to upload an image and get real-time predictions using a deep learning model trained on the CIFAR-10 dataset. The model is able to classify images into 10 categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

🚀 Features
Upload an image and get an instant prediction of the class.

Real-time visualization of the uploaded image.

Backend powered by a Convolutional Neural Network (CNN) trained on CIFAR-10.

Built using Python 3.10 and TensorFlow.

Deployable as a standalone Streamlit web app.

📁 Project Structure
bash
Copy
Edit
├── Test Images                 # Example images for testing 
├── app.py                      # Streamlit app file
├── my_model.h5                 # Trained Keras model file
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview (this file)
⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/cifar10-image-classifier.git
cd cifar10-image-classifier
2. Create a Virtual Environment (Python 3.10)
Make sure Python 3.10 is installed and set in your environment.

bash
Copy
Edit
python -m venv myenv
myenv\Scripts\activate   # On Windows
source myenv/bin/activate  # On Linux/Mac
3. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt
Sample requirements.txt:

makefile
Copy
Edit
streamlit==1.32.2
tensorflow==2.12.0
numpy
opencv-python
Pillow
4. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
📦 Model Training (Optional)
The CNN model was trained on CIFAR-10 using TensorFlow. If you want to retrain the model:

python
Copy
Edit
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10 data...
Save your trained model as my_model.h5:

python
Copy
Edit
model.save("my_model.h5")
📌 Notes
Ensure your image is clearly representative of one of the 10 classes.

The model was trained on 32x32 RGB images; uploaded images are resized internally.

Image normalization (/255.0) is applied before prediction.

🧠 Credits
Created by Anik Pal
Trained on CIFAR-10 dataset
Interface powered by Streamlit

Let me know if you want to add a project screenshot, GitHub badge, or deployment section as well!
