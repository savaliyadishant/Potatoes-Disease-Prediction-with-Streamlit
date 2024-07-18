import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# Ensure you use the correct path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'potatoes.h5')

# Load your TensorFlow model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prediction function
def predict_disease(image):
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class == 0:
        return "No Disease"
    else:
        return "Disease Detected"

# Streamlit UI
st.title("Disease Detection from Image")
st.write("Upload an image and the model will predict whether it has a disease or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    prediction = predict_disease(image)
    st.write(f"Prediction: {prediction}")
