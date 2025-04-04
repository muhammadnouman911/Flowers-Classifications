import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load Model
model = load_model('animal10_model.h5')
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Replace with actual category names

# Streamlit App
st.title("FLower Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = categories[np.argmax(predictions)]

    st.write(f"Predicted Class: {predicted_class}")




    #streamlit run app.py
#python -m streamlit run app.py

