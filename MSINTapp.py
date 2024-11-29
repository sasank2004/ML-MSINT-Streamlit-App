import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model once when app starts
model = load_model('numrec_model.h5')

# Preprocess image function
def preprocess_image(image):
    # Convert image to grayscale if not already
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize and normalize the image
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1] / 255.0  # Thresholding & normalizing
    
    # Invert the image (black and white flip)
    inverted_image = 1 - image
    
    # Expand dimensions for batch size and channels (1 sample, 28x28, 1 channel)
    return np.expand_dims(inverted_image, axis=-1).reshape(1, 28, 28, 1), inverted_image

# Streamlit app interface
st.title("Handwritten Digit Recognition")
st.subheader("Using Neural Net by Sasank")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read the uploaded image using OpenCV
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
    
    # Preprocess the image and get the inverted image
    preprocessed_image, inverted_image = preprocess_image(image)
    
    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")
    
    # Predict button
    if st.button('Predict'):
        # Make prediction on the inverted image
        prediction = np.argmax(model.predict(preprocessed_image), axis=-1)[0]
        
        # Display the prediction in a large font
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>Predicted Class: {prediction}</h1>", unsafe_allow_html=True)
        
        # Show the inverted image after prediction
        st.image(inverted_image, caption="Inverted Image", use_container_width=True, clamp=True)
