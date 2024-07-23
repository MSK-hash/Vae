import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the VAE model
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to the required dimensions
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = image.reshape(1, 128, 128, 1)  # Reshape for the model
    return image

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("VAE Model Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict(image)
    st.write(f"Prediction: {prediction}")
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the VAE model
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to the required dimensions
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = image.reshape(1, 128, 128, 1)  # Reshape for the model
    return image

def generate_image():
    # Assuming the model's decoder is accessible via the model object
    # Generate random latent vector
    latent_dim = 2  # Replace with your actual latent dimension
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    
    # Generate image from latent vector
    generated_image = model.decoder(random_latent_vector, training=False).numpy()
    
    # Post-process generated image
    generated_image = generated_image.squeeze()  # Remove batch dimension
    generated_image = (generated_image * 255).astype(np.uint8)  # Convert to 0-255 range
    return Image.fromarray(generated_image)

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("VAE Model Prediction and Image Generation")

# Button to generate image
if st.button('Generate Image'):
    generated_image = generate_image()
    st.image(generated_image, caption='Generated Image', use_column_width=True)

# File uploader for classification
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict(image)
    st.write(f"Prediction: {prediction}")
