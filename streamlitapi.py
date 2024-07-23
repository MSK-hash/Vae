
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the VAE model
with open('model2.pkl', 'rb') as file:
    vae_model = pickle.load(file)

# Function to generate new samples using the VAE model
def generate_samples(vae_model, num_samples):
    # Assuming your VAE model has a method 'sample' to generate new data
    samples = vae_model.sample(num_samples)
    return samples

# Streamlit app
st.title("VAE Model Deployment with Streamlit")

# Input: Number of samples to generate
num_samples = st.number_input("Number of samples to generate", min_value=1, max_value=100, value=10)

if st.button("Generate Samples"):
    samples = generate_samples(vae_model, num_samples)
    
    # Display the generated samples
    st.write("Generated Samples:")
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i, sample in enumerate(samples):
        if num_samples == 1:
            axes.imshow(sample, cmap='gray')
        else:
            axes[i].imshow(sample, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
