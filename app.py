import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import requests

# Load model and processor
st.title("Depth Estimation using DPT")
st.write("Upload an image to estimate its depth map.")

@st.cache_resource
def load_model():
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return processor, model

processor, model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    # Convert to NumPy array
    output = prediction.squeeze().cpu().numpy()
    normalized_depth = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
    
    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(normalized_depth, cmap="inferno")
    ax[1].set_title("Predicted Depth Map")
    ax[1].axis("off")
    
    # Display result
    st.pyplot(fig)