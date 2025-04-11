import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import requests

# Load model and processor
@st.cache_resource
def load_model():
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return processor, model


def estimate_image(image: Image.Image, processor, model):
    """Run inference and return segmentation labels."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    return prediction


def visualize_estimation(prediction):
    """Visualize the segmentation output."""
    output = prediction.squeeze().cpu().numpy()
    normalized_depth = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
    plt.subplot(1, 2, 2)
    plt.imshow(normalized_depth, cmap="jet", alpha=0.7)
    plt.axis("off")
    plt.title("Segmented Image")
    
    st.pyplot(plt)


def sidebar_profile():
    # Sidebar info with custom profile section
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <style>
            .custom-sidebar {
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                width: 650px;
                padding: 10px;
            }
            .profile-container {
                display: flex;
                flex-direction: row;
                align-items: flex-start;
                width: 100%;
            }
            .profile-image {
                width: 200px;
                height: auto;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                margin-right: 15px;
            }
            .profile-details {
                font-size: 14px;
                width: 100%;
            }
            .profile-details h3 {
                margin: 0 0 10px;
                font-size: 18px;
                color: #333;
            }
            .profile-details p {
                margin: 10px 0;
                display: flex;
                align-items: center;
            }
            .profile-details a {
                text-decoration: none;
                color: #1a73e8;
            }
            .profile-details a:hover {
                text-decoration: underline;
            }
            .icon-img {
                width: 18px;
                height: 18px;
                margin-right: 6px;
            }
        </style>

        <div class="custom-sidebar">
            <div class="profile-container">
                <img class="profile-image" src="https://res.cloudinary.com/dwhfxqolu/image/upload/v1744014185/pnhnaejyt3udwalrmnhz.jpg" alt="Profile Image">
                <div class="profile-details">
                    <h3>👨‍💻 Developed by:<br> Tahir Abbas Shaikh</h3>
                    <p>
                        <img class="icon-img" src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail">
                        <strong>Email:</strong> <a href="mailto:tahirabbasshaikh555@gmail.com">tahirabbasshaikh555@gmail.com</a>
                    </p>
                    <p>📍 <strong>Location:</strong> Sukkur, Sindh, Pakistan</p>
                    <p>
                        <img class="icon-img" src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="GitHub">
                        <strong>GitHub:</strong> <a href="https://github.com/Tahir-Abbas-555" target="_blank">Tahir-Abbas-555</a>
                    </p>
                    <p>
                        <img class="icon-img" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace">
                        <strong>HuggingFace:</strong> <a href="https://huggingface.co/Tahir5" target="_blank">Tahir5</a>
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------- MAIN UI ----------------------

def main():
    # Set page config FIRST
    
    st.title("📏 Depth Estimation using DPT")

    st.markdown(
        """
        Upload an image, and this app will estimate its **depth map** using the 
        [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) model. 
        This transformer-based model provides high-quality monocular depth estimation from a single RGB image.
        """
    )

    # Load the model only once
    processor, model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("🧠 Processing with Segformer..."):
            labels = estimate_image(image, processor, model)

        with col2:
            st.markdown("#### 🖼️ Segmentation Output")
            visualize_estimation(labels)

        st.success("✅ Segmentation completed successfully!")

    else:
        st.info("Please upload an image to start face parsing.")

# ---------------------- LAUNCH APP ----------------------

if __name__ == "__main__":
    sidebar_profile()
    main()