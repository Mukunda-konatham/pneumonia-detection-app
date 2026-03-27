import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import random

from model import ClassicalCNN, HybridQNN

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_classical():
    model = ClassicalCNN()
    model.load_state_dict(torch.load("classical.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_hybrid():
    model = HybridQNN()
    model.load_state_dict(torch.load("hybrid.pth", map_location="cpu"))
    model.eval()
    return model

# -----------------------------
# Noise Functions
# -----------------------------
def add_gaussian_noise(img):
    std = random.uniform(0.0, 0.4)   # RANDOM
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0., 1.), std

def add_salt_pepper_noise(img):
    prob = random.uniform(0.0, 0.2)  # RANDOM
    noisy = img.clone()
    rand = torch.rand_like(img)

    noisy[rand < prob] = 1.0
    noisy[rand > 1 - prob] = 0.0

    return noisy, prob

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Pneumonia Detection System")

st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Classical CNN", "Hybrid QNN"]
)

noise_type = st.sidebar.multiselect(
    "Select Noise Type",
    ["Gaussian", "Salt & Pepper"]
)
gaussian_level = st.slider("Gaussian Noise Level", 0.0, 0.4, 0.2)
sp_level = st.slider("Salt & Pepper Noise Level", 0.0, 0.1, 0.05)


already vunna code lo ivi add cheyyi ra,appudu adjust cheyyataniki avuthadhi

# -----------------------------
# Image Processing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    img = transform(image).unsqueeze(0)

    gaussian_val = None
    sp_val = None

    # Apply Random Noise
    if "Gaussian" in noise_type:
        img, gaussian_val = add_gaussian_noise(img)

    if "Salt & Pepper" in noise_type:
        img, sp_val = add_salt_pepper_noise(img)

    # Show processed image
    st.subheader("Processed Image")
    st.image(img.squeeze().numpy(), clamp=True)

    # Load model
    if model_choice == "Classical CNN":
        model = load_classical()
    else:
        model = load_hybrid()

    # Prediction
    output = model(img)
    _, pred = torch.max(output, 1)

    result = "PNEUMONIA 🫁" if pred.item() == 1 else "NORMAL ✅"

    st.subheader("Prediction")
    st.success(result)

    # Show random values used
    if gaussian_val is not None:
        st.write(f"Gaussian Noise (random): {round(gaussian_val, 3)}")

    if sp_val is not None:
        st.write(f"Salt & Pepper Noise (random): {round(sp_val, 3)}")
