import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

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
def add_gaussian_noise(img, std):
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0., 1.)

def add_salt_pepper_noise(img, prob):
    noisy = img.clone()
    rand = torch.rand_like(img)
    noisy[rand < prob] = 1.0
    noisy[rand > 1 - prob] = 0.0
    return noisy

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Pneumonia Detection System")

st.sidebar.title("Settings")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Classical CNN", "Hybrid QNN"]
)

# Noise selection
noise_type = st.sidebar.multiselect(
    "Select Noise Type",
    ["Gaussian", "Salt & Pepper"]
)

# Gaussian slider
gaussian_std = st.sidebar.slider(
    "Gaussian Noise Level",
    min_value=0.0,
    max_value=0.4,
    value=0.2,
    step=0.05
)

# Salt & Pepper slider
sp_prob = st.sidebar.slider(
    "Salt & Pepper Intensity",
    min_value=0.0,
    max_value=0.2,
    value=0.05,
    step=0.01
)

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

    # Apply Noise
    if "Gaussian" in noise_type:
        img = add_gaussian_noise(img, gaussian_std)

    if "Salt & Pepper" in noise_type:
        img = add_salt_pepper_noise(img, sp_prob)

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

    # Show noise values
    st.write(f"Gaussian Level: {gaussian_std}")
    st.write(f"Salt & Pepper Level: {sp_prob}")
