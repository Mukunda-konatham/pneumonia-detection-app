import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import ClassicalCNN, HybridQNN

# -----------------------------
# Load models
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
# UI
# -----------------------------
st.title("🩺 Pneumonia Detection System")

st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Classical CNN", "Hybrid QNN"]
)

noise_option = st.sidebar.selectbox(
    "Noise",
    ["None", "Gaussian", "Salt & Pepper"]
)

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

def add_gaussian_noise(img):
    noise = torch.randn_like(img) * 0.2
    return torch.clamp(img + noise, 0., 1.)

def add_salt_pepper_noise(img):
    noisy = img.clone()
    rand = torch.rand_like(img)
    noisy[rand < 0.05] = 1.0
    noisy[rand > 0.95] = 0.0
    return noisy

# -----------------------------
# Upload image
# -----------------------------
file = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    if noise_option == "Gaussian":
        img = add_gaussian_noise(img)
    elif noise_option == "Salt & Pepper":
        img = add_salt_pepper_noise(img)

    # Load model
    if model_choice == "Classical CNN":
        model = load_classical()
    else:
        model = load_hybrid()

    # Prediction
    output = model(img)
    _, pred = torch.max(output, 1)

    result = "PNEUMONIA 🫁" if pred.item() == 1 else "NORMAL ✅"

    st.subheader("Prediction:")
    st.success(result)