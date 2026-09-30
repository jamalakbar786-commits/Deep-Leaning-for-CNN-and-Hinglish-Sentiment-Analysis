import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Solar Cell Defect Classification",
    page_icon="☀️",
    layout="centered"
)

# --------------------------------------------------
# Model Architecture
# --------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):  # FIXED: Syntax correction
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.leaky(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.leaky(out)
        return out

class DecoupledMultiScaleFusionCNN(nn.Module):
    def __init__(self):  # FIXED: Syntax correction
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.layer1 = ResidualBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)

        self.local_track = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.context_track = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.defect_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

        self.material_head = nn.Sequential(
            nn.Linear(128, 48),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(48, 2)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.pool1(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)

        feat_local = self.local_track(x)
        feat_context = self.context_track(x)

        fused = torch.cat([feat_local, feat_context], dim=1)

        shared_features = self.flatten(
            self.global_pool(fused)
        )

        ord_logits = self.defect_head(shared_features)
        mat_logits = self.material_head(shared_features)

        return ord_logits, mat_logits

# --------------------------------------------------
# Class Names & Settings
# --------------------------------------------------
CLASS_NAMES = [
    "0.00 Mono",
    "0.00 Poly",
    "0.33 & 0.66 Mono",
    "0.33 & 0.66 Poly",
    "1.00 Mono",
    "1.00 Poly"
]

DEVICE = torch.device("cpu")

# --------------------------------------------------
# Load Model & Configurations Safely
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = DecoupledMultiScaleFusionCNN()
    if os.path.exists("solar_cnn_weights.pth"):
        model.load_state_dict(torch.load("solar_cnn_weights.pth", map_location=DEVICE))
    model.eval()
    return model

@st.cache_data
def load_config():
    if os.path.exists("app_config.json"):
        with open("app_config.json", "r") as f:
            return json.load(f)
    # Default fallback threshold metrics if metadata document is missing
    return {"best_th_medium": 0.5, "best_th_high": 0.5}

# --------------------------------------------------
# Transform Pipeline
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_image(img):
    model = load_model()
    config = load_config()

    th_medium = config["best_th_medium"]
    th_high = config["best_th_high"]

    # Transform input pipeline elements
    img = img.convert("L")
    tensor = transform(img)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        ord_logits, mat_logits = model(tensor)
        ord_probs = torch.sigmoid(ord_logits).cpu().numpy()[0]
        mat_probs = torch.softmax(mat_logits, dim=1).cpu().numpy()[0]

    p0 = ord_probs[0]
    p1 = min(ord_probs[0], ord_probs[1])

    medium_trigger = float(p0 >= th_medium)
    high_trigger = float(p1 >= th_high)

    defect_probs = np.zeros(3)
    defect_probs[0] = 1.0 - medium_trigger
    defect_probs[1] = medium_trigger - high_trigger
    defect_probs[2] = high_trigger

    defect_class = np.argmax(defect_probs)
    material_class = np.argmax(mat_probs)

    final_class = defect_class * 2 + material_class

    return CLASS_NAMES[final_class], ord_probs, mat_probs

# --------------------------------------------------
# UI Component Layer
# --------------------------------------------------
st.title("☀️ Solar Cell Defect Classification")
st.write("Upload a solar-cell image and classify its defect severity and material type.")

uploaded_file = st.file_uploader("Choose Image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Predict"):
        with st.spinner("Running Multi-Head CNN Inference..."):
            pred_class, ord_probs, mat_probs = predict_image(image)

        st.success(f"Predicted Class: {pred_class}")