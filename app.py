
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import io
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        input = noise * self.label_embed(labels)
        return self.model(input).view(-1, 1, 28, 28)


@st.cache_resource
def load_generator():
    model = Generator(latent_dim=100, num_classes=10)
    model.load_state_dict(torch.load("mnist_gan_generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def generate_images(generator, digit, num_images=5):
    z = torch.randn(num_images, 100)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    with torch.no_grad():
        images = generator(z, labels)
    return images

# --------------------
# Streamlit UI
# --------------------

st.set_page_config(page_title="MNIST Digit Generator", layout="centered")
st.title("🧠 MNIST Digit Generator")
st.markdown("Generate handwritten-style digits using a GAN trained on MNIST.")

selected_digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    generator = load_generator()
    imgs = generate_images(generator, selected_digit)
    grid = make_grid(imgs, nrow=5, normalize=True, pad_value=1)

    # Convert grid to image for display
    np_img = grid.permute(1, 2, 0).numpy()
    st.image(np_img, caption=f"Generated digit: {selected_digit}", use_container_width=False)
else:
    st.info("👆 Select a digit and click the button above to generate images.")
