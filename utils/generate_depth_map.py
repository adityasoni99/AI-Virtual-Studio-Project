from transformers import pipeline
from PIL import Image
import torch
import os

# Ensure examples directory exists
os.makedirs("examples", exist_ok=True)

# Set device to MPS if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device set to use {device}")

# Initialize depth estimation pipeline with fast processor
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large", device=device, use_fast=True)

# Load input image
input_path = "examples/sample_person.png"
output_path = "examples/depth_map_example.png"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input image {input_path} not found. Please provide a sample image.")

img = Image.open(input_path).convert("RGB")

# Generate depth map
depth = depth_estimator(img)["depth"]
depth = depth.resize((512, 512))  # Resize to match ControlNet input size
depth.save(output_path)
print(f"Depth map saved to {output_path}")