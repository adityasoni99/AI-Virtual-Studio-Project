#!/opt/anaconda3/envs/ai-studio/bin/python
import cv2
from PIL import Image
import numpy as np
import os

# Ensure examples directory exists
os.makedirs("examples", exist_ok=True)

# Load a sample image
input_path = "examples/sample_person.png"
output_path = "examples/canny_edge_example.png"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input image {input_path} not found. Please provide a sample image.")

img = cv2.imread(input_path)
if img is None:
    raise ValueError(f"Failed to load image from {input_path}.")

# Generate Canny edges
edges = cv2.Canny(img, 100, 200)
canny_img = Image.fromarray(edges).resize((512, 512))
canny_img.save(output_path)
print(f"Canny edge image saved to {output_path}")