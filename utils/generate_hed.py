import cv2
import numpy as np
from PIL import Image

for input_file in ["portrait.jpg", "landscape.jpg", "group.jpg"]:
    img = cv2.imread(f"examples/{input_file}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    hed_img = Image.fromarray(edges).resize((1024, 1024))  # Updated to 1024x1024
    hed_img.save(f"examples/hed_{input_file.replace('.jpg', '.png')}")
    print(f"HED control image saved to examples/hed_{input_file.replace('.jpg', '.png')}")