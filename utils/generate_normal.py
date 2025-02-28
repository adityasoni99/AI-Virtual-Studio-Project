# utils/generate_normal.py
import cv2
import numpy as np
from PIL import Image

for input_file in ["portrait.jpg", "landscape.jpg", "group.jpg"]:
    img = cv2.imread(f"examples/{input_file}", cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    normal = np.dstack((sobelx, sobely, np.ones_like(img) * 255))
    normal_img = Image.fromarray(normal.astype(np.uint8)).resize((512, 512))
    normal_img.save(f"examples/normal_{input_file.replace('.jpg', '.png')}")
    print(f"Normal map saved to examples/normal_{input_file.replace('.jpg', '.png')}")