import cv2
import numpy as np
from PIL import Image

img = cv2.imread("examples/sample_person.png", cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
normal = np.dstack((sobelx, sobely, np.ones_like(img) * 255))
normal_img = Image.fromarray(normal.astype(np.uint8)).resize((512, 512))
normal_img.save("examples/normal_map_example.png")