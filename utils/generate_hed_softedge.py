import cv2
import numpy as np
from PIL import Image

img = cv2.imread("examples/sample_person.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
hed_img = Image.fromarray(edges).resize((512, 512))
hed_img.save("examples/hed_example.png")