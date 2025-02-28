# utils/generate_openpose.py
from PIL import Image
from controlnet_aux import OpenposeDetector
import os

pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
for input_file in ["portrait.jpg", "landscape.jpg", "group.jpg"]:
    input_path = f"examples/{input_file}"
    output_path = f"examples/openpose_{input_file.replace('.jpg', '.png')}"
    if not os.path.exists(input_path):
        print(f"Skipping {input_file}: not found")
        continue
    input_image = Image.open(input_path).convert("RGB")
    pose_image = pose_estimator(input_image, hand_and_face=True)
    pose_image.save(output_path)
    print(f"OpenPose control image saved to {output_path}")