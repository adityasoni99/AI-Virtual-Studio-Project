import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from photoshoot_studio.image_generation.generate import ImageGenerator
import pytest
from PIL import Image

def test_image_generator_init():
    generator = ImageGenerator()
    assert generator.device in ["mps", "cpu"]
    assert generator.pipe is not None

def test_generate_image():
    generator = ImageGenerator()
    prompt = "A simple test image"
    image = generator.generate_image(prompt, width=256, height=256, num_inference_steps=10)
    assert image.size == (256, 256)

def test_generate_image_with_canny_controlnet():
    generator = ImageGenerator(controlnet_model_ids=["canny"])
    prompt = "A simple test image"
    control_image = Image.new("L", (256, 256), color=0)
    image = generator.generate_image(prompt, width=256, height=256, num_inference_steps=10, control_images=control_image, control_weights=1.0)
    assert image.size == (256, 256)

def test_generate_image_with_hed_controlnet():
    generator = ImageGenerator(controlnet_model_ids=["hed"])
    prompt = "A simple test image"
    control_image = Image.new("L", (256, 256), color=0)
    image = generator.generate_image(prompt, width=256, height=256, num_inference_steps=10, control_images=control_image, control_weights=1.0)
    assert image.size == (256, 256)

def test_generate_image_with_openpose_controlnet():
    generator = ImageGenerator(controlnet_model_ids=["openpose"])
    prompt = "A superhero posing"
    # Use the real OpenPose image if available, otherwise a dummy
    control_image_path = "examples/openpose_example.png"
    control_image = Image.open(control_image_path).resize((256, 256)) if os.path.exists(control_image_path) else Image.new("RGB", (256, 256), color=(0, 0, 0))
    image = generator.generate_image(prompt, width=256, height=256, num_inference_steps=10, control_images=control_image, control_weights=1.0)
    assert image.size == (256, 256)

def test_generate_image_with_normal_controlnet():
    generator = ImageGenerator(controlnet_model_ids=["normal"])
    prompt = "A simple test image"
    control_image = Image.new("RGB", (256, 256), color=(128, 128, 255))
    image = generator.generate_image(prompt, width=256, height=256, num_inference_steps=10, control_images=control_image, control_weights=1.0)
    assert image.size == (256, 256)

def test_generate_image_with_multi_controlnet():
    generator = ImageGenerator(controlnet_model_ids=["hed", "openpose", "normal"])
    prompt = "A superhero in a city"
    control_images = [
        Image.new("L", (256, 256), color=0),          # HED
        Image.open("examples/openpose_example.png").resize((256, 256)) if os.path.exists("examples/openpose_example.png") else Image.new("RGB", (256, 256), color=(0, 0, 0)),  # OpenPose
        Image.new("RGB", (256, 256), color=(128, 128, 255))  # Normal
    ]
    image = generator.generate_image(
        prompt, width=256, height=256, num_inference_steps=10, control_images=control_images, control_weights=[0.7, 0.8, 0.5]
    )
    assert image.size == (256, 256)

# Existing tests for brightness, contrast, etc. remain unchanged...
def test_adjust_brightness():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color="gray")
    bright_image = generator.adjust_brightness(image, factor=2.0)
    assert bright_image.size == image.size
    orig_pixel = image.getpixel((128, 128))
    bright_pixel = bright_image.getpixel((128, 128))
    assert bright_pixel[0] > orig_pixel[0]

def test_adjust_contrast():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color="gray")
    contrast_image = generator.adjust_contrast(image, factor=2.0)
    assert contrast_image.size == image.size

def test_adjust_color_balance():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color=(100, 150, 200))
    color_image = generator.adjust_color_balance(image, factor=2.0)
    assert color_image.size == image.size
    orig_pixel = image.getpixel((128, 128))
    color_pixel = color_image.getpixel((128, 128))
    assert color_pixel != orig_pixel

def test_adjust_saturation():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color=(100, 150, 200))
    sat_image = generator.adjust_saturation(image, factor=2.0)
    assert sat_image.size == image.size
    orig_pixel = image.getpixel((128, 128))
    sat_pixel = sat_image.getpixel((128, 128))
    assert sat_pixel != orig_pixel

def test_save_image_png():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color="white")
    filepath = "output/test_output.png"
    generator.save_image(image, filepath, format="PNG")
    assert os.path.exists(filepath)
    os.remove(filepath)

def test_save_image_jpeg():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color="white")
    filepath = "output/test_output.jpg"
    generator.save_image(image, filepath, format="JPEG", quality=75)
    assert os.path.exists(filepath)
    os.remove(filepath)

def test_save_image_invalid_format():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color="white")
    with pytest.raises(ValueError):
        generator.save_image(image, "output/test_output.xyz", format="XYZ")