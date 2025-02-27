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

def test_adjust_brightness():
    generator = ImageGenerator()
    # Create a dummy image for testing
    image = Image.new("RGB", (256, 256), color="gray")
    bright_image = generator.adjust_brightness(image, factor=2.0)
    assert bright_image.size == image.size
    # Check if brightness increased (simple pixel value check)
    orig_pixel = image.getpixel((128, 128))
    bright_pixel = bright_image.getpixel((128, 128))
    assert bright_pixel[0] > orig_pixel[0]  # Red channel should be brighter

def test_adjust_contrast():
    generator = ImageGenerator()
    image = Image.new("RGB", (256, 256), color="gray")
    contrast_image = generator.adjust_contrast(image, factor=2.0)
    assert contrast_image.size == image.size
    # Basic check for contrast change is harder; verify it runs without error