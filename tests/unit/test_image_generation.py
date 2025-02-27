import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from photoshoot_studio.image_generation.generate import ImageGenerator
import pytest

def test_image_generator_init():
    generator = ImageGenerator()
    assert generator.device in ["mps", "cpu"]
    assert generator.pipe is not None

def test_generate_image():
    generator = ImageGenerator()
    prompt = "A simple test image"
    image = generator.generate_image(prompt, width=256, height=256)
    assert image.size == (256, 256)