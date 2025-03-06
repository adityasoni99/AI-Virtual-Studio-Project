import pytest
from fastapi.testclient import TestClient
from backend.main import app
import os
from io import BytesIO
from PIL import Image
import torch

client = TestClient(app)


def test_generate_image_api_jpeg():
    response = client.post(
        "/api/generate",
        data={
            "prompt": "A simple test image",
            "negative_prompt": "blurry, low quality",
            "width": 512,  # Test custom resolution
            "height": 768,
            "format": "JPEG",
            "quality": 75
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_generate_image_api_png():
    response = client.post(
        "/api/generate",
        data={
            "prompt": "A simple test image",
            "negative_prompt": "blurry, low quality",
            "width": 256,  # Test smaller resolution
            "height": 256,
            "format": "PNG",
            "quality": 50
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


def test_generate_image_api_invalid_format():
    response = client.post(
        "/api/generate",
        data={
            "prompt": "A simple test image",
            "format": "GIF"
        }
    )
    assert response.status_code == 400
    assert "Format must be 'JPEG' or 'PNG'" in response.json()["detail"]


def test_generate_image_api_invalid_quality():
    response = client.post(
        "/api/generate",
        data={
            "prompt": "A simple test image",
            "format": "JPEG",
            "quality": 101
        }
    )
    assert response.status_code == 400
    assert "Quality must be between 0 and 100" in response.json()["detail"]


def test_generate_image_api_invalid_resolution():
    response = client.post(
        "/api/generate",
        data={
            "prompt": "A simple test image",
            "width": 4096,  # Beyond limit
            "height": 1024,
            "format": "JPEG"
        }
    )
    assert response.status_code == 400
    assert "Width and height must be between 64 and 2048" in response.json()["detail"]


def test_generate_image_api_controlnet():
    os.makedirs("examples", exist_ok=True)
    if not os.path.exists("examples/openpose_portrait.png"):
        Image.new("RGB", (512, 768)).save("examples/openpose_portrait.png")
    if not os.path.exists("examples/hed_portrait.png"):
        Image.new("RGB", (512, 768)).save("examples/hed_portrait.png")

    files = [
        ("control_images", ("openpose.png", open("examples/openpose_portrait.png", "rb"), "image/png")),
        ("control_images", ("hed.png", open("examples/hed_portrait.png", "rb"), "image/png"))
    ]
    response = client.post(
        "/api/generate",
        data={
            "prompt": "A simple test image",
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 768,
            "format": "JPEG",
            "quality": 75,
            "control_weights": "[0.9, 0.7]"
        },
        files=files
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0