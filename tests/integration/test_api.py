import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_generate_image_api_jpeg():
    response = client.post(
        "/api/generate",
        json={
            "prompt": "A simple test image",
            "negative_prompt": "blurry, low quality",
            "width": 128,
            "height": 128,
            "format": "JPEG",
            "quality": 75  # Test specific quality
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0

def test_generate_image_api_png():
    response = client.post(
        "/api/generate",
        json={
            "prompt": "A simple test image",
            "negative_prompt": "blurry, low quality",
            "width": 128,
            "height": 128,
            "format": "PNG",
            "quality": 50  # Ignored for PNG
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

def test_generate_image_api_invalid_format():
    response = client.post(
        "/api/generate",
        json={
            "prompt": "A simple test image",
            "format": "GIF"
        }
    )
    assert response.status_code == 400
    assert "Format must be 'JPEG' or 'PNG'" in response.json()["detail"]

def test_generate_image_api_invalid_quality():
    response = client.post(
        "/api/generate",
        json={
            "prompt": "A simple test image",
            "format": "JPEG",
            "quality": 101  # Out of range
        }
    )
    assert response.status_code == 400
    assert "Quality must be between 0 and 100" in response.json()["detail"]