# AI Virtual Studio Features

## Overview
AI Virtual Studio is a platform for generating high-quality, photorealistic images using advanced AI models. It leverages Stable Diffusion XL (SDXL) with a refiner pipeline for detailed outputs, optimized for performance on constrained hardware like the M3 Max with 36GB RAM.

## Photoshoot Studio
The core image generation module, providing photorealistic rendering capabilities.

- **Image Generation**:
  - **Model**: Stable Diffusion XL (SDXL) with refiner.
  - **Resolution**: Default 1024x1024, configurable via API.
  - **Base Pipeline**:
    - Steps: 100
    - Guidance Scale: 9.5
    - Negative Prompt: Tailored per scenario (e.g., "blurry, low quality, dark, distorted, unrealistic, flat lighting")
  - **Refiner Pipeline**:
    - Steps: 30 (~9 effective with `strength=0.3`)
    - Guidance Scale: 7.5
    - Strength: 0.3
  - **Post-Processing**:
    - Brightness: 1.25
    - Saturation: 1.45
    - Contrast: 1.1
  - **Memory Optimization**: Uses `enable_model_cpu_offload` and `enable_vae_slicing` (~15-20GB peak usage).

- **Export System**:
  - **Formats**: JPEG (lossy), PNG (lossless).
  - **Quality**: Configurable for JPEG (0-100), defaults to 90; ignored for PNG.
  - **Example**: `save_image(image, "output/test_image.jpg", format="JPEG", quality=75)`.

- **API**:
  - **Endpoint**: `POST /api/generate`
  - **Request**:
    - `prompt`: Text description (required).
    - `negative_prompt`: Optional, defaults to "blurry, low quality, dark, distorted, unrealistic".
    - `width`: Optional, defaults to 1024.
    - `height`: Optional, defaults to 1024.
    - `format`: Optional, defaults to "JPEG", options: "JPEG", "PNG".
    - `quality`: Optional, defaults to 90, range 0-100 (JPEG only).
  - **Response**: Generated image file in specified format and quality.
  - **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/generate" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "A superhero", "format": "JPEG", "quality": 75}' \
         --output image.jpg
  - **Status Codes**:
    - `200 OK`: Image generated successfully.
    - `400 Bad Request`: Invalid format or quality specified.
    - `500 Internal Server Error`: Unexpected generation failure.