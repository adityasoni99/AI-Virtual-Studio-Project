# VirtualAIStudio

A modular Python-based AI platform for creating virtual influencers, AI photoshoots, and virtual try-on experiences using Stable Diffusion XL (SDXL) and ControlNet-guided pipelines.

## What this is
AI Virtual Studio produces high-quality, photorealistic images and photoshoot-style renders from text prompts and optional control images (OpenPose/HED/depth/normal maps). It is intended for researchers and developers building virtual influencer assets, product photography, and virtual try-on workflows.

### Stack
- **Language(s):** Python (primary)
- **Framework / runtime:** FastAPI (serves the generation API)
- **Notable libraries:** torch, torchvision, diffusers, transformers, controlnet_aux
- Supporting tooling shown in requirements.txt: uvicorn, OpenCV (opencv-python), Pillow, mediapipe, numpy

## How it's organized
Top-level (important files/directories):
```
README.md               - This file
requirements.txt        - Python dependencies
backend/                - FastAPI application and API entrypoint
  main.py               - App start (uvicorn entry)
  api/                  - API endpoints (router included at /api)
docs/                   - Design & usage notes (features.md, api.md, validation.md)
examples/               - Input/control images and example assets (openpose, hed, normal maps)
output/                 - Generated outputs (images)
photoshoot_studio/      - Core image-generation modules (SDXL/refiner pipelines, exporters)
utils/                  - Utilities and helpers
tests/                  - pytest-based tests and test config
```

How it fits together:
- The FastAPI server defined in `backend/main.py` mounts the API router from `backend.api` at `/api`.
- Core generation logic, model loading, and export utilities live in `photoshoot_studio/` and are used by the API endpoints to run SDXL/refiner pipelines with optional ControlNet inputs.
- `examples/` contains sample control images (OpenPose, HED, depth, normal maps) for producing guided generations; `output/` holds generated images.

## Features (summary)
- Image generation with Stable Diffusion XL (SDXL) + optional refiner pipeline
  - Configurable resolution (64x64 up to 2048x2048; default 1024x1024)
  - Base pipeline: ~100 steps, guidance scale ~9.5
  - Refiner pipeline: ~30 steps (effective ~9 with strength=0.3), guidance ~7.5
- ControlNet support (OpenPose, HED/Normal, depth) with per-control-image weights
- Post-processing and export (JPEG/PNG) with configurable quality and basic color adjustments (brightness/saturation/contrast)
- Memory optimizations noted for large resolutions (model CPU offload, VAE slicing)

(See docs/features.md for full details and parameter examples.)

## API (exact parameters & validation)
Endpoint: POST /api/generate
- Content type: multipart/form-data

Form fields / parameters:
- prompt (string, required)
- negative_prompt (string, optional) — default: "blurry, low quality, dark, distorted, unrealistic"
- width (int, optional) — default: 1024, allowed range: 64–2048
- height (int, optional) — default: 1024, allowed range: 64–2048
- format (string, optional) — default: "JPEG", allowed values: "JPEG" or "PNG" (case-insensitive)
- quality (int, optional) — default: 90; when format is JPEG, must be between 0 and 100. Ignored for PNG.
- control_weights (string, optional) — JSON-encoded list of floats (e.g., "[0.9, 0.7]"). If provided, length must match number of control images.
- control_images (files, optional) — upload N image files; if provided, N must match initialized ControlNet models in the repository (default in code: ["openpose", "hed"]).

Behavior & validation (from backend/api/endpoints.py):
- `format` is validated to upper-case and must be either JPEG or PNG.
- `quality` validated only for JPEG to be between 0 and 100.
- `width` and `height` validated to be within 64–2048.
- If control images are provided, they must match the number of configured ControlNet models. If `control_weights` is provided it must match the number of control images.

Response:
- On success: returns the generated image file (FileResponse). Media type is `image/jpeg` for JPEG or `image/png` for PNG. Files are written to `output/generated_<id>.<ext>`.
- 400: Bad Request for validation errors (detailed message returned).
- 500: Internal Server Error for unexpected failures.

Example curl (multipart form-data):
```bash
curl -X POST "http://localhost:8000/api/generate" \
     -H "Content-Type: multipart/form-data" \
     -F "prompt=A superhero" \
     -F "width=512" \
     -F "height=768" \
     -F "format=JPEG" \
     -F "quality=75" \
     --output image.jpg
```

Notes from implementation (photoshoot_studio/image_generation/generate.py):
- Default model IDs: base `stabilityai/stable-diffusion-xl-base-1.0`, refiner `stabilityai/stable-diffusion-xl-refiner-1.0`.
- Supported ControlNets mapped in code: `openpose` -> `thibaud/controlnet-openpose-sdxl-1.0`, `hed`/`normal` -> `xinsir/controlnet-union-sdxl-1.0`.
- Device selection: CUDA > MPS > CPU. For non-CPU devices the code uses float16 and enables model CPU offload and VAE slicing to reduce peak memory.

## How to run (shortest path)
1. Clone and create a virtual environment:
```bash
git clone https://github.com/adityasoni99/AI-Virtual-Studio-Project.git
cd AI-Virtual-Studio-Project
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows (PowerShell)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
- Option A (recommended; uses uvicorn directly):
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Option B (run the module which calls uvicorn):
```bash
python backend/main.py
```

4. Generate an image (example curl shown above).

## Highlights
- Inference & deployment: End-to-end inference API using FastAPI + Uvicorn (backend/main.py, backend/api/endpoints.py) demonstrating production-serving patterns.
- PyTorch & GPU usage: Uses PyTorch-based diffusers pipelines (photoshoot_studio/image_generation/generate.py) with explicit device selection (CUDA/MPS/CPU) and fp16 usage.
- Memory/edge awareness: Model CPU offload, VAE slicing, dtype switching and other memory optimizations for constrained devices.
- Generative pipeline orchestration: SDXL base + refiner + ControlNet orchestration and post-processing pipelines provide experience with complex model orchestration.

## Examples & outputs
- Input control images and examples live in `examples/` (openpose_example.png, hed_example.png, normal_map_example.png, etc.) — helpful when testing ControlNet-guided generation.
- Generated images are stored under `output/` (sample images are already present and referenced in this repository).

## Tests
- pytest configuration is present (pytest.ini), and tests are in `tests/`. Run:
```bash
pytest
```

## Recent activity
- Latest recorded commit: 431afada0f0f (author: adityasoni99). For the most up-to-date activity, check the repository commits page.

## Contributing
See CONTRIBUTING.md for contribution guidelines.

## License
This repository is licensed under the MIT License — see LICENSE for details.

## Roadmap
See `docs/roadmap.md` for a prioritized roadmap to expand inference/runtime/agent capabilities and optimization work.
