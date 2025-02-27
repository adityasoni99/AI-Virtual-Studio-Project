### Photoshoot Studio
- **Image Generation**: Generates photorealistic images from text prompts using Stable Diffusion.
- **Scene Editing**:
  - Brightness adjustment (factor: 0.0 to >1.0).
  - Contrast adjustment (factor: 0.0 to >1.0).
- **Export System**:
  - Supports PNG (lossless) and JPEG (with quality setting, 0-100).
  - Example: `save_image(image, "output.jpg", format="JPEG", quality=85)`.