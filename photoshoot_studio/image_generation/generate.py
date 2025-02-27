import torch
import os
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance


class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Initialize the Stable Diffusion model."""
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id)
        self.pipe = self.pipe.to(self.device)
        self.pipe.safety_checker = None  # Disable NSFW filter (optional)

    def generate_image(self, prompt: str, width: int = 512, height: int = 512, num_inference_steps: int = 50) -> Image:
        """Generate an image from a text prompt."""
        with torch.no_grad():
            image = self.pipe(prompt, width=width, height=height, num_inference_steps=num_inference_steps).images[0]
        return image

    def adjust_brightness(self, image: Image, factor: float = 1.0) -> Image:
        """Adjust image brightness (0.0 = black, 1.0 = original, >1.0 = brighter)."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image: Image, factor: float = 1.0) -> Image:
        """Adjust image contrast (0.0 = gray, 1.0 = original, >1.0 = higher contrast)."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def save_image(self, image: Image, filepath: str, format: str = "PNG", quality: int = 95) -> None:
        """Save the generated image with custom format and quality."""
        format = format.upper()
        if format not in ["PNG", "JPEG"]:
            raise ValueError("Unsupported format. Use 'PNG' or 'JPEG'.")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format == "JPEG":
            image.save(filepath, format="JPEG", quality=quality)
        else:
            image.save(filepath, format="PNG")
        print(f"Image saved to {filepath} as {format}")


if __name__ == "__main__":
    generator = ImageGenerator()
    prompt = "A futuristic cityscape at sunset, photorealistic style"
    image = generator.generate_image(prompt)
    image = generator.adjust_brightness(image, factor=1.2)
    image = generator.adjust_contrast(image, factor=1.3)
    generator.save_image(image, "output/test_image_enhanced.jpg", format="JPEG", quality=85)