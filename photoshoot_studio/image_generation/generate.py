import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Initialize the Stable Diffusion model."""
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id)
        self.pipe = self.pipe.to(self.device)
        self.pipe.safety_checker = None  # Disable NSFW filter (optional)

    def generate_image(self, prompt: str, width: int = 512, height: int = 512) -> Image:
        """Generate an image from a text prompt."""
        with torch.no_grad():
            image = self.pipe(prompt, width=width, height=height).images[0]
        return image

    def save_image(self, image: Image, filepath: str):
        """Save the generated image to disk."""
        image.save(filepath)
        print(f"Image saved to {filepath}")

if __name__ == "__main__":
    generator = ImageGenerator()
    prompt = "A futuristic cityscape at sunset, photorealistic style"
    image = generator.generate_image(prompt)
    generator.save_image(image, "output/test_image.png")