import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageEnhance

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", controlnet_model_id=None):
        """Initialize the Stable Diffusion model with optional ControlNet support."""
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if controlnet_model_id:
            controlnet = ControlNetModel.from_pretrained(controlnet_model_id).to(self.device)
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id, controlnet=controlnet
            ).to(self.device)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        self.pipe.safety_checker = None  # Disable NSFW filter (optional)

    def generate_image(self, prompt: str, width: int = 512, height: int = 512, num_inference_steps: int = 50, control_image: Image = None) -> Image:
        """Generate an image from a text prompt, optionally with ControlNet conditioning."""
        with torch.no_grad():
            if control_image and isinstance(self.pipe, StableDiffusionControlNetPipeline):
                image = self.pipe(
                    prompt, image=control_image, width=width, height=height, num_inference_steps=num_inference_steps
                ).images[0]
            else:
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

    def adjust_color_balance(self, image: Image, factor: float = 1.0) -> Image:
        """Adjust color balance (0.0 = grayscale, 1.0 = original, >1.0 = more vibrant)."""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def adjust_saturation(self, image: Image, factor: float = 1.0) -> Image:
        """Adjust saturation (0.0 = grayscale, 1.0 = original, >1.0 = more saturated)."""
        enhancer = ImageEnhance.Color(image)
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
    # Example with ControlNet (pose control)
    controlnet_model_id = "lllyasviel/sd-controlnet-openpose"  # Pre-trained OpenPose ControlNet
    generator = ImageGenerator(controlnet_model_id=controlnet_model_id)
    prompt = "A futuristic cityscape at sunset with a person standing in the foreground, photorealistic style"
    # Load a sample pose image (you’ll need an OpenPose-generated image)
    control_image = Image.open("controlNet_demo_image.png").resize((512, 512))  # Placeholder; provide your own
    image = generator.generate_image(prompt, control_image=control_image, num_inference_steps=20)
    image = generator.adjust_brightness(image, factor=1.2)
    image = generator.adjust_saturation(image, factor=1.4)
    generator.save_image(image, "output/test_image_controlnet.jpg", format="JPEG", quality=85)