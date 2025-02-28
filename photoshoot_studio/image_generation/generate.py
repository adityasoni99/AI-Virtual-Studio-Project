import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageEnhance
from typing import List, Optional, Union


class ImageGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 controlnet_model_ids: Optional[List[str]] = None):
        """Initialize the Stable Diffusion model with optional ControlNet support."""
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        if controlnet_model_ids:
            # Load multiple ControlNet models
            controlnets = [ControlNetModel.from_pretrained(model_id).to(self.device) for model_id in
                           controlnet_model_ids]
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id, controlnet=controlnets if len(controlnets) > 1 else controlnets[0]
            ).to(self.device)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)

        self.pipe.safety_checker = None  # Disable NSFW filter (optional)

    def generate_image(
            self,
            prompt: str,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 50,
            control_images: Optional[Union[Image.Image, List[Image.Image]]] = None,
            control_weights: Optional[Union[float, List[float]]] = None
    ) -> Image:
        """Generate an image from a text prompt, optionally with ControlNet conditioning."""
        with torch.no_grad():
            if control_images and isinstance(self.pipe, StableDiffusionControlNetPipeline):
                # Ensure control_images is a list
                control_images = [control_images] if isinstance(control_images, Image.Image) else control_images
                # Handle control_weights: float for single ControlNet, list for multiple
                if control_weights is None:
                    control_weights = 1.0 if len(control_images) == 1 else [1.0] * len(control_images)
                elif isinstance(control_weights, float) and len(control_images) > 1:
                    control_weights = [control_weights] * len(control_images)
                elif isinstance(control_weights, list) and len(control_weights) != len(control_images):
                    raise ValueError(
                        f"Length of `control_weights` ({len(control_weights)}) must match length of `control_images` ({len(control_images)})."
                    )

                # Pass appropriate controlnet_conditioning_scale based on number of ControlNets
                conditioning_scale = control_weights if len(control_images) > 1 else control_weights[0] if isinstance(
                    control_weights, list) else control_weights

                image = self.pipe(
                    prompt,
                    image=control_images,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=conditioning_scale
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
    # Example with multiple ControlNets (Canny and Depth)
    controlnet_model_ids = [
        "lllyasviel/sd-controlnet-canny",
        "lllyasviel/sd-controlnet-depth"
    ]
    generator = ImageGenerator(controlnet_model_ids=controlnet_model_ids)

    # Load real example control images
    canny_image = Image.open("examples/canny_edge_example.png").resize((512, 512))
    depth_image = Image.open("examples/depth_map_example.png").resize((512, 512))

    prompt = "full-body, a young female, highlights in hair, dancing outside a restaurant, brown eyes, wearing jeans"
    image = generator.generate_image(
        prompt,
        control_images=[canny_image, depth_image],
        control_weights=[0.8, 0.6],
        num_inference_steps=20
    )
    image = generator.adjust_brightness(image, factor=1.2)
    image = generator.adjust_saturation(image, factor=1.4)
    generator.save_image(image, "output/test_image_multi_controlnet.jpg", format="JPEG", quality=85)