import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image, ImageEnhance
from typing import List, Optional, Union

class ImageGenerator:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0"):
        """Initialize the SDXL model with refiner and memory optimization."""
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initialized with device: {self.device}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.enable_model_cpu_offload(device=self.device)
        self.pipe.enable_vae_slicing()
        print(f"Base pipeline device: {self.pipe.device}")

        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.refiner.enable_model_cpu_offload(device=self.device)
        self.refiner.enable_vae_slicing()
        print(f"Refiner pipeline device: {self.refiner.device}")

    def generate_image(
            self,
            prompt: str,
            width: int = 1024,
            height: int = 1024,
            num_inference_steps: int = 100,
            negative_prompt: str = "blurry, low quality, dark, distorted, unrealistic",
            control_images: Optional[Union[Image.Image, List[Image.Image]]] = None,
            control_weights: Optional[Union[float, List[float]]] = None,
            control_types: Optional[List[str]] = None
    ) -> Image:
        """Generate and refine an image from a text prompt."""
        with torch.no_grad():
            initial_image = self.pipe(
                prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=9.5,
                negative_prompt=negative_prompt,
            ).images[0]

            refined_image = self.refiner(
                prompt,
                image=initial_image,
                num_inference_steps=30,
                strength=0.3,
                guidance_scale=7.5,
                negative_prompt=negative_prompt
            ).images[0]

            enhanced_image = self.adjust_brightness(refined_image, factor=1.25)
            enhanced_image = self.adjust_saturation(enhanced_image, factor=1.45)
            enhanced_image = self.adjust_contrast(enhanced_image, factor=1.1)

            torch.mps.empty_cache()
            return enhanced_image

    def adjust_contrast(self, image: Image, factor: float = 1.1) -> Image:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_brightness(self, image: Image, factor: float = 1.25) -> Image:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_saturation(self, image: Image, factor: float = 1.45) -> Image:
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def save_image(self, image: Image, filepath: str, format: str = "JPEG", quality: int = 90) -> None:
        format = format.upper()
        if format not in ["PNG", "JPEG"]:
            raise ValueError("Unsupported format. Use 'PNG' or 'JPEG'.")
        image.save(filepath, format=format, quality=quality)
        print(f"Image saved to {filepath}")

if __name__ == "__main__":
    generator = ImageGenerator()

    scenarios = [
        {
            "name": "portrait",
            "prompt": "A highly detailed superhero standing confidently, cinematic lighting, intricate costume design, photorealistic style",
            "negative": "blurry, low quality, dark, distorted, unrealistic, flat lighting"
        },
        {
            "name": "landscape",
            "prompt": "A futuristic cityscape at sunset with intricate details, vibrant colors, towering skyscrapers, photorealistic style",
            "negative": "blurry, low quality, dark, distorted, unrealistic, dull colors"
        },
        {
            "name": "group",
            "prompt": "A team of heroes posing together in a dynamic scene, detailed costumes, dramatic lighting, photorealistic style",
            "negative": "blurry, low quality, dark, distorted, unrealistic, misaligned poses"
        }
    ]

    for scenario in scenarios:
        image = generator.generate_image(scenario["prompt"], negative_prompt=scenario["negative"])
        generator.save_image(image, f"output/test_{scenario['name']}_sdxl_refined.jpg")