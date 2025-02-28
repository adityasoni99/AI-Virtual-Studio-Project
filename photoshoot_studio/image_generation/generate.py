import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageEnhance
from typing import List, Optional, Union


class ImageGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 controlnet_model_ids: Optional[List[str]] = None):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.supported_controlnets = {
            "canny": "lllyasviel/sd-controlnet-canny",
            "depth": "lllyasviel/sd-controlnet-depth",
            "hed": "lllyasviel/sd-controlnet-hed",
            "openpose": "lllyasviel/sd-controlnet-openpose",
            "normal": "lllyasviel/sd-controlnet-normal"
        }
        if controlnet_model_ids:
            resolved_model_ids = [self.supported_controlnets.get(model_id, model_id) for model_id in
                                  controlnet_model_ids]
            controlnets = [ControlNetModel.from_pretrained(model_id).to(self.device) for model_id in resolved_model_ids]
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id, controlnet=controlnets if len(controlnets) > 1 else controlnets[0]
            ).to(self.device)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        self.pipe.safety_checker = None

    def generate_image(
            self,
            prompt: str,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 50,
            control_images: Optional[Union[Image.Image, List[Image.Image]]] = None,
            control_weights: Optional[Union[float, List[float]]] = None
    ) -> Image:
        with torch.no_grad():
            if control_images and isinstance(self.pipe, StableDiffusionControlNetPipeline):
                control_images = [control_images] if isinstance(control_images, Image.Image) else control_images
                if control_weights is None:
                    control_weights = 1.0 if len(control_images) == 1 else [1.0] * len(control_images)
                elif isinstance(control_weights, float) and len(control_images) > 1:
                    control_weights = [control_weights] * len(control_images)
                elif isinstance(control_weights, list) and len(control_weights) != len(control_images):
                    raise ValueError(
                        f"Length of `control_weights` ({len(control_weights)}) must match length of `control_images` ({len(control_images)})."
                    )
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
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image: Image, factor: float = 1.0) -> Image:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_color_balance(self, image: Image, factor: float = 1.0) -> Image:
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def adjust_saturation(self, image: Image, factor: float = 1.0) -> Image:
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def save_image(self, image: Image, filepath: str, format: str = "PNG", quality: int = 95) -> None:
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
    # Test scenarios with HED, OpenPose, and Normal ControlNets
    controlnet_model_ids = ["hed", "openpose", "normal"]
    generator = ImageGenerator(controlnet_model_ids=controlnet_model_ids)

    scenarios = [
        {
            "name": "portrait",
            "prompt": "A superhero standing confidently, photorealistic style",
            "weights": [0.7, 0.95, 0.5]  # Strong OpenPose for pose, moderate HED, light Normal
        },
        {
            "name": "landscape",
            "prompt": "A futuristic cityscape at sunset, photorealistic style",
            "weights": [0.9, 0.1, 0.8]  # Strong HED for edges, light OpenPose, moderate Normal
        },
        {
            "name": "group",
            "prompt": "A team of heroes posing together, photorealistic style",
            "weights": [0.5, 0.85, 0.6]  # Balanced HED, strong OpenPose, moderate Normal
        }
    ]

    for scenario in scenarios:
        control_images = [
            Image.open(f"examples/hed_{scenario['name']}.png").resize((512, 512)),
            Image.open(f"examples/openpose_{scenario['name']}.png").resize((512, 512)),
            Image.open(f"examples/normal_{scenario['name']}.png").resize((512, 512))
        ]
        image = generator.generate_image(
            scenario["prompt"],
            control_images=control_images,
            control_weights=scenario["weights"],
            num_inference_steps=20
        )
        image = generator.adjust_brightness(image, factor=1.2)
        image = generator.adjust_saturation(image, factor=1.4)
        generator.save_image(image, f"output/test_{scenario['name']}_enhanced.jpg", format="JPEG", quality=85)