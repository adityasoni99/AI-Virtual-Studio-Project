import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetPipeline, \
    ControlNetModel
from PIL import Image, ImageEnhance
from typing import List, Optional, Union


class ImageGenerator:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 controlnet_model_ids: Optional[List[str]] = None):
        """Initialize the SDXL model with refiner and optional ControlNet support, GPU-agnostic."""
        # Dynamically select device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()  # Clear CUDA memory
        elif torch.backends.mps.is_available():
            self.device = "mps"
            torch.mps.empty_cache()  # Clear MPS memory
        else:
            self.device = "cpu"
        print(f"Initialized with device: {self.device}")

        # ControlNet setup
        self.supported_controlnets = {
            "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
            "hed": "xinsir/controlnet-union-sdxl-1.0",
            "normal": "xinsir/controlnet-union-sdxl-1.0"
        }
        self.controlnet_model_ids = controlnet_model_ids or []

        # Load base pipeline (without ControlNet initially)
        self.base_pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,  # FP16 for GPU/MPS, FP32 for CPU
            variant="fp16",
            use_safetensors=True
        )
        self.base_pipe.enable_model_cpu_offload(device=self.device)
        self.base_pipe.enable_vae_slicing()

        # Load ControlNet models if provided
        self.controlnets = None
        if self.controlnet_model_ids:
            self.controlnets = [ControlNetModel.from_pretrained(
                self.supported_controlnets.get(cid, cid),
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            ) for cid in self.controlnet_model_ids]
            self.control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_id,
                controlnet=self.controlnets if len(self.controlnets) > 1 else self.controlnets[0],
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16",
                use_safetensors=True
            )
            self.control_pipe.enable_model_cpu_offload(device=self.device)
            self.control_pipe.enable_vae_slicing()

        # Default to base pipeline
        self.pipe = self.base_pipe
        print(f"Base pipeline device: {self.pipe.device}")

        # Refiner pipeline
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
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
            control_images: Optional[List[Image.Image]] = None,
            control_weights: Optional[List[float]] = None
    ) -> Image:
        """Generate and refine an image from a text prompt with optional ControlNet."""
        with torch.no_grad():
            if control_images and self.controlnets:
                if len(control_images) != len(self.controlnets):
                    raise ValueError(
                        f"Number of control images ({len(control_images)}) must match number of ControlNet models ({len(self.controlnets)})")
                self.pipe = self.control_pipe
                control_weights = control_weights or [1.0] * len(control_images)
                if len(control_weights) != len(control_images):
                    raise ValueError(
                        f"Number of control weights ({len(control_weights)}) must match number of control images ({len(control_images)})")

                initial_image = self.pipe(
                    prompt,
                    image=control_images,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=9.5,
                    negative_prompt=negative_prompt,
                    controlnet_conditioning_scale=control_weights
                ).images[0]
            else:
                self.pipe = self.base_pipe
                initial_image = self.pipe(
                    prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=9.5,
                    negative_prompt=negative_prompt
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

            # Clear device-specific memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
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
        image.save(filepath, format=format, quality=quality if format == "JPEG" else None)
        print(f"Image saved to {filepath}")


if __name__ == "__main__":
    generator = ImageGenerator(controlnet_model_ids=["openpose", "hed"])
    # Test code remains unchanged