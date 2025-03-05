from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from photoshoot_studio.image_generation.generate import ImageGenerator
from pydantic import BaseModel, field_validator, ValidationError
import os
import io
from typing import List, Optional
import json
from PIL import Image

router = APIRouter()
generator = ImageGenerator(controlnet_model_ids=["openpose", "hed"])


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, dark, distorted, unrealistic"
    width: int = 1024
    height: int = 1024
    format: str = "JPEG"
    quality: int = 90
    control_weights: Optional[List[float]] = None

    @field_validator("format")
    def validate_format(cls, v):
        if v.upper() not in ["JPEG", "PNG"]:
            raise ValueError("Format must be 'JPEG' or 'PNG'")
        return v.upper()

    @field_validator("quality")
    def validate_quality(cls, v, values):
        if "format" in values.data and values.data["format"] == "JPEG" and (v < 0 or v > 100):
            raise ValueError("Quality must be between 0 and 100 for JPEG")
        return v


@router.post("/generate", response_class=FileResponse)
async def generate_image(
        prompt: str = Form(...),
        negative_prompt: str = Form("blurry, low quality, dark, distorted, unrealistic"),
        width: int = Form(1024),
        height: int = Form(1024),
        format: str = Form("JPEG"),
        quality: int = Form(90),
        control_weights: Optional[str] = Form(None),
        control_images: List[UploadFile] = File(None)
):
    try:
        weights = json.loads(control_weights) if control_weights else None

        try:
            request = GenerateRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                format=format,
                quality=quality,
                control_weights=weights
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        control_images_list = None
        if control_images:
            if not generator.controlnet_model_ids:
                raise HTTPException(status_code=400, detail="ControlNet not initialized in ImageGenerator")
            control_images_list = [Image.open(io.BytesIO(await file.read())) for file in control_images]
            if len(control_images_list) != len(generator.controlnet_model_ids):
                raise HTTPException(status_code=400,
                                    detail=f"Number of control images ({len(control_images_list)}) must match ControlNet models ({len(generator.controlnet_model_ids)})")
            if request.control_weights and len(request.control_weights) != len(control_images_list):
                raise HTTPException(status_code=400,
                                    detail=f"Number of control weights ({len(request.control_weights)}) must match control images ({len(control_images_list)})")

        image = generator.generate_image(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            negative_prompt=request.negative_prompt,
            control_images=control_images_list if control_images_list else None,
            control_weights=request.control_weights
        )

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        ext = request.format.lower()
        filename = f"{output_dir}/generated_{id(image)}.{ext}"
        quality = request.quality if request.format == "JPEG" else None
        generator.save_image(image, filename, format=request.format, quality=quality if quality is not None else 90)

        media_type = "image/jpeg" if request.format == "JPEG" else "image/png"
        return FileResponse(filename, media_type=media_type, filename=os.path.basename(filename))
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")