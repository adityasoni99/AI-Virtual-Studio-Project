from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from photoshoot_studio.image_generation.generate import ImageGenerator
from pydantic import BaseModel
import os

router = APIRouter()
generator = ImageGenerator()


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, dark, distorted, unrealistic"
    width: int = 1024
    height: int = 1024
    format: str = "JPEG"  # Default to JPEG


@router.post("/generate", response_class=FileResponse)
async def generate_image(request: GenerateRequest):
    """Generate an image from a text prompt and return it in the specified format."""
    try:
        # Validate format early
        format_upper = request.format.upper()
        if format_upper not in ["JPEG", "PNG"]:
            raise HTTPException(status_code=400, detail="Format must be 'JPEG' or 'PNG'")

        # Generate image
        image = generator.generate_image(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            negative_prompt=request.negative_prompt
        )

        # Save image
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        ext = format_upper.lower()  # e.g., "jpeg" or "png"
        filename = f"{output_dir}/generated_{id(image)}.{ext}"
        generator.save_image(image, filename, format=format_upper)

        # Return file response
        media_type = "image/jpeg" if format_upper == "JPEG" else "image/png"
        return FileResponse(filename, media_type=media_type, filename=os.path.basename(filename))
    except HTTPException as e:
        # Ensure HTTPException propagates correctly
        raise e
    except Exception as e:
        # Log unexpected errors but keep them as 500
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")