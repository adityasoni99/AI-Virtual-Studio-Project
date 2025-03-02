import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI
from backend.api.endpoints import router as api_router

app = FastAPI(title="AI Virtual Studio API")

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    # Use import string for Uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)