# FastAPI Server for Bounding Box Detection

# Endpoints:
# - GET  /health             → Health check
# - POST /analyze            → Analyze uploaded image
# - POST /analyze/base64     → Analyze base64 encoded image

import base64
import logging
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bbox_detector import __version__
from bbox_detector.detector import analyze_bytes, DetectionResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bounding Box Detector API",
    description="Detect punctuation and other content in bounding boxes",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    version: str


class AnalysisResponse(BaseModel):
    result: str
    is_empty: bool
    is_punctuation: Optional[bool]
    confidence: float
    total_area: int
    component_count: int


class Base64Request(BaseModel):
    image: str  # Base64
    filename: Optional[str] = None


# Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs"""
    return {"message": "Bounding Box Detector API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version=__version__)


@app.post("/analyze", response_model=AnalysisResponse, tags=["Detection"])
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image file.

    Supported formats: PNG, JPG, JPEG, TIF, TIFF

    Returns detection result:
    - EMPTY: Bounding box is empty
    - FILLED_PUNCT: Contains punctuation mark
    - FILLED_OTHER: Contains other content (signature, etc.)
    """
    try:
        contents = await file.read()
        result = analyze_bytes(contents)

        logger.info(f"Analyzed {file.filename}: {result.result.value}")

        return AnalysisResponse(
            result=result.result.value,
            is_empty=result.is_empty,
            is_punctuation=result.is_punctuation,
            confidence=result.confidence,
            total_area=result.total_area,
            component_count=result.component_count
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze/base64", response_model=AnalysisResponse, tags=["Detection"])
async def analyze_base64(request: Base64Request):
    """
    Analyze a base64 encoded image.

    Request body:
    ```json
    {
        "image": "<base64_encoded_image>",
        "filename": "optional_filename.png"
    }
    ```
    """
    try:
        # Decode base64
        try:
            image_bytes = base64.b64decode(request.image)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Invalid base64 encoding")

        result = analyze_bytes(image_bytes)

        filename = request.filename or "base64_image"
        logger.info(f"Analyzed {filename}: {result.result.value}")

        return AnalysisResponse(
            result=result.result.value,
            is_empty=result.is_empty,
            is_punctuation=result.is_punctuation,
            confidence=result.confidence,
            total_area=result.total_area,
            component_count=result.component_count
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing base64 image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Run with: uvicorn bbox_detector.api.server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
