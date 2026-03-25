"""
FastAPI backend for Car Damage Detection
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Car Damage Detection API",
    description="YOLOv8 instance segmentation for automated vehicle damage detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = Path("models/best.pt")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

logger.info(f"Loading model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
logger.info("Model loaded successfully")

# Severity mapping
SEVERITY_MAP = {
    "Broken part": "CRITICAL",
    "Missing part": "CRITICAL",
    "Cracked": "SEVERE",
    "Corrosion": "SEVERE",
    "Dent": "MODERATE",
    "Flaking": "MODERATE",
    "Paint chip": "MINOR",
    "Scratch": "MINOR",
}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Car Damage Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detect car damage in uploaded image
    
    Args:
        file: Image file (JPG, PNG)
        conf_threshold: Confidence threshold (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
    
    Returns:
        JSON with detections, severity, and confidence scores
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG are supported."
            )
        
        # Read and process image
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        
        # Run inference
        results = model.predict(
            image_np,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        severity_counts = {"CRITICAL": 0, "SEVERE": 0, "MODERATE": 0, "MINOR": 0}
        
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])
                severity = SEVERITY_MAP.get(cls_name, "UNKNOWN")
                
                # Get bounding box coordinates
                bbox = box.xyxy[0].tolist()
                
                detection = {
                    "id": i + 1,
                    "class": cls_name,
                    "confidence": round(confidence, 3),
                    "severity": severity,
                    "bbox": {
                        "x1": round(bbox[0], 2),
                        "y1": round(bbox[1], 2),
                        "x2": round(bbox[2], 2),
                        "y2": round(bbox[3], 2)
                    }
                }
                
                detections.append(detection)
                severity_counts[severity] += 1
        
        logger.info(f"Detected {len(detections)} damages")
        
        return {
            "success": True,
            "filename": file.filename,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "detections": detections,
            "summary": {
                "total_damages": len(detections),
                "by_severity": severity_counts,
                "has_critical_damage": severity_counts["CRITICAL"] > 0
            },
            "parameters": {
                "confidence_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "YOLOv8n-seg",
        "model_path": str(MODEL_PATH),
        "classes": model.names,
        "num_classes": len(model.names),
        "severity_mapping": SEVERITY_MAP
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
