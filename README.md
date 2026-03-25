# 🚗 Car Damage Detection System

**YOLOv8 Instance Segmentation for Automated Vehicle Damage Assessment**

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ahmedomar10/car-damage-detection)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://github.com/dridi10331/car-damage-detection-yolov8)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Deep learning system for detecting and classifying 8 types of vehicle damage with pixel-level segmentation and automatic severity assessment.

## 🎯 Key Features

- **Instance Segmentation**: Pixel-perfect damage localization with YOLOv8n-seg
- **8 Damage Classes**: Broken part, Missing part, Dent, Scratch, Cracked, Corrosion, Paint chip, Flaking
- **Severity Classification**: Automatic assessment (CRITICAL/SEVERE/MODERATE/MINOR)
- **Production Ready**: FastAPI backend, Docker deployment, live demo
- **Fast Inference**: 57ms per image on CPU

## 🚀 Quick Start

### Try the Live Demo
**[🤗 Hugging Face Space](https://huggingface.co/spaces/ahmedomar10/car-damage-detection)** - Upload an image and get instant results!

### Local Installation

```bash
# Clone repository
git clone https://github.com/dridi10331/car-damage-detection-yolov8.git
cd car-damage-detection-yolov8

# Install dependencies
pip install -r requirements.txt

# Run inference
python scripts/inter.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### API Usage

```bash
# Start FastAPI server
python -m uvicorn api.main:app --reload

# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -F "file=@car_image.jpg" \
  -F "conf_threshold=0.25"
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Overall mAP@0.5 | 13.1% |
| Critical Damage mAP | 44.3% |
| Inference Speed | 57ms (CPU) |
| Model Size | 6.8 MB |
| Parameters | 3.26M |

### Per-Class Performance

| Class | Severity | mAP@0.5 | Samples |
|-------|----------|---------|---------|
| Missing part | 🔴 CRITICAL | 44.3% | 632 |
| Broken part | 🔴 CRITICAL | 25.9% | 1,500 |
| Dent | 🟡 MODERATE | 15.2% | 1,664 |
| Cracked | 🟠 SEVERE | 7.5% | 76 |
| Scratch | 🟢 MINOR | 5.4% | 3,239 |
| Paint chip | 🟢 MINOR | 4.5% | 1,355 |
| Flaking | 🟡 MODERATE | 1.5% | 337 |
| Corrosion | 🟠 SEVERE | 0.3% | 277 |

## 🏗️ Project Structure

```
car-damage-detection-yolov8/
├── api/                          # FastAPI backend
│   └── main.py                  # API endpoints
├── scripts/                      # Training & inference
│   ├── convert_to_yolov8_seg.py # Dataset conversion
│   ├── train_seg.py             # Model training
│   ├── evaluate.py              # Evaluation
│   └── inter.py                 # Inference
├── models/                       # Trained weights
│   ├── best.pt                  # Best model (13.1% mAP)
│   └── last.pt                  # Last checkpoint
├── config/                       # Configuration files
│   └── dataset.yaml             # Dataset config
├── dataset/                      # YOLOv8 format data
│   ├── train/                   # 569 images (70%)
│   ├── val/                     # 122 images (15%)
│   └── test/                    # 123 images (15%)
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Usage

### 1. Dataset Preparation

```bash
# Convert Supervisely format to YOLOv8
python scripts/convert_to_yolov8_seg.py
```

### 2. Training

```bash
# Train model (50 epochs on CPU, ~2.4 hours)
python scripts/train_seg.py

# For GPU training, modify batch size in script
```

**Training Configuration:**
- Model: YOLOv8n-seg
- Epochs: 50
- Batch size: 4 (CPU) / 16 (GPU)
- Image size: 640x640
- Augmentation: Mosaic, mixup, copy-paste, HSV

### 3. Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py

# Outputs:
# - evaluation/evaluation_report.json
# - evaluation/per_class_metrics.csv
```

### 4. Inference

```bash
# Run detection on images
python scripts/inter.py

# Outputs:
# - inference_results/*.png (annotated images)
# - inference_results/*.json (detection results)
```

## 🌐 API Documentation

### Endpoints

#### `GET /`
Root endpoint with API information

#### `GET /health`
Health check endpoint

#### `POST /predict`
Detect damage in uploaded image

**Parameters:**
- `file`: Image file (JPG, PNG)
- `conf_threshold`: Confidence threshold (default: 0.25)
- `iou_threshold`: IoU threshold (default: 0.45)

**Response:**
```json
{
  "success": true,
  "filename": "car.jpg",
  "detections": [
    {
      "id": 1,
      "class": "Broken part",
      "confidence": 0.85,
      "severity": "CRITICAL",
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    }
  ],
  "summary": {
    "total_damages": 1,
    "by_severity": {"CRITICAL": 1, "SEVERE": 0, "MODERATE": 0, "MINOR": 0},
    "has_critical_damage": true
  }
}
```

#### `GET /model/info`
Get model information and statistics

### Interactive Documentation

Access Swagger UI at: **http://localhost:8000/docs**

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

### Using Docker

```bash
# Build image
docker build -t car-damage-api .

# Run container
docker run -p 8000:8000 car-damage-api
```

## 📦 Dependencies

```
ultralytics>=8.2.34
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
```

## 🎓 Model Details

**Architecture:** YOLOv8n-seg  
**Framework:** Ultralytics 8.4.14  
**Input Size:** 640x640  
**Parameters:** 3.26M  
**Training Data:** 814 images, 9,084 annotations  
**Training Time:** 2.4 hours (CPU) / 30-45 min (GPU)

## 🔍 Severity Classification

| Severity | Classes | Use Case |
|----------|---------|----------|
| 🔴 CRITICAL | Broken part, Missing part | Immediate repair required |
| 🟠 SEVERE | Cracked, Corrosion | Urgent attention needed |
| 🟡 MODERATE | Dent, Flaking | Repair recommended |
| 🟢 MINOR | Scratch, Paint chip | Cosmetic fixes |

## 📈 Improvement Recommendations

**Quick Wins:**
- Lower confidence threshold to 0.15 for better recall
- Train for 100+ epochs
- Use larger model (yolov8s-seg or yolov8m-seg)

**Data Improvements:**
- Collect more samples for rare classes (corrosion, flaking, cracked)
- Balance dataset with oversampling
- Add more aggressive augmentation for small damages

**Architecture:**
- Implement two-stage detection → classification
- Multi-scale training for different damage sizes
- Ensemble multiple models

## 🚀 Deployment Options

- **Hugging Face Spaces**: Live demo (already deployed)
- **Docker**: Local or cloud deployment
- **Railway**: One-click deployment
- **Render**: Free tier available
- **AWS/GCP/Azure**: Production deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **Dataset** from Supervisely format
- **Framework** built with FastAPI and PyTorch

## 📧 Contact

- **GitHub**: [@dridi10331](https://github.com/dridi10331)
- **Hugging Face**: [@ahmedomar10](https://huggingface.co/ahmedomar10)

---

**⭐ If you find this project useful, please consider giving it a star!**

