# 🚀 Deployment Guide

## Local Development

### Option 1: Run API Directly

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test API
python test_api.py

# Access API docs
open http://localhost:8000/docs
```

### Option 2: Run with Docker

```bash
# Build and run
docker-compose up --build

# Test API
python test_api.py

# Stop
docker-compose down
```

### Option 3: Run Gradio App

```bash
# Install dependencies
pip install -r requirements.txt

# Run Gradio
python app.py

# Access at http://localhost:7860
```

---

## Deploy to Hugging Face Spaces

### Step 1: Create Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `car-damage-detection`
4. SDK: `Gradio`
5. Hardware: `CPU basic` (free)

### Step 2: Push Code

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/car-damage-detection
cd car-damage-detection

# Copy files
cp ../app.py .
cp ../requirements.txt .
cp -r ../models .
cp -r ../test_results .

# Create README for HF
cat > README.md << 'EOF'
---
title: Car Damage Detection
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Car Damage Detection

YOLOv8 instance segmentation for automated vehicle damage detection.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
EOF

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### Step 3: Wait for Build

- HF will automatically build and deploy
- Check logs at: https://huggingface.co/spaces/YOUR_USERNAME/car-damage-detection/logs
- Access app at: https://huggingface.co/spaces/YOUR_USERNAME/car-damage-detection

---

## Deploy to Railway

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
railway login
```

### Step 2: Deploy

```bash
# Initialize
railway init

# Deploy
railway up

# Get URL
railway domain
```

---

## Deploy to Render

### Step 1: Create render.yaml

```yaml
services:
  - type: web
    name: car-damage-api
    env: docker
    plan: free
    healthCheckPath: /health
```

### Step 2: Deploy

1. Go to https://render.com
2. Connect GitHub repo
3. Select "Docker"
4. Deploy

---

## API Usage Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg" \
  -F "conf_threshold=0.25"
```

### Python

```python
import requests

# Predict
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        data={"conf_threshold": 0.25}
    )

result = response.json()
print(f"Detected {result['summary']['total_damages']} damages")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## Monitoring

### Check Logs

```bash
# Docker
docker-compose logs -f

# Local
tail -f logs/api.log
```

### Health Check

```bash
# Check if API is running
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true}
```

---

## Troubleshooting

### Model not found

```bash
# Ensure model exists
ls -lh models/best.pt

# Should be ~6.8 MB
```

### Port already in use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api.main:app --port 8001
```

### Out of memory

```bash
# Reduce batch size or use smaller model
# Edit api/main.py and reduce image size
```

---

## Performance Optimization

### 1. Use GPU (if available)

```python
# In api/main.py
model = YOLO('models/best.pt', device='cuda')
```

### 2. Enable Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_prediction(image_hash):
    # Cache predictions
    pass
```

### 3. Use Gunicorn (Production)

```bash
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## Security

### 1. Add Rate Limiting

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
    pass
```

### 2. Add Authentication

```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/predict")
async def predict(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify token
    pass
```

---

## Next Steps

1. ✅ Test locally
2. ✅ Deploy to Hugging Face Spaces
3. ⭐ Add monitoring (Prometheus)
4. ⭐ Add tests (pytest)
5. ⭐ Add CI/CD (GitHub Actions)
