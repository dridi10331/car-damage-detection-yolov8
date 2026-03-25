from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

print("="*60)
print("Car Damage Detection - Training")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("="*60)

# Configuration
DEVICE = "0" if torch.cuda.is_available() else "cpu"
DATA_CONFIG = "config/dataset.yaml"
MODEL_SIZE = "yolov8n-seg.pt"  # nano for CPU, use yolov8s-seg.pt for GPU
PROJECT = "training_output"  # Output in project folder
NAME = "damage_seg"

# Training hyperparameters optimized for segmentation
EPOCHS = 100 if torch.cuda.is_available() else 50
BATCH = 16 if torch.cuda.is_available() else 4
IMG_SIZE = 640
PATIENCE = 20

# Data augmentation (critical for small datasets)
AUGMENTATION = {
    'hsv_h': 0.015,      # Hue augmentation
    'hsv_s': 0.7,        # Saturation augmentation
    'hsv_v': 0.4,        # Value augmentation
    'degrees': 10.0,     # Rotation
    'translate': 0.1,    # Translation
    'scale': 0.5,        # Scale
    'shear': 0.0,        # Shear
    'perspective': 0.0,  # Perspective
    'flipud': 0.0,       # Flip up-down
    'fliplr': 0.5,       # Flip left-right
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.1,        # Mixup augmentation
    'copy_paste': 0.1,   # Copy-paste augmentation
}

# Verify dataset exists
if not Path(DATA_CONFIG).exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA_CONFIG}")

# Load model
print(f"\nLoading model: {MODEL_SIZE}")
model = YOLO(MODEL_SIZE)

# Train with optimized parameters
print(f"\nStarting training...")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH}")
print(f"  Image size: {IMG_SIZE}")
print(f"  Device: {DEVICE}")
print(f"  Patience: {PATIENCE}")
print("="*60 + "\n")

results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    
    # Optimization
    optimizer="AdamW",
    lr0=0.001,           # Initial learning rate
    lrf=0.01,            # Final learning rate (lr0 * lrf)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Regularization
    box=7.5,             # Box loss gain
    cls=0.5,             # Class loss gain
    dfl=1.5,             # DFL loss gain
    dropout=0.0,
    
    # Data augmentation
    **AUGMENTATION,
    
    # Training settings
    patience=PATIENCE,
    save=True,
    save_period=10,
    cache=False,         # Don't cache on CPU
    workers=2 if DEVICE == "cpu" else 8,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    pretrained=True,
    verbose=True,
    seed=42,
    deterministic=True,
    
    # Validation
    val=True,
    plots=True,
)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")
print(f"Results: {PROJECT}/{NAME}/results.csv")
print("="*60)
