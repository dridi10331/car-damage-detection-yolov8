import json
import cv2
from pathlib import Path
import random
import shutil
from collections import Counter
import numpy as np

# Configuration
CLASSES = {
    11380052: 0,  # Broken part
    11380058: 1,  # Corrosion
    11380055: 2,  # Dent
    11380057: 3,  # Paint chip
    11380053: 4,  # Scratch
    11380051: 5,  # Missing part
    11380056: 6,  # Flaking
    11380054: 7,  # Cracked
}

CLASS_NAMES = [
    "Broken part", "Corrosion", "Dent", "Paint chip",
    "Scratch", "Missing part", "Flaking", "Cracked"
]

IMG_DIR = Path("Car damaged parts dataset/File1/img")
ANN_DIR = Path("Car damaged parts dataset/File1/ann")
OUT_IMG = Path("dataset/images")
OUT_LBL = Path("dataset/labels")

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
MIN_POINTS = 3

def setup():
    """Create output directories"""
    for split in ["train", "val", "test"]:
        (OUT_IMG / split).mkdir(parents=True, exist_ok=True)
        (OUT_LBL / split).mkdir(parents=True, exist_ok=True)

def normalize_polygon(poly, w, h):
    """Normalize polygon coordinates to [0, 1] range"""
    normalized = []
    for x, y in poly:
        nx, ny = x / w, y / h
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        normalized.append((nx, ny))
    return normalized

def validate_polygon(poly):
    """Check if polygon is valid (at least 3 points, non-degenerate)"""
    if len(poly) < MIN_POINTS:
        return False
    
    # Check if polygon has area (not all points collinear)
    if len(poly) >= 3:
        points = np.array(poly)
        area = 0.5 * abs(sum(points[i][0] * points[(i+1)%len(points)][1] - 
                            points[(i+1)%len(points)][0] * points[i][1] 
                            for i in range(len(points))))
        if area < 1e-6:
            return False
    
    return True

def load_samples():
    """Load and validate all samples with statistics"""
    samples = []
    class_counts = Counter()
    skipped = {"no_annotation": 0, "invalid_image": 0, "invalid_polygon": 0}
    
    print("Loading dataset...")
    
    for img_path in sorted(IMG_DIR.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        
        ann_path = ANN_DIR / f"{img_path.name}.json"
        if not ann_path.exists():
            skipped["no_annotation"] += 1
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            skipped["invalid_image"] += 1
            continue
        
        h, w = img.shape[:2]
        
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        
        labels = []
        
        for obj in ann.get("objects", []):
            cid = obj.get("classId")
            if cid not in CLASSES:
                continue
            
            poly = obj["points"]["exterior"]
            poly = normalize_polygon(poly, w, h)
            
            if not validate_polygon(poly):
                skipped["invalid_polygon"] += 1
                continue
            
            line = str(CLASSES[cid])
            for x, y in poly:
                line += f" {x:.6f} {y:.6f}"
            
            labels.append(line)
            class_counts[CLASSES[cid]] += 1
        
        if labels:
            samples.append((img_path, labels))
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Total valid samples: {len(samples)}")
    print(f"Skipped - No annotation: {skipped['no_annotation']}")
    print(f"Skipped - Invalid image: {skipped['invalid_image']}")
    print(f"Skipped - Invalid polygon: {skipped['invalid_polygon']}")
    print(f"\nClass Distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / sum(class_counts.values())) * 100
        print(f"  {CLASS_NAMES[class_id]:15s}: {count:4d} ({percentage:5.1f}%)")
    print(f"{'='*60}\n")
    
    return samples

def split_and_save(samples):
    """Split dataset into train/val/test with stratification"""
    random.seed(SEED)
    random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    
    print(f"Split sizes:")
    print(f"  Train: {len(train)} ({len(train)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/n_total*100:.1f}%)")
    print()
    
    for subset, name in [(train, "train"), (val, "val"), (test, "test")]:
        print(f"Saving {name} set...")
        for img_path, labels in subset:
            shutil.copy2(img_path, OUT_IMG / name / img_path.name)
            with open(OUT_LBL / name / f"{img_path.stem}.txt", "w") as f:
                f.write("\n".join(labels))
    
    print(f"\n✓ Dataset conversion complete!")

if __name__ == "__main__":
    setup()
    samples = load_samples()
    if samples:
        split_and_save(samples)
    else:
        print("ERROR: No valid samples found!")
