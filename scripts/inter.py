from ultralytics import YOLO
import cv2
from pathlib import Path
import json
import numpy as np

# Configuration
MODEL_PATH = "models/best.pt"
TEST_DIR = Path("dataset/images/test")
OUTPUT_DIR = Path("inference_results")  # Output in project folder
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Damage severity mapping
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

def setup():
    """Create output directories"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "json").mkdir(exist_ok=True)

def load_model():
    """Load trained model with error handling"""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Please train the model first using: python scripts/train_seg.py"
        )
    
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    return model

def process_image(model, img_path):
    """Process single image and return results"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"✗ Could not load: {img_path}")
        return None
    
    # Run inference
    results = model(
        img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )[0]
    
    # Extract detections
    detections = []
    if results.masks is not None:
        for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get mask polygon
            mask_coords = mask.xy[0]  # Polygon coordinates
            
            detection = {
                "class": class_name,
                "confidence": round(confidence, 3),
                "severity": SEVERITY_MAP.get(class_name, "UNKNOWN"),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "area": int((x2 - x1) * (y2 - y1)),
                "mask_points": len(mask_coords)
            }
            detections.append(detection)
    
    return results, detections

def save_results(img_path, results, detections):
    """Save annotated image and JSON report"""
    # Save annotated image
    annotated = results.plot(
        conf=True,
        line_width=2,
        font_size=12,
        labels=True,
        boxes=True,
        masks=True
    )
    
    output_img = OUTPUT_DIR / "images" / img_path.name
    cv2.imwrite(str(output_img), annotated)
    
    # Save JSON report
    report = {
        "image": img_path.name,
        "total_damages": len(detections),
        "detections": detections,
        "severity_summary": {
            "CRITICAL": sum(1 for d in detections if d["severity"] == "CRITICAL"),
            "SEVERE": sum(1 for d in detections if d["severity"] == "SEVERE"),
            "MODERATE": sum(1 for d in detections if d["severity"] == "MODERATE"),
            "MINOR": sum(1 for d in detections if d["severity"] == "MINOR"),
        }
    }
    
    output_json = OUTPUT_DIR / "json" / f"{img_path.stem}.json"
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def print_summary(all_reports):
    """Print overall summary statistics"""
    total_images = len(all_reports)
    total_damages = sum(r["total_damages"] for r in all_reports)
    
    severity_totals = {
        "CRITICAL": sum(r["severity_summary"]["CRITICAL"] for r in all_reports),
        "SEVERE": sum(r["severity_summary"]["SEVERE"] for r in all_reports),
        "MODERATE": sum(r["severity_summary"]["MODERATE"] for r in all_reports),
        "MINOR": sum(r["severity_summary"]["MINOR"] for r in all_reports),
    }
    
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Images processed: {total_images}")
    print(f"Total damages detected: {total_damages}")
    print(f"Average damages per image: {total_damages/total_images:.1f}")
    print(f"\nSeverity Breakdown:")
    for severity, count in severity_totals.items():
        percentage = (count / total_damages * 100) if total_damages > 0 else 0
        print(f"  {severity:10s}: {count:3d} ({percentage:5.1f}%)")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Annotated images: {OUTPUT_DIR}/images/")
    print(f"  - JSON reports: {OUTPUT_DIR}/json/")
    print("="*60)

def main():
    """Main inference pipeline"""
    setup()
    model = load_model()
    
    # Get test images
    test_images = list(TEST_DIR.glob("*"))
    test_images = [p for p in test_images if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not test_images:
        print(f"✗ No test images found in: {TEST_DIR}")
        return
    
    print(f"\nProcessing {len(test_images)} test images...")
    print("="*60)
    
    all_reports = []
    
    for i, img_path in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] {img_path.name}...", end=" ")
        
        result = process_image(model, img_path)
        if result is None:
            continue
        
        results, detections = result
        report = save_results(img_path, results, detections)
        all_reports.append(report)
        
        print(f"✓ {len(detections)} damages detected")
    
    if all_reports:
        print_summary(all_reports)
    else:
        print("\n✗ No images processed successfully")

if __name__ == "__main__":
    main()
