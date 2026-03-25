from ultralytics import YOLO
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configuration
MODEL_PATH = "models/best.pt"
DATA_CONFIG = "config/dataset.yaml"
OUTPUT_DIR = Path("evaluation")  # Output in project folder

def setup():
    """Create output directory"""
    OUTPUT_DIR.mkdir(exist_ok=True)

def load_model():
    """Load trained model"""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    print(f"Loading model: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

def evaluate_model(model):
    """Run comprehensive evaluation on test set"""
    print("\n" + "="*60)
    print("Running Model Evaluation")
    print("="*60)
    
    # Validate on test set
    metrics = model.val(
        data=DATA_CONFIG,
        split="test",
        save_json=True,
        save_hybrid=True,
        conf=0.001,
        iou=0.6,
        max_det=300,
        plots=True,
        verbose=True
    )
    
    return metrics

def analyze_per_class(metrics):
    """Analyze performance per damage class"""
    print("\n" + "="*60)
    print("Per-Class Performance Analysis")
    print("="*60)
    
    # Extract per-class metrics
    class_names = metrics.names
    
    # Box detection metrics
    box_p = metrics.box.p  # Precision
    box_r = metrics.box.r  # Recall
    box_map50 = metrics.box.map50  # mAP@0.5
    box_map = metrics.box.map  # mAP@0.5:0.95
    
    # Mask segmentation metrics
    mask_p = metrics.seg.p
    mask_r = metrics.seg.r
    mask_map50 = metrics.seg.map50
    mask_map = metrics.seg.map
    
    # Create DataFrame
    results = []
    for i, name in class_names.items():
        # Handle both array and scalar metrics
        def get_metric(metric, idx):
            try:
                return metric[idx] if hasattr(metric, '__getitem__') else metric
            except:
                return metric
        
        results.append({
            "Class": name,
            "Box_Precision": f"{get_metric(box_p, i):.3f}",
            "Box_Recall": f"{get_metric(box_r, i):.3f}",
            "Box_mAP50": f"{get_metric(box_map50, i):.3f}",
            "Box_mAP": f"{get_metric(box_map, i):.3f}",
            "Mask_Precision": f"{get_metric(mask_p, i):.3f}",
            "Mask_Recall": f"{get_metric(mask_r, i):.3f}",
            "Mask_mAP50": f"{get_metric(mask_map50, i):.3f}",
            "Mask_mAP": f"{get_metric(mask_map, i):.3f}",
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR}/per_class_metrics.csv")
    
    return df

def print_summary(metrics):
    """Print overall summary"""
    print("\n" + "="*60)
    print("Overall Performance Summary")
    print("="*60)
    
    print("\nBox Detection:")
    print(f"  Precision:    {metrics.box.mp:.3f}")
    print(f"  Recall:       {metrics.box.mr:.3f}")
    print(f"  mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    
    print("\nMask Segmentation:")
    print(f"  Precision:    {metrics.seg.mp:.3f}")
    print(f"  Recall:       {metrics.seg.mr:.3f}")
    print(f"  mAP@0.5:      {metrics.seg.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.seg.map:.3f}")
    
    print("\nSpeed:")
    print(f"  Preprocess:   {metrics.speed['preprocess']:.1f} ms")
    print(f"  Inference:    {metrics.speed['inference']:.1f} ms")
    print(f"  Postprocess:  {metrics.speed['postprocess']:.1f} ms")
    
    print("="*60)

def save_summary_report(metrics, df):
    """Save comprehensive evaluation report"""
    report = {
        "model": str(MODEL_PATH),
        "overall_metrics": {
            "box_detection": {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
            },
            "mask_segmentation": {
                "precision": float(metrics.seg.mp),
                "recall": float(metrics.seg.mr),
                "mAP50": float(metrics.seg.map50),
                "mAP50_95": float(metrics.seg.map),
            },
        },
        "per_class_metrics": df.to_dict(orient="records"),
        "speed": {
            "preprocess_ms": float(metrics.speed['preprocess']),
            "inference_ms": float(metrics.speed['inference']),
            "postprocess_ms": float(metrics.speed['postprocess']),
        }
    }
    
    with open(OUTPUT_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved: {OUTPUT_DIR}/evaluation_report.json")

def main():
    """Main evaluation pipeline"""
    setup()
    model = load_model()
    metrics = evaluate_model(model)
    df = analyze_per_class(metrics)
    print_summary(metrics)
    save_summary_report(metrics, df)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
