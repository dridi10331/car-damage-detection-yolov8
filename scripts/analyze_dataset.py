"""
Dataset Analysis Tool
Provides comprehensive statistics and visualizations
"""

import json
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np

# Configuration
IMG_DIR = Path("Car damaged parts dataset/File1/img")
ANN_DIR = Path("Car damaged parts dataset/File1/ann")
OUTPUT_DIR = Path("dataset_analysis")

CLASSES = {
    11380052: "Broken part",
    11380058: "Corrosion",
    11380055: "Dent",
    11380057: "Paint chip",
    11380053: "Scratch",
    11380051: "Missing part",
    11380056: "Flaking",
    11380054: "Cracked",
}

def setup():
    """Create output directory"""
    OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_images():
    """Analyze image properties"""
    print("Analyzing images...")
    
    widths, heights, aspects = [], [], []
    formats = Counter()
    
    for img_path in IMG_DIR.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        aspects.append(w / h)
        formats[img_path.suffix.lower()] += 1
    
    return {
        "count": len(widths),
        "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths), "std": np.std(widths)},
        "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights), "std": np.std(heights)},
        "aspect_ratio": {"min": min(aspects), "max": max(aspects), "mean": np.mean(aspects), "std": np.std(aspects)},
        "formats": dict(formats),
    }

def analyze_annotations():
    """Analyze annotation properties"""
    print("Analyzing annotations...")
    
    class_counts = Counter()
    objects_per_image = []
    polygon_sizes = []
    class_polygons = defaultdict(list)
    
    for ann_path in ANN_DIR.glob("*.json"):
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        
        objects = ann.get("objects", [])
        objects_per_image.append(len(objects))
        
        for obj in objects:
            cid = obj.get("classId")
            if cid in CLASSES:
                class_counts[CLASSES[cid]] += 1
                
                poly = obj["points"]["exterior"]
                polygon_sizes.append(len(poly))
                class_polygons[CLASSES[cid]].append(len(poly))
    
    return {
        "total_annotations": len(list(ANN_DIR.glob("*.json"))),
        "total_objects": sum(class_counts.values()),
        "class_distribution": dict(class_counts),
        "objects_per_image": {
            "min": min(objects_per_image) if objects_per_image else 0,
            "max": max(objects_per_image) if objects_per_image else 0,
            "mean": np.mean(objects_per_image) if objects_per_image else 0,
            "std": np.std(objects_per_image) if objects_per_image else 0,
        },
        "polygon_complexity": {
            "min": min(polygon_sizes) if polygon_sizes else 0,
            "max": max(polygon_sizes) if polygon_sizes else 0,
            "mean": np.mean(polygon_sizes) if polygon_sizes else 0,
            "std": np.std(polygon_sizes) if polygon_sizes else 0,
        },
        "class_polygons": {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in class_polygons.items()},
    }

def plot_class_distribution(ann_stats):
    """Plot class distribution"""
    class_dist = ann_stats["class_distribution"]
    
    plt.figure(figsize=(12, 6))
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    
    colors = sns.color_palette("husl", len(classes))
    bars = plt.bar(classes, counts, color=colors)
    
    plt.xlabel("Damage Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
    print(f"✓ Saved: {OUTPUT_DIR}/class_distribution.png")

def plot_image_sizes(img_stats):
    """Plot image size distribution"""
    # This would require storing all widths/heights, simplified for now
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Width distribution
    axes[0].axvline(img_stats["width"]["mean"], color='r', linestyle='--', label=f'Mean: {img_stats["width"]["mean"]:.0f}')
    axes[0].set_xlabel("Width (pixels)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Image Width Distribution", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Height distribution
    axes[1].axvline(img_stats["height"]["mean"], color='r', linestyle='--', label=f'Mean: {img_stats["height"]["mean"]:.0f}')
    axes[1].set_xlabel("Height (pixels)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Image Height Distribution", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "image_sizes.png", dpi=300)
    print(f"✓ Saved: {OUTPUT_DIR}/image_sizes.png")

def print_summary(img_stats, ann_stats):
    """Print comprehensive summary"""
    print("\n" + "="*70)
    print("DATASET ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nImage Statistics:")
    print(f"  Total images: {img_stats['count']}")
    print(f"  Width:  {img_stats['width']['min']:.0f} - {img_stats['width']['max']:.0f} (mean: {img_stats['width']['mean']:.0f})")
    print(f"  Height: {img_stats['height']['min']:.0f} - {img_stats['height']['max']:.0f} (mean: {img_stats['height']['mean']:.0f})")
    print(f"  Aspect ratio: {img_stats['aspect_ratio']['mean']:.2f} ± {img_stats['aspect_ratio']['std']:.2f}")
    print(f"  Formats: {img_stats['formats']}")
    
    print("\nAnnotation Statistics:")
    print(f"  Total annotations: {ann_stats['total_annotations']}")
    print(f"  Total objects: {ann_stats['total_objects']}")
    print(f"  Objects per image: {ann_stats['objects_per_image']['mean']:.1f} ± {ann_stats['objects_per_image']['std']:.1f}")
    print(f"  Polygon complexity: {ann_stats['polygon_complexity']['mean']:.1f} ± {ann_stats['polygon_complexity']['std']:.1f} points")
    
    print("\nClass Distribution:")
    total = sum(ann_stats['class_distribution'].values())
    for cls, count in sorted(ann_stats['class_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"  {cls:15s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\nClass Balance:")
    counts = list(ann_stats['class_distribution'].values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 5:
        print("  ⚠ WARNING: Significant class imbalance detected!")
    
    print("="*70)

def save_report(img_stats, ann_stats):
    """Save analysis report to JSON"""
    report = {
        "images": img_stats,
        "annotations": ann_stats,
    }
    
    with open(OUTPUT_DIR / "dataset_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved: {OUTPUT_DIR}/dataset_analysis.json")

def main():
    """Main analysis pipeline"""
    setup()
    
    img_stats = analyze_images()
    ann_stats = analyze_annotations()
    
    print_summary(img_stats, ann_stats)
    
    plot_class_distribution(ann_stats)
    plot_image_sizes(img_stats)
    
    save_report(img_stats, ann_stats)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
