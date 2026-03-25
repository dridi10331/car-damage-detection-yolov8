"""
Gradio app for Car Damage Detection - Hugging Face Spaces
"""
import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json

# Load model
model = YOLO('models/best.pt')

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

SEVERITY_COLORS = {
    "CRITICAL": "🔴",
    "SEVERE": "🟠",
    "MODERATE": "🟡",
    "MINOR": "🟢"
}


def predict(image, conf_threshold, iou_threshold):
    """
    Predict car damage from image
    
    Args:
        image: PIL Image
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        Annotated image and JSON results
    """
    if image is None:
        return None, "Please upload an image"
    
    # Run inference
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    # Get annotated image
    annotated_image = results[0].plot()
    annotated_image = Image.fromarray(annotated_image)
    
    # Parse results
    detections = []
    severity_counts = {"CRITICAL": 0, "SEVERE": 0, "MODERATE": 0, "MINOR": 0}
    
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])
            severity = SEVERITY_MAP.get(cls_name, "UNKNOWN")
            
            detections.append({
                "id": i + 1,
                "class": cls_name,
                "confidence": f"{confidence:.2%}",
                "severity": severity
            })
            
            severity_counts[severity] += 1
    
    # Create summary text
    summary = f"""
## 🚗 Detection Summary

**Total Damages Detected:** {len(detections)}

### By Severity:
- 🔴 **CRITICAL:** {severity_counts['CRITICAL']} (Broken/Missing parts)
- 🟠 **SEVERE:** {severity_counts['SEVERE']} (Cracks/Corrosion)
- 🟡 **MODERATE:** {severity_counts['MODERATE']} (Dents/Flaking)
- 🟢 **MINOR:** {severity_counts['MINOR']} (Scratches/Paint chips)

### Detected Damages:
"""
    
    for det in detections:
        emoji = SEVERITY_COLORS[det['severity']]
        summary += f"\n{emoji} **{det['class']}** - {det['confidence']} confidence"
    
    if len(detections) == 0:
        summary = "✅ No damage detected in this image."
    
    # Create JSON output
    json_output = {
        "total_damages": len(detections),
        "severity_counts": severity_counts,
        "detections": detections
    }
    
    return annotated_image, summary, json.dumps(json_output, indent=2)


# Create Gradio interface
with gr.Blocks(title="Car Damage Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 Car Damage Detection System
    
    **YOLOv8 Instance Segmentation for Automated Vehicle Damage Detection**
    
    Upload an image of a damaged vehicle to detect and classify 8 types of damage with automatic severity assessment.
    
    ### Damage Classes:
    - 🔴 **CRITICAL:** Broken part, Missing part
    - 🟠 **SEVERE:** Cracked, Corrosion
    - 🟡 **MODERATE:** Dent, Flaking
    - 🟢 **MINOR:** Scratch, Paint chip
    
    ### Model Performance:
    - **Overall mAP@0.5:** 13.1%
    - **Critical Damage mAP:** 44.3%
    - **Inference Speed:** 57ms on CPU
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Car Image")
            
            with gr.Accordion("Advanced Settings", open=False):
                conf_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold"
                )
                iou_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="IoU Threshold"
                )
            
            predict_btn = gr.Button("🔍 Detect Damage", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Detected Damage")
            output_summary = gr.Markdown(label="Summary")
    
    with gr.Accordion("JSON Output", open=False):
        output_json = gr.Code(language="json", label="Detailed Results")
    
    # Examples
    gr.Examples(
        examples=[
            ["test_results/result_Car damages 1085.png", 0.25, 0.45],
            ["test_results/result_Car damages 1061.png", 0.25, 0.45],
        ],
        inputs=[input_image, conf_threshold, iou_threshold],
        outputs=[output_image, output_summary, output_json],
        fn=predict,
        cache_examples=False
    )
    
    # Connect button
    predict_btn.click(
        fn=predict,
        inputs=[input_image, conf_threshold, iou_threshold],
        outputs=[output_image, output_summary, output_json]
    )
    
    gr.Markdown("""
    ---
    ### 📊 Technical Details
    - **Model:** YOLOv8n-seg (3.26M parameters)
    - **Training Data:** 814 images, 8 damage classes
    - **Framework:** Ultralytics YOLOv8
    - **GitHub:** [car-damage-detection-yolov8](https://github.com/dridi10331/car-damage-detection-yolov8)
    
    ### 🔗 Links
    - [Documentation](https://github.com/dridi10331/car-damage-detection-yolov8#readme)
    - [API Endpoint](https://github.com/dridi10331/car-damage-detection-yolov8#api-usage)
    - [Model Weights](https://github.com/dridi10331/car-damage-detection-yolov8/tree/main/models)
    """)

# Launch
if __name__ == "__main__":
    demo.launch()
