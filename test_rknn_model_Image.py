from ultralytics import YOLO
import json
import os
from datetime import datetime

# Load the exported RKNN model
rknn_model = YOLO("./best_yolo11n_rknn_model")

# Input image path
image_path = "00001_TANK_test.jpg"

# Run inference
results = rknn_model(image_path)

# Create output directory if it doesn't exist
output_dir = "inference_results"
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = os.path.splitext(os.path.basename(image_path))[0]

# 1. Save annotated image with detections
output_image_path = os.path.join(output_dir, f"{base_name}_detected_{timestamp}.jpg")
results[0].save(output_image_path)
print(f"Annotated image saved to: {output_image_path}")

# 2. Save detection results to JSON file
detections_data = []
for result in results:
    for box in result.boxes:
        detection = {
            "class_id": int(box.cls[0]),
            "class_name": result.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": {
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3])
            }
        }
        detections_data.append(detection)

# Save to JSON
json_output_path = os.path.join(output_dir, f"{base_name}_detections_{timestamp}.json")
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump({
        "image_path": image_path,
        "timestamp": timestamp,
        "detections_count": len(detections_data),
        "detections": detections_data
    }, f, indent=2, ensure_ascii=False)
print(f"Detection results saved to: {json_output_path}")

# 3. Save detection results to text file
txt_output_path = os.path.join(output_dir, f"{base_name}_detections_{timestamp}.txt")
with open(txt_output_path, 'w', encoding='utf-8') as f:
    f.write(f"Image: {image_path}\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Total detections: {len(detections_data)}\n")
    f.write("-" * 50 + "\n")
    
    for i, detection in enumerate(detections_data, 1):
        f.write(f"Detection {i}:\n")
        f.write(f"  Class: {detection['class_name']} (ID: {detection['class_id']})\n")
        f.write(f"  Confidence: {detection['confidence']:.4f}\n")
        f.write(f"  Bounding box: ({detection['bbox']['x1']:.1f}, {detection['bbox']['y1']:.1f}, "
                f"{detection['bbox']['x2']:.1f}, {detection['bbox']['y2']:.1f})\n")
        f.write("\n")

print(f"Detection summary saved to: {txt_output_path}")

# 4. Print summary to console
print("\n=== DETECTION SUMMARY ===")
print(f"Image: {image_path}")
print(f"Total detections: {len(detections_data)}")

if detections_data:
    print("\nDetected objects:")
    for i, detection in enumerate(detections_data, 1):
        print(f"  {i}. {detection['class_name']} - {detection['confidence']:.4f}")
else:
    print("No objects detected.")

print(f"\nAll results saved in: {output_dir}/")