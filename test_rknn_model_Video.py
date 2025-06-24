from ultralytics import YOLO
import json
import os
import cv2
from datetime import datetime
from collections import defaultdict

# Load the exported RKNN model
rknn_model = YOLO("./best_yolo11n_rknn_model")

# Input video path
video_path = "./video_and_photo/BMP.MP4"  # Замініть на ваш відеофайл

# Create output directory if it doesn't exist
output_dir = "video_inference_results"
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = os.path.splitext(os.path.basename(video_path))[0]

# Output paths
output_video_path = os.path.join(output_dir, f"{base_name}_detected_{timestamp}.mp4")
json_output_path = os.path.join(output_dir, f"{base_name}_detections_{timestamp}.json")
txt_output_path = os.path.join(output_dir, f"{base_name}_summary_{timestamp}.txt")

print(f"Processing video: {video_path}")
print(f"Output video will be saved to: {output_video_path}")

# Open input video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Variables for tracking
frame_detections = []
detection_stats = defaultdict(int)
total_detections = 0
frame_count = 0

print("Processing frames...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on frame
        results = rknn_model(frame, verbose=False)
        
        # Process detections for current frame
        frame_data = {
            "frame_number": frame_count,
            "detections": []
        }
        
        frame_detection_count = 0
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = float(box.conf[0])
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                
                frame_data["detections"].append(detection)
                detection_stats[class_name] += 1
                frame_detection_count += 1
        
        total_detections += frame_detection_count
        frame_detections.append(frame_data)
        
        # Draw detections on frame
        annotated_frame = results[0].plot()
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Progress indicator
        if frame_count % 30 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")

finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("\nVideo processing completed!")
print(f"Processed {frame_count} frames")
print(f"Total detections: {total_detections}")

# Save detailed JSON results
print(f"Saving detailed results to: {json_output_path}")
video_results = {
    "video_path": video_path,
    "timestamp": timestamp,
    "video_info": {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": frame_count
    },
    "summary": {
        "total_detections": total_detections,
        "detection_stats": dict(detection_stats),
        "avg_detections_per_frame": total_detections / frame_count if frame_count > 0 else 0
    },
    "frame_detections": frame_detections
}

with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(video_results, f, indent=2, ensure_ascii=False)

# Save summary text file
print(f"Saving summary to: {txt_output_path}")
with open(txt_output_path, 'w', encoding='utf-8') as f:
    f.write("VIDEO DETECTION SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Video: {video_path}\n")
    f.write(f"Processed: {timestamp}\n")
    f.write(f"Output video: {output_video_path}\n\n")
    
    f.write("VIDEO INFORMATION:\n")
    f.write(f"Resolution: {width}x{height}\n")
    f.write(f"FPS: {fps}\n")
    f.write(f"Total frames: {total_frames}\n")
    f.write(f"Processed frames: {frame_count}\n")
    f.write(f"Duration: {total_frames/fps:.2f} seconds\n\n")
    
    f.write("DETECTION STATISTICS:\n")
    f.write(f"Total detections: {total_detections}\n")
    f.write(f"Average detections per frame: {total_detections/frame_count:.2f}\n\n")
    
    if detection_stats:
        f.write("DETECTED OBJECTS:\n")
        for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {class_name}: {count} times\n")
    else:
        f.write("No objects detected in video.\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Files saved in: {output_dir}/\n")

# Print final summary
print("\n=== VIDEO PROCESSING SUMMARY ===")
print(f"Input video: {video_path}")
print(f"Output video: {output_video_path}")
print(f"Frames processed: {frame_count}/{total_frames}")
print(f"Total detections: {total_detections}")
print(f"Average per frame: {total_detections/frame_count:.2f}")

if detection_stats:
    print("\nDetected objects:")
    for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} times")

print(f"\nAll results saved in: {output_dir}/")