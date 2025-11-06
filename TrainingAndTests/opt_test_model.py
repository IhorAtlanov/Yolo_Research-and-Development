#!/usr/bin/env python3
"""
Optimized script for testing the performance of the YOLO model on video
Maximally optimized for speed without visualization
"""

import cv2
import time
import argparse
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path


def optimize_model(model):
    """Optimizing the model for inference"""
    # Switching to evaluation mode
    model.model.eval()

    if torch.cuda.is_available():
        model.model.cuda()
        print(f" GPU used: {torch.cuda.get_device_name()}")
    else:
        print(" CPU used")
    
    # Optimization for inference
    torch.backends.cudnn.benchmark = True
    return model


def test_yolo_video(video_path, model_path="./_.pt", conf_threshold=0.5):
    """
    Testing the YOLO model on video with maximum optimization

    Args:
        video_path: path to the video file
        model_path: path to the YOLO model
        conf_threshold: confidence threshold for detection
    """
    
    print(f" Model loading: {model_path}")

    model = YOLO(model_path)
    model = optimize_model(model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не вдалося відкрити відео: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Video: {Path(video_path).name}")
    print(f"   Personnel: {total_frames}")
    print(f"   FPS video: {video_fps:.1f}")
    print(f"   Confidence threshold: {conf_threshold}")
    print("\n Start of testing...\n")

    frame_times = []
    processed_frames = 0
    total_detections = 0

    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame processing start time
            frame_start = time.time()
            
            # Inference without saving visualization results
            with torch.no_grad():
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    verbose=False,
                    save=False,
                    show=False,
                    stream=False
                )
            
            # Detection count (optional)
            if results and len(results) > 0:
                total_detections += len(results[0].boxes) if results[0].boxes is not None else 0
            
            # Frame processing completion time
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            processed_frames += 1
            
            # Progress every 100 frames
            if processed_frames % 100 == 0:
                current_fps = 1.0 / np.mean(frame_times[-100:])
                print(f"Frame {processed_frames}/{total_frames} | FPS: {current_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n Testing interrupted by user")
    
    finally:
        cap.release()
    
    # Calculation of final metrics
    total_time = time.time() - start_time
    
    if frame_times:
        avg_frame_time = np.mean(frame_times)
        avg_fps = 1.0 / avg_frame_time
        min_frame_time = np.min(frame_times)
        max_frame_time = np.max(frame_times)
        std_frame_time = np.std(frame_times)
    else:
        avg_frame_time = avg_fps = min_frame_time = max_frame_time = std_frame_time = 0

    print("\n" + "="*60)
    print(" TEST RESULTS")
    print("="*60)
    print(f"Processed frames:           {processed_frames}")
    print(f"Total time:              {total_time:.2f} сек")
    print(f"Average FPS:               {avg_fps:.2f}")
    print(f"Average frame time:         {avg_frame_time*1000:.2f} ms")
    print(f"Minimum frame time:              {min_frame_time*1000:.2f} ms")
    print(f"Maximum frame time:             {max_frame_time*1000:.2f} ms")
    print(f"Standard deviation:      {std_frame_time*1000:.2f} ms")
    print(f"Total number of detections: {total_detections}")
    print(f"Average detection per frame:  {total_detections/processed_frames:.1f}")
    
    # Порівняння з реальним FPS відео
    if video_fps > 0:
        realtime_ratio = avg_fps / video_fps
        print(f"Real-time coefficient:  {realtime_ratio:.2f}x")
        if realtime_ratio >= 1.0:
            print(" Real-time processing achieved!")
        else:
            print(" Processing slower than real time")


def main():
    parser = argparse.ArgumentParser(description="YOLO video performance test")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-m", "--model", default="yolo.pt",
                       help="Path to the YOLO model")
    parser.add_argument("-c", "--conf", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f" Video file not found: {args.video}")
        return
    
    try:
        test_yolo_video(args.video, args.model, args.conf)
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    VIDEO_PATH = "./_.mp4"
    MODEL_PATH = "./_.pt"
    
    if Path(VIDEO_PATH).exists():
        test_yolo_video(VIDEO_PATH, MODEL_PATH, conf_threshold=0.5)
    else:
        print("To run the script:")
        print("python script.py your_video.mp4 -m yolo.pt -c 0.5")
        print("\nOr replace VIDEO_PATH in the code with your video file")