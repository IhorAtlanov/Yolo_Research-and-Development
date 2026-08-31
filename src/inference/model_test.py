"""
Script for testing the performance of a YOLO model on images and videos.
Supports processing individual images, video files, and webcam streams.
Additional features:
- Processing every nth frame to improve performance
- Saving individual frames with detections as images
- Displaying the average processing time for all frames
- Monitoring memory usage (RAM and VRAM)
- Extended statistics and metrics

Usage:
    # For image processing:
    python model_test.py --source path/to/image.jpg --model path/to/best.pt

    # For video processing:
    python model_test.py --source path/to/video.mp4 --model path/to/best.pt

    # For webcam capture:
    python model_test.py --source 0 --model path/to/best.pt

    # Additional options:
    --conf 0.25           # Confidence threshold (default 0.25)
    --iou 0.45            # IoU threshold for NMS (default 0.45)
    --device 0            # Inference device (CPU: 'cpu', GPU: 0,1,2...)
    --save-results        # Save video results
    --output-dir results  # Directory to save video results
    --frame-skip 5        # Process every nth frame (default 5)
    --save-frames         # Save individual frames with detections
    --frames-dir frames   # Directory to save frames
    --memory-monitor      # Detailed memory monitoring
"""

import os
import gc
import cv2
import time
import torch
import psutil
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt


class MemoryMonitor:
    def __init__(self, device='cpu'):
        self.device = device
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available() and device != 'cpu'
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self):
        """Get current memory usage."""
        memory_info = {}

        ram_info = self.process.memory_info()
        memory_info['ram_mb'] = ram_info.rss / 1024 / 1024
        memory_info['ram_percent'] = self.process.memory_percent()
        peak_mb = None
        if hasattr(ram_info, 'peak_wset'):
            peak_mb = ram_info.peak_wset / 1024 / 1024
        elif hasattr(ram_info, 'rss'):
            peak_mb = ram_info.rss / 1024 / 1024
        memory_info['ram_peak_mb'] = peak_mb

        if self.gpu_available:
            try:
                memory_info['vram_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                memory_info['vram_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
                memory_info['vram_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            except Exception:
                memory_info['vram_allocated_mb'] = 0
                memory_info['vram_cached_mb'] = 0
                memory_info['vram_max_allocated_mb'] = 0

        return memory_info

    def get_memory_delta(self):
        """Get memory change since initialization."""
        current = self.get_memory_usage()
        delta = {}
        for key in current:
            if key in self.initial_memory:
                delta[f"delta_{key}"] = current[key] - self.initial_memory[key]
            else:
                delta[f"delta_{key}"] = current[key]
        return current, delta

    def print_memory_info(self, title="Memory usage"):
        """Print memory usage information."""
        current, delta = self.get_memory_delta()
        print(f"\n--- {title} ---")
        print(f"RAM: {current['ram_mb']:.1f} MB ({current['ram_percent']:.1f}%)")
        print(f"RAM change: {delta['delta_ram_mb']:+.1f} MB")
        print(f"RAM peak: {current['ram_peak_mb']:.1f} MB")
        print(f"RAM peak change: {delta['delta_ram_peak_mb']:+.1f} MB")

        if self.gpu_available:
            print(f"VRAM allocated: {current['vram_allocated_mb']:.1f} MB")
            print(f"VRAM cached: {current['vram_cached_mb']:.1f} MB")
            print(f"VRAM peak: {current['vram_max_allocated_mb']:.1f} MB")
            print(f"VRAM change: {delta['delta_vram_allocated_mb']:+.1f} MB")


class PerformanceTracker:
    """Class for tracking performance metrics."""

    def __init__(self):
        self.processing_times = []
        self.confidence_scores = []
        self.detection_counts = []
        self.start_time = None
        self.last_processing_time = 0

    def start_timing(self):
        self.start_time = time.perf_counter()

    def end_timing(self, detections=None):
        if self.start_time is None:
            return 0
        processing_time = time.perf_counter() - self.start_time
        self.processing_times.append(processing_time)
        self.last_processing_time = processing_time

        if detections:
            self.detection_counts.append(len(detections))
            confidences = [d['confidence'] for d in detections]
            self.confidence_scores.extend(confidences)
        else:
            self.detection_counts.append(0)

        self.start_time = None
        return processing_time

    def get_avg_processing_time(self):
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0

    def get_instant_fps(self):
        return 1 / self.last_processing_time if self.last_processing_time > 0 else 0

    def get_avg_fps(self):
        avg_time = self.get_avg_processing_time()
        return 1 / avg_time if avg_time > 0 else 0

    def get_stats(self):
        stats = {
            'last_processing_time': self.last_processing_time,
            'avg_processing_time': self.get_avg_processing_time(),
            'instant_fps': self.get_instant_fps(),
            'avg_fps': self.get_avg_fps(),
            'total_detections': sum(self.detection_counts),
            'avg_detections_per_frame': sum(self.detection_counts) / len(self.detection_counts) if self.detection_counts else 0,
            'avg_confidence': sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
            'min_confidence': min(self.confidence_scores) if self.confidence_scores else 0,
            'max_confidence': max(self.confidence_scores) if self.confidence_scores else 0
        }
        return stats


def get_model_info(model):
    """Retrieve model information."""
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / 1024 / 1024
        info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'model_type': type(model.model).__name__
        }
        return info
    except Exception as e:
        print(f"Error getting model info: {e}")
        return {}


def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO tank detector (enhanced version)')

    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or webcam index (0, 1, 2...)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model (.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default 0.45)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device for inference (CPU: "cpu", GPU: 0,1,2...)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save video results')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save video results')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Display results')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every nth frame (default 1 - no skipping)')
    parser.add_argument('--save-frames', action='store_true',
                        help='Save individual frames with detections')
    parser.add_argument('--frames-dir', type=str, default='frames',
                        help='Directory to save frames')
    parser.add_argument('--memory-monitor', action='store_true',
                        help='Detailed memory monitoring')
    
    return parser.parse_args()

def process_image(args, memory_monitor=None):
    print("Model loading...")
    model = YOLO(args.model) 

    model_info = get_model_info(model) 
    if model_info: 
        print("\n--- Model information ---")
        print(f"Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"Total number of parameters: {model_info.get('total_params', 0):,}")
        print(f"Learning parameters: {model_info.get('trainable_params', 0):,}")
        print(f"Model size: {model_info.get('model_size_mb', 0):.1f} MB")
     
    if memory_monitor: 
        memory_monitor.print_memory_info("After loading the model")

    if not os.path.exists(args.source): 
        print(f"Error: Path {args.source} not found")
        return None

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    if os.path.isfile(args.source):
        image_files = [args.source]
    elif os.path.isdir(args.source):
        for file in os.listdir(args.source):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(args.source, file))
    
    if not image_files:
        print("Error: No images found for processing")
        return None
    
    print(f"Images found for processing: {len(image_files)}")
    
    # Warming up the model [WARMUP] in the first image
    warmup_iters = getattr(args, 'warmup_iters', 5)
    if image_files:
        first_img = image_files[0]
        img = cv2.imread(first_img)
        if img is not None:
            print(f"[WARMUP] Let's do it {warmup_iters} non-public inferences for warming up the model in the image {first_img}...")
            for _ in range(warmup_iters):
                _ = model.predict(img, conf=args.conf, iou=args.iou, verbose=False)
            print("[WARMUP] Warm-up complete, moving on to image processing.")
        else:
            print(f"[WARMUP] Failed to load image {first_img}; skip warmup.")

    # Common variables for statistics
    all_processing_times = []
    all_detections_count = []
    all_confidences = []
    processed_images = 0
    
    # Processing each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\nImage processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Creating a performance tracker for each image
            tracker = PerformanceTracker()
            
            # Inference
            tracker.start_timing()
            results = model.predict(image_path, conf=args.conf, iou=args.iou)
            
            # Obtaining an image with detections
            result_image = results[0].plot()
            
            # Creating a list of detections
            boxes = results[0].boxes
            detections = [
                {
                    'box': box.xyxy.cpu().numpy()[0],
                    'confidence': box.conf.cpu().numpy()[0],
                    'class': box.cls.cpu().numpy()[0]
                }
                for box in boxes
            ]
            
            tracker.end_timing(detections)
            stats = tracker.get_stats()
            
            # Statistics collection
            all_processing_times.append(stats['avg_processing_time'])
            all_detections_count.append(len(detections))
            
            if detections:
                confidences = [det['confidence'] for det in detections]
                all_confidences.extend(confidences)
            
            print(f"  Objects found: {len(detections)}")
            print(f"  Processing time: {stats['avg_processing_time']:.4f} seconds")
            
            # Adding information to images
            info_text = f"Objects: {len(detections)}  Time: {stats['avg_processing_time']:.3f}s"
            cv2.putText(result_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Preserving results
            if args.save_results:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(
                    args.output_dir, 
                    f"detection_{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(output_path, result_image)
                print(f"  Result saved in {output_path}")
            
            processed_images += 1
            
        except Exception as e:
            print(f"  Error during processing {image_path}: {str(e)}")
            continue
    
    # Calculation and derivation of average metrics
    if processed_images > 0:
        avg_processing_time = sum(all_processing_times) / len(all_processing_times)
        avg_detections = sum(all_detections_count) / len(all_detections_count)
        total_detections = sum(all_detections_count)
        
        print(f"\n--- Overall results for {processed_images} images ---")
        print(f"Average processing time: {avg_processing_time:.4f} seconds")
        print(f"Average number of objects per image: {avg_detections:.1f}")
        print(f"Total number of objects detected: {total_detections}")
        
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            min_confidence = min(all_confidences)
            max_confidence = max(all_confidences)
            print(f"Average confidence across all detections: {avg_confidence:.3f}")
            print(f"Min/max confidence: {min_confidence:.3f}/{max_confidence:.3f}")
        else:
            print("No objects found")
    else:
        print("No image was successfully processed")
        
    if memory_monitor: 
        memory_monitor.print_memory_info("After processing all images")
    
    return processed_images, total_detections if processed_images > 0 else 0

def process_video(args, memory_monitor=None):
    """Processing video or webcam streams with frame skipping and frame saving support"""
    print("Model loading...")
    model = YOLO(args.model)

    model_info = get_model_info(model)
    if model_info:
        print("\n--- Model information ---")
        print(f"Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"Total number of parameters: {model_info.get('total_params', 0):,}")
        print(f"Model size: {model_info.get('model_size_mb', 0):.1f} MB")
    
    if memory_monitor:
        memory_monitor.print_memory_info("After loading the model")
    
    # Identifying the source of the video
    try:
        if args.source.isdigit():
            cap = cv2.VideoCapture(int(args.source))
            source_name = f"webcam {args.source}"
        else:
            if not os.path.exists(args.source):
                print(f"Error: File {args.source} not found")
                return
            cap = cv2.VideoCapture(args.source)
            source_name = os.path.basename(args.source)
    except Exception as e:
        print(f"Error opening video: {e}")
        return

    if not cap.isOpened():
        print("Error: Unable to open video")
        return
    
    # Obtaining information about videos
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video processing: {source_name}, size: {width}x{height}, FPS: {fps:.1f}")
    if total_frames > 0:
        print(f"Total number of frames: {total_frames}")
    print(f"Processing of each {args.frame_skip}-his staff")
    
    # Preparing a directory for storing personnel records
    if args.save_frames:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        frames_path = os.path.join(
            args.frames_dir, 
            f"{Path(args.source).stem if not args.source.isdigit() else 'webcam'}_{timestamp}"
        )
        os.makedirs(frames_path, exist_ok=True)
        print(f"The footage will be stored in {frames_path}")
    
    # Preparation for recording the result
    video_writer = None
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir, 
            f"detection_{Path(args.source).stem if not args.source.isdigit() else 'webcam'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        effective_fps = max(1, fps / args.frame_skip)
        video_writer = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))

    tracker = PerformanceTracker()
    
    # === Logging settings ===
    # Create a directory for storing logs (if it does not already exist)
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    # Forming the log file name with a timestamp
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(logs_dir, f"video_log_{log_timestamp}.csv")
    # Open the file for writing (in CSV format)
    log_file = open(log_filepath, "w", encoding="utf-8")
    # Write down the header (column names)
    log_file.write("Timestamp,Frame,ProcessingTime,InstantFPS,Detections,AvgConfidence,MinConfidence,MaxConfidence,AvgBoxArea,FrameRes,InputRes,SkippedFrames\n")
    # === End of logging configuration ===

    # [WARMUP]
    ret, warmup_frame = cap.read()
    if ret:
        # Number of warm-up iterations (can be configured via args or hardcoded)
        warmup_iters = 5
        print(f"[WARMUP] Let's do it {warmup_iters} non-public inferences for model warm-up...")
        for _ in range(warmup_iters):
            _ = model.predict(warmup_frame, conf=args.conf, iou=args.iou, verbose=False)
        # We rewind the video back to the beginning so as not to lose the frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("[WARMUP] Warm-up complete, returning to video processing.")
    else:
        print("[WARMUP] Failed to get the first frame for warmup; skipping warmup.")

    #! Synchronising the GPU before starting the measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Variables for tracking speed and statistics
    frame_count = 0
    processed_frames = 0
    skipped_frames = 0
    start_time_total = time.perf_counter()
    last_stats_time = time.perf_counter()

    #DEBUG:
    timesINF = []

    # Frame processing
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Frame skipping
            for _ in range(args.frame_skip - 1):
                if not cap.grab():
                    break
                skipped_frames += 1
            
            processed_frames += 1
            
            # Standard inference
            tracker.start_timing()
            results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
            result_frame = results[0].plot()
            
            # Creating a list of detections
            boxes = results[0].boxes
            detections = [
                {
                    'box': box.xyxy.cpu().numpy()[0],
                    'confidence': box.conf.cpu().numpy()[0],
                    'class': box.cls.cpu().numpy()[0]
                }
                for box in boxes
            ]

            processing_time = tracker.end_timing(detections)
            instant_fps = tracker.get_instant_fps()
            det_count = len(detections)

            confidences_frame = [d['confidence'] for d in detections]
            avg_conf_frame = sum(confidences_frame) / det_count if det_count > 0 else 0.0
            max_conf = max(confidences_frame) if confidences_frame else 0.0
            min_conf = min(confidences_frame) if confidences_frame else 0.0

            # Calculation of the average area of objects
            areas = [
                (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1])
                for d in detections
            ]
            avg_area = sum(areas) / det_count if det_count > 0 else 0.0

            # Frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Input image dimensions for the model (assuming that model.predict(...) returns a scaled result)
            input_resolution = results[0].orig_shape  # (height, width)

            log_time = datetime.now().isoformat(sep=' ', timespec='seconds')
            log_file.write(
                f"{log_time},"  # Timestamp
                f"{(processed_frames - 1) * args.frame_skip + 1},"  # Frame
                f"{processing_time:.4f},"  # ProcessingTime
                f"{instant_fps:.2f},"  # InstantFPS
                f"{det_count},"  # Detections
                f"{avg_conf_frame:.3f},"  # AvgConfidence
                f"{min_conf:.3f},"  # MinConfidence
                f"{max_conf:.3f},"  # MaxConfidence
                f"{avg_area:.1f},"  # AvgBoxArea
                f"{frame_width}x{frame_height},"  # FrameRes
                f"{input_resolution[1]}x{input_resolution[0]},"  # InputRes
                f"{skipped_frames}"  # SkippedFrames
                "\n"
            )
            
            # Obtaining current statistics
            current_stats = tracker.get_stats()
                
            # Statistics displayed every 5 seconds
            current_time = time.perf_counter()
            if current_time - last_stats_time > 5.0:
                if memory_monitor:
                    memory_monitor.print_memory_info(f"Frame {frame_count}")
                print(f"Frame {frame_count}: FPS={current_stats['instant_fps']:.1f}, "
                      f"Objects={len(detections)}, "
                      f"Processing time={current_stats['avg_processing_time']:.4f}с")
                last_stats_time = current_time
            
            real_index = (processed_frames - 1)*args.frame_skip + 1

            # Adding information to a frame
            info_text = f"Frame: {real_index} | Objects: {len(detections)} | FPS: {current_stats['instant_fps']:.1f}"
            cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Saving a frame as an image
            if args.save_frames:
                if len(detections) > 0:
                    frame_path = os.path.join(
                        frames_path,
                        f"frame_{real_index:06d}_det{len(detections)}.jpg"
                    )
                else:
                    frame_path = os.path.join(
                        frames_path,
                        f"frame_{real_index:06d}_no_det.jpg"
                    )
                cv2.imwrite(frame_path, result_frame)
      
            # Recording the result in a video
            if video_writer:
                video_writer.write(result_frame)
                
            # Displaying results
            if args.show:
                cv2.imshow("Result", result_frame)
                
                # Exit by pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            #DEBUG:
            start = time.perf_counter()
            model.predict(frame)
            time_taken = (time.perf_counter() - start)*1000
            #print("Model-only inference time:", time_taken, "ms")
            timesINF.append(time_taken)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Close the log file (very important!)
        log_file.close()

        # Calculation of final statistics
        total_time = time.perf_counter() - start_time_total
        final_stats = tracker.get_stats()
        
        # Freeing up resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Calculating actual speed
        effective_fps = processed_frames / total_time if total_time > 0 else 0
        
        # Final statistics
        print("\n" + "="*50)
        print("VIDEO PROCESSING RESULTS")
        print("="*50)
        print(f"Total number of frames: {frame_count:,}")
        print(f"Processed frames: {processed_frames:,}")
        print(f"Missing frames: {skipped_frames:,}")
        print(f"Objects found: {final_stats['total_detections']:,}")
        print(f"Average number of objects per frame: {final_stats['avg_detections_per_frame']:.2f}")
        
        print("\nDEBUG:")
        print(f"Total time: {total_time:.3f} sec")
        print(f"Processed frames: {processed_frames}")
        print(f"FPS calculation: {processed_frames / total_time:.2f}")
        print(f"Time per frame: {1000 * total_time / processed_frames:.2f} ms")

        if timesINF:
            average_time_inf_ms = sum(timesINF) / len(timesINF)
            print("Average inference time:", average_time_inf_ms, "ms")
        else:
            print("No frames processed.")
        
        average_time_inf_s = average_time_inf_ms / 1000.0
        theoretical_model_FPS = 1 / average_time_inf_s

        print("\n--- Productivity ---")
        print(f"Average frame processing time: {final_stats['avg_processing_time']:.4f} seconds")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Instantaneous speed (based on the last processed frame): {final_stats['instant_fps']:.2f} FPS")
        print(f"Effective speed: {effective_fps:.2f} FPS")
        print(f"Average FPS: {final_stats['avg_fps']:.2f} FPS")
        print(f"Theoretical processing speed of the model: {theoretical_model_FPS:.2f} FPS")
        print(f"Acceleration coefficient: {args.frame_skip}x")
        
        if final_stats['avg_confidence'] > 0:
            print("\n--- Accuracy ---")
            print(f"Average confidence: {final_stats['avg_confidence']:.3f}")
            print(f"Min/max confidence: {final_stats['min_confidence']:.3f}/{final_stats['max_confidence']:.3f}")
        
        if memory_monitor:
            memory_monitor.print_memory_info("Final memory usage")
        
        if args.save_results and video_writer:
            print(f"\nThe video result is saved in {output_path}")
        
        if args.save_frames:
            print(f"Footage saved in {frames_path}")
        
        # Building and maintaining performance charts
        try:
            os.makedirs(args.output_dir, exist_ok=True)

            # 1. FPS per frame
            fps_values = [1/t for t in tracker.processing_times if t > 0.0]
            plt.figure(figsize=(10, 5))
            plt.plot(fps_values, label="FPS per frame")

            # We use the get_avg_fps() method instead of manual calculation.
            avg_fps = tracker.get_avg_fps()
            plt.axhline(avg_fps, color='red', linestyle='--', label=f"Avg FPS: {avg_fps:.2f}")

            plt.xlabel("Processed frames")
            plt.ylabel("FPS")
            plt.title("Frame processing speed (FPS)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "fps_plot.png"))
            plt.close()

            # 2. Histogramme de confiance
            plt.figure(figsize=(8, 5))
            plt.hist(tracker.confidence_scores, edgecolor='black')
            avg_conf = np.mean(tracker.confidence_scores)
            plt.axvline(avg_conf, color='red', linestyle='--', label=f"Avg: {avg_conf:.2f}")
            plt.legend()
            plt.xlabel("Confidence")
            plt.ylabel("Quantity")
            plt.title("Confidence histogram of detections")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "confidence_histogram.png"))
            plt.close()

            print("\nGraphs are saved in the directory:", args.output_dir)

        except Exception as e:
            print(f"Error when constructing graphs: {e}")

def main():
    args = parse_arguments()
    
    # Installation of a computing device
    if args.device.lower() != 'cpu' and torch.cuda.is_available():
        device = f"cuda:{args.device}" if args.device.isdigit() else args.device
        print(f"The device is used: {device}")
    else:
        device = "cpu"
        print(f"The device is used: {device}")
    
    # Initialization of memory monitoring
    memory_monitor = None
    if args.memory_monitor:
        memory_monitor = MemoryMonitor(device)
        memory_monitor.print_memory_info("Initial state")
        
        # Clearing your memory before you start
        gc.collect()
        if device != 'cpu':
            torch.cuda.empty_cache()
    
    try:
        # Determining the type of input data
        if args.source.isdigit() or args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            process_video(args, memory_monitor)
        else:
            process_image(args, memory_monitor)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        
    finally:
        # Final memory cleanup
        if memory_monitor:
            gc.collect()
            if device != 'cpu':
                torch.cuda.empty_cache()
            memory_monitor.print_memory_info("After clearing the memory")

if __name__ == "__main__":
    main()