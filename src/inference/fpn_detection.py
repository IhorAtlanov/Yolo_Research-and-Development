# python fpn-detection.py --source .\1.mp4 --model .\best.pt --use-fpn --fpn-scales 0.5,1.0,1.5 --save-results

import os
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import torch.nn.functional as F


def parse_arguments():
    parser = argparse.ArgumentParser(description='Tank detector based on YOLO with FPN')

    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or webcam number (0, 1, 2...)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model (.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device for inference (CPU: "cpu", GPU: 0,1,2...)')
    parser.add_argument('--process-large', action='store_true',
                        help='Enable processing of large images by slicing')
    parser.add_argument('--use-fpn', action='store_true',
                        help='Enable Feature Pyramid Network for better small object detection')
    parser.add_argument('--fpn-scales', type=str, default='0.5,1.0,1.5',
                        help='Scales for FPN (comma-separated, e.g. 0.5,1.0,1.5)')
    parser.add_argument('--slice-size', type=int, default=640,
                        help='Slice size for large image splitting')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap ratio for slicing (0-1)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Display results')

    return parser.parse_args()


def process_fpn(model, image, scales=None, conf=0.25, iou=0.45):
    """
    Process image using Feature Pyramid Network (FPN)

    Args:
        model: YOLO model
        image: Input image (numpy array)
        scales: FPN scales
        conf: Confidence threshold
        iou: IoU threshold for NMS

    Returns:
        Image with detections
        List of detections
    """
    if scales is None:
        scales = [0.5, 1.0, 1.5]
    original_height, original_width = image.shape[:2]
    result_image = image.copy()
    all_detections = []

    temp_dir = os.path.join(os.getcwd(), 'temp_fpn')
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Applying FPN with scales: {scales}")

    for i, scale in enumerate(scales):
        if scale != 1.0:
            scaled_width = int(original_width * scale)
            scaled_height = int(original_height * scale)
            scaled_img = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_img = image.copy()
            scaled_width, scaled_height = original_width, original_height

        scaled_img_path = os.path.join(temp_dir, f'scale_{i}_{scale}.jpg')
        cv2.imwrite(scaled_img_path, scaled_img)

        results = model.predict(scaled_img_path, conf=conf, iou=iou, verbose=False)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, confidence, cls in zip(boxes, confidences, classes):
                adjusted_box = [
                    box[0] / scale,
                    box[1] / scale,
                    box[2] / scale,
                    box[3] / scale
                ]
                all_detections.append({
                    'box': adjusted_box,
                    'confidence': confidence,
                    'class': cls,
                    'scale': scale
                })

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    if all_detections:
        boxes = np.array([d['box'] for d in all_detections])
        confidences = np.array([d['confidence'] for d in all_detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf, iou)
        filtered_detections = [all_detections[i] for i in indices.flatten()]

        for detection in filtered_detections:
            x1, y1, x2, y2 = [int(c) for c in detection['box']]
            confidence = detection['confidence']
            scale = detection['scale']

            if scale < 1.0:
                color = (0, 0, 255)  # red
            elif scale == 1.0:
                color = (0, 255, 0)  # green
            else:
                color = (255, 0, 0)  # blue

            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"Tank {confidence:.2f} (x{scale})"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_image, filtered_detections

    return result_image, []


def process_large_image(model, image, slice_size=640, overlap=0.2, conf=0.25, iou=0.45):
    """
    Process large image by slicing it into smaller parts
    """
    original_height, original_width = image.shape[:2]
    stride = int(slice_size * (1 - overlap))
    result_image = image.copy()
    all_detections = []

    temp_dir = os.path.join(os.getcwd(), 'temp_slices')
    os.makedirs(temp_dir, exist_ok=True)

    slice_count = 0
    for y in range(0, original_height, stride):
        for x in range(0, original_width, stride):
            x2 = min(x + slice_size, original_width)
            y2 = min(y + slice_size, original_height)
            if x2 == original_width:
                x = max(0, x2 - slice_size)
            if y2 == original_height:
                y = max(0, y2 - slice_size)
            slice_img = image[y:y2, x:x2]
            temp_slice_path = os.path.join(temp_dir, f'slice_{slice_count}.jpg')
            cv2.imwrite(temp_slice_path, slice_img)
            slice_count += 1

            results = model.predict(temp_slice_path, conf=conf, iou=iou, verbose=False)

            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                for box, confidence, cls in zip(boxes, confidences, classes):
                    adjusted_box = [
                        box[0] + x,
                        box[1] + y,
                        box[2] + x,
                        box[3] + y,
                    ]
                    all_detections.append({
                        'box': adjusted_box,
                        'confidence': confidence,
                        'class': cls
                    })

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    if all_detections:
        boxes = np.array([d['box'] for d in all_detections])
        confidences = np.array([d['confidence'] for d in all_detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf, iou)
        filtered_detections = [all_detections[i] for i in indices.flatten()]

        for detection in filtered_detections:
            x1, y1, x2, y2 = [int(c) for c in detection['box']]
            confidence = detection['confidence']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Tank {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_image, filtered_detections

    return result_image, []


def process_image(args):
    """Process a single image"""
    model = YOLO(args.model)

    if not os.path.exists(args.source):
        print(f"Error: File {args.source} not found")
        return

    image = cv2.imread(args.source)
    start_time = time.time()

    if args.use_fpn:
        scales = [float(s) for s in args.fpn_scales.split(',')]
        result_image, detections = process_fpn(model, image, scales, args.conf, args.iou)
    elif args.process_large and (image.shape[1] > 1280 or image.shape[0] > 1280):
        print(f"Processing large image ({image.shape[1]}x{image.shape[0]}) using slicing...")
        result_image, detections = process_large_image(model, image, args.slice_size, args.overlap, args.conf, args.iou)
    else:
        results = model.predict(args.source, conf=args.conf, iou=args.iou)
        result_image = results[0].plot()
        boxes = results[0].boxes
        detections = [{
            'box': box.xyxy.cpu().numpy()[0],
            'confidence': box.conf.cpu().numpy()[0],
            'class': box.cls.cpu().numpy()[0]
        } for box in boxes]

    processing_time = time.time() - start_time
    print(f"Detected {len(detections)} objects")
    print(f"Processing time: {processing_time:.2f} seconds")

    info_text = f"Objects: {len(detections)}  Time: {processing_time:.2f}s"
    cv2.putText(result_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"detection_{Path(args.source).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")

    if args.show:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_image, detections


def process_video(args):
    """Process video or webcam stream"""
    model = YOLO(args.model)

    try:
        if args.source.isdigit():
            cap = cv2.VideoCapture(int(args.source))
            source_name = f"Webcam {args.source}"
        else:
            if not os.path.exists(args.source):
                print(f"Error: File {args.source} not found")
                return
            cap = cv2.VideoCapture(args.source)
            source_name = os.path.basename(args.source)
    except Exception as e:
        print(f"Video open error: {e}")
        return

    if not cap.isOpened():
        print("Error: Unable to open video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Processing video: {source_name}, size: {width}x{height}, FPS: {fps}")

    processing_mode = "FPN" if args.use_fpn else ("Slicing" if args.process_large else "Standard")
    print(f"Processing mode: {processing_mode}")

    video_writer = None
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"detection_{Path(args.source).stem if not args.source.isdigit() else 'webcam'}_{processing_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0
    fps_display = 0

    if args.use_fpn:
        scales = [float(s) for s in args.fpn_scales.split(',')]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        if args.use_fpn:
            result_frame, detections = process_fpn(model, frame, scales, args.conf, args.iou)
        elif args.process_large and (width > 1280 or height > 1280):
            result_frame, detections = process_large_image(model, frame, args.slice_size, args.overlap, args.conf,
                                                           args.iou)
        else:
            results = model.predict(frame, conf=args.conf, iou=args.iou)
            result_frame = results[0].plot()
            boxes = results[0].boxes
            detections = [{
                'box': box.xyxy.cpu().numpy()[0],
                'confidence': box.conf.cpu().numpy()[0],
                'class': box.cls.cpu().numpy()[0]
            } for box in boxes] if len(results[0].boxes) > 0 else []

        processing_time = time.time() - start_time
        total_time += processing_time
        if frame_count % 10 == 0:
            fps_display = 10 / total_time
            total_time = 0

        info_text = f"Objects: {len(detections)}  FPS: {fps_display:.1f} Mode: {processing_mode}"
        cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if video_writer:
            video_writer.write(result_frame)

        if args.show:
            cv2.imshow("Result", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f"Video processing completed. Average FPS: {frame_count / (total_time + 0.001):.1f}")

    if args.save_results and video_writer:
        print(f"Result saved to {output_path}")


def main():
    args = parse_arguments()

    if args.device.lower() != 'cpu' and torch.cuda.is_available():
        device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    else:
        device = "cpu"

    print(f"Using device: {device}")

    if args.source.isdigit() or args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(args)
    else:
        process_image(args)


if __name__ == "__main__":
    main()
