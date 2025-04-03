#!/usr/bin/env python3
import cv2
import os
import argparse
from pathlib import Path
import numpy as np

def resize_and_crop(image, target_size=640):
    """
    Resize and crop the image to target_size x target_size while maintaining aspect ratio.
    
    Args:
        image: The input image (numpy array)
        target_size: Target width and height (default: 640)
    
    Returns:
        Resized and cropped image
    """
    height, width = image.shape[:2]
    
    # Determine the scaling factor to resize the image
    scale = max(target_size / width, target_size / height)
    
    # Resize the image
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Crop the image to get the center portion
    y_start = max(0, int((new_height - target_size) / 2))
    x_start = max(0, int((new_width - target_size) / 2))
    cropped = resized[y_start:y_start + target_size, x_start:x_start + target_size]
    
    # If the cropped image is smaller than target_size (rare case), pad it
    h, w = cropped.shape[:2]
    if h < target_size or w < target_size:
        # Create a black canvas of target_size x target_size
        result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        # Calculate where to paste the image on the canvas
        y_offset = (target_size - h) // 2
        x_offset = (target_size - w) // 2
        # Paste the image
        result[y_offset:y_offset + h, x_offset:x_offset + w] = cropped
        return result
    
    return cropped

def process_images(input_dir, output_dir, target_size=640):
    """
    Process all images in the input directory and save them to the output directory.
    
    Args:
        input_dir: Path to the input directory
        output_dir: Path to the output directory
        target_size: Target size for width and height (default: 640)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all files in input directory
    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Count total images for progress reporting
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process")
    
    # Process each image
    for i, file_path in enumerate(image_files, 1):
        try:
            # Read the image
            image = cv2.imread(str(file_path))
            
            if image is None:
                print(f"Warning: Could not read {file_path}, skipping")
                continue
                
            # Process the image
            processed = resize_and_crop(image, target_size)
            
            # Create output filename
            output_filename = os.path.join(output_dir, file_path.name)
            
            # Save the processed image
            cv2.imwrite(output_filename, processed)
            
            # Print progress
            print(f"Processed {i}/{total_images}: {file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"All done! Processed images saved to {output_dir}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Format images to 640x640 using OpenCV')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for processed images')
    parser.add_argument('--size', '-s', type=int, default=640, help='Target size (both width and height)')
    
    args = parser.parse_args()
    
    # Process the images
    process_images(args.input, args.output, args.size)

if __name__ == "__main__":
    main()