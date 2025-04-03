import cv2
import os
import re
import glob
import argparse
import numpy as np
import random

def extract_number_from_filename(filename):
    """
    Extract the number from a filename like '1.jpg', '2.jpg', etc.
    """
    base_name = os.path.basename(filename)
    name, ext = os.path.splitext(base_name)
    
    # Check if the entire name is a number
    if name.isdigit():
        return int(name), ext
    
    # Otherwise find the last sequence of digits
    matches = list(re.finditer(r'\d+', name))
    if not matches:
        return None, ext
    
    last_match = matches[-1]
    number = int(last_match.group())
    return number, ext

def get_image_files(folder_path):
    """
    Get all image files in the folder, avoiding duplicates.
    Returns a sorted list of image files with their numbers.
    """
    # Get all image files
    all_files = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        pattern = os.path.join(folder_path, f'*.{ext}')
        found_files = glob.glob(pattern)
        print(f"Found {len(found_files)} files with pattern {pattern}")
        all_files.extend(found_files)
    
    # Extract numbers and filter files
    numbered_files = []
    for file_path in all_files:
        number, _ = extract_number_from_filename(file_path)
        if number is not None:
            numbered_files.append((number, file_path))
    
    # Sort by number and remove duplicates (keep first occurrence of each number)
    numbered_files.sort()
    unique_files = []
    seen_numbers = set()
    
    for number, file_path in numbered_files:
        if number not in seen_numbers:
            unique_files.append((number, file_path))
            seen_numbers.add(number)
    
    return unique_files

def mirror_image(image):
    """Mirror an image horizontally using OpenCV."""
    return cv2.flip(image, 1)  # 1 for horizontal flip

def apply_crop(image, min_scale=0.0, max_scale=0.2):
    """Apply random crop to the image."""
    height, width = image.shape[:2]
    
    # Calculate scale factor
    scale = random.uniform(min_scale, max_scale)
    
    # Calculate crop dimensions
    crop_width = int(width * scale)
    crop_height = int(height * scale)
    
    # Calculate random crop position
    if crop_width > 0:
        start_x = random.randint(0, crop_width)
    else:
        start_x = 0
        
    if crop_height > 0:
        start_y = random.randint(0, crop_height)
    else:
        start_y = 0
    
    # Perform crop
    cropped = image[start_y:height-crop_height+start_y, start_x:width-crop_width+start_x]
    
    # Resize back to original dimensions
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

def apply_rotation(image, min_angle=-15, max_angle=15):
    """Apply random rotation to the image."""
    height, width = image.shape[:2]
    angle = random.uniform(min_angle, max_angle)
    
    # Get rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def apply_shift(image, horizontal_shift=0.15, vertical_shift=0.15):
    """Apply random shift to the image."""
    height, width = image.shape[:2]
    
    # Calculate shift amount
    tx = random.uniform(-horizontal_shift, horizontal_shift) * width
    ty = random.uniform(-vertical_shift, vertical_shift) * height
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    shifted = cv2.warpAffine(image, translation_matrix, (width, height), 
                            borderMode=cv2.BORDER_REFLECT)
    return shifted

def apply_hue_shift(image, min_hue=-25, max_hue=25):
    """Apply random hue shift to the image."""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply hue shift
    hue_shift = random.uniform(min_hue, max_hue)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Convert back to BGR
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_brightness(image, min_brightness=-0.25, max_brightness=0.25):
    """Apply random brightness adjustment to the image."""
    brightness = random.uniform(min_brightness, max_brightness)
    
    if brightness > 0:
        # Increase brightness
        return np.clip(image * (1 + brightness), 0, 255).astype(np.uint8)
    else:
        # Decrease brightness
        return np.clip(image * (1 + brightness), 0, 255).astype(np.uint8)

def apply_exposure(image, min_exposure=-0.25, max_exposure=0.25):
    """Apply random exposure adjustment to the image."""
    exposure = random.uniform(min_exposure, max_exposure)
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust V channel (brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + exposure), 0, 255)
    
    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_noise(image, max_noise_percentage=0.05):
    """Apply random noise to the image."""
    noise_percentage = random.uniform(0, max_noise_percentage)
    
    # Create noise matrix
    noise = np.zeros(image.shape, np.uint8)
    cv2.randu(noise, 0, 255)
    
    # Create mask for noise pixels
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.randu(mask, 0, 100)
    mask = mask < (noise_percentage * 100)
    
    # Apply noise only to selected pixels
    noisy_image = image.copy()
    noisy_image[mask] = noise[mask]
    
    return noisy_image

def apply_augment_plus(image, min_augs=3, max_augs=5):
    """
    Apply a random subset of augmentation types with random parameters.
    
    Args:
        image: The input image
        min_augs: Minimum number of augmentations to apply
        max_augs: Maximum number of augmentations to apply
    """
    # Create list of all augmentation functions with their names
    all_augmentations = [
        ("crop", lambda img: apply_crop(img, min_scale=0.0, max_scale=0.2)),
        ("rotation", lambda img: apply_rotation(img, min_angle=-15, max_angle=15)),
        ("shift", lambda img: apply_shift(img, horizontal_shift=0.15, vertical_shift=0.15)),
        ("hue_shift", lambda img: apply_hue_shift(img, min_hue=-25, max_hue=25)),
        ("brightness", lambda img: apply_brightness(img, min_brightness=-0.25, max_brightness=0.25)),
        ("exposure", lambda img: apply_exposure(img, min_exposure=-0.25, max_exposure=0.25)),
        ("noise", lambda img: apply_noise(img, max_noise_percentage=0.05))
    ]
    
    # Determine how many augmentations to apply
    num_augs = random.randint(min_augs, min(max_augs, len(all_augmentations)))
    
    # Randomly select which augmentations to apply
    selected_augs = random.sample(all_augmentations, num_augs)
    
    # Apply selected augmentations in sequence
    result = image.copy()
    
    print(f"Applying {num_augs} random augmentations:")
    for i, (aug_name, aug_func) in enumerate(selected_augs):
        print(f"  - {i+1}. {aug_name}")
        result = aug_func(result)
    
    return result

def augment_images(folder_path, output_folder=None, aug_type='mirror', min_augs=3, max_augs=5):
    """
    Augment images in folder_path based on the augmentation type and save with sequential numbering.
    
    aug_type can be:
    - 'mirror': simple horizontal mirroring
    - 'augmentplus': apply random subset of augmentations with random parameters
    
    min_augs and max_augs control how many augmentations are applied when using 'augmentplus'
    """
    print(f"Starting augmentation process in folder: {folder_path}")
    print(f"Augmentation type: {aug_type}")
    
    if output_folder is None:
        output_folder = folder_path
    
    # Create output folder if it doesn't exist
    if output_folder != folder_path:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")
    
    # Get all image files with their numbers
    numbered_files = get_image_files(folder_path)
    
    if not numbered_files:
        print(f"No numbered image files found in {folder_path}")
        return
    
    # Find the highest number
    highest_number = numbered_files[-1][0]
    # Get the extension from the highest numbered file
    _, highest_file = numbered_files[-1]
    _, common_ext = os.path.splitext(highest_file)
    
    print(f"Found {len(numbered_files)} unique numbered files.")
    print(f"Highest number found: {highest_number}")
    print(f"Will create augmented images numbered from {highest_number + 1} to {highest_number + len(numbered_files)}")
    
    # Start numbering from the next number after the highest
    next_number = highest_number + 1
    augmented_count = 0
    
    # Process each image
    for i, (original_number, file_path) in enumerate(numbered_files):
        print(f"Processing file {i+1}/{len(numbered_files)}: {os.path.basename(file_path)} (Number: {original_number})")
        
        # Read the image
        original_image = cv2.imread(file_path)
        if original_image is None:
            print(f"Failed to read image: {file_path}")
            continue
        
        # Apply the chosen augmentation
        if aug_type.lower() == 'mirror':
            augmented_image = mirror_image(original_image)
        elif aug_type.lower() == 'augmentplus':
            augmented_image = apply_augment_plus(original_image, min_augs=min_augs, max_augs=max_augs)
        else:
            print(f"Unknown augmentation type: {aug_type}. Using mirror as default.")
            augmented_image = mirror_image(original_image)
        
        # Create new filename with sequential numbering
        new_number = next_number + i
        output_filename = f"{new_number}{common_ext}"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the augmented image
        success = cv2.imwrite(output_path, augmented_image)
        if success:
            print(f"Saved augmented image: {output_path}")
            augmented_count += 1
        else:
            print(f"Failed to save image: {output_path}")
    
    print(f"Augmentation complete. Created {augmented_count} augmented images from {len(numbered_files)} original images.")
    print(f"The augmented images are numbered from {next_number} to {next_number + augmented_count - 1}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment images with various transformations and sequential numbering.')
    parser.add_argument('folder_path', help='Path to the folder containing images')
    parser.add_argument('--output', help='Path to save augmented images (optional)')
    parser.add_argument('--type', choices=['mirror', 'augmentplus'], default='mirror',
                        help='Type of augmentation to apply (default: mirror)')
    parser.add_argument('--min-augs', type=int, default=3, 
                        help='Minimum number of augmentations to apply when using augmentplus (default: 3)')
    parser.add_argument('--max-augs', type=int, default=5, 
                        help='Maximum number of augmentations to apply when using augmentplus (default: 5)')
    parser.add_argument('--noise', type=float, default=0.05, 
                        help='Maximum noise percentage for noise augmentation (0.0-1.0, default: 0.05)')
    
    args = parser.parse_args()
    
    # Update the noise parameter in the apply_noise function
    apply_noise.__defaults__ = (args.noise,)
    
    # Make sure path is properly formatted
    folder_path = os.path.abspath(args.folder_path)
    output_path = os.path.abspath(args.output) if args.output else None
    
    print(f"Starting script with folder path: {folder_path}")
    print(f"Using augmentation type: {args.type}")
    if args.type.lower() == 'augmentplus':
        print(f"Will apply between {args.min_augs} and {args.max_augs} random augmentations")
        print(f"Maximum noise percentage: {args.noise}")
    
    augment_images(folder_path, output_path, args.type, args.min_augs, args.max_augs)