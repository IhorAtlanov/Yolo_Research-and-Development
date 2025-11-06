import cv2
import os


def create_output_folder(folder_path):
    """Creates a folder for saving frames if it doesn't already exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a video at a specified interval

    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save frames
        frame_interval (int): Interval between frames (1 = every frame, 5 = every 5th frame)
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found - {video_path}")
        return False

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Failed to open video - {video_path}")
        return False

    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print("Video information:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extraction interval: every {frame_interval} frame(s)")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame only if it matches the interval
        if frame_count % frame_interval == 0:
            # Create filename
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = os.path.join(output_folder, filename)

            # Save the frame
            cv2.imwrite(filepath, frame)
            saved_count += 1

            if saved_count % 100 == 0:  # Show progress every 100 frames
                print(f"Frames saved: {saved_count}")

        frame_count += 1

    cap.release()

    print("Extraction complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Frames saved to: {output_folder}")

    return True


def get_video_info(video_path):
    """Retrieves video information without extracting frames"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Failed to open video - {video_path}")
        return None

    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

    info['duration'] = info['total_frames'] / info['fps']

    cap.release()
    return info


def main():
    """Main function with usage examples"""

    # Settings (change for your needs)
    video_file = "./_.mp4"  # Path to input video
    output_dir = "####"  # Folder to save frames
    interval = 25  # Every 25th frame

    print("=== Script for extracting frames from a video ===\n")

    # Get video info
    print("1. Retrieving video information...")
    video_info = get_video_info(video_file)

    if video_info:
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   Frames: {video_info['total_frames']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print(f"   Duration: {video_info['duration']:.2f} sec")
        print(f"   Approximately {video_info['total_frames'] // interval} frames will be extracted\n")
    else:
        print("   Failed to get video information. Check the file path.\n")
        return

    # Create output folder
    print("2. Creating output folder...")
    create_output_folder(output_dir)
    print()

    # Extract frames
    print("3. Extracting frames...")
    success = extract_frames(video_file, output_dir, interval)

    if success:
        print("\n Process completed successfully!")
    else:
        print("\n An error occurred during frame extraction.")


# Additional usage example
def example_different_intervals():
    """Example of extraction with different intervals"""
    video_file = "./_.mp4"
    create_output_folder("####")
    extract_frames(video_file, "####", 25)


if __name__ == "__main__":
    main()
    example_different_intervals()
