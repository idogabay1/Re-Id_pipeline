import cv2
import sys

def crop_first_20_frames(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < 150:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Saved first {frame_count} frames to {output_path}")

if __name__ == "__main__":
    input_path = "./videos/splits/part1.mp4"  # Replace with your input video path
    output_path = "./videos/splits/part1_cropped.mp4"  # Replace with your desired output path
    crop_first_20_frames(input_path, output_path)