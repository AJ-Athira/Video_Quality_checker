import cv2
import numpy as np
import os

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def generate_visual_sharpness_video(input_path, output_path, sharpness_threshold=100.0):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)

        # Visual indicator for blur
        if sharpness < sharpness_threshold:
            # Blend the frame with red to show it's blurry
            red_tint = np.zeros_like(frame)
            red_tint[:, :, 2] = 255  # Full red channel
            frame = cv2.addWeighted(frame, 0.7, red_tint, 0.3, 0)

        # Overlay sharpness score
        text = f"Sharpness: {sharpness:.2f}"
        color = (0, 255, 0) if sharpness >= sharpness_threshold else (0, 0, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 8)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Sharpness visualization video saved to: {output_path}")

# Example usage
input_video = 'test_videos/vid1.mp4'
output_video = 'output_videos/vid1_sharpness.mp4'
generate_visual_sharpness_video(input_video, output_video)
