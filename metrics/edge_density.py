import cv2
import numpy as np
import os

def calculate_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    density = edge_pixels / total_pixels
    return density * 100  # Return percentage

def process_edge_density_video(input_path, output_path):
    if not os.path.exists('output_videos'):
        os.makedirs('output_videos')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        density = calculate_edge_density(frame)
        threshold = 3  # you can tune this threshold
        color = (0, 255, 0) if density >= threshold else (0, 0, 255)
        label = f"Edge Density: {density:.2f}%"

        cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        out.write(frame)

    cap.release()
    out.release()
    print(f"[✔] Edge Density video saved to {output_path}")

if __name__ == "__main__":
    input_video = 'test_videos/vid1.mp4'  # Change this to your input video
    output_video = 'output_videos/edge_density_output 1.mp4'
    process_edge_density_video(input_video, output_video)
