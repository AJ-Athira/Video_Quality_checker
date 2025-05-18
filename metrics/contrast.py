import cv2
import numpy as np
import os

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def process_contrast_video(input_path, output_path):
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

        contrast = calculate_contrast(frame)

        # Threshold for good contrast (you can tune this)
        threshold = 50  
        color = (0, 255, 0) if contrast >= threshold else (0, 0, 255)
        label = f"Contrast: {contrast:.2f}"

        cv2.putText(frame, label, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)


        out.write(frame)
        cv2.imshow('Contrast Visualization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = 'test_videos/vid2.mp4'  # change this to your video path
    output_video = 'output_videos/contrast_output 2.mp4'
    process_contrast_video(input_video, output_video)
