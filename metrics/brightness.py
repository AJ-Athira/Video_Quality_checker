import cv2
import numpy as np
import os

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def process_brightness_video(input_path, output_path):
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

        brightness = calculate_brightness(frame)

        threshold = 100  # tune as needed
        color = (0, 255, 0) if brightness >= threshold else (0, 0, 255)
        label = f"Brightness: {brightness:.2f}"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        cv2.imshow('Brightness Visualization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = 'test_videos/vid2.mp4'
    output_video = 'output_videos/brightness_output 2.mp4'
    process_brightness_video(input_video, output_video)
