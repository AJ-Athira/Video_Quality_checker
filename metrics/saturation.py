import cv2
import numpy as np
import os

def calculate_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return saturation.mean()

def process_saturation_video(input_path, output_path):
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

        saturation = calculate_saturation(frame)

        threshold = 50  # tune this
        color = (0, 255, 0) if saturation >= threshold else (0, 0, 255)
        label = f"Saturation: {saturation:.2f}"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        cv2.imshow('Saturation Visualization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = 'test_videos/vid2.mp4'
    output_video = 'output_videos/saturation_output 2.mp4'
    process_saturation_video(input_video, output_video)
