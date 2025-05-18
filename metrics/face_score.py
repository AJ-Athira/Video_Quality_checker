import cv2
import numpy as np
import os

def get_face_score(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return 0.0
    h, w = frame.shape[:2]
    max_area = 0
    for (x, y, fw, fh) in faces:
        area = fw * fh
        if area > max_area:
            max_area = area
    face_score = max_area / (w * h)
    return min(face_score, 1.0)

def process_face_score_video(input_path, output_path):
    if not os.path.exists('output_videos'):
        os.makedirs('output_videos')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

        face_score = get_face_score(frame, face_cascade)

        threshold = 0.05  
        color = (0, 255, 0) if face_score >= threshold else (0, 0, 255)
        label = f"Face Score: {face_score:.2f}"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Optionally draw rectangle around largest face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            (x, y, fw, fh) = max(faces, key=lambda rect: rect[2]*rect[3])
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)

        out.write(frame)
        cv2.imshow('Face Score Visualization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = 'test_videos/vid2.mp4'
    output_video = 'output_videos/face_score_output 2.mp4'
    process_face_score_video(input_video, output_video)
