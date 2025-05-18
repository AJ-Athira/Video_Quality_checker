import cv2
import numpy as np
import os

def variance_of_laplacian(image):
    """Compute the Laplacian variance for sharpness."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def tenengrad(image):
    """Calculate sharpness using the Tenengrad method (gradient magnitude)."""
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    fm = gx**2 + gy**2
    return fm.mean()

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def calculate_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return saturation.mean()

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
    return min(max_area / (w * h), 1.0)

def motion_blur_metric(gray):
    """Estimate motion blur by analyzing FFT spectrum variance."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum.var()

def evaluate_video(video_path):
    if not os.path.exists(video_path):
        return None, f"FileNotFound: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Cannot open video: {video_path}"

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    laplacian_list = []
    tenengrad_list = []
    motion_blur_list = []
    contrast_list = []
    brightness_list = []
    saturation_list = []
    face_scores = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_sample = min(frame_count, 30)
    frame_indices = np.linspace(0, frame_count - 1, frames_to_sample).astype(int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        laplacian_val = variance_of_laplacian(gray)
        tenengrad_val = tenengrad(gray)
        motion_val = motion_blur_metric(gray)

        laplacian_list.append(laplacian_val)
        tenengrad_list.append(tenengrad_val)
        motion_blur_list.append(motion_val)

        contrast_list.append(calculate_contrast(frame))
        brightness_list.append(calculate_brightness(frame))
        saturation_list.append(calculate_saturation(frame))
        face_scores.append(get_face_score(frame, face_cascade))

    cap.release()

    if len(laplacian_list) == 0:
        return None, "No frames processed"

    avg_laplacian = np.mean(laplacian_list)
    avg_tenengrad = np.mean(tenengrad_list)
    avg_motion_blur = np.mean(motion_blur_list)
    avg_contrast = np.mean(contrast_list)
    avg_brightness = np.mean(brightness_list)
    avg_saturation = np.mean(saturation_list)
    avg_face_score = np.mean(face_scores) if face_scores else 0

    # Combine perceptual blur scores (higher laplacian and tenengrad = sharper)
    perceptual_sharpness = (avg_laplacian + avg_tenengrad) / 2

    # Penalize perceptual sharpness by motion blur (higher motion blur variance means less blur)
    # So invert motion blur metric for penalty:
    motion_blur_penalty = 1 / (avg_motion_blur + 1e-5)

    # Final clarity score adjusted for motion blur:
    clarity = perceptual_sharpness * motion_blur_penalty

    # Normalize contrast, brightness, saturation roughly to [0, 255]
    norm_contrast = avg_contrast / 128  # rough scale
    norm_brightness = avg_brightness / 255
    norm_saturation = avg_saturation / 255

    # Composite score with weights 
    score = (
        0.4 * clarity +
        0.15 * norm_contrast +
        0.1 * norm_saturation +
        0.1 * norm_brightness +
        0.25 * avg_face_score
    )

    return {
        'score': score,
        'clarity': clarity,
        'laplacian_sharpness': avg_laplacian,
        'tenengrad_sharpness': avg_tenengrad,
        'motion_blur_var': avg_motion_blur,
        'contrast': avg_contrast,
        'brightness': avg_brightness,
        'saturation': avg_saturation,
        'face_score': avg_face_score
    }, "Success"


def main():
    videos = ['test_videos/vid1.mp4', 'test_videos/vid2.mp4']
    results = {}

    for v in videos:
        metrics, status = evaluate_video(v)
        if metrics is None:
            print(f"❌ Error for {v}: {status}")
            results[v] = None
            continue

        print(f"{os.path.basename(v)} → ✅ Score: {metrics['score']:.2f} | "
              f"Clarity: {metrics['clarity']:.2f} | "
              f"Laplacian: {metrics['laplacian_sharpness']:.2f} | "
              f"Tenengrad: {metrics['tenengrad_sharpness']:.2f} | "
              f"Motion Blur Var: {metrics['motion_blur_var']:.2f} | "
              f"Contrast: {metrics['contrast']:.2f} | "
              f"Brightness: {metrics['brightness']:.2f} | "
              f"Saturation: {metrics['saturation']:.2f} | "
              f"Face Score: {metrics['face_score']:.2f}")

        results[v] = metrics['score']

    filtered_results = {k: v for k, v in results.items() if v is not None}
    if not filtered_results:
        print("No valid videos found for evaluation.")
        return

    best_video = max(filtered_results, key=filtered_results.get)
    print("\n🎯 Best Video:", os.path.basename(best_video))


if __name__ == "__main__":
    main()
