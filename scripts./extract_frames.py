# extract_frames.py
import os
import cv2
from tensorflow.keras.utils import img_to_array

def is_violent_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_intensity = cv2.mean(gray)[0]
    return avg_intensity > 100

def extract_frames(video_path, violence_output_folder, non_violence_output_folder, frames_per_second=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(violence_output_folder, exist_ok=True)
    os.makedirs(non_violence_output_folder, exist_ok=True)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_interval = max(1, fps // frames_per_second)

    frame_count, saved_count = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            folder = violence_output_folder if is_violent_frame(frame) else non_violence_output_folder
            filename = f"{video_name}_frame{saved_count}.jpg"
            cv2.imwrite(os.path.join(folder, filename), frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"✅ Extracted {saved_count} frames from {video_name}")
