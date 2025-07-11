# env_variation_test.py
import cv2
import numpy as np
import os

def adjust_brightness_contrast(image, brightness=30, contrast=30):
    return cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)

def add_night_vision_effect(image):
    green = np.zeros_like(image)
    green[:, :, 1] = image[:, :, 1]
    return green

def apply_environmental_effects(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate blur, contrast, and night vision
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = adjust_brightness_contrast(frame, brightness=40, contrast=-20)
        frame = add_night_vision_effect(frame)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Environmental effect video saved to {output_path}")
