# adversarial_test.py
import cv2
import numpy as np
import os

def add_occlusion(frame, box_size=(30, 30)):
    x, y = np.random.randint(0, frame.shape[1] - box_size[0]), np.random.randint(0, frame.shape[0] - box_size[1])
    frame[y:y + box_size[1], x:x + box_size[0]] = 0
    return frame

def simulate_motion_blur(frame, kernel_size=10):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(frame, -1, kernel)

def apply_adversarial_conditions(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = add_occlusion(frame)
        frame = simulate_motion_blur(frame)
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Adversarial test video saved to {output_path}")
