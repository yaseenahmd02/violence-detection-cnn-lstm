# test_model.py
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array

MODEL_PATH = "/content/drive/My Drive/mini_violence/models/optimized_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def extract_frames(video_path, sequence_length=5, target_size=(32, 32)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = img_to_array(frame) / 255.0
        frames.append(frame)
    cap.release()
    if len(frames) < sequence_length:
        return None
    return [frames[i:i+sequence_length] for i in range(len(frames)-sequence_length+1)]

def predict_video(video_path):
    sequences = extract_frames(video_path)
    if sequences is None:
        return "Not enough frames!"
    predictions = model.predict(np.array(sequences))
    avg_pred = np.mean(predictions, axis=0)
    class_idx = np.argmax(avg_pred)
    label = ["Violence", "Non-Violence"][class_idx]
    return f"Prediction: {label} ({avg_pred[class_idx]*100:.2f}%)"

# Example
# result = predict_video("/path/to/video.avi")
# print(result)
