# Violence Detection using CNN + LSTM

An end-to-end deep learning pipeline to detect violence in videos using CNN + LSTM, frame extraction, adversarial/environmental testing, and real-time Telegram alerts.

## 📁 Scripts Overview

| Script | Description |
|--------|-------------|
| `extract_frames.py` | Extracts and categorizes frames from videos |
| `train_model.py` | Trains a CNN + LSTM model on frame sequences |
| `test_model.py` | Predicts violence in new videos using the trained model |
| `balance_dataset.py` | Ensures balanced dataset by limiting video counts per class |
| `adversarial_test.py` | Adds occlusions and motion blur to test robustness |
| `env_variation_test.py` | Applies lighting, blur, and night effects |
| `telegram_alert.py` | Sends alerts via Telegram when violence is detected |

## 🚀 Quick Start

1. Extract frames: `python scripts/extract_frames.py`
2. Train model: `python scripts/train_model.py`
3. Test model: `python scripts/test_model.py`

## 📦 Requirements

- TensorFlow
- OpenCV
- NumPy
- Requests (for Telegram bot)
