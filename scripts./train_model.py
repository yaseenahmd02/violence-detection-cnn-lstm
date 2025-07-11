# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

# Custom generator
class FrameSequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=4, sequence_length=5, target_size=(32, 32)):
        self.directory = directory
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.classes = ["Violence", "Non-Violence"]
        self.data = self.load_images()

    def load_images(self):
        data = []
        for label, category in enumerate(self.classes):
            path = os.path.join(self.directory, category)
            if os.path.exists(path):
                files = sorted(os.listdir(path))
                for i in range(len(files) - self.sequence_length):
                    data.append((files[i:i+self.sequence_length], label))
        return data

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []
        for sequence, label in batch:
            frames = []
            for frame in sequence:
                img = tf.keras.utils.load_img(os.path.join(self.directory, self.classes[label], frame), target_size=self.target_size)
                img = tf.keras.utils.img_to_array(img) / 255.0
                frames.append(img)
            X.append(frames)
            y.append(label)
        return np.array(X), tf.keras.utils.to_categorical(y, num_classes=2)

# Paths
DATASET_PATH = "/content/drive/My Drive/mini_violence/dataset/train_frames"
MODEL_SAVE_PATH = "/content/drive/My Drive/mini_violence/models/optimized_model.h5"
HISTORY_SAVE_PATH = "/content/drive/My Drive/mini_violence/models/training_history.npy"

# Load data
train_gen = FrameSequenceGenerator(DATASET_PATH)
val_gen = FrameSequenceGenerator(DATASET_PATH)

# Define model
model = Sequential([
    Input(shape=(5, 32, 32, 3)),
    TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5)
]

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)
model.save(MODEL_SAVE_PATH)
np.save(HISTORY_SAVE_PATH, history.history)
