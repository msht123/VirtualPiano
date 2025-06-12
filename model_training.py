# Model Training Scripts for Multimodal Virtual Piano System

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

# --------------------------- CONFIG ----------------------------
LANDMARK_DIR = "./landmarks"
LABEL_DIR = "./piano_notes"
MFCC_DIR = "./audio_segments"
VIDEO_MODEL_PATH = "./trained_models/video_finger_key_model.keras"
AUDIO_MODEL_PATH = "./trained_models/audio_tap_model.keras"
os.makedirs("./trained_models", exist_ok=True)


# --------------------- VIDEO MODEL TRAINING --------------------
def load_video_data():
    X, y = [], []
    for txt_file in os.listdir(LABEL_DIR):
        if not txt_file.endswith(".txt") or not txt_file.startswith("tap_"):
            continue
        key = txt_file.replace("tap_", "").replace(".txt", "")
        label_df = pd.read_csv(os.path.join(LABEL_DIR, txt_file), sep="\t", names=["start", "end", "key", "finger"], header=None)
        label_df["finger"] = label_df["finger"].astype(int)
        landmark_file = os.path.join(LANDMARK_DIR, f"{key}_landmarks.npy")
        if not os.path.exists(landmark_file):
            continue
        landmarks = np.load(landmark_file)  # shape: (n_frames, 63)
        fps = 30  # approximate
        for i, row in label_df.iterrows():
            start_frame = int(row["start"] * fps)
            end_frame = int(row["end"] * fps)
            label = f"{row['finger']}_{row['key']}"
            for f in range(start_frame, min(end_frame + 1, len(landmarks))):
                X.append(landmarks[f])
                y.append(label)
    return np.array(X), np.array(y)

print("Loading video training data...")
X_video, y_video = load_video_data()
video_scaler = StandardScaler()
X_video_scaled = video_scaler.fit_transform(X_video)
video_label_encoder = LabelEncoder()
y_video_encoded = video_label_encoder.fit_transform(y_video)
y_video_categorical = to_categorical(y_video_encoded)

Xv_train, Xv_val, yv_train, yv_val = train_test_split(X_video_scaled, y_video_categorical, test_size=0.2, random_state=42)

video_model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_video_categorical.shape[1], activation='softmax')
])

video_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

video_model.fit(Xv_train, yv_train, validation_data=(Xv_val, yv_val), epochs=20, batch_size=32)
video_model.save(VIDEO_MODEL_PATH)
np.save("./trained_models/label_encoder_classes.npy", video_label_encoder.classes_)
joblib.dump(video_scaler, "./trained_models/video_scaler.joblib")

# --------------------- AUDIO MODEL TRAINING --------------------
def load_audio_data():
    X_audio, y_audio = [], []
    for file in os.listdir(MFCC_DIR):
        if not file.endswith(".npy"):
            continue
        mfcc = np.load(os.path.join(MFCC_DIR, file))
        mfcc_flat = mfcc[:, :12].T.flatten()  # ensure fixed shape (13x12)
        X_audio.append(mfcc_flat)
        y_audio.append(1)  # tap
    # Add dummy background samples (you should replace this with actual no-tap samples)
    for _ in range(len(X_audio)):
        X_audio.append(np.random.normal(0, 0.05, len(X_audio[0])))
        y_audio.append(0)  # no tap
    return np.array(X_audio), np.array(y_audio)

print("Loading audio training data...")
X_audio, y_audio = load_audio_data()
audio_scaler = StandardScaler()
X_audio_scaled = audio_scaler.fit_transform(X_audio)
Xa_train, Xa_val, ya_train, ya_val = train_test_split(X_audio_scaled, y_audio, test_size=0.2, random_state=42)

audio_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_audio.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

audio_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

audio_model.fit(Xa_train, ya_train, validation_data=(Xa_val, ya_val), epochs=15, batch_size=32)
audio_model.save(AUDIO_MODEL_PATH)
joblib.dump(audio_scaler, "./trained_models/audio_scaler.joblib")

print("✅ Training complete.")
