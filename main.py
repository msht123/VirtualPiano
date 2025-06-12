import os
import cv2
import json
import numpy as np
import sounddevice as sd
import queue
import librosa
import joblib
import mediapipe as mp
import threading
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# ----------------------------- CONFIG -----------------------------
VIDEO_MODEL_PATH = "./trained_models/video_finger_key_model.keras"
AUDIO_MODEL_PATH = "./trained_models/audio_tap_model.keras"
LABEL_ENCODER_PATH = "./trained_models/label_encoder_classes.npy"
VIDEO_SCALER_PATH = "./trained_models/video_scaler.joblib"
AUDIO_SCALER_PATH = "./trained_models/audio_scaler.joblib"
KEY_REGIONS_FILE_PATH = "./piano_notes/key_regions.json"

CAMERA_INDEX = 1
AUDIO_INPUT_DEVICE_INDEX = 1
SAMPLE_RATE = 44100
N_MFCC = 13
FIXED_MFCC_TIME_STEPS = 12
AUDIO_WINDOW_DURATION_S = 0.2
HOP_LENGTH = 512
NOTE_DURATION_SECONDS = 0.3
VOLUME = 0.5

NOTE_FREQUENCIES = {
    "C": 261.63, "Csharp": 277.18, "D": 293.66, "Dsharp": 311.13,
    "E": 329.63, "F": 349.23, "Fsharp": 369.99, "G": 392.00,
    "Gsharp": 415.30, "A": 440.00, "Asharp": 466.16, "B": 493.88
}

# ------------------------- LOAD MODELS ---------------------------
video_model = keras.models.load_model(VIDEO_MODEL_PATH)
audio_model = keras.models.load_model(AUDIO_MODEL_PATH)
video_scaler = joblib.load(VIDEO_SCALER_PATH)
audio_scaler = joblib.load(AUDIO_SCALER_PATH)
video_label_encoder = LabelEncoder()
video_label_encoder.classes_ = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

with open(KEY_REGIONS_FILE_PATH, 'r') as f:
    key_regions = json.load(f)

# ------------------------- SETUP --------------------------
audio_queue = queue.Queue()
note_queue = queue.Queue()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(CAMERA_INDEX)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ------------------- AUDIO CALLBACK ------------------------
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

audio_stream = sd.InputStream(callback=audio_callback, device=AUDIO_INPUT_DEVICE_INDEX,
                               channels=1, samplerate=SAMPLE_RATE)
audio_stream.start()

# ------------------ PLAYBACK WORKER ------------------------
def playback_worker():
    while True:
        freq = note_queue.get()
        if freq is None:
            break
        t = np.linspace(0, NOTE_DURATION_SECONDS, int(NOTE_DURATION_SECONDS * SAMPLE_RATE), False)
        fade_in, fade_out = int(0.01 * SAMPLE_RATE), int(0.05 * SAMPLE_RATE)
        envelope = np.ones_like(t)
        envelope[:fade_in] = np.linspace(0, 1, fade_in)
        envelope[-fade_out:] = np.linspace(1, 0, fade_out)
        amplitude = np.sin(2 * np.pi * freq * t) * VOLUME * envelope
        try:
            sd.play(amplitude.astype(np.float32), samplerate=SAMPLE_RATE, blocking=True)
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
        note_queue.task_done()

threading.Thread(target=playback_worker, daemon=True).start()

# ------------------- INFERENCE HELPERS ------------------------
def predict_finger_key(landmarks):
    x = video_scaler.transform(np.array(landmarks).flatten().reshape(1, -1))
    preds = video_model.predict(x, verbose=0)
    label = video_label_encoder.inverse_transform([np.argmax(preds)])[0]
    return label

def predict_tap(mfcc):
    x = audio_scaler.transform(mfcc.reshape(1, -1))
    p = audio_model.predict(x, verbose=0)[0][0]
    return p > 0.9, p

def determine_key_from_position(x, y):
    for key, region in key_regions.items():
        if region['x_min'] <= x <= region['x_max'] and region['y_min'] <= y <= region['y_max']:
            return key
    return None

# ------------------- MAIN LOOP ------------------------
audio_buffer = np.zeros(int(SAMPLE_RATE * AUDIO_WINDOW_DURATION_S))
print("Starting fusion-based inference with UI and audio output...")

while True:
    ret, frame_original = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame_original.copy(), -1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_label = None
    finger_name, predicted_key = None, None
    cx, cy = None, None

    for key_name, region in key_regions.items():
        x1, y1, x2, y2 = int(region['x_min']), int(region['y_min']), int(region['x_max']), int(region['y_max'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, key_name, (x1 + 5, y1 - 5), FONT, 0.4, (255, 255, 255), 1)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            predicted_label = predict_finger_key(landmarks)
            finger_name, predicted_key = predicted_label.split("_")
            cx, cy = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
            screen_x, screen_y = int(cx * frame.shape[1]), int(cy * frame.shape[0])
            region_key = determine_key_from_position(screen_x, screen_y)
            if region_key:
                cv2.putText(frame, f"{finger_name} -> {region_key}", (screen_x, screen_y - 10),
                            FONT, 0.7, (0, 255, 0), 2)

    if not audio_queue.empty():
        audio_chunk = audio_queue.get()
        audio_buffer = np.roll(audio_buffer, -len(audio_chunk))
        audio_buffer[-len(audio_chunk):] = audio_chunk.flatten()
        mfcc = librosa.feature.mfcc(y=audio_buffer, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc = mfcc[:, :FIXED_MFCC_TIME_STEPS].T.flatten()
        is_tap, confidence = predict_tap(mfcc)
        if is_tap and predicted_label:
            print(f"Tap Detected (p={confidence:.2f}) → {predicted_label}")
            note = predicted_key
            freq = NOTE_FREQUENCIES.get(note)
            if freq:
                note_queue.put(freq)
                cv2.putText(frame, f"Playing: {note} ({finger_name})", (10, 30), FONT, 0.8, (0, 255, 255), 2)

    cv2.imshow("Virtual Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
audio_stream.stop()
audio_stream.close()
note_queue.put(None)
note_queue.join()
cv2.destroyAllWindows()
