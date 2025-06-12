# Feature Extraction Scripts for Multimodal Virtual Piano System

import os
import cv2
import numpy as np
import pandas as pd
import librosa
import mediapipe as mp
from tqdm import tqdm

# ---------------------------- CONFIG ----------------------------
VIDEO_DIR = "./piano_notes"
LABEL_DIR = "./piano_notes"
OUTPUT_LANDMARK_DIR = "./landmarks"
OUTPUT_AUDIO_DIR = "./audio_segments"

os.makedirs(OUTPUT_LANDMARK_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 44100
AUDIO_WINDOW_S = 0.2  # duration of tap window
N_MFCC = 13
HOP_LENGTH = 512
FIXED_MFCC_TIME_STEPS = int(librosa.time_to_frames(AUDIO_WINDOW_S, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)) + 1

# ---------------------- MEDIAPIPE SETUP ------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# -------------------- VIDEO FEATURE EXTRACTION -----------------
def extract_video_landmarks():
    for file in os.listdir(LABEL_DIR):
        if file.endswith(".txt") and file.startswith("tap_"):
            key = file.replace("tap_", "").replace(".txt", "")
            video_path = os.path.join(VIDEO_DIR, f"tap_{key}.MOV")
            if not os.path.exists(video_path):
                video_path = os.path.join(VIDEO_DIR, f"tap_{key}.mov")
            if not os.path.exists(video_path):
                print(f"[SKIPPED] No video for {key}")
                continue

            print(f"Processing {video_path}...")
            cap = cv2.VideoCapture(video_path)
            landmarks = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    frame_landmarks = []
                    for lm in hand.landmark:
                        frame_landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    frame_landmarks = [0.0] * 63
                landmarks.append(frame_landmarks)
            cap.release()
            landmarks = np.array(landmarks)
            np.save(os.path.join(OUTPUT_LANDMARK_DIR, f"{key}_landmarks.npy"), landmarks)

# --------------------- AUDIO FEATURE EXTRACTION ----------------
def extract_audio_mfccs():
    for file in os.listdir(LABEL_DIR):
        if file.endswith(".txt") and file.startswith("tap_"):
            key = file.replace("tap_", "").replace(".txt", "")
            video_path = os.path.join(VIDEO_DIR, f"tap_{key}.MOV")
            if not os.path.exists(video_path):
                continue
            label_path = os.path.join(LABEL_DIR, file)
            df = pd.read_csv(label_path, header=None, sep="\t", names=["start", "end", "key", "finger"])
            df["start"] = df["start"].astype(float)
            df["end"] = df["end"].astype(float)
            df["finger"] = df["finger"].astype(int)


            print(f"Extracting audio from {video_path}...")
            # Load audio from video file
            y, sr = librosa.load(video_path, sr=SAMPLE_RATE)

            for idx, row in df.iterrows():
                center_time = (row["start"] + row["end"]) / 2
                start_sample = int((center_time - AUDIO_WINDOW_S/2) * sr)
                end_sample = int((center_time + AUDIO_WINDOW_S/2) * sr)
                if start_sample < 0 or end_sample > len(y):
                    continue
                window = y[start_sample:end_sample]
                mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=N_MFCC)
                mfcc = mfcc[:, :FIXED_MFCC_TIME_STEPS]  # trim or pad to fixed length
                np.save(os.path.join(OUTPUT_AUDIO_DIR, f"{key}_tap_{idx}.npy"), mfcc)

# ---------------------------- MAIN ------------------------------
if __name__ == "__main__":
    print("Extracting video landmarks...")
    #extract_video_landmarks()
    print("Extracting audio MFCCs...")
    extract_audio_mfccs()
    print("Feature extraction complete.")
