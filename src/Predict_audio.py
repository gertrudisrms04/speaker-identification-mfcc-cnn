import os
import json
import pickle
import numpy as np
import librosa
import tensorflow as tf

from tensorflow.keras.models import load_model

# =========================
# PATH
# =========================
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

# GANTI INI dengan file audio yang mau dites
AUDIO_PATH = "data/50_speakers_audio_data/Speaker0037/Speaker0037_053.wav"

# =========================
# LOAD CONFIG, LABEL, MODEL
# =========================
with open(os.path.join(PROCESSED_DIR, "dataset_config.json"), "r") as f:
    config = json.load(f)

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

model_path = os.path.join(MODEL_DIR, "best_model.keras")
model = load_model(model_path)

TARGET_SR = config["target_sr"]
SEGMENT_DURATION = config["segment_duration"]
SAMPLES_PER_SEGMENT = config["samples_per_segment"]

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160

print("Model berhasil diload:", model_path)
print("Audio yang dites:", AUDIO_PATH)

# =========================
# FUNGSI EKSTRAK MFCC
# =========================
def extract_segments_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    total_segments = len(audio) // SAMPLES_PER_SEGMENT
    mfcc_list = []

    for i in range(total_segments):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        segment = audio[start:end]

        if len(segment) != SAMPLES_PER_SEGMENT:
            continue

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            center=False
        ).T

        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        mfcc = mfcc[..., np.newaxis].astype(np.float32)

        mfcc_list.append(mfcc)

    return np.array(mfcc_list)

# =========================
# PREDIKSI
# =========================
if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(f"File audio tidak ditemukan: {AUDIO_PATH}")

X_audio = extract_segments_mfcc(AUDIO_PATH)

if len(X_audio) == 0:
    raise ValueError("Audio terlalu pendek atau tidak bisa diproses menjadi segmen.")

pred_probs = model.predict(X_audio)

# Prediksi per segmen
pred_labels = np.argmax(pred_probs, axis=1)

# Voting mayoritas dari semua segmen
final_label_id = np.bincount(pred_labels).argmax()
final_speaker = label_encoder.inverse_transform([final_label_id])[0]

# Confidence rata-rata untuk speaker final
final_confidence = np.mean(pred_probs[:, final_label_id])

print("\n=== HASIL PREDIKSI ===")
print("Jumlah segmen audio:", len(X_audio))
print("Prediksi speaker:", final_speaker)
print("Confidence rata-rata:", round(final_confidence * 100, 2), "%")

print("\nPrediksi per segmen:")
for i, label_id in enumerate(pred_labels):
    speaker = label_encoder.inverse_transform([label_id])[0]
    confidence = pred_probs[i][label_id]
    print(f"Segmen {i+1}: {speaker} | confidence {confidence*100:.2f}%")