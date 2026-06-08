import os
import json
import pickle
import wave
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# PATH
# =========================
DATASET_DIR = "data/50_speakers_audio_data"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# =========================
# PARAMETER DATASET
# =========================
TARGET_SR = 16000
SEGMENT_DURATION = 3
SAMPLES_PER_SEGMENT = TARGET_SR * SEGMENT_DURATION

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# =========================
# CARI SEMUA FILE WAV
# =========================
audio_files = []

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            speaker = os.path.basename(root)

            audio_files.append({
                "file_path": file_path,
                "speaker": speaker,
                "file_name": file
            })

df = pd.DataFrame(audio_files)

if len(df) == 0:
    raise ValueError("Tidak ada file .wav ditemukan. Cek kembali DATASET_DIR.")

print("Total file audio:", len(df))
print("Total speaker:", df["speaker"].nunique())

# =========================
# HITUNG DURASI DAN JUMLAH SEGMEN
# =========================
durations = []
segment_counts = []

for file_path in df["file_path"]:
    try:
        with wave.open(file_path, "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            duration = frames / float(rate)

            durations.append(duration)
            segment_counts.append(int(duration // SEGMENT_DURATION))
    except Exception:
        durations.append(0)
        segment_counts.append(0)

df["duration_seconds"] = durations
df["segment_count"] = segment_counts

# Buang audio yang terlalu pendek
df = df[df["segment_count"] > 0].reset_index(drop=True)

print("Total file valid:", len(df))
print("Estimasi total segmen:", df["segment_count"].sum())

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["speaker"])

num_classes = len(label_encoder.classes_)

print("Jumlah kelas:", num_classes)

# =========================
# SPLIT TRAIN, VAL, TEST BERDASARKAN FILE
# =========================
train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_val_df["label"]
)

print("\nJumlah file:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nEstimasi jumlah segmen:")
print("Train:", train_df["segment_count"].sum())
print("Validation:", val_df["segment_count"].sum())
print("Test:", test_df["segment_count"].sum())

# =========================
# SIMPAN HASIL SPLIT
# =========================
train_df.to_csv(os.path.join(PROCESSED_DIR, "train_files.csv"), index=False)
val_df.to_csv(os.path.join(PROCESSED_DIR, "val_files.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test_files.csv"), index=False)
df.to_csv(os.path.join(PROCESSED_DIR, "all_files.csv"), index=False)

# Simpan label encoder
with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# Simpan config
config = {
    "dataset_dir": DATASET_DIR,
    "target_sr": TARGET_SR,
    "segment_duration": SEGMENT_DURATION,
    "samples_per_segment": SAMPLES_PER_SEGMENT,
    "test_size": TEST_SIZE,
    "val_size": VAL_SIZE,
    "random_state": RANDOM_STATE,
    "total_audio_files": int(len(df)),
    "total_speakers": int(num_classes),
    "total_segments_estimation": int(df["segment_count"].sum()),
    "speakers": label_encoder.classes_.tolist()
}

with open(os.path.join(PROCESSED_DIR, "dataset_config.json"), "w") as f:
    json.dump(config, f, indent=4)

print("\nFile berhasil disimpan ke data/processed/:")
print("- all_files.csv")
print("- train_files.csv")
print("- val_files.csv")
print("- test_files.csv")
print("- label_encoder.pkl")
print("- dataset_config.json")