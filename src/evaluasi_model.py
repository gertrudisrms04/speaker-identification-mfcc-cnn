import os
import json
import pickle
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# =========================
# PATH
# =========================
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# =========================
# LOAD DATA TEST, CONFIG, LABEL
# =========================
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_files.csv"))

with open(os.path.join(PROCESSED_DIR, "dataset_config.json"), "r") as f:
    config = json.load(f)

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

TARGET_SR = config["target_sr"]
SEGMENT_DURATION = config["segment_duration"]
SAMPLES_PER_SEGMENT = config["samples_per_segment"]
NUM_CLASSES = config["total_speakers"]

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
BATCH_SIZE = 16

print("Test files:", len(test_df))
print("Jumlah speaker:", NUM_CLASSES)

# =========================
# INPUT SHAPE
# =========================
dummy_audio = np.zeros(SAMPLES_PER_SEGMENT, dtype=np.float32)

dummy_mfcc = librosa.feature.mfcc(
    y=dummy_audio,
    sr=TARGET_SR,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    center=False
).T

TIME_STEPS = dummy_mfcc.shape[0]
INPUT_SHAPE = (TIME_STEPS, N_MFCC, 1)

print("Input shape:", INPUT_SHAPE)

# =========================
# GENERATOR MFCC UNTUK TEST
# =========================
def mfcc_generator(dataframe):
    for _, row in dataframe.iterrows():
        file_path = row["file_path"]
        label = int(row["label"])

        try:
            audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            print("Gagal load:", file_path, "|", e)
            continue

        total_segments = len(audio) // SAMPLES_PER_SEGMENT

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

            yield mfcc, label


def make_dataset(dataframe):
    dataset = tf.data.Dataset.from_generator(
        lambda: mfcc_generator(dataframe),
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


test_ds = make_dataset(test_df)

# =========================
# LOAD MODEL TERBAIK
# =========================
best_model_path = os.path.join(MODEL_DIR, "best_model.keras")

if not os.path.exists(best_model_path):
    raise FileNotFoundError("models/best_model.keras tidak ditemukan. Jalankan training dulu.")

model = load_model(best_model_path)

print("Model berhasil diload:", best_model_path)

# =========================
# EVALUASI TEST LOSS & ACCURACY
# =========================
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print("\n=== HASIL EVALUASI TEST ===")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# =========================
# PREDIKSI TEST UNTUK REPORT
# =========================
y_true = []
y_pred = []

for batch_x, batch_y in test_ds:
    pred_prob = model.predict(batch_x, verbose=0)
    pred_label = np.argmax(pred_prob, axis=1)

    y_true.extend(batch_y.numpy())
    y_pred.extend(pred_label)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# SIMPAN METRICS RINGKAS
# =========================
metrics_df = pd.DataFrame([{
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "accuracy_score": accuracy_score(y_true, y_pred),
    "total_test_segments": len(y_true),
    "total_speakers": NUM_CLASSES
}])

metrics_path = os.path.join(REPORT_DIR, "metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(REPORT_DIR, "classification_report.csv")
report_df.to_csv(report_path)

print("\nClassification report:")
print(report_df)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_csv_path = os.path.join(REPORT_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)

# Simpan gambar confusion matrix
plt.figure(figsize=(16, 14))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Speaker Identification")
plt.colorbar()

tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=90, fontsize=6)
plt.yticks(tick_marks, label_encoder.classes_, fontsize=6)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

cm_png_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
plt.savefig(cm_png_path, dpi=300)
plt.close()

print("\nFile evaluasi berhasil disimpan:")
print("-", metrics_path)
print("-", report_path)
print("-", cm_csv_path)
print("-", cm_png_path)