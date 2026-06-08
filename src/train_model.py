import os
import json
import pickle
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =========================
# PATH
# =========================
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =========================
# LOAD DATA SPLIT DAN CONFIG
# =========================
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_files.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val_files.csv"))

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
EPOCHS = 5
RANDOM_STATE = 42

print("Train files:", len(train_df))
print("Validation files:", len(val_df))
print("Jumlah speaker:", NUM_CLASSES)

# =========================
# INPUT SHAPE MFCC
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
# GENERATOR MFCC
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


def make_dataset(dataframe, shuffle=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: mfcc_generator(dataframe),
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=RANDOM_STATE)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = make_dataset(train_df, shuffle=True)
val_ds = make_dataset(val_df, shuffle=False)

print("Dataset training siap.")

# =========================
# MODEL CNN
# =========================
model = models.Sequential([
    layers.Input(shape=INPUT_SHAPE),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACK
# =========================
best_model_path = os.path.join(MODEL_DIR, "best_model.keras")

callbacks = [
    ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================
# TRAINING
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# SIMPAN MODEL FINAL
# =========================
final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
model.save(final_model_path)

# =========================
# SIMPAN TRAINING HISTORY
# =========================
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(REPORT_DIR, "training_history.csv"), index=False)

# =========================
# SIMPAN GRAFIK ACCURACY
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "accuracy_plot.png"))
plt.close()

# =========================
# SIMPAN GRAFIK LOSS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "loss_plot.png"))
plt.close()

print("\nTraining selesai.")
print("Model terbaik disimpan di:", best_model_path)
print("Model final disimpan di:", final_model_path)
print("Training history disimpan di: reports/training_history.csv")
print("Grafik disimpan di:")
print("- reports/accuracy_plot.png")
print("- reports/loss_plot.png")