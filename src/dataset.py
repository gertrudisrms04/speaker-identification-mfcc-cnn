import os
import wave
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATH DATASET
# =========================
DATASET_DIR = "/Users/gertrudisrms/Documents/Kuliah/Semester 6/Speech Processing/Projek_FInal_Speaker_Identification/data/50_speakers_audio_data"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# =========================
# CARI FILE AUDIO
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

print("=== INFORMASI DATASET ===")
print("Total file audio:", len(df))
print("Total speaker:", df["speaker"].nunique())

print("\nContoh data:")
print(df.head())

# =========================
# CEK JUMLAH FILE PER SPEAKER
# =========================
speaker_counts = df["speaker"].value_counts().reset_index()
speaker_counts.columns = ["speaker", "jumlah_file"]

print("\nJumlah file per speaker:")
print(speaker_counts)

# Simpan ringkasan jumlah file speaker ke CSV
speaker_counts.to_csv(
    os.path.join(REPORT_DIR, "speaker_file_counts.csv"),
    index=False
)

# =========================
# CEK DURASI AUDIO
# =========================
durations = []

for file_path in df["file_path"]:
    try:
        with wave.open(file_path, "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            duration = frames / float(rate)
            durations.append(duration)
    except Exception:
        durations.append(None)

df["duration_seconds"] = durations

print("\nStatistik durasi audio:")
print(df["duration_seconds"].describe())

# Simpan info dataset lengkap
df.to_csv(
    os.path.join(REPORT_DIR, "dataset_info.csv"),
    index=False
)

# =========================
# GRAFIK JUMLAH FILE PER SPEAKER
# =========================
plt.figure(figsize=(14, 5))
plt.bar(speaker_counts["speaker"], speaker_counts["jumlah_file"])
plt.xticks(rotation=90)
plt.title("Jumlah File Audio per Speaker")
plt.xlabel("Speaker")
plt.ylabel("Jumlah File")
plt.tight_layout()

plot_path = os.path.join(REPORT_DIR, "speaker_file_counts.png")
plt.savefig(plot_path)
plt.show()

print("\nFile hasil eksplorasi disimpan di folder reports:")
print("- speaker_file_counts.csv")
print("- dataset_info.csv")
print("- speaker_file_counts.png")