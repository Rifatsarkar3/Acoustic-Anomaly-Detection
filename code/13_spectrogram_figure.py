"""
SCRIPT 13 - Log-Mel spectrogram exemplar figure
================================================
Plots a normal vs. abnormal 64-band Log-Mel spectrogram for each of the
four MIMII machine types at -6 dB SNR (2 rows x 4 columns). No training.

Output: results/spectrogram_exemplars.png
"""

import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "MIMII_Dataset_-6dB")
OUT_PATH = os.path.join(ROOT, "results", "spectrogram_exemplars.png")
MACHINES = ["fan", "pump", "slider", "valve"]
N_MELS   = 64


def first_wav(machine, condition):
    mdir = os.path.join(DATA_DIR, machine)
    for mid in sorted(os.listdir(mdir)):
        cp = os.path.join(mdir, mid, condition)
        if os.path.isdir(cp):
            wavs = sorted(f for f in os.listdir(cp) if f.endswith(".wav"))
            if wavs:
                return os.path.join(cp, wavs[0]), mid
    raise FileNotFoundError(f"no {condition} wav for {machine}")


def logmel(path):
    sig, sr = librosa.load(path, sr=16000)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=N_MELS), ref=np.max)
    return mel, sr


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=False)
    for col, machine in enumerate(MACHINES):
        for row, cond in enumerate(["normal", "abnormal"]):
            path, mid = first_wav(machine, cond)
            mel, sr = logmel(path)
            ax = axes[row, col]
            img = librosa.display.specshow(
                mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma",
                vmin=-80, vmax=0)
            ax.set_title(f"{machine.capitalize()} ({mid}) - {cond}",
                         fontsize=11, fontweight="bold")
            if col > 0:
                ax.set_ylabel("")
            if row == 0:
                ax.set_xlabel("")
    cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.9,
                        format="%+2.0f dB")
    cbar.set_label("Log-Mel power (dB)", fontsize=10)
    fig.suptitle("Log-Mel spectrograms (64 bands) of normal vs. abnormal "
                 "recordings, MIMII -6 dB SNR", fontsize=13, fontweight="bold")
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {OUT_PATH}")
