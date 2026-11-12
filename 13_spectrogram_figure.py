"""
SCRIPT 13 - Log-Mel spectrogram exemplar figure
================================================
Normal vs. abnormal 64-band log-Mel spectrogram for each MIMII machine type at -6 dB SNR
(2 rows x 4 columns), drawn at the printed width (13.86 cm) with labelled axis units. No training.

Exemplar = first file (sorted) of the first machine ID, as before. It is read from the unzipped
dataset if present, otherwise from the exemplar WAVs saved by utils/cache_features.py.

Output: results/spectrogram_exemplars.png
"""

import glob, os
import numpy as np
import librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "MIMII_Dataset_-6dB")
EXEMPLARS = os.environ.get("MIMII_EXEMPLARS", r"E:\MIMII\exemplars\m6")
OUT_PATH = os.path.join(ROOT, "results", "spectrogram_exemplars.png")
MACHINES = ["fan", "pump", "slider", "valve"]
N_MELS   = 64
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def first_wav(machine, condition):
    mdir = os.path.join(DATA_DIR, machine)
    if os.path.isdir(mdir):
        for mid in sorted(os.listdir(mdir)):
            cp = os.path.join(mdir, mid, condition)
            if os.path.isdir(cp):
                wavs = sorted(f for f in os.listdir(cp) if f.endswith(".wav"))
                if wavs:
                    return os.path.join(cp, wavs[0]), mid
    hits = sorted(glob.glob(os.path.join(EXEMPLARS, f"{machine}_id_*_{condition}_*.wav")))
    if hits:
        return hits[0], "id_" + os.path.basename(hits[0]).split("_id_")[1][:2]
    raise FileNotFoundError(f"no {condition} wav for {machine}")


def logmel(path):
    sig, sr = librosa.load(path, sr=16000)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=N_MELS), ref=np.max)
    return mel, sr


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 4, figsize=(13.86 / 2.54, 3.3), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.12, right=0.83, bottom=0.13, top=0.92, wspace=0.08, hspace=0.12))
    for col, machine in enumerate(MACHINES):
        for row, cond in enumerate(["normal", "abnormal"]):
            path, mid = first_wav(machine, cond)
            mel, sr = logmel(path)
            ax = axes[row, col]
            img = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax,
                                           cmap="magma", vmin=-80, vmax=0)
            if row == 0:
                ax.set_title(f"{machine.capitalize()} ({mid})", fontsize=7.5, pad=3)
            ax.set_xlabel("Time (s)" if row == 1 else "")
            ax.set_ylabel(f"{cond.capitalize()}\nFrequency (Hz)" if col == 0 else "")
            ax.tick_params(labelsize=6.5, length=2)
    cax = fig.add_axes([0.845, 0.13, 0.015, 0.79])
    cbar = fig.colorbar(img, cax=cax, format="%+2.0f dB")
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Log-Mel power (dB)", fontsize=7)
    plt.savefig(OUT_PATH, dpi=600, facecolor="white")
    print(f"[SAVED] {OUT_PATH}")
