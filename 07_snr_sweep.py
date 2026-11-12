import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score
from audio_loader import MIMII_AcousticDataset
from cpu_model import AcousticAnomalyDetector

# IMPORTANT: Check your MIMII_Dataset folder.
# Each SNR level is a separate folder extracted from its zip.
# Expected structure:
#   MIMII_Dataset/
#     -6_dB_fan/fan/...    (already done)
#     0_dB_fan/fan/...     (extract 0_dB_fan.zip)
#     6_dB_fan/fan/...     (extract 6_dB_fan.zip)
#
# If all SNR folders extract into the same root, adjust SNR_DIRS below.

BASE_DIR = r"E:\\Yolo-Thermal\\Acoustic Anomaly Detection"
SNR_DIRS = {
    "-6 dB": os.path.join(BASE_DIR, "MIMII_Dataset_-6dB"),     
    "0 dB" : os.path.join(BASE_DIR, "MIMII_Dataset_0dB"),  
    "+6 dB": os.path.join(BASE_DIR, "MIMII_Dataset_6dB"),  
}
MACHINES = ["fan", "pump", "slider", "valve"]
SEED = 42; EPOCHS = 10; LR = 1e-4; BATCH = 32
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_eval_snr(data_dir, machine):
    device = torch.device("cpu")
    try:
        ds = MIMII_AcousticDataset(data_dir=data_dir, machine_type=machine)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None

    train_sz = int(0.8 * len(ds))
    gen = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = random_split(ds, [train_sz, len(ds)-train_sz], generator=gen)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = AcousticAnomalyDetector().to(device)
    pw = torch.tensor([CLASS_RATIOS[machine]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        model.train()
        for mfccs, labels in train_ld:
            optimizer.zero_grad(); criterion(model(mfccs), labels).backward(); optimizer.step()

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for mfccs, labels in test_ld:
            probs = torch.sigmoid(model(mfccs)).squeeze().numpy()
            true  = labels.squeeze().numpy()
            if probs.ndim == 0: probs = np.array([probs]); true = np.array([true])
            all_probs.extend(probs.tolist()); all_labels.extend(true.tolist())

    y_true = np.array(all_labels); y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(float)
    return dict(auc=roc_auc_score(y_true,y_prob),
                f1=f1_score(y_true,y_pred,zero_division=0),
                acc=(y_pred==y_true).mean()*100)


if __name__ == "__main__":
    print("SCRIPT 7 — SNR Robustness Sweep: -6dB, 0dB, +6dB")
    print("NOTE: Ensure 0dB and +6dB MIMII data is extracted first.")
    print("Update SNR_DIRS paths if your folder structure differs.\n")

    all_results = {snr: {} for snr in SNR_DIRS}
    for snr_label, data_dir in SNR_DIRS.items():
        print(f"\n[SNR: {snr_label}]")
        for machine in MACHINES:
            print(f"  Training {machine}...", end=" ", flush=True)
            r = train_eval_snr(data_dir, machine)
            all_results[snr_label][machine] = r
            if r: print(f"AUC={r['auc']:.4f}  F1={r['f1']:.4f}")

    out_txt = os.path.join(OUT_DIR, "snr_sweep_results.txt")
    with open(out_txt, "w") as f:
        f.write("SNR ROBUSTNESS SWEEP — MIMII Dataset | pos_weight | Seed=42\n")
        f.write("="*65+"\n\n")
        f.write(f"{'SNR':<8} {'Machine':<10} {'AUC':>7} {'F1':>7} {'Acc%':>7}\n")
        f.write("-"*45+"\n")
        for snr in SNR_DIRS:
            for m in MACHINES:
                r = all_results[snr][m]
                if r:
                    f.write(f"{snr:<8} {m:<10} {r['auc']:>7.4f} "
                            f"{r['f1']:>7.4f} {r['acc']:>7.2f}\n")
                else:
                    f.write(f"{snr:<8} {m:<10} {'N/A':>7} {'N/A':>7} {'N/A':>7}\n")

    # Plot: AUC vs SNR per machine
    snr_labels = list(SNR_DIRS.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"fan":"#2E86AB","pump":"#E63946","slider":"#2D6A4F","valve":"#6A0572"}
    for machine in MACHINES:
        aucs = [all_results[snr][machine]["auc"]
                if all_results[snr][machine] else None for snr in snr_labels]
        valid = [(x,y) for x,y in zip(snr_labels, aucs) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", color=colors[machine], linewidth=2.2,
                    markersize=8, label=machine.upper())
    ax.axhline(0.64, color="gray", linewidth=1.5, linestyle="--", alpha=0.7,
               label="AE Baseline (Purohit 2019) = 0.64")
    ax.set_xlabel("SNR Level", fontsize=12); ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("AUC-ROC vs SNR — CPU-Bound 2D-CNN\n(pos_weight applied, MIMII dataset)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"snr_sweep_plot.png"),
                dpi=300,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"\n[SAVED] {out_txt}")