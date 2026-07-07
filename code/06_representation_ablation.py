import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np, librosa
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from audio_loader import MIMII_AcousticDataset

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
MACHINES = ["fan", "pump", "slider", "valve"]
SEED = 42; EPOCHS = 10; LR = 1e-4; BATCH = 32
N_MELS = 64; N_MFCC = 40; MAX_LEN = 400
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}

class MIMII_LogMelDataset(Dataset):
    def __init__(self, data_dir, machine_type):
        self.data_dir = os.path.join(data_dir, machine_type)
        self.samples = []
        for mid in os.listdir(self.data_dir):
            id_path = os.path.join(self.data_dir, mid)
            if not os.path.isdir(id_path): continue
            for cond, lbl in [("normal", 0.0), ("abnormal", 1.0)]:
                cp = os.path.join(id_path, cond)
                if not os.path.exists(cp): continue
                for f in os.listdir(cp):
                    if f.endswith(".wav"):
                        self.samples.append({"path": os.path.join(cp, f), "label": lbl})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        y, sr = librosa.load(s["path"], sr=16000)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS), ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        if mel.shape[1] > MAX_LEN:
            mel = mel[:, :MAX_LEN]
        else:
            mel = np.pad(mel, ((0,0),(0, MAX_LEN-mel.shape[1])), mode="constant")
        return (torch.tensor(mel, dtype=torch.float32).unsqueeze(0),
                torch.tensor([s["label"]], dtype=torch.float32))


def build_cnn(n_freq_bins):
    flat = 64 * (n_freq_bins // 8) * (MAX_LEN // 8)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2))
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(flat,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
        def forward(self, x): return self.classifier(self.features(x))
    return CNN()


def train_eval(ds_class, n_bins, machine, label):
    device = torch.device("cpu")
    ds = ds_class(DATA_DIR, machine)
    train_sz = int(0.8 * len(ds))
    gen = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = random_split(ds, [train_sz, len(ds)-train_sz], generator=gen)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)
    model = build_cnn(n_bins).to(device)
    pw = torch.tensor([CLASS_RATIOS[machine]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train()
        for feat, lbl in train_ld:
            optimizer.zero_grad(); criterion(model(feat), lbl).backward(); optimizer.step()
        if (ep+1) % 5 == 0:
            print(f"  [{label}] {machine} epoch {ep+1}/{EPOCHS}")
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for feat, lbl in test_ld:
            probs = torch.sigmoid(model(feat)).squeeze().numpy()
            true  = lbl.squeeze().numpy()
            if probs.ndim == 0: probs = np.array([probs]); true = np.array([true])
            all_probs.extend(probs.tolist()); all_labels.extend(true.tolist())
    y_true = np.array(all_labels); y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(float)
    return dict(auc=roc_auc_score(y_true,y_prob), f1=f1_score(y_true,y_pred,zero_division=0),
                recall=recall_score(y_true,y_pred,zero_division=0),
                acc=(y_pred==y_true).mean()*100)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("SCRIPT 6 — Representation Ablation: MFCC vs Log-Mel")
    configs = {
        "MFCC (40 bins)":     (MIMII_AcousticDataset, N_MFCC),
        "Log-Mel (64 bands)": (MIMII_LogMelDataset,   N_MELS),
    }
    all_results = {}
    for feat_name, (ds_cls, n_bins) in configs.items():
        print(f"\n[{feat_name}]"); all_results[feat_name] = {}
        for machine in MACHINES:
            r = train_eval(ds_cls, n_bins, machine, feat_name)
            all_results[feat_name][machine] = r
            print(f"  {machine}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Recall={r['recall']:.4f}")

    out = os.path.join(OUT_DIR, "representation_ablation.txt")
    with open(out, "w") as f:
        f.write("REPRESENTATION ABLATION — MIMII -6dB | pos_weight | Seed=42\n")
        f.write("="*65+"\n\n")
        for feat_name in configs:
            f.write(f"[{feat_name}]\n")
            f.write(f"  {'Machine':<10} {'AUC':>7} {'F1':>7} {'Recall':>7} {'Acc%':>7}\n")
            aucs = []
            for m in MACHINES:
                r = all_results[feat_name][m]; aucs.append(r["auc"])
                f.write(f"  {m:<10} {r['auc']:>7.4f} {r['f1']:>7.4f} "
                        f"{r['recall']:>7.4f} {r['acc']:>7.2f}\n")
            f.write(f"  {'MEAN':<10} {np.mean(aucs):>7.4f}\n\n")

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    x = np.arange(len(MACHINES)); w = 0.35
    color_map = {"MFCC (40 bins)":"#2E86AB","Log-Mel (64 bands)":"#E63946"}
    for ax_idx, metric in enumerate(["auc","f1"]):
        ax = axes[ax_idx]
        for i,(feat_name,color) in enumerate(color_map.items()):
            vals = [all_results[feat_name][m][metric] for m in MACHINES]
            ax.bar(x+i*w, vals, w, label=feat_name, color=color, alpha=0.85)
        ax.set_xlabel("Machine Type",fontsize=11); ax.set_ylabel(metric.upper(),fontsize=11)
        ax.set_title(f"{metric.upper()} — MFCC vs Log-Mel",fontsize=11,fontweight="bold")
        ax.set_xticks(x+w/2); ax.set_xticklabels([m.upper() for m in MACHINES])
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3,linestyle="--",axis="y")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Input Representation Ablation",fontsize=11,fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"representation_ablation_plot.png"),
                dpi=300,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"\n[SAVED] {out}")