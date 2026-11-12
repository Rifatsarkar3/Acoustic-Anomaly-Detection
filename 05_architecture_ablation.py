"""
REVIEW SCRIPT 5 — CNN Architecture Ablation (Resumable)
============================================
Tests 6 architecture variants to prove the 3-block design is intentional:

Variants:
  A: 2 Conv blocks + FC-128 (under-capacity)
  B: 3 Conv blocks + FC-128 (proposed — current model)
  C: 4 Conv blocks + FC-128 (over-capacity)
  D: 3 Conv blocks + FC-64  (smaller classifier)
  E: 3 Conv blocks + FC-256 (larger classifier)
  F: 2 Conv blocks + FC-64  (minimum viable)

Reports: AUC, F1, Acc, Latency (ms), Parameter count

Outputs:
  - architecture_ablation_checkpoint.json (state tracking)
  - architecture_ablation.txt
  - architecture_ablation_plot.png

Usage:
    python 05_architecture_ablation.py
"""

import os, time, torch, json, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score
from audio_loader import MIMII_AcousticDataset

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
MACHINES = ["fan", "pump", "slider", "valve"]
SEED     = 42
EPOCHS   = 10
LR       = 1e-4
BATCH    = 32
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}

# ── Dynamic architecture builder ─────────────────────────────
def build_model(n_blocks, fc_size):
    """Builds a 2D-CNN with n_blocks conv blocks and fc_size hidden units."""
    channels = [1] + [16 * (2**i) for i in range(n_blocks)]
    layers   = []
    for i in range(n_blocks):
        in_c  = channels[i]
        out_c = channels[i+1]
        layers += [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ]

    # Compute flattened size: input (1, 40, 400) → MaxPool2d(2,2) × n_blocks
    freq_out = 40 // (2 ** n_blocks)
    time_out = 400 // (2 ** n_blocks)
    flat_dim = channels[-1] * freq_out * time_out

    class DynamicCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features   = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, fc_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fc_size, 1)
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    model     = DynamicCNN()
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_params, flat_dim


VARIANTS = {
    "2B-FC64" : (2, 64),
    "2B-FC128": (2, 128),
    "3B-FC64" : (3, 64),
    "3B-FC128": (3, 128),   # ← proposed
    "3B-FC256": (3, 256),
    "4B-FC128": (4, 128),
}


def train_eval_variant(n_blocks, fc_size, machine):
    device = torch.device("cpu")
    model, n_params, _ = build_model(n_blocks, fc_size)
    model = model.to(device)

    full_ds  = MIMII_AcousticDataset(data_dir=DATA_DIR, machine_type=machine)
    train_sz = int(0.8 * len(full_ds))
    gen      = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = random_split(
        full_ds, [train_sz, len(full_ds)-train_sz], generator=gen
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    pw        = torch.tensor([CLASS_RATIOS[machine]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        model.train()
        for mfccs, labels in train_loader:
            optimizer.zero_grad()
            criterion(model(mfccs), labels).backward()
            optimizer.step()

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for mfccs, labels in test_loader:
            probs = torch.sigmoid(model(mfccs)).squeeze().numpy()
            true  = labels.squeeze().numpy()
            if probs.ndim == 0:
                probs = np.array([probs]); true = np.array([true])
            all_probs.extend(probs.tolist()); all_labels.extend(true.tolist())

    y_true = np.array(all_labels); y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(float)

    # Latency benchmark (50 cycles, batch=16)
    torch.set_num_threads(2)
    dummy = torch.randn(16, 1, 40, 400)
    for _ in range(10): model(dummy)
    t0    = time.perf_counter()
    for _ in range(100): model(dummy)
    lat_ms = (time.perf_counter() - t0) / 100 * 1000

    return dict(
        auc      = roc_auc_score(y_true, y_prob),
        f1       = f1_score(y_true, y_pred, zero_division=0),
        acc      = (y_pred == y_true).mean() * 100,
        lat_ms   = lat_ms,
        n_params = n_params,
    )


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 65)
    print("REVIEW SCRIPT 5 — Architecture Ablation (Resumable)")
    print("=" * 65)

    ckpt_file = os.path.join(OUT_DIR, "architecture_ablation_checkpoint.json")
    all_results = {}

    # ── Load State Checkpoint ────────────────────────────────
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"[INFO] Resuming from checkpoint. Data found for {len(all_results)} variants.\n")
        except Exception as e:
            print(f"[WARN] Failed to read checkpoint. Starting fresh. ({e})\n")

    for var_name, (n_blocks, fc_size) in VARIANTS.items():
        if var_name not in all_results:
            all_results[var_name] = {}
            
        print(f"\n[Variant: {var_name}]  blocks={n_blocks}  fc={fc_size}")
        
        for machine in MACHINES:
            # Check if this specific machine run is already cached
            if machine in all_results[var_name]:
                r = all_results[var_name][machine]
                print(f"  {machine:<8}: [SKIPPED] AUC={r['auc']:.4f}  F1={r['f1']:.4f} (Loaded from checkpoint)")
                continue

            # Run training and evaluation
            r = train_eval_variant(n_blocks, fc_size, machine)
            all_results[var_name][machine] = r
            
            print(f"  {machine:<8}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
                  f"Lat={r['lat_ms']:.1f}ms  Params={r['n_params']:,}")
            
            # Save state immediately to prevent data loss on crash
            with open(ckpt_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)

    # ── Save Final Text File ─────────────────────────────────
    out_txt = os.path.join(OUT_DIR, "architecture_ablation.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CNN ARCHITECTURE ABLATION — MIMII -6dB | pos_weight applied\n")
        f.write("Proposed model: 3B-FC128  |  Seed=42  |  Epochs=10\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Variant':<12} {'Machine':<10} {'AUC':>7} {'F1':>7} "
                f"{'Acc%':>7} {'Lat(ms)':>9} {'Params':>10}\n")
        f.write("-" * 70 + "\n")
        for var_name in VARIANTS:
            for machine in MACHINES:
                # Ensure we only write if data successfully completed
                if machine in all_results.get(var_name, {}):
                    r = all_results[var_name][machine]
                    marker = " \u25c4" if var_name == "3B-FC128" else "" 
                    f.write(f"{var_name:<12} {machine:<10} {r['auc']:>7.4f} "
                            f"{r['f1']:>7.4f} {r['acc']:>7.2f} "
                            f"{r['lat_ms']:>9.1f} {r['n_params']:>10,}{marker}\n")
            f.write("\n")

    # ── Plot: AUC vs Latency trade-off ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#AAAAAA','#AAAAAA','#AAAAAA','#E63946','#AAAAAA','#AAAAAA']

    avg_aucs = []
    avg_lats = []
    
    completed_variants = [v for v in VARIANTS if len(all_results.get(v, {})) == len(MACHINES)]
    
    for var_name in completed_variants:
        aucs = [all_results[var_name][m]['auc'] for m in MACHINES]
        lats = [all_results[var_name][m]['lat_ms'] for m in MACHINES]
        avg_aucs.append(np.mean(aucs))
        avg_lats.append(np.mean(lats))

    for ax_idx, (ax, metric_vals, ylabel) in enumerate(
        [(axes[0], avg_aucs, 'Mean AUC-ROC'),
         (axes[1], avg_lats, 'Mean Batch Latency (ms)')]
    ):
        bars = ax.bar(completed_variants, metric_vals,
                      color=['#E63946' if v == '3B-FC128' else '#B5D4F4'
                             for v in completed_variants],
                      width=0.6, alpha=0.9)
        for bar, val in zip(bars, metric_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.005,
                    f'{val:.3f}' if ax_idx == 0 else f'{val:.1f}ms',
                    ha='center', fontsize=8, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} by Architecture Variant', fontsize=11, fontweight='bold')
        ax.set_xticklabels(completed_variants, rotation=30, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Architecture Ablation — MIMII −6 dB SNR (red bar = proposed)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out_plot = os.path.join(OUT_DIR, "architecture_ablation_plot.png")
    plt.savefig(out_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[SAVED] {out_txt}")
    print(f"[SAVED] {out_plot}")
    print(f"[SAVED] {ckpt_file}")