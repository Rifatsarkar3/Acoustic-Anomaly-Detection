import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score
from audio_loader import MIMII_AcousticDataset
from cpu_model import AcousticAnomalyDetector

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
MACHINES = ["fan", "pump", "slider", "valve"]
SEED = 42; EPOCHS = 10; LR = 1e-4; BATCH = 32
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}
SPLITS = {
    "70/30": 0.70,
    "80/20": 0.80,  # baseline
    "85/15": 0.85,
}

def train_eval_split(machine, train_ratio):
    device = torch.device("cpu")
    ds = MIMII_AcousticDataset(data_dir=DATA_DIR, machine_type=machine)
    train_sz = int(train_ratio * len(ds))
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
    return dict(
        auc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
        acc=(y_pred == y_true).mean() * 100,
        n_test=len(test_ds)
    )

def generate_sensitivity_plots(results_dict, output_dir):
    """Generates high-res scientific plots for AUC, F1, and Accuracy trends."""
    print("\n[INFO] Generating publication-ready sensitivity plots...")
    
    splits = list(SPLITS.keys())
    metrics = [('auc', 'AUC Score', 'upper left'), 
               ('f1', 'F1 Score', 'lower left'), 
               ('acc', 'Accuracy (%)', 'lower left')]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Sensitivity Across Train/Test Splits (-6dB SNR)', fontsize=16, fontweight='bold', y=1.05)
    
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for ax_idx, (metric_key, metric_title, leg_loc) in enumerate(metrics):
        ax = axes[ax_idx]
        
        for m_idx, machine in enumerate(MACHINES):
            # Extract the metric data for this specific machine across all splits
            y_values = [results_dict[s][machine][metric_key] for s in splits]
            
            ax.plot(splits, y_values, marker=markers[m_idx], markersize=8, color=colors[m_idx], 
                    linewidth=2, label=machine.capitalize())
            
        ax.set_title(metric_title, fontsize=14)
        ax.set_xlabel('Train/Test Split Ratio', fontsize=12)
        ax.set_ylabel(metric_title, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc=leg_loc)
        
        # Highlight the 80/20 baseline split on the x-axis
        ax.axvline(x='80/20', color='black', linestyle=':', alpha=0.5, label='Baseline')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "split_sensitivity_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # 300 DPI for paper publication
    print(f"[SAVED GRAPH] {plot_path}")
    plt.close()

if __name__ == "__main__":
    print("SCRIPT 9 — Split Sensitivity: 70/30 vs 80/20 vs 85/15")
    
    checkpoint_file = os.path.join(OUT_DIR, "split_checkpoint.json")
    all_results = {}
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
            print("[INFO] Loaded previous progress from checkpoint.")

    for split_label, ratio in SPLITS.items():
        print(f"\n[{split_label} split]")
        if split_label not in all_results:
            all_results[split_label] = {}
            
        for machine in MACHINES:
            if machine in all_results[split_label]:
                r = all_results[split_label][machine]
                print(f"  {machine}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
                      f"Acc={r['acc']:.2f}%  TestN={r['n_test']} (Loaded from checkpoint)")
                continue
                
            r = train_eval_split(machine, ratio)
            all_results[split_label][machine] = r
            print(f"  {machine}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
                  f"Acc={r['acc']:.2f}%  TestN={r['n_test']}")
            
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)

    out = os.path.join(OUT_DIR, "split_sensitivity.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("SPLIT SENSITIVITY — MIMII -6dB | pos_weight | Seed=42\n")
        f.write("="*60+"\n\n")
        f.write(f"{'Split':<8} {'Machine':<10} {'AUC':>7} {'F1':>7} "
                f"{'Acc%':>7} {'TestN':>7}\n")
        f.write("-"*50+"\n")
        for split_label in SPLITS:
            for machine in MACHINES:
                r = all_results[split_label][machine]
                marker = " ◄ baseline" if split_label == "80/20" else ""
                f.write(f"{split_label:<8} {machine:<10} {r['auc']:>7.4f} "
                        f"{r['f1']:>7.4f} {r['acc']:>7.2f} {r['n_test']:>7}{marker}\n")
            f.write("\n")
    print(f"\n[SAVED TEXT] {out}")
    
    # Generate the plots at the very end
    generate_sensitivity_plots(all_results, OUT_DIR)