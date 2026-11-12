"""
SCRIPT 20 - Published results used in the literature comparison table
=====================================================================
Per-ID values are transcribed from the cited papers (source table given per entry) and averaged
here, so every number in the manuscript's comparison table can be traced to a printed table.
No training. Output: results/literature_comparison.json and .txt
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MACH = ["fan", "pump", "slider", "valve"]

# Purohit et al., DCASE 2019 (ref [1]), Table 3, AUC at -6 dB; the released dataset contains IDs 00/02/04/06
MIMII_AE_M6 = {"fan": {"00": .57, "02": .68, "04": .57, "06": .83}, "pump": {"00": .58, "02": .52, "04": .93, "06": .61},
               "slider": {"00": .93, "02": .74, "04": .61, "06": .52}, "valve": {"00": .62, "02": .57, "04": .50, "06": .53}}
# Mueller et al., ICAART 2021 (ref [14]), Table 1, ResNet34 + GMM (best extractor and detector), -6 dB, AUC %
MUELLER_R34_GMM = {"fan": [58.7, 65.6, 57.0, 90.9], "pump": [78.4, 66.8, 87.9, 63.2],
                   "slider": [99.6, 90.4, 82.5, 69.1], "valve": [73.0, 79.1, 60.1, 61.9]}
# DCASE 2020 Task 2 development set (dev IDs 00/02/04/06): baseline from Koizumi et al. 2020 (ref [3]),
# Table 1 (mean of 10 trials); the other rows from Liu et al. ICASSP 2022 (ref [7]), Table 2 (AUC %)
DCASE20_AE = {"fan": [54.41, 73.40, 61.61, 73.92], "pump": [67.15, 61.53, 88.33, 74.55],
              "slider": [96.19, 78.97, 94.30, 69.59], "valve": [68.76, 68.18, 74.30, 53.90]}
STGRAM_TABLE2 = {  # machine-type averages as printed
    "IDNN [10]": {"fan": 67.71, "pump": 73.76, "slider": 86.45, "valve": 84.09},
    "Self-supervised MobileNetV2 [6]": {"fan": 80.19, "pump": 82.53, "slider": 95.27, "valve": 88.65},
    "STgram-MFN (ArcFace) [7]": {"fan": 94.04, "pump": 91.94, "slider": 99.55, "valve": 99.64},
}
# Zhang and Lai, Machines 2026 (ref [8]), Table 3: Tiny-AST, pump subset, -6 dB, IDs pooled, random 7:2:1 split
TINY_AST_PUMP_M6 = 0.994595

rows = []
rows.append(dict(method="Autoencoder baseline [1]", data="MIMII, -6 dB", labels="normal only",
                 **{m: float(np.mean(list(MIMII_AE_M6[m].values()))) for m in MACH}))
rows.append(dict(method="Image transfer learning, ResNet34 + GMM [14]", data="MIMII, -6 dB", labels="normal only",
                 **{m: float(np.mean(MUELLER_R34_GMM[m]) / 100) for m in MACH}))
rows.append(dict(method="Tiny-AST [8]", data="MIMII pump, -6 dB", labels="normal + faults",
                 fan=None, pump=TINY_AST_PUMP_M6, slider=None, valve=None))
rows.append(dict(method="Autoencoder baseline [3]", data="DCASE 2020 dev.", labels="normal only",
                 **{m: float(np.mean(DCASE20_AE[m]) / 100) for m in MACH}))
for name, d in STGRAM_TABLE2.items():
    rows.append(dict(method=name, data="DCASE 2020 dev.", labels="normal only (+ machine-ID labels)",
                     **{m: d[m] / 100 for m in MACH}))
for r in rows:
    vals = [r[m] for m in MACH if r[m] is not None]
    r["mean"] = float(np.mean(vals)) if len(vals) == 4 else None

json.dump(rows, open(os.path.join(OUT, "literature_comparison.json"), "w"), indent=1)
with open(os.path.join(OUT, "literature_comparison.txt"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(f"{r['method']:<46} {r['data']:<18} " + " ".join(
            f"{m}={r[m]:.3f}" if r[m] is not None else f"{m}=  -  " for m in MACH + ["mean"]) + "\n")
print(open(os.path.join(OUT, "literature_comparison.txt"), encoding="utf-8").read())
