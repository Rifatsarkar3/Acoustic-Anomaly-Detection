"""
SCRIPT 16 - Consistent model complexity / CPU-throughput benchmark
==================================================================
Times the proposed 3B-FC128 (log-Mel) against the MobileNetV3 and
STgram-MFN(-Lite) baselines under ONE consistent setup (CPU, 2 threads,
batch 16, log-Mel input 1x64x400) so the params/latency/throughput
comparison in the manuscript is measured the same way for every model.
Inference only (no training).

Usage:
    python 16_model_complexity.py
"""
import os, sys, time
import numpy as np
import torch, torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "v2_scripts"))
from models import MobileNetV3Audio, STgramMFN_Lite  # noqa: E402

N_MELS, MAX_LEN = 64, 400
BATCH, THREADS = 16, 2
WARMUP, ITERS = 10, 100

torch.set_num_threads(THREADS)


def build_proposed():
    flat = 64 * (N_MELS // 8) * (MAX_LEN // 8)  # 64*8*50 = 25600
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def bench(name, model):
    model.eval()
    x = torch.randn(BATCH, 1, N_MELS, MAX_LEN)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)  # ms/batch
    ts = np.array(ts)
    lat = ts.mean()
    fps = BATCH / (lat / 1000.0)
    print(f"{name:<22} {nparams(model):>11,} {lat:>10.2f} {fps:>10.1f}")
    return nparams(model), lat, fps


if __name__ == "__main__":
    print(f"CPU complexity benchmark | threads={THREADS} batch={BATCH} "
          f"input=(1,{N_MELS},{MAX_LEN}) warmup={WARMUP} iters={ITERS}")
    print(f"{'Model':<22} {'Params':>11} {'Lat(ms)':>10} {'FPS':>10}")
    print("-" * 56)
    bench("MobileNetV3", MobileNetV3Audio(num_classes=1))
    bench("STgram-MFN (Lite)", STgramMFN_Lite(num_classes=1))
    bench("Proposed 3B-FC128", build_proposed())
