"""
SCRIPT 18 - Deployment benchmark of the PROPOSED (log-Mel) model, with the MFCC variant alongside
==================================================================================================
Revision experiment (reviewer requests: evaluate the log-Mel configuration under concurrent GPU
load; distinguish batch latency from single-prediction latency).

One campaign, both 3B-FC128 input variants, measured identically:
  A. Thread sweep (1, 2, 3, 4, 6 threads), batch 16, isolated vs. concurrent GPU stress,
     3 repetitions with alternating condition order (run-to-run variability).
  B. Latency distribution at 2 threads, 1,000 cycles, batch 16 AND batch 1 (batch 1 = response
     time of one prediction), isolated vs. concurrent.
  C. Feature-extraction time for one 10-s clip (log-Mel and MFCC, librosa), isolated vs. concurrent.
  D. CPU (2 threads) vs. GPU inference, batch 16, with GPU memory use.
GPU stress = ResNet-50 (random weights, batch 16, 224x224x3 FP32) in a separate process, back-to-back
forward passes. GPU utilisation and CPU load are sampled during every measurement and saved.

Run on an otherwise idle machine:  python 18_deployment_benchmark.py
Outputs: results/deployment_benchmark.json (all raw latencies + stats), results/deployment_benchmark.txt
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")     # librosa/numpy BLAS: same 2-thread budget as the model
os.environ.setdefault("MKL_NUM_THREADS", "2")
import glob, json, platform, threading, time
import numpy as np
import psutil
import torch, torch.nn as nn
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
EXEMPLAR = sorted(glob.glob(r"E:\MIMII\exemplars\m6\valve_id_00_normal_*.wav"))[0]
THREADS = [1, 2, 3, 4, 6]
VARIANTS = {"logmel": 64, "mfcc": 40}
MAX_LEN, WARMUP = 400, 50
QUICK = os.environ.get("BENCH_QUICK") == "1"   # functional dry run: few cycles, separate output
N = (lambda n: max(5, n // 50)) if QUICK else (lambda n: n)
TAG = "_quick" if QUICK else ""


def build_cnn(n_bins):
    """3B-FC128 as in scripts 12 (log-Mel) and cpu_model.py (MFCC)."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(), nn.Linear(64 * (n_bins // 8) * (MAX_LEN // 8), 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 1)).eval()


def gpu_stress(ready):
    import torchvision
    m = torchvision.models.resnet50(weights=None).cuda().eval()
    x = torch.randn(16, 3, 224, 224, device="cuda")
    with torch.no_grad():
        for _ in range(20):
            m(x)
        torch.cuda.synchronize(); ready.set()
        while True:
            m(x)


class Monitor:
    """Samples system CPU load and GPU utilisation every 0.25 s while a measurement runs."""
    def __init__(self):
        import pynvml
        pynvml.nvmlInit(); self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)

    def __enter__(self):
        self.cpu, self.gpu, self.stop = [], [], False
        psutil.cpu_percent(None)
        def loop():
            while not self.stop:
                time.sleep(0.25)
                self.cpu.append(psutil.cpu_percent(None))
                self.gpu.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
        self.t = threading.Thread(target=loop, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self.stop = True; self.t.join()

    def summary(self):
        return dict(cpu_load_mean=float(np.mean(self.cpu)) if self.cpu else None,
                    gpu_util_mean=float(np.mean(self.gpu)) if self.gpu else None,
                    gpu_util_min=float(np.min(self.gpu)) if self.gpu else None)


def time_model(n_bins, threads, batch, cycles, device="cpu"):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build_cnn(n_bins).to(device)
    x = torch.randn(batch, 1, n_bins, MAX_LEN, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    lats = []
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        sync()
        for _ in range(cycles):
            t0 = time.perf_counter(); model(x); sync()
            lats.append((time.perf_counter() - t0) * 1000.0)
    return np.array(lats)


def stats(lats, batch):
    return dict(batch=batch, cycles=len(lats), clips_per_s=float(len(lats) * batch / (lats.sum() / 1000.0)),
                mean_ms=float(lats.mean()), sd_ms=float(lats.std()), p50_ms=float(np.percentile(lats, 50)),
                p95_ms=float(np.percentile(lats, 95)), p99_ms=float(np.percentile(lats, 99)),
                max_ms=float(lats.max()))


def time_features(cycles=None):
    cycles = cycles or N(200)
    import librosa
    sig, sr = librosa.load(EXEMPLAR, sr=16000)
    out = {}
    for name, fn in [("logmel", lambda: librosa.power_to_db(librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=64), ref=np.max)),
                     ("mfcc", lambda: librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=40))]:
        for _ in range(10):
            fn()
        lats = []
        for _ in range(cycles):
            t0 = time.perf_counter(); fn(); lats.append((time.perf_counter() - t0) * 1000.0)
        out[name] = np.array(lats)
    return out


class Stress:
    def __enter__(self):
        ctx = mp.get_context("spawn"); ready = ctx.Event()
        self.p = ctx.Process(target=gpu_stress, args=(ready,), daemon=True); self.p.start()
        assert ready.wait(120), "GPU stress process did not start"
        time.sleep(2.0); return self

    def __exit__(self, *a):
        self.p.terminate(); self.p.join(); time.sleep(2.0)


def measure(results, key, cond, fn):
    with Monitor() as mon:
        val = fn()
    results.setdefault(key, {})[cond] = dict(val, **mon.summary()) if isinstance(val, dict) else val
    return results[key][cond]


if __name__ == "__main__":
    mp.freeze_support()
    assert not torch.cuda.is_initialized()
    res = dict(meta=dict(cpu=platform.processor(), logical_cpus=psutil.cpu_count(), torch=torch.__version__,
                         gpu=torch.cuda.get_device_name(0), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                         idle_cpu_load_before=psutil.cpu_percent(interval=3.0), exemplar=os.path.basename(EXEMPLAR)))
    print("idle CPU load before start: %.1f%%" % res["meta"]["idle_cpu_load_before"], flush=True)

    # A. thread sweep, batch 16, 3 repetitions, alternating which condition comes first
    for rep in range(3):
        order = ["isolated", "concurrent"] if rep % 2 == 0 else ["concurrent", "isolated"]
        for cond in order:
            ctx = Stress() if cond == "concurrent" else None
            if ctx: ctx.__enter__()
            try:
                for t in THREADS:
                    for v, nb in VARIANTS.items():
                        r = measure(res, f"A|{v}|t{t}|rep{rep}", cond, lambda: stats(time_model(nb, t, 16, N(500)), 16))
                        print(f"A rep{rep} {cond:<10} {v:<6} t={t}: {r['clips_per_s']:7.1f} clips/s  "
                              f"mean {r['mean_ms']:6.2f} ms  gpu {r['gpu_util_mean']:.0f}%  cpu {r['cpu_load_mean']:.0f}%", flush=True)
            finally:
                if ctx: ctx.__exit__()

    # B. latency distributions at 2 threads (batch 16 and batch 1); C. feature extraction; both conditions
    raw = {}
    for cond in ["isolated", "concurrent"]:
        ctx = Stress() if cond == "concurrent" else None
        if ctx: ctx.__enter__()
        try:
            for v, nb in VARIANTS.items():
                for b in (16, 1):
                    with Monitor() as mon:
                        lats = time_model(nb, 2, b, N(1000))
                    r = res.setdefault(f"B|{v}|b{b}", {})[cond] = dict(stats(lats, b), **mon.summary())
                    raw[f"B|{v}|b{b}|{cond}"] = np.round(lats, 4).tolist()
                    print(f"B {cond:<10} {v:<6} batch {b:>2}: mean {r['mean_ms']:6.2f} ms  p99 {r['p99_ms']:6.2f} ms  "
                          f"{r['clips_per_s']:7.1f} clips/s", flush=True)
            torch.set_num_threads(2)
            with Monitor() as mon:
                feats = time_features()
            for v, lats in feats.items():
                r = res.setdefault(f"C|{v}", {})[cond] = dict(stats(lats, 1), **mon.summary())
                print(f"C {cond:<10} {v:<6} feature extraction: mean {r['mean_ms']:6.2f} ms  p99 {r['p99_ms']:6.2f} ms", flush=True)
        finally:
            if ctx: ctx.__exit__()

    # D. CPU (2 threads) vs GPU, batch 16, isolated
    for v, nb in VARIANTS.items():
        measure(res, f"D|{v}|cpu", "isolated", lambda: stats(time_model(nb, 2, 16, N(500)), 16))
        torch.cuda.reset_peak_memory_stats()
        r = measure(res, f"D|{v}|gpu", "isolated", lambda: stats(time_model(nb, 2, 16, N(500), "cuda"), 16))
        r["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        print(f"D {v}: CPU {res[f'D|{v}|cpu']['isolated']['clips_per_s']:.1f} vs GPU {r['clips_per_s']:.1f} clips/s, "
              f"GPU memory {r['gpu_mem_mb']:.1f} MB", flush=True)

    # E. parameters and batch-16 latency of the six architecture variants (log-Mel input, 2 threads, isolated)
    def variant(nb_blocks, fc, n_bins=64):
        ch = [1] + [16 * 2 ** i for i in range(nb_blocks)]
        layers = []
        for i in range(nb_blocks):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(), nn.MaxPool2d(2, 2)]
        flat = ch[-1] * (n_bins // 2 ** nb_blocks) * (MAX_LEN // 2 ** nb_blocks)
        return nn.Sequential(*layers, nn.Flatten(), nn.Linear(flat, fc), nn.ReLU(), nn.Dropout(0.3), nn.Linear(fc, 1)).eval()
    torch.set_num_threads(2)
    for name, (nbk, fc) in {"2B-FC64": (2, 64), "2B-FC128": (2, 128), "3B-FC64": (3, 64), "3B-FC128": (3, 128),
                            "3B-FC256": (3, 256), "4B-FC128": (4, 128)}.items():
        m = variant(nbk, fc); x = torch.randn(16, 1, 64, MAX_LEN); lats = []
        with torch.no_grad():
            for _ in range(WARMUP):
                m(x)
            for _ in range(N(300)):
                t0 = time.perf_counter(); m(x); lats.append((time.perf_counter() - t0) * 1000.0)
        r = res.setdefault(f"E|{name}", {})["isolated"] = dict(stats(np.array(lats), 16),
                                                               params=sum(p.numel() for p in m.parameters()))
        print(f"E {name}: {r['params']:,} params, {r['mean_ms']:.2f} ms/batch", flush=True)

    res["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUT, f"deployment_benchmark{TAG}.json"), "w") as f:
        json.dump(dict(results=res, raw_latencies_ms=raw), f)
    with open(os.path.join(OUT, f"deployment_benchmark{TAG}.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(res["meta"], indent=1) + "\n\n")
        for key in sorted(k for k in res if k != "meta"):
            for cond, r in res[key].items():
                f.write(f"{key:<22} {cond:<10} " + "  ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()) + "\n")
    print(f"[SAVED] results/deployment_benchmark{TAG}.json, results/deployment_benchmark{TAG}.txt")
