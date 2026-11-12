import os, time, torch
import numpy as np
from cpu_model import AcousticAnomalyDetector

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 16
NUM_CYCLES = 500
WARMUP     = 20


def benchmark_device(device_name):
    device = torch.device(device_name)
    model  = AcousticAnomalyDetector().to(device)
    model.eval()
    dummy  = torch.randn(BATCH_SIZE, 1, 40, 400).to(device)

    # Record VRAM before
    vram_before = torch.cuda.memory_allocated() if device_name == "cuda" else 0

    for _ in range(WARMUP):
        with torch.no_grad(): _ = model(dummy)

    if device_name == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(NUM_CYCLES):
        if device_name == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): _ = model(dummy)
        if device_name == "cuda": torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    total_time = sum(latencies) / 1000
    fps        = (NUM_CYCLES * BATCH_SIZE) / total_time
    lats       = np.array(latencies)
    vram_after = torch.cuda.memory_allocated() if device_name == "cuda" else 0

    return dict(
        device      = device_name.upper(),
        fps         = fps,
        lat_mean_ms = np.mean(lats),
        lat_p95_ms  = np.percentile(lats, 95),
        lat_p99_ms  = np.percentile(lats, 99),
        vram_bytes  = vram_after,
        vram_mb     = vram_after / (1024**2),
    )


if __name__ == "__main__":
    print("SCRIPT 8 — CPU vs GPU Acoustic Inference Benchmark")
    print("="*55)

    results = {}

    # CPU benchmark (2 threads — proposed config)
    torch.set_num_threads(2)
    print("\nBenchmarking CPU (2 threads)...")
    results["CPU"] = benchmark_device("cpu")
    r = results["CPU"]
    print(f"  FPS={r['fps']:.2f}  Mean={r['lat_mean_ms']:.2f}ms  "
          f"P95={r['lat_p95_ms']:.2f}ms  VRAM={r['vram_bytes']}B")

    # GPU benchmark
    if torch.cuda.is_available():
        print("\nBenchmarking GPU (CUDA)...")
        results["GPU"] = benchmark_device("cuda")
        r = results["GPU"]
        print(f"  FPS={r['fps']:.2f}  Mean={r['lat_mean_ms']:.2f}ms  "
              f"P95={r['lat_p95_ms']:.2f}ms  VRAM={r['vram_mb']:.2f}MB")
    else:
        print("\n[GPU] CUDA not available.")
        results["GPU"] = {"fps": "N/A", "lat_mean_ms": "N/A",
                          "lat_p95_ms": "N/A", "vram_mb": "N/A",
                          "device": "GPU"}

    out = os.path.join(OUT_DIR, "cpu_vs_gpu_inference.txt")
    with open(out, "w") as f:
        f.write("CPU vs GPU ACOUSTIC INFERENCE — AcousticAnomalyDetector\n")
        f.write(f"Batch={BATCH_SIZE}, Cycles={NUM_CYCLES}, Warmup={WARMUP}\n")
        f.write("="*55+"\n\n")
        for label, r in results.items():
            f.write(f"[{label}]\n")
            for k, v in r.items():
                if k != "device":
                    f.write(f"  {k:<18}: {v}\n")
            f.write("\n")
        # Key comparison
        if "GPU" in results and isinstance(results["GPU"]["fps"], float):
            cpu_fps = results["CPU"]["fps"]
            gpu_fps = results["GPU"]["fps"]
            f.write(f"Throughput ratio (CPU/GPU): {cpu_fps/gpu_fps:.2f}x\n")
            f.write(f"VRAM savings vs GPU: "
                    f"{results['GPU']['vram_mb']:.2f} MB freed by CPU offload\n")
            f.write(f"\nKey finding: CPU acoustic inference at {cpu_fps:.1f} FPS with\n")
            f.write(f"0 bytes VRAM — enables concurrent GPU visual stream without\n")
            f.write(f"any memory contention.\n")

    print(f"\n[SAVED] {out}")