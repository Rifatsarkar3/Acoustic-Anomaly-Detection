"""Run the script-19 experiments in priority order as each machine's cached features appear.
Re-checks every minute; each (experiment, machine) call resumes from its JSON, so it can be restarted."""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.environ.get("MIMII_FEATURES", r"E:\MIMII\features")
ORDER = ["equiv", "factorial", "arch", "split", "snr", "enroll", "valcal"]
MACHINES = ["valve", "fan", "pump", "slider"]
N_RUNS = {"equiv": 20, "factorial": 20, "arch": 50, "split": 20, "snr": 20, "enroll": 60, "valcal": 30}  # per machine


def ready(exp, m):
    tags = ["0", "p6"] if exp == "snr" else ["m6"]
    return all(os.path.exists(os.path.join(FEAT, t, f"{m}_logmel.npy")) for t in tags)


def complete(exp, m):
    p = os.path.join(ROOT, "results", f"revision_{exp}.json")
    return os.path.exists(p) and sum(r["machine"] == m for r in json.load(open(p))) >= N_RUNS[exp]


while True:
    todo = [(e, m) for e in ORDER for m in MACHINES if not complete(e, m)]
    if not todo:
        print("QUEUE DONE", flush=True); break
    runnable = [(e, m) for e, m in todo if ready(e, m)]
    if not runnable:
        time.sleep(60); continue
    e, m = runnable[0]
    print(time.strftime("%H:%M:%S"), "->", e, m, flush=True)
    subprocess.run([sys.executable, "-u", os.path.join(ROOT, "19_revision_experiments.py"), "--exp", e, "--machines", m])
