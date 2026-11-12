"""
SCRIPT 22 - STgram-MFN (Liu et al., ICASSP 2022) reproduced on the ten-seed partitions of this study
====================================================================================================
Self-supervised anomalous sound detection as published: a classifier is trained on NORMAL recordings only to
recognise the machine identifier (ArcFace loss), and a test clip is scored by the negative log-probability of its
own identifier. Network, input transform, loss, optimiser, schedule and anomaly score follow the official code
(github.com/liuyoude/STgram-MFN, commit b559782: net.py, trainer.py, run.py, dataset.py, utils.py, config.yaml):
  TgramNet + MobileFaceNet (DCASE2022 bottleneck setting), 128-band log-Mel (n_fft = win = 1024, hop 512, power 2,
  HTK mel scale, no filter normalisation, dB with amin 1e-10) + raw waveform, ArcFace m = 0.7, s = 30, cross-entropy,
  Adam lr 1e-4, batch 128, epochs 0..300, cosine annealing (T_max 300, eta_min 1e-5) stepped from epoch 20,
  score = -log_softmax(ArcFace logits incl. margin)[true identifier].
Differences, all stated in the manuscript:
  * data: MIMII -6 dB, four machine types x four units = 16 identifiers (official: DCASE 2020, 41 identifiers);
    training = the normal clips of each machine type's 80 % partition of the seed, test = its 20 % partition
    (normal + abnormal), i.e. exactly the partitions of every other model in this study; ten seeds;
  * the model after the last epoch is evaluated (the official trainer keeps the epoch with the best AUC on the
    test data, which would select on the test set);
  * the batch of 128 is processed as two micro-batches of 64 with gradient accumulation (the official setting runs
    on two GPUs with DataParallel, i.e. batch normalisation over 64 clips per GPU);
  * the mel transform is re-implemented with torch.stft (torchaudio is not installed), with torchaudio's formulas;
  * waveforms are read from a float16 cache (utils/cache_waveforms.py);
  * for speed, the convolutions run in bfloat16 mixed precision with channels-last tensors; the mel transform,
    the ArcFace margin and the loss stay in float32. cuDNN runs in deterministic mode, so runs are repeatable.
Metrics: AUC and pAUC (FPR <= 0.1) pooled over all test clips of a machine type (protocol of this study) and, for
reference, averaged over the four identifiers (DCASE convention).

Usage: python 22_stgram_mfn_reproduction.py [--seeds 42 ...] [--epochs 300] [--tag smoke]
Output: results/stgram_mfn_reproduction[_tag].json (one record per seed; resumable; checkpoints every 25 epochs)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, math, queue, threading, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
FEAT = os.environ.get("MIMII_FEATURES", r"E:\MIMII\features")
CKPT = os.environ.get("STGRAM_CKPT", r"E:\MIMII\stgram_ckpt")
OUT = os.path.join(ROOT, "results")
MACHINES = ["fan", "pump", "slider", "valve"]
SEEDS = [42, 0, 123, 456, 789, 1, 2024, 999, 7, 314]
SR, N_FFT, WIN, HOP, N_MELS, N_SAMPLES = 16000, 1024, 1024, 512, 128, 160000
BATCH, MICRO, LR, ARC_M, ARC_S, SCHED_START = 128, 64, 1e-4, 0.7, 30.0, 20
EVAL_EPOCHS = (50, 100, 150, 200, 250)                 # progress record only, never used for selection
DEV = torch.device("cuda")
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------------- official network (net.py)
class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False), nn.BatchNorm2d(inp * expansion), nn.PReLU(inp * expansion),
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion), nn.PReLU(inp * expansion),
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        return x + self.conv(x) if self.connect else self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super().__init__()
        self.linear = linear
        self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False) if dw else nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x if self.linear else self.prelu(x)


BOTTLENECK_SETTING = [[2, 128, 2, 2], [4, 128, 2, 2], [4, 128, 2, 2]]     # t, c, n, s (DCASE2022 Task 2 top-1)


class MobileFaceNet(nn.Module):
    def __init__(self, num_class, bottleneck_setting=BOTTLENECK_SETTING):
        super().__init__()
        self.conv1 = ConvBlock(2, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        layers = []
        for t, c, n, s in bottleneck_setting:
            for i in range(n):
                layers.append(Bottleneck(self.inplanes, c, s if i == 0 else 1, t)); self.inplanes = c
        self.blocks = nn.Sequential(*layers)
        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        self.fc_out = nn.Linear(128, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x, label):
        x = self.linear1(self.linear7(self.conv2(self.blocks(self.dw_conv1(self.conv1(x))))))
        feature = x.view(x.size(0), -1)
        return self.fc_out(feature), feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super().__init__()
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(*[nn.Sequential(nn.LayerNorm(313), nn.LeakyReLU(0.2, inplace=True),
                                                          nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False))
                                            for _ in range(num_layer)])

    def forward(self, x):
        return self.conv_encoder(self.conv_extrctor(x))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super().__init__()
        self.in_features, self.out_features, self.s, self.m, self.sub = in_features, out_features, s, m, sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > 0, phi, cosine) if self.easy_margin else torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s


class STgramMFN(nn.Module):
    def __init__(self, num_classes, c_dim=128, win_len=1024, hop_len=512, use_arcface=False, m=0.5, s=30, sub=1):
        super().__init__()
        self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes, m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes)

    def forward(self, x_wav, x_mel, label=None):
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):                 # speed: convolutions in bfloat16
            x = torch.cat((x_mel, self.tgramnet(x_wav).unsqueeze(1)), dim=1).contiguous(memory_format=torch.channels_last)
            out, feature = self.mobilefacenet(x, label)
        feature = feature.float()                                           # ArcFace margin and loss in float32
        if self.arcface:
            out = self.arcface(feature, label)
        return out.float(), feature


# ----------------------------------------------------------------------------- official Wave2Mel (utils.py)
class Wave2Mel(nn.Module):
    """torchaudio MelSpectrogram(16 kHz, n_fft 1024, win 1024, hop 512, 128 mels, power 2) + AmplitudeToDB('power'),
    re-implemented with torchaudio's own formulas (Hann window, centred reflect padding, HTK mel, norm None)."""

    def __init__(self):
        super().__init__()
        n_freqs = N_FFT // 2 + 1
        hz2mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
        all_freqs = torch.linspace(0, SR // 2, n_freqs)
        m_pts = torch.linspace(hz2mel(0.0), hz2mel(float(SR // 2)), N_MELS + 2)
        f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
        fb = torch.max(torch.zeros(1), torch.min((-1.0 * slopes[:, :-2]) / f_diff[:-1], slopes[:, 2:] / f_diff[1:]))
        self.register_buffer("fb", fb)                                   # (n_freqs, n_mels)
        self.register_buffer("window", torch.hann_window(WIN))

    def forward(self, x):                                                # (B, N) -> (B, 128, 313)
        spec = torch.stft(x, N_FFT, HOP, WIN, self.window, center=True, pad_mode="reflect", normalized=False,
                          onesided=True, return_complex=True).abs().pow(2.0)
        mel = torch.matmul(spec.transpose(-1, -2), self.fb).transpose(-1, -2)
        return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))


# ----------------------------------------------------------------------------- data
class Waves:
    """Row reader over the float16 waveform caches; file reads release the GIL, so a thread can prefetch."""

    def __init__(self):
        self.files, self.offset, self.meta = {}, {}, {}
        for m in MACHINES:
            path = os.path.join(FEAT, "m6", f"{m}_wave16.npy")
            mm = np.load(path, mmap_mode="r"); assert mm.shape[1] == N_SAMPLES and mm.dtype == np.float16
            self.offset[m] = mm.offset; del mm
            self.files[m] = path
            self.meta[m] = np.genfromtxt(os.path.join(FEAT, "m6", f"{m}_meta.csv"), delimiter=",", names=True,
                                         dtype=None, encoding="utf-8")
        self.row_bytes = N_SAMPLES * 2

    def read(self, items, out):
        """items: list of (machine, row); out: pinned float16 tensor (len(items), N_SAMPLES)."""
        buf = out.numpy()
        handles = {}
        for j, (m, r) in sorted(enumerate(items), key=lambda e: (e[1][0], e[1][1])):
            fh = handles.get(m) or handles.setdefault(m, open(self.files[m], "rb", buffering=0))
            fh.seek(self.offset[m] + r * self.row_bytes)
            fh.readinto(memoryview(buf[j]).cast("B"))
        for fh in handles.values():
            fh.close()
        return out


def split_8020(n, seed, frac=0.8):                                       # identical to scripts 12/15/19
    gen = torch.Generator().manual_seed(seed)
    tr, te = random_split(range(n), [int(frac * n), n - int(frac * n)], generator=gen)
    return list(tr.indices), list(te.indices)


def partitions(W, seed):
    train, tests, id_names = [], {}, []
    for mi, m in enumerate(MACHINES):
        meta = W.meta[m]
        ids = sorted(set(meta["machine_id"].astype(str)))
        id_names += [f"{m}-{i}" for i in ids]
        cls = np.array([mi * len(ids) + ids.index(i) for i in meta["machine_id"].astype(str)])
        tr, te = split_8020(len(meta), seed)
        train += [(m, r, int(cls[r])) for r in tr if meta["label"][r] == 0]          # normal clips only
        tests[m] = dict(rows=te, cls=cls[te], y=meta["label"][te].astype(int), uid=meta["machine_id"].astype(str)[te])
    return train, tests, id_names


def batches(W, items, order, pinned):
    """Yields (wave16 on GPU, class labels on GPU) per micro-batch; one batch is read ahead in a thread."""
    q = queue.Queue(maxsize=1)

    def producer():
        for k, start in enumerate(range(0, len(order), MICRO)):
            idx = order[start:start + MICRO]
            buf = pinned[k % 3][:len(idx)]                            # k in transfer, k+1 queued, k+2 being read
            W.read([(items[i][0], items[i][1]) for i in idx], buf)
            q.put((buf, torch.tensor([items[i][2] for i in idx])))
        q.put(None)

    th = threading.Thread(target=producer, daemon=True); th.start()
    while True:
        got = q.get()
        if got is None:
            break
        buf, lab = got
        xb = buf.to(DEV, non_blocking=True).float(); lb = lab.to(DEV)
        torch.cuda.current_stream().synchronize()                          # buffer may be reused by the producer
        yield xb, lb
    th.join()


@torch.no_grad()
def scores(net, w2m, W, test, pinned):
    net.eval()
    items = [(None, r, c) for r, c in zip(test["rows"], test["cls"])]
    out = []
    machine = test["machine"]
    for start in range(0, len(items), MICRO):
        chunk = items[start:start + MICRO]
        buf = W.read([(machine, it[1]) for it in chunk], pinned[0][:len(chunk)])
        x = buf.to(DEV).float(); lb = torch.tensor([it[2] for it in chunk], device=DEV)
        logits, _ = net(x, w2m(x), lb)
        out.append((-torch.log_softmax(logits, dim=1)).gather(1, lb.view(-1, 1)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def metrics(test, s):
    y, uid = test["y"], test["uid"]
    units = [u for u in sorted(set(uid)) if len(set(y[uid == u])) == 2]    # a unit needs both classes in its test part
    per_id = [(roc_auc_score(y[uid == u], s[uid == u]), roc_auc_score(y[uid == u], s[uid == u], max_fpr=0.1)) for u in units]
    return {"auc": roc_auc_score(y, s), "pauc": roc_auc_score(y, s, max_fpr=0.1),
            "auc_id_mean": float(np.mean([a for a, _ in per_id])), "pauc_id_mean": float(np.mean([p for _, p in per_id])),
            "auc_per_id": {u: a for u, (a, _) in zip(units, per_id)}}


def evaluate(net, w2m, W, tests, pinned):
    res = {}
    for m, t in tests.items():
        t = dict(t, machine=m)
        res[m] = metrics(t, scores(net, w2m, W, t, pinned))
    net.train()
    return res


def run_seed(W, seed, epochs, tag, log):
    train, tests, id_names = partitions(W, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    net = STgramMFN(num_classes=len(id_names), use_arcface=True, m=ARC_M, s=ARC_S, sub=1)   # init on the CPU
    net = net.to(DEV, memory_format=torch.channels_last); w2m = Wave2Mel().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.1 * LR)
    gen = torch.Generator().manual_seed(seed)
    ce = nn.CrossEntropyLoss()
    os.makedirs(CKPT, exist_ok=True)
    ck = os.path.join(CKPT, f"seed{seed}{'_' + tag if tag else ''}.pt")
    start, losses, progress = 0, [], {}
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=True)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        gen.set_state(st["gen"]); start, losses, progress = st["epoch"] + 1, st["losses"], st["progress"]
        log(f"  seed {seed}: resumed after epoch {st['epoch']}")
    pinned = [torch.empty((MICRO, N_SAMPLES), dtype=torch.float16).pin_memory() for _ in range(3)]
    per_batch = BATCH // MICRO
    t0 = time.time()
    for ep in range(start, epochs + 1):                                    # epochs 0..E inclusive, as the official loop
        net.train()
        order = torch.randperm(len(train), generator=gen).tolist()
        n, last = len(order), math.ceil(len(order) / MICRO) - 1
        tot = torch.zeros((), device=DEV)
        opt.zero_grad()
        for mi, (xb, lb) in enumerate(batches(W, train, order, pinned)):
            bsize = min(BATCH, n - (mi // per_batch) * BATCH)               # the last batch may be smaller
            logits, _ = net(xb, w2m(xb), lb)
            loss = ce(logits, lb)
            (loss * len(lb) / bsize).backward()                             # = mean loss over the whole batch
            tot += loss.detach() * len(lb)
            if (mi + 1) % per_batch == 0 or mi == last:
                opt.step(); opt.zero_grad()
        losses.append(float(tot) / n)
        if ep >= SCHED_START:
            sched.step()
        if ep in EVAL_EPOCHS:
            progress[str(ep)] = {m: v["auc"] for m, v in evaluate(net, w2m, W, tests, pinned).items()}
        if ep % 10 == 0 or ep == epochs:
            log(f"  seed {seed} epoch {ep:>3}: loss {losses[-1]:.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                f"[{time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB]"
                + (f"  test AUC {progress[str(ep)]}" if str(ep) in progress else ""))
        if (ep % 25 == 0 and ep > 0) or ep == epochs:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "gen": gen.get_state(), "epoch": ep, "losses": losses, "progress": progress}, ck)
    final = evaluate(net, w2m, W, tests, pinned)
    return {"seed": seed, "epochs": epochs, "n_train": len(train), "n_classes": len(id_names),
            "n_test": {m: len(t["rows"]) for m, t in tests.items()}, "final": final, "progress": progress,
            "train_loss": losses, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    path = os.path.join(OUT, f"stgram_mfn_reproduction{'_' + a.tag if a.tag else ''}.json")
    done = json.load(open(path)) if os.path.exists(path) else []
    log = lambda s: print(time.strftime("%H:%M:%S ") + s, flush=True)
    log(f"STgram-MFN reproduction | torch {torch.__version__} | {torch.cuda.get_device_name(0)} | epochs {a.epochs} "
        f"| {len(done)} seeds already in {os.path.basename(path)}")
    W = Waves()
    for seed in a.seeds:
        if any(r["seed"] == seed for r in done):
            continue
        r = run_seed(W, seed, a.epochs, a.tag, log)
        done.append(r); json.dump(done, open(path, "w"), indent=0)
        log(f"seed {seed} done: " + "  ".join(f"{m} AUC {v['auc']:.4f} (per-ID {v['auc_id_mean']:.4f})"
                                              for m, v in r["final"].items()) + f"  [{r['seconds']}s]")
    log("DONE")
