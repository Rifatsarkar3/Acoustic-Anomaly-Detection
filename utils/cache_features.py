"""Stream WAVs out of a MIMII zip and cache the exact features used by the experiment scripts.

Usage: python utils/cache_features.py <zip> <snr_tag>   e.g. -6_dB_valve.zip m6  (tags: m6, 0, p6)
Writes to E:\\MIMII\\features\\<snr_tag>\\:
  <machine>_logmel.npy  (N,1,64,400) float32   - identical to scripts 12/15/17 extract_features()
  <machine>_mfcc.npy    (N,1,40,400) float32   - identical to audio_loader.MIMII_AcousticDataset
  <machine>_meta.csv    index, machine_id, condition, file, label, zip timestamp
Order matches the original os.listdir enumeration on NTFS: sorted machine IDs, then normal, then
abnormal, files sorted by name (NTFS returns these names in sorted order), so random_split with a
given seed reproduces the original partitions.
Also saves the first normal/abnormal WAV of the first machine ID (Figure 1 exemplars).
"""
import csv, io, os, sys, time, zipfile
import librosa
import numpy as np

N_MELS, N_MFCC, MAX_LEN, SR = 64, 40, 400, 16000
OUT = os.environ.get('MIMII_FEATURES', r'E:\MIMII\features')


def fit(m):
    return m[:, :MAX_LEN] if m.shape[1] > MAX_LEN else np.pad(m, ((0, 0), (0, MAX_LEN - m.shape[1])), mode='constant')


def logmel(sig, sr):                                  # scripts 12/15/17
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=N_MELS), ref=np.max)
    return fit((mel - mel.mean()) / (mel.std() + 1e-8))


def mfcc(sig, sr):                                    # audio_loader._extract_mfcc
    return fit(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=N_MFCC))


def main(zpath, tag):
    z = zipfile.ZipFile(zpath)
    entries = {}
    for info in z.infolist():
        parts = info.filename.replace('\\', '/').strip('/').split('/')
        if not parts[-1].endswith('.wav') or len(parts) < 4:
            continue
        machine, mid, cond, fname = parts[-4:]
        entries.setdefault(machine, []).append((mid, cond, fname, info))
    assert len(entries) == 1, entries.keys()
    machine, items = next(iter(entries.items()))
    order = []
    for mid in sorted({e[0] for e in items}):
        for cond, label in (('normal', 0), ('abnormal', 1)):
            order += [(mid, cond, f, info, label) for (m, c, f, info) in sorted(
                (e for e in items if e[0] == mid and e[1] == cond), key=lambda e: e[2])]
    n = len(order)
    X_lm = np.zeros((n, 1, N_MELS, MAX_LEN), np.float32)
    X_mf = np.zeros((n, 1, N_MFCC, MAX_LEN), np.float32)
    od = os.path.join(OUT, tag); os.makedirs(od, exist_ok=True)
    ex_dir = os.path.join(os.path.dirname(OUT), 'exemplars', tag); os.makedirs(ex_dir, exist_ok=True)
    first_id = order[0][0]
    t0 = time.time()
    with open(os.path.join(od, f'{machine}_meta.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['index', 'machine_id', 'condition', 'file', 'label', 'zip_datetime'])
        for i, (mid, cond, fname, info, label) in enumerate(order):
            data = z.read(info)
            sig, sr = librosa.load(io.BytesIO(data), sr=SR)
            X_lm[i, 0], X_mf[i, 0] = logmel(sig, sr), mfcc(sig, sr)
            w.writerow([i, mid, cond, fname, label, '%04d-%02d-%02d %02d:%02d:%02d' % info.date_time])
            if mid == first_id and fname == min(e[2] for e in items if e[0] == mid and e[1] == cond):
                open(os.path.join(ex_dir, f'{machine}_{mid}_{cond}_{fname}'), 'wb').write(data)
            if (i + 1) % 1000 == 0:
                print(f'  {machine}: {i + 1}/{n} [{time.time() - t0:.0f}s]', flush=True)
    np.save(os.path.join(od, f'{machine}_logmel.npy'), X_lm)
    np.save(os.path.join(od, f'{machine}_mfcc.npy'), X_mf)
    print(f'[done] {tag}/{machine}: {n} clips ({sum(o[4] for o in order)} abnormal) in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
