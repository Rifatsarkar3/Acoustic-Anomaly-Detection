"""Cache the mono 16-kHz waveforms of one MIMII zip for the STgram-MFN reproduction (script 22).

Usage: python utils/cache_waveforms.py <zip> <snr_tag>      e.g. E:\\MIMII\\zips\\-6_dB_valve.zip m6
Writes E:\\MIMII\\features\\<snr_tag>\\<machine>_wave16.npy, shape (N, 160000), float16, written through a memory map
(no large RAM use). Rows follow <machine>_meta.csv written by utils/cache_features.py (checked file by file).
Loading matches the official STgram-MFN dataset: librosa.load(sr=16000, mono=True), first 10 s.
float16 keeps about 11 bits of mantissa (quantization noise about 66 dB below the signal), which is negligible at
the -6 dB SNR of the recordings.
"""
import csv, io, os, sys, time, zipfile
import librosa
import numpy as np

SR, N_SAMPLES = 16000, 160000
OUT = os.environ.get('MIMII_FEATURES', r'E:\MIMII\features')


def main(zpath, tag):
    z = zipfile.ZipFile(zpath)
    by_name = {}
    machine = None
    for info in z.infolist():
        parts = info.filename.replace('\\', '/').strip('/').split('/')
        if parts[-1].endswith('.wav') and len(parts) >= 4:
            machine = parts[-4]
            by_name[tuple(parts[-3:])] = info                   # (machine_id, condition, file)
    meta = list(csv.DictReader(open(os.path.join(OUT, tag, f'{machine}_meta.csv'), newline='')))
    assert len(meta) == len(by_name), (len(meta), len(by_name))
    path = os.path.join(OUT, tag, f'{machine}_wave16.npy')
    W = np.lib.format.open_memmap(path + '.part', mode='w+', dtype=np.float16, shape=(len(meta), N_SAMPLES))
    t0 = time.time()
    for i, row in enumerate(meta):
        assert int(row['index']) == i
        info = by_name[(row['machine_id'], row['condition'], row['file'])]
        sig, _ = librosa.load(io.BytesIO(z.read(info)), sr=SR, mono=True)
        assert len(sig) >= N_SAMPLES, (row, len(sig))
        W[i] = sig[:N_SAMPLES].astype(np.float16)
        if (i + 1) % 1000 == 0:
            print(f'  {machine}: {i + 1}/{len(meta)} [{time.time() - t0:.0f}s]', flush=True)
    W.flush(); del W
    os.replace(path + '.part', path)
    print(f'[done] {tag}/{machine}: {len(meta)} waveforms in {time.time() - t0:.0f}s -> {path}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
