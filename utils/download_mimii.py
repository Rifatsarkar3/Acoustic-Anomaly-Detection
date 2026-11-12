"""Parallel, resumable downloader for MIMII zips from Zenodo (record 3384388).

Usage: python utils/download_mimii.py -6_dB_fan.zip -6_dB_pump.zip ...
Each file is fetched in 32 MB HTTP range chunks by 12 workers into a pre-sized
file; finished chunks are recorded in <file>.parts.json so an interrupted run
resumes. A file is renamed from .partial to its final name only after its MD5
matches the checksum published by Zenodo.
"""
import concurrent.futures as cf
import hashlib, json, os, sys, threading, time, urllib.request

API = 'https://zenodo.org/api/records/3384388'
OUT = os.environ.get('MIMII_ZIPS', r'E:\MIMII\zips')
CHUNK, WORKERS = 32 * 1024 * 1024, 12
HDRS = {'User-Agent': 'mimii-research-downloader/1.0'}


def get(url, start=None, end=None, timeout=90):
    h = dict(HDRS, **({'Range': f'bytes={start}-{end}'} if start is not None else {}))
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def fetch_chunk(url, start, end):
    err = None
    for k in range(30):
        try:
            data = get(url, start, end)
            if len(data) == end - start + 1:
                return data
            err = f'short read {len(data)}'
        except Exception as e:                       # network hiccup, 429, 5xx: back off and retry
            err = e
        time.sleep(min(120, 3 * 2 ** min(k, 5)))
    raise RuntimeError(f'chunk {start}-{end} failed after retries: {err}')


def download(meta):
    name, size, url = meta['key'], meta['size'], meta['links']['self']
    md5 = meta['checksum'].split(':', 1)[1]
    final, part, side = os.path.join(OUT, name), os.path.join(OUT, name + '.partial'), os.path.join(OUT, name + '.parts.json')
    if os.path.exists(final):
        print(f'[skip] {name} already complete', flush=True); return
    done = set(json.load(open(side))) if os.path.exists(side) else set()
    if not os.path.exists(part):
        with open(part, 'wb') as f:
            f.truncate(size)
    chunks = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    todo = [c for c in chunks if c[0] not in done]
    lock, t0, got = threading.Lock(), time.time(), [0]
    print(f'[get] {name}: {size/1e9:.2f} GB, {len(todo)}/{len(chunks)} chunks to fetch', flush=True)

    def work(c):
        data = fetch_chunk(url, *c)
        with lock:
            with open(part, 'r+b') as f:
                f.seek(c[0]); f.write(data)
            done.add(c[0]); got[0] += len(data)
            json.dump(sorted(done), open(side, 'w'))
            if len(done) % 20 == 0:
                rate = got[0] / max(1e-9, time.time() - t0) / 1e6
                print(f'    {name}: {len(done)}/{len(chunks)} chunks, {rate:.1f} MB/s', flush=True)

    with cf.ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    h = hashlib.md5()
    with open(part, 'rb') as f:
        for blk in iter(lambda: f.read(64 * 1024 * 1024), b''):
            h.update(blk)
    if h.hexdigest() != md5:
        os.remove(side)                              # force a clean re-download next run
        raise RuntimeError(f'{name}: MD5 mismatch ({h.hexdigest()} != {md5})')
    os.replace(part, final); os.remove(side)
    print(f'[ok] {name} verified (md5 {md5}) in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    files = {f['key']: f for f in json.loads(get(API))['files']}
    for n in sys.argv[1:]:
        download(files[n])
    print('ALL DONE', flush=True)
