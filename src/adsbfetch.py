"""Fetch adsb.lol globe_history days with parallel byte-range connections.

GitHub's release-asset host caps each connection (~130 KB/s on 2026-10-14,
against a 500 Mbit/s line) but not how many there are, so `adsb.py download`,
which streams one connection per day, can take days for a month. This fetches
each day's tar into a local file over many range requests, then hands it to the
unchanged `adsb.extract_day` as a file:// URL: the same bytes through the same
parser. Checked on 2026-01-31 against the cached streamed extraction: the same
798,900 points with identical values (the cached file predates the float32
`alt` cast and holds another row order), and identical trace features on all
4,602 departures (`adsb.dep_features` sorts the points first). It lives outside
adsb.py because the trace-feature cache keys hash adsb.py's source.

    python src/adsbfetch.py --start 2026-02-01 --end 2026-02-28 --streams 2 --conns 96 --alt-cut legacy
"""
import argparse
import os
import pathlib
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import adsb

# small ranges keep every connection busy to the end of a day: with 32 MB ranges
# a day's last few ranges ran on a handful of connections for minutes
CHUNK = 8 * 1024 * 1024
_local = {}                      # day -> local tar, read by the patched release_urls
_lock = threading.Lock()
_release_urls = adsb.release_urls


def _local_urls(day):
    with _lock:
        path = _local.get(day)
    return [path.resolve().as_uri()] if path is not None else _release_urls(day)


adsb.release_urls = _local_urls


def _size(url):
    req = urllib.request.Request(url, method="HEAD")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except OSError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"cannot size {url}")


def _fetch_range(url, path, offset, start, end):
    """Bytes start..end (inclusive) of url into path at offset + start, resuming
    after a dropped connection."""
    pos = start
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "r+b") as f:
                if r.status != 206:
                    raise RuntimeError(f"range not honoured (HTTP {r.status})")
                f.seek(offset + pos)
                while pos <= end:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b[:end + 1 - pos])
                    pos += len(b)
            if pos > end:
                return end - start + 1
        except OSError:
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"cannot fetch {url} bytes {pos}-{end}")


def fetch(urls, path, conns):
    """The day's assets, concatenated in order, into one local file."""
    sizes = [_size(u) for u in urls]
    with open(path, "wb") as f:
        f.truncate(sum(sizes))
    jobs, offset = [], 0
    for u, n in zip(urls, sizes):
        jobs += [(u, offset, s, min(s + CHUNK, n) - 1) for s in range(0, n, CHUNK)]
        offset += n
    with ThreadPoolExecutor(max_workers=conns) as ex:
        got = sum(ex.map(lambda j: _fetch_range(j[0], path, *j[1:]), jobs))
    if got != sum(sizes) or os.path.getsize(path) != sum(sizes):
        raise RuntimeError(f"{path}: {got} bytes fetched, {os.path.getsize(path)} on disk, "
                           f"{sum(sizes)} expected")
    return sum(sizes)


def fetch_day(day, conns, out_dir, legacy_cut):
    out_dir = pathlib.Path(out_dir)
    if (out_dir / f"{day}.parquet").exists():
        return f"{day}: exists"
    t = time.time()
    tar = out_dir / f"{day}.{os.getpid()}.tar.part"  # a second fetcher may meet on one day
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        n = fetch(_release_urls(day), tar, conns)
        print(f"{day}: fetched {n / 1e9:.2f} GB in {time.time() - t:.0f}s "
              f"({n / 1e6 / max(time.time() - t, 1):.1f} MB/s)", flush=True)
        with _lock:
            _local[day] = tar
        return adsb.extract_day(day, workers=6, out_dir=out_dir, legacy_cut=legacy_cut)
    finally:
        with _lock:
            _local.pop(day, None)
        for attempt in range(5):
            try:
                tar.unlink(missing_ok=True)
                break
            except PermissionError:  # the parser's handle may close a moment later
                time.sleep(2)


def _safe(day, conns, out_dir, legacy_cut):
    try:
        return fetch_day(day, conns, out_dir, legacy_cut)
    except Exception as e:
        return f"{day}: FAILED {e!r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--streams", type=int, default=2, help="days in flight at once")
    ap.add_argument("--conns", type=int, default=96, help="range connections per day")
    ap.add_argument("--reverse", action="store_true",
                    help="newest day first, to meet a forward fetcher over the same range")
    ap.add_argument("--out-dir", default=str(adsb.ADSB_DIR))
    ap.add_argument("--alt-cut", choices=["field", "legacy"], default="field",
                    help="as adsb.py download: every cached day so far used legacy")
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out_dir)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, a.end)]
    days = [d for d in days if not (out_dir / f"{d}.parquet").exists()]
    days = days[::-1] if a.reverse else days
    print(f"{len(days)} days, {a.streams} at once, {a.conns} connections each, into {out_dir}"
          f"{' (legacy altitude cut)' if a.alt_cut == 'legacy' else ''}", flush=True)
    with ThreadPoolExecutor(max_workers=a.streams) as ex:
        for msg in ex.map(lambda d: _safe(d, a.conns, out_dir, a.alt_cut == "legacy"), days):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
