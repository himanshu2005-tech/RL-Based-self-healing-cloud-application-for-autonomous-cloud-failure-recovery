"""
Download parts of the public Google cluster trace 2011 (Borg `task_usage` table) and
reduce each one to 5-minute CPU and memory demand per job. The raw part is deleted after
it is reduced, so only the small aggregate is kept.

    python data_analysis/fetch_google_trace.py --parts 0 36 --jobs 3

The project uses parts 0-35 (about 2.5 days, starting at the beginning of the trace).

Output: data_analysis/trace/usage_part-XXXXX.csv.gz with columns
    window (5-minute index since trace start), job_id, cpu, mem
where cpu / mem are the summed task CPU rate / canonical memory usage, weighted by the
fraction of the window each measurement covers (normalised units, as in the trace).
Trace documentation: https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md
"""
import argparse
import os
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

URL = "https://commondatastorage.googleapis.com/clusterdata-2011-2/task_usage/part-{:05d}-of-00500.csv.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "trace")
WINDOW_US = 300_000_000
# task_usage columns: 0 start, 1 end, 2 job ID, 3 task index, 4 machine ID, 5 mean CPU rate,
# 6 canonical memory usage, ...
COLS = {0: "start", 1: "end", 2: "job_id", 5: "cpu", 6: "mem"}


def reduce_part(path):
    rows = []
    for chunk in pd.read_csv(path, header=None, usecols=list(COLS), names=None, chunksize=2_000_000):
        chunk = chunk.rename(columns=COLS)
        window = chunk.start // WINDOW_US
        # Weight each measurement by the share of its 5-minute window it covers
        weight = ((chunk.end - chunk.start) / WINDOW_US).clip(0, 1)
        g = pd.DataFrame({"window": window, "job_id": chunk.job_id,
                          "cpu": chunk.cpu.fillna(0) * weight, "mem": chunk.mem.fillna(0) * weight})
        rows.append(g.groupby(["window", "job_id"], as_index=False).sum())
    return pd.concat(rows).groupby(["window", "job_id"], as_index=False).sum()


def fetch(part, tmp_dir):
    out = os.path.join(OUT_DIR, f"usage_part-{part:05d}.csv.gz")
    if os.path.exists(out):
        return f"skip {part}"
    raw = os.path.join(tmp_dir, f"part-{part:05d}.csv.gz")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(URL.format(part), timeout=120) as r, open(raw, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            break
        except OSError as e:
            if attempt == 2:
                return f"FAILED {part}: {e}"
    try:
        df = reduce_part(raw)
        df.to_csv(out + ".tmp", index=False, compression="gzip")
        os.replace(out + ".tmp", out)
    finally:
        os.remove(raw)
    return f"ok {part}: windows {df.window.min()}-{df.window.max()}, {df.job_id.nunique()} jobs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=int, nargs=2, default=[0, 36], metavar=("FIRST", "STOP"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tmp", default=None, help="directory for raw downloads (default: system temp)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = args.tmp or tempfile.gettempdir()
    with ThreadPoolExecutor(args.jobs) as pool:
        for msg in pool.map(lambda p: fetch(p, tmp_dir), range(*args.parts)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
