"""
Combine the reduced Google 2011 trace parts (fetch_google_trace.py) into one demand series:
total CPU usage of the cluster per 5-minute window.

    python data_analysis/build_trace_series.py

Output: data_analysis/trace/google2011_cpu.csv (window, hours, load). The env normalises
`load` to mean 1 over the training split (environment/trace.py).
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(HERE, "trace")


def main():
    parts = sorted(glob.glob(os.path.join(TRACE_DIR, "usage_part-*.csv.gz")))
    df = pd.concat(pd.read_csv(p) for p in parts)
    # A window can be split across two consecutive parts, so sum again after concatenating
    series = df.groupby("window").cpu.sum().sort_index()
    # Drop the partially covered first and last windows
    series = series.iloc[1:-1]
    full = np.arange(series.index.min(), series.index.max() + 1)
    missing = len(full) - len(series)
    series = series.reindex(full).interpolate()
    out = pd.DataFrame({"window": series.index, "hours": series.index * 5 / 60, "load": series.values})
    out.to_csv(os.path.join(TRACE_DIR, "google2011_cpu.csv"), index=False)

    x = out.load.to_numpy()
    def autocorr(lag):
        a = x - x.mean()
        return float((a[:-lag] * a[lag:]).sum() / (a * a).sum())
    print(f"{len(parts)} parts -> {len(out)} windows ({len(out) * 5 / 60 / 24:.2f} days), {missing} interpolated")
    print(f"load mean {x.mean():.0f}, cv {x.std() / x.mean():.3f}, min/max ratio {x.min() / x.mean():.2f}/{x.max() / x.mean():.2f}")
    print("autocorrelation  5 min %.2f | 1 h %.2f | 6 h %.2f | 24 h %.2f"
          % (autocorr(1), autocorr(12), autocorr(72), autocorr(288) if len(x) > 600 else float("nan")))


if __name__ == "__main__":
    main()
