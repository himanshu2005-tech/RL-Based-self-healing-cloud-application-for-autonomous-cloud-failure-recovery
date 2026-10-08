"""
Load series derived from the Google cluster trace 2011 (Borg), used to drive the v2 env.

The series is built by data_analysis/build_trace_series.py. Every algorithm uses the same
fixed chronological split: the first 50% of the windows for training, the next 20% for
validation (checkpoint selection, baseline tuning) and the last 30% for testing.
"""
import functools
import os

import numpy as np
import pandas as pd

SPLIT_FRACTIONS = {"train": (0.0, 0.5), "val": (0.5, 0.7), "test": (0.7, 1.0)}
TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_analysis", "trace")


@functools.lru_cache(maxsize=None)
def load_series(name):
    """Demand per 5-minute window, normalised to mean 1 over the training split."""
    df = pd.read_csv(os.path.join(TRACE_DIR, f"{name}.csv"))
    x = df["load"].to_numpy(dtype=float)
    lo, hi = split_bounds(len(x), "train")
    return x / x[lo:hi].mean()


def split_bounds(n, split):
    a, b = SPLIT_FRACTIONS[split]
    return int(round(a * n)), int(round(b * n))
