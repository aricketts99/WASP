#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 08:37:36 2026

@author: andrew
"""

from pathlib import Path
import pickle
import zstandard as zstd
import pandas as pd


import numpy as np
import sys

# Folder containing the .zst files
results_dir = Path("results_07/")

dctx = zstd.ZstdDecompressor()

rows = []

for file in results_dir.glob("*.zst"):

    try:
        with open(file, "rb") as f:
            with dctx.stream_reader(f) as zf:
                output = pickle.load(zf)

    except EOFError:
        print(f"EOFError — incomplete pickle: {file}")
        continue

    except Exception as e:
        print(f"{type(e).__name__} — {file}")
        print(f"    {e}")
        continue

    setup = output["setup"]
    min_per_class = output["results"]

    for L, result in min_per_class.items():
        rows.append({
            "job_id": setup["job_id"],
            "seed": setup["seed"],
            "p": setup["p"],
            "active_p": setup["active_p"],
            "L": L,
            "objective": result[0],
            
            'weighted_sum_over_K':result[-4],
            'weighted_sum_over_L':result[-3],
            'weighted_R2':result[-2],
            'weighted_R2_test':result[-1],
        })

df = pd.DataFrame(rows)
rows = []

print(df.head())
print()
print(f"{len(df)} rows")

df.to_csv("out_file.csv", index=False)
