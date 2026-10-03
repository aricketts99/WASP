#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:00:53 2026

@author: andrew
"""

from pathlib import Path
import pickle
import zstandard as zstd
import pandas as pd


import numpy as np
import sys

sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.multiarray"] = np.core.multiarray
sys.modules["numpy._core.numeric"] = np.core.numeric

# import matplotlib.pyplot as plt
# import seaborn as sns

# Folder containing the .zst files
results_dir = Path("results/temp/experiment_overlap")

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

            # R2
            "r2_OLS_error": result[-13],
            "r2_OLS_agg": result[-12],
            "r2_post_error": result[-11],
            "r2_post_agg": result[-10],

            # Hamming
            "weighted_hamming": result[-9],

            # OLS coefficient errors
            "error_OLS_error_comp": result[-8],
            "error_OLS_error_pop": result[-7],
            "error_OLS_agg_comp": result[-6],
            "error_OLS_agg_pop": result[-5],

            # Posterior/shrunk coefficient errors
            "error_post_error_comp": result[-4],
            "error_post_error_pop": result[-3],
            "error_post_agg_comp": result[-2],
            "error_post_agg_pop": result[-1],
        })

df = pd.DataFrame(rows)
rows = []

print(df.head())
print()
print(f"{len(df)} rows")

# # Average repeated observations
# plot_df = (
#     df.groupby(["L", "active", "p"], as_index=False)["objective"]
#       .mean()
# )

# # One subplot for each L
# Ls = sorted(plot_df["L"].unique())

# fig, axes = plt.subplots(
#     1, len(Ls),
#     figsize=(5 * len(Ls), 4),
#     sharex=True,
#     sharey=True
# )

# # Handle the case of only one L
# if len(Ls) == 1:
#     axes = [axes]

# for ax, L in zip(axes, Ls):
#     tmp = plot_df[plot_df["L"] == L]

#     heatmap = tmp.pivot(
#         index="p",
#         columns="active",
#         values="objective"
#     )

#     sns.heatmap(
#         heatmap,
#         ax=ax,
#         cmap="viridis",
#         annot=False,
#         cbar=True
#     )

#     ax.set_title(f"L = {L}")
#     ax.set_xlabel("Active")
#     ax.set_ylabel("p")

# plt.tight_layout()
# plt.show()


