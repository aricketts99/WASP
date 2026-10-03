#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:15:30 2026

@author: andrew
"""

import numpy as np
import pickle
import zstandard as zstd
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Load fixed posterior samples
# ============================================================

def load_zst(filename):
    dctx = zstd.ZstdDecompressor()

    with open(filename, "rb") as f:
        with dctx.stream_reader(f) as reader:
            return pickle.load(reader)


samples = load_zst("samples_fixed.zst")

print("samples shape:", samples.shape)


# ============================================================
# BFDR / BFNR calculation
# ============================================================

def compute_bfdr_bfnr(samples, weights, neighbourhoods, alpha):
    """
    Compute posterior BFDR and BFNR for a Wasserstein posterior approximation.

    Parameters
    ----------
    samples : ndarray (n_samples, p)
        Original posterior binary samples.

    weights : ndarray (L,)
        Wasserstein cluster weights.

    neighbourhoods : ndarray (n_samples,)
        Cluster assignment for each posterior sample.

    alpha : float
        Hamming loss parameter.

    Returns
    -------
    bfdr : float
    bfnr : float
    """

    fp = 0.0
    tp = 0.0
    fn = 0.0

    L = len(weights)

    for k in range(L):

        # samples assigned to cluster k
        cluster = samples[neighbourhoods == k]

        if cluster.shape[0] == 0:
            continue

        # cluster-specific posterior inclusion probabilities
        pi_k = cluster.mean(axis=0)

        # Bayes decision under weighted Hamming loss
        selected = pi_k >= alpha / 2


        # expected false discoveries
        fp_k = np.sum(1 - pi_k[selected])

        # expected true discoveries
        tp_k = np.sum(pi_k[selected])

        # expected false non-discoveries
        fn_k = np.sum(pi_k[~selected])


        # weight by Wasserstein posterior mass
        fp += weights[k] * fp_k
        tp += weights[k] * tp_k
        fn += weights[k] * fn_k


    bfdr = fp / max(fp + tp, 1e-12)

    bfnr = fn / max(fn + tp, 1e-12)

    return bfdr, bfnr



# ============================================================
# Load all alpha results
# ============================================================

files = sorted(glob.glob("results/results_bfdr_stuff/results_alpha_*.zst"))

print("Number of result files:", len(files))


records = []


for file in files:

    output = load_zst(file)

    alpha = output["setup"]["alpha"]

    results = output["results"]


    for L, result in results.items():

        loss, weights, centres, neighbourhoods, B, l, _ = result

        bfdr, bfnr = compute_bfdr_bfnr(
            samples,
            weights,
            neighbourhoods,
            alpha
        )

        records.append(
            {
                "alpha": alpha,
                "L": int(L[0]),
                "BFDR": bfdr,
                "BFNR": bfnr,
            }
        )


df = pd.DataFrame(records)

df = df.sort_values(["L", "alpha"])

print(df.head())

df.to_csv("bfdr_bfnr_results.csv", index=False)



# ============================================================
# Plot BFDR vs alpha
# ============================================================

plt.figure(figsize=(8,6))

for L in sorted(df.L.unique()):

    temp = df[df.L == L]

    plt.plot(
        temp.alpha,
        temp.BFDR,
        marker="o",
        markersize=3,
        label=f"L={L}"
    )


plt.xlabel(r"$\alpha$")
plt.ylabel("BFDR")
plt.title("BFDR versus alpha")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig(
#     "BFDR_vs_alpha.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.close()



# ============================================================
# Plot BFNR vs alpha
# ============================================================

plt.figure(figsize=(8,6))

for L in sorted(df.L.unique()):

    temp = df[df.L == L]

    plt.plot(
        temp.alpha,
        temp.BFNR,
        marker="o",
        markersize=3,
        label=f"L={L}"
    )


plt.xlabel(r"$\alpha$")
plt.ylabel("BFNR")
plt.title("BFNR versus alpha")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig(
#     "BFNR_vs_alpha.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.close()



# ============================================================
# Plot BFDR vs BFNR (Pareto curves)
# ============================================================

plt.figure(figsize=(18,18))

for L in sorted(df.L.unique()):

    temp = df[df.L == L].sort_values("alpha")
    plt.plot(
        temp.BFDR,
        temp.BFNR,
        marker="o",
        markersize=3,
        label=f"L={L}"
    )


plt.xlabel("BFDR")
plt.ylabel("BFNR")
plt.title("BFDR-BFNR trade-off")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig(
#     "BFDR_vs_BFNR.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.close()


import matplotlib as mpl


# ============================================================
# Scatter plot selected L values: BFDR vs BFNR
# Colourblind seaborn palette
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Find alpha closest to origin for each L
# ============================================================

optimal_points = []

for L in sorted(df.L.unique()):

    temp = df[df.L == L].copy()

    # distance to (0,0)
    temp["distance"] = np.sqrt(
        temp.BFDR**2 + temp.BFNR**2
    )

    best = temp.loc[temp.distance.idxmin()]

    optimal_points.append(best)


optimal_df = pd.DataFrame(optimal_points)

print(optimal_df[[
    "L",
    "alpha",
    "BFDR",
    "BFNR",
    "distance"
]])


selected_L = [1, 2, 3, 4, 5, 10, 15]

colors = sns.color_palette("colorblind", len(selected_L))


plt.figure(figsize=(12, 12))


for colour, L in zip(colors, selected_L):

    if L not in df.L.unique():
        continue

    temp = df[df.L == L].sort_values("alpha")

    plt.plot(
        temp.BFDR,
        temp.BFNR,
        color=colour,
        linewidth=1.0,   # thin line
        alpha=0.8,
    )
    
    plt.scatter(
        temp.BFDR,
        temp.BFNR,
        color=colour,
        s=50,            # larger points
        alpha=0.8,
        label=f"$L={L}$"
    )


plt.scatter(
    optimal_df[optimal_df.L.isin(selected_L)].BFDR,
    optimal_df[optimal_df.L.isin(selected_L)].BFNR,
    s=150,
    marker="*",
    color="black",
    label="closest to (0,0)"
)

# ------------------------------------------------------------
# Red stars: alpha = 1
# ------------------------------------------------------------
alpha1_points = []

for L in selected_L:

    if L not in df.L.unique():
        continue

    temp = df[df.L == L].copy()

    # choose the alpha value closest to 1
    idx = (temp.alpha - 1).abs().idxmin()
    alpha1_points.append(temp.loc[idx])

alpha1_df = pd.DataFrame(alpha1_points)

plt.scatter(
    alpha1_df.BFDR,
    alpha1_df.BFNR,
    s=150,
    marker="*",
    color="red",
    label=r"$\alpha=1$"
)


plt.xlabel("BFDR", fontsize=15)
plt.ylabel("BFNR", fontsize=15)

plt.xlim(0, None)
plt.ylim(0, None)

plt.title(
    "BFDR-BFNR trade-off",
    fontsize=16
)

plt.tick_params(
    axis="both",
    labelsize=12
)

plt.grid(True, alpha=0.3)

plt.legend(
    title="Number of centres",
    fontsize=12,
    title_fontsize=12,
    loc="best"
)

plt.tight_layout()

plt.savefig(
    "BFDR_vs_BFNR_scatter_colourblind.pdf",
    format='pdf',
    bbox_inches="tight"
)

plt.show()