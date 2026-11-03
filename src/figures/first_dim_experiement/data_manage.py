#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:16:09 2026

@author: andrew
"""

from pathlib import Path
import pickle
import zstandard as zstd
import pandas as pd

# Folder containing the .zst files
results_dir = Path("results/miss_mixture")

dctx = zstd.ZstdDecompressor()

rows = []

for file in results_dir.glob("*.zst"):

    with open(file, "rb") as f:
        with dctx.stream_reader(f) as zf:
            output = pickle.load(zf)

    setup = output["setup"]
    min_per_class = output["results"]

    for L, result in min_per_class.items():

        rows.append({
            "job_id": setup["job_id"],
            "seed": setup["seed"],
            "p": setup["p"],
            "active": setup["n_active"],
            "active_p": setup["active_p"],
            "L": L,
            "result": result,
        })

df = pd.DataFrame(rows)

print(df.head())
print()
print(f"{len(df)} rows")

from pathlib import Path
import pickle
import zstandard as zstd

dctx = zstd.ZstdDecompressor()

samples_dict = {}

for file in Path("results/miss_mixture").glob("*.zst"):

    with open(file, "rb") as f:
        with dctx.stream_reader(f) as zf:
            output = pickle.load(zf)

    setup = output["setup"]

    key = (
        setup["p"],
        setup["active"],
        setup["seed"],
    )

    samples_dict[key] = output["samples"]
import math
import numpy as np
import matplotlib.pyplot as plt

# Extract the objective (first element of the result tuple)
df["objective"] = df["result"].apply(lambda x: x[0])

ps = np.sort(df["p"].unique())
actives = np.sort(df["active"].unique())
Ls = sorted(df["L"].unique())

# Compute common colour scale from the averaged heatmaps
vmin = np.inf
vmax = -np.inf

for L in Ls:

    table = (
        df[df["L"] == L]
        .groupby(["active", "p"])["objective"]
        .mean()
        .unstack()
        .reindex(index=actives, columns=ps)
    )

    vmin = min(vmin, np.nanmin(table.values))
    vmax = max(vmax, np.nanmax(table.values))

# Layout
ncols = 4
nrows = math.ceil(len(Ls) / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(4.5 * ncols, 3.8 * nrows),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)

axes = np.atleast_1d(axes).ravel()

pcm = None

# Plot one heatmap for each L
for ax, L in zip(axes, Ls):

    table = (
        df[df["L"] == L]
        .groupby(["active", "p"])["objective"]
        .mean()
        .unstack()
        .reindex(index=actives, columns=ps)
    )

    pcm = ax.pcolormesh(
        ps,
        actives,
        table.values,
        shading="nearest",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"$L={L}$")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("Active proportion")

# Hide unused axes
for ax in axes[len(Ls):]:
    ax.set_visible(False)

# One shared colourbar
cbar = fig.colorbar(
    pcm,
    ax=axes.tolist(),
    shrink=0.9,
    pad=0.02,
)
cbar.set_label("Mean objective")

plt.savefig(
    "big_objective_heatmaps.pdf",
    bbox_inches="tight",
)

# plt.savefig(
#     "big_objective_heatmaps.png",
#     dpi=600,
#     bbox_inches="tight",
# )

plt.show()

def weighted_difference(row):
    weights = np.asarray(row["result"][1])
    summaries = np.asarray(row["result"][2])
    


    # true binary vector
    truth = np.zeros(row["p"], dtype=int)
    truth[:int(row["active_p"])] = 1
    

    # Hamming error for each summary vector
    errors = np.sum(np.abs(summaries - truth), axis=1)

    # weight errors by their posterior weights
    return np.sum(weights * errors)


df["weighted_difference"] = df.apply(weighted_difference, axis=1)
df["weighted_difference"] = df["result"].apply(lambda x: x[-2])
ps = np.sort(df["p"].unique())
actives = np.sort(df["active"].unique())
Ls = sorted(df["L"].unique())

# Compute common colour scale from the averaged heatmaps
vmin = np.inf
vmax = -np.inf

for L in Ls:

    table = (
        df[df["L"] == L]
        .groupby(["active", "p"])["weighted_difference"]
        .mean()
        .unstack()
        .reindex(index=actives, columns=ps)
    )

    vmin = min(vmin, np.nanmin(table.values))
    vmax = max(vmax, np.nanmax(table.values))

# Layout
ncols = 4
nrows = math.ceil(len(Ls) / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(4.5 * ncols, 3.8 * nrows),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)

axes = np.atleast_1d(axes).ravel()

pcm = None

# Plot one heatmap for each L
for ax, L in zip(axes, Ls):

    table = (
        df[df["L"] == L]
        .groupby(["active", "p"])["weighted_difference"]
        .mean()
        .unstack()
        .reindex(index=actives, columns=ps)
    )

    pcm = ax.pcolormesh(
        ps,
        actives,
        table.values,
        shading="nearest",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"$L={L}$")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("Active proportion")

# Hide unused axes
for ax in axes[len(Ls):]:
    ax.set_visible(False)

# One shared colourbar
cbar = fig.colorbar(
    pcm,
    ax=axes.tolist(),
    shrink=0.9,
    pad=0.02,
)
cbar.set_label("Mean weighted difference")

plt.savefig(
    "big_true_bin_heatmaps.pdf",
    bbox_inches="tight",
)

# plt.savefig(
#     "big_true_bin_heatmaps.png",
#     dpi=600,
#     bbox_inches="tight",
# )

plt.show()


import numpy as np
from src.utils import lrrsg
# def prediction_error(row):

#     # Recreate the simulated dataset
#     active_p = row["active_p"]

#     prng = np.random.default_rng(12345)
#     beta0 = prng.choice([-3, 3], size=active_p)

#     beta, y, X = lrrsg(
#         n=200,
#         p=row["p"],
#         beta0=beta0,
#         SNR=1.5,
#         rho=0.35,
#         seed=row["seed"],
#     )

#     mu_true = X @ beta

#     # Extract posterior summary
#     weights = np.asarray(row["result"][1], dtype=float)
#     summaries = np.asarray(row["result"][2])

#     # Ensure summaries has shape (L, p)
#     if summaries.shape[0] != len(weights):
#         summaries = summaries.T

#     mu_pred = np.zeros_like(mu_true)

#     for w, gamma in zip(weights, summaries):

#         active = gamma.astype(bool)

#         if active.sum() == 0:
#             mu_l = np.zeros_like(mu_true)
#         else:
#             beta_hat, *_ = np.linalg.lstsq(
#                 X[:, active],
#                 y,
#                 rcond=None,
#             )

#             mu_l = X[:, active] @ beta_hat

#         mu_pred += w * mu_l

#     return np.mean((mu_true - mu_pred) ** 2)

import numpy as np
from joblib import Parallel, delayed


def prediction_error(p, active_p, seed, result):

    p = int(p)
    active_p = int(active_p)
    seed = int(seed)

    # Recreate simulated dataset
    prng = np.random.default_rng(12345)
    beta0 = prng.choice([-3, 3], size=active_p)

    beta, y, X = lrrsg(
        n=1000,
        p=p,
        beta0=beta0,
        SNR=2.5,
        rho=0.0,
        seed=seed,
    )

    mu_true = X @ beta

    weights = np.asarray(result[1], dtype=float)
    summaries = np.asarray(result[2])

    if summaries.shape[0] != len(weights):
        summaries = summaries.T

    mu_pred = np.zeros_like(mu_true)

    for w, gamma in zip(weights, summaries):

        active = gamma.astype(bool)

        if not active.any():
            continue

        Xa = X[:, active]

        beta_hat = np.linalg.lstsq(
            Xa,
            y,
            rcond=None,
        )[0]

        mu_pred += w * (Xa @ beta_hat)

    return np.mean((mu_true - mu_pred) ** 2)
#df["prediction_error"] = df.apply(prediction_error, axis=1)
from joblib import Parallel, delayed
errors = Parallel(
    n_jobs=8,
    prefer="threads",
    verbose=10,
)(
    delayed(prediction_error)(
        row.p,
        row.active_p,
        row.seed,
        row.result,
    )
    for row in df.itertuples(index=False)
)

df["prediction_error"] = errors
ps = np.sort(df["p"].unique())
actives = np.sort(df["active"].unique())
Ls = sorted(df["L"].unique())

# Compute common colour scale from the averaged heatmaps
vmin = np.inf
vmax = -np.inf

for L in Ls:

    table = (
        df[df["L"] == L]
        .groupby(["active", "p"])["prediction_error"]
        .mean()
        .unstack()
        .reindex(index=actives, columns=ps)
    )

    vmin = min(vmin, np.nanmin(table.values))
    vmax = max(vmax, np.nanmax(table.values))

# Layout
ncols = 4
nrows = math.ceil(len(Ls) / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(4.5 * ncols, 3.8 * nrows),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)

axes = np.atleast_1d(axes).ravel()

pcm = None

# Plot one heatmap for each L
for ax, L in zip(axes, Ls):

    table = (
        df[df["L"] == L]
        .groupby(["active", "p"])["prediction_error"]
        .mean()
        .unstack()
        .reindex(index=actives, columns=ps)
    )

    pcm = ax.pcolormesh(
        ps,
        actives,
        table.values,
        shading="nearest",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"$L={L}$")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("Active proportion")

# Hide unused axes
for ax in axes[len(Ls):]:
    ax.set_visible(False)

# One shared colourbar
cbar = fig.colorbar(
    pcm,
    ax=axes.tolist(),
    shrink=0.9,
    pad=0.02,
)
cbar.set_label("Mean weighted difference")

plt.savefig(
    "pred_error_heatmaps.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "pred_error_heatmaps.png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()