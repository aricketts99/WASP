#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:18:55 2026

@author: andrew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/experiment_overlap_2.csv")

#df = df[df.L.isin([1, 5, 10, 15])]
#df["r^2"] = df["r^2"] / df["p"]

# Average repeated observations
plot_df = (
    df.groupby(["L", "active", "p"], as_index=False)["r^2"]
      .mean()
)



Ls = sorted(plot_df["L"].unique())

fig, axes = plt.subplots(
    1,
    len(Ls),
    figsize=(16, 6),
    sharex=True,
    sharey=True
)

if len(Ls) == 1:
    axes = [axes]

# Common colour scale
vmin = plot_df["r^2"].min()
vmax = plot_df["r^2"].max()

# Grid values
p_values = sorted(plot_df["p"].unique())
active_values = sorted(plot_df["active"].unique())

for ax, L in zip(axes, Ls):

    tmp = plot_df[plot_df["L"] == L]

    sc = ax.scatter(
        tmp["p"],
        tmp["active"],
        c=tmp["r^2"],
        marker="s",
        s=220,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        edgecolors="none"
    )

    ax.set_title(f"$L = {L}$", fontsize=14)

    ax.set_xlabel("$p$", fontsize=12)

    if ax is axes[0]:
        ax.set_ylabel("Active", fontsize=12)

    ax.set_xticks(p_values[::5])
    ax.set_yticks(active_values)

    ax.tick_params(labelsize=9)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.3
    )

# Leave room on the right for colourbar
fig.subplots_adjust(
    left=0.06,
    right=0.88,
    bottom=0.12,
    top=0.88,
    wspace=0.08
)

# Colourbar on the right
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])

cbar = fig.colorbar(
    sc,
    cax=cbar_ax
)

cbar.set_label(
    "Mean r^2",
    fontsize=12
)

plt.show()