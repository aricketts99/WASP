#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 21:18:46 2026

@author: andrew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/experiment_overlap_betas.csv")

# # Experiment parameters
# params = pd.read_csv("parameters_mixture_contam_3.csv")

# # Add active using job_id
# df = df.merge(
#     params[["job_id", "active"]],
#     on="job_id",
#     how="left"
# )
# df = df[df.L.isin([1, 5, 10, 15])]
# df["objective"] =(df["objective"]/df['active_p'])

# Average repeated observations
plot_df = (
    df.groupby(["L", "active", "p"], as_index=False)[["objective", "r^2"]]
      .mean()
)

Ls = sorted(plot_df["L"].unique())

fig, axes = plt.subplots(
    2,
    len(Ls),
    figsize=(16, 10),
    sharex=True,
    sharey=True)

# Make axes 2D if there is only one L
if len(Ls) == 1:
    axes = np.asarray(axes).reshape(2, 1)

# Common colour scales for each row
objective_vmin = plot_df["objective"].min()
objective_vmax = plot_df["objective"].max()

r2_vmin = plot_df["r^2"].min()
r2_vmax = plot_df["r^2"].max()

# Grid values
p_values = sorted(plot_df["p"].unique())
active_values = sorted(plot_df["active"].unique())

for j, L in enumerate(Ls):

    tmp = plot_df[plot_df["L"] == L]

    # --------------------------------------------------
    # Convert data to grids
    # --------------------------------------------------

    objective_grid = tmp.pivot(
        index="active",
        columns="p",
        values="objective"
    ).reindex(
        index=active_values,
        columns=p_values
    )

    r2_grid = tmp.pivot(
        index="active",
        columns="p",
        values="r^2"
    ).reindex(
        index=active_values,
        columns=p_values
    )

    # --------------------------------------------------
    # Top row: objective
    # --------------------------------------------------

    ax = axes[0, j]

    mesh_obj = ax.pcolormesh(
        p_values,
        active_values,
        objective_grid.values,
        cmap="coolwarm",
        vmin=objective_vmin,
        vmax=objective_vmax,
        shading="nearest"
    )

    ax.set_title(
        f"$L = {L}$",
        fontsize=14
    )

    if j == 0:
        ax.set_ylabel(
            "Active",
            fontsize=12
        )

    ax.tick_params(labelsize=9)

    # --------------------------------------------------
    # Bottom row: r^2
    # --------------------------------------------------

    ax = axes[1, j]

    mesh_r2 = ax.pcolormesh(
        p_values,
        active_values,
        r2_grid.values,
        cmap="coolwarm",
        vmin=r2_vmin,
        vmax=r2_vmax,
        shading="nearest"
    )

    ax.set_xlabel(
        "$p$",
        fontsize=12
    )

    if j == 0:
        ax.set_ylabel(
            "Active",
            fontsize=12
        )

    ax.tick_params(labelsize=9)

# --------------------------------------------------
# X-axis ticks
# --------------------------------------------------

for ax in axes.flat:
    ax.set_xticks(p_values[::25])
    ax.set_yticks(active_values[::5])

# --------------------------------------------------
# Figure layout
# --------------------------------------------------

fig.subplots_adjust(
    left=0.07,
    right=0.88,
    bottom=0.10,
    top=0.92,
    hspace=0.15,
    wspace=0.08
)

# --------------------------------------------------
# Colourbar: objective
# --------------------------------------------------

cbar_ax_obj = fig.add_axes(
    [0.90, 0.54, 0.02, 0.34]
)

cbar_obj = fig.colorbar(
    mesh_obj,
    cax=cbar_ax_obj
)

cbar_obj.set_label(
    "Mean objective",
    fontsize=12
)

# --------------------------------------------------
# Colourbar: r^2
# --------------------------------------------------

cbar_ax_r2 = fig.add_axes(
    [0.90, 0.12, 0.02, 0.34]
)

cbar_r2 = fig.colorbar(
    mesh_r2,
    cax=cbar_ax_r2
)

cbar_r2.set_label(
    "Mean $r^2$",
    fontsize=12
)

# plt.savefig(
#     "results/experiment_1_high_fid.pdf",
#     bbox_inches="tight"
# )

plt.show()