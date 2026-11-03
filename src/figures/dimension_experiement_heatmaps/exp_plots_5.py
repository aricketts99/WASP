#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare objective ratios across three experiments.

Rows:
    Experiment 1
    Experiment 2
    Experiment 3 (overlap)

Columns:
    L = 5
    L = 10
    L = 15

Each cell shows

    J_L / J_1

where J_1 is the objective at L = 1 for the same
(active, p) within the same experiment.

A single colour scale is used across all 9 panels,
with the minimum and maximum calculated over the
entire combined dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# File names
# ============================================================

file_1 = "results/experiment_1_high_fid.csv"
file_2 = "results/experiment_2_high_fid.csv"
file_3 = "results/experiment_overlap.csv"

params_file = "parameters_mixture_contam_3.csv"


# ============================================================
# Load data
# ============================================================

df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
df3 = pd.read_csv(file_3)

params = pd.read_csv(params_file)


# ============================================================
# Add active to experiments 1 and 2
# ============================================================

df1 = df1.merge(
    params[["job_id", "active"]],
    on="job_id",
    how="left"
)

df2 = df2.merge(
    params[["job_id", "active"]],
    on="job_id",
    how="left"
)


# ============================================================
# Add experiment indicator
# ============================================================

df1["experiment"] = 1
df2["experiment"] = 2
df3["experiment"] = 3


# ============================================================
# Combine experiments
# ============================================================

df = pd.concat(
    [df1, df2, df3],
    ignore_index=True,
    sort=False
)


# ============================================================
# Normalise objective by active_p
# ============================================================

df["objective"] = (
    df["objective"] / df["active_p"]
)


# ============================================================
# Average repeated observations
# ============================================================

plot_df = (
    df.groupby(
        ["experiment", "L", "active", "p"],
        as_index=False
    )[["objective"]]
    .mean()
)


# ============================================================
# Get L = 1 baseline separately for each experiment
# ============================================================

baseline = (
    plot_df[plot_df["L"] == 1]
    [["experiment", "active", "p", "objective"]]
    .rename(
        columns={
            "objective": "objective_L1"
        }
    )
)


# ============================================================
# Merge L = 1 baseline onto each experiment
# ============================================================

plot_df = plot_df.merge(
    baseline,
    on=["experiment", "active", "p"],
    how="left"
)


# ============================================================
# Remove observations without an L = 1 baseline
# ============================================================

missing_baseline = plot_df[
    plot_df["objective_L1"].isna()
]

if not missing_baseline.empty:

    print(
        "Warning: some observations do not have "
        "an L = 1 baseline and will be removed."
    )

    plot_df = plot_df[
        ~plot_df["objective_L1"].isna()
    ].copy()


# ============================================================
# Check positivity
# ============================================================

if (plot_df["objective"] <= 0).any():
    raise ValueError(
        "The objective contains zero or negative values."
    )

if (plot_df["objective_L1"] <= 0).any():
    raise ValueError(
        "The L = 1 objective contains zero or negative values."
    )


# ============================================================
# Calculate objective ratio
# ============================================================

plot_df["objective_ratio"] = (
    plot_df["objective"]
    / plot_df["objective_L1"]
)


# ============================================================
# Only plot L > 1
# ============================================================

Ls = [5, 10, 20]

plot_df = plot_df[
    plot_df["L"].isin(Ls)
].copy()


# ============================================================
# Experiments
# ============================================================

experiments = [1, 2, 3]


# ============================================================
# Global colour scale
#
# IMPORTANT:
# min/max are calculated over the ENTIRE combined
# dataframe, not separately for each experiment or L.
# ============================================================

ratio_vmin = plot_df["objective_ratio"].min()
ratio_vmax = plot_df["objective_ratio"].max()

print("\nGlobal objective ratio range:")
print(f"Minimum: {ratio_vmin:.4f}")
print(f"Maximum: {ratio_vmax:.4f}")


# ============================================================
# Grid values
# ============================================================

p_values = sorted(
    plot_df["p"].unique()
)

active_values = sorted(
    plot_df["active"].unique()
)


# ============================================================
# Create 3 x 3 figure
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(15, 12),
    sharex=True,
    sharey=True
)


# ============================================================
# Plot
# ============================================================

for i, experiment in enumerate(experiments):

    for j, L in enumerate(Ls):

        ax = axes[i, j]

        tmp = plot_df[
            (plot_df["experiment"] == experiment)
            & (plot_df["L"] == L)
        ]


        # ----------------------------------------------------
        # Convert to grid
        # ----------------------------------------------------

        if not tmp.empty:

            ratio_grid = tmp.pivot(
                index="active",
                columns="p",
                values="objective_ratio"
            ).reindex(
                index=active_values,
                columns=p_values
            )


            mesh = ax.pcolormesh(
                p_values,
                active_values,
                ratio_grid.values,
                cmap="coolwarm",
                vmin=ratio_vmin,
                vmax=ratio_vmax,
                shading="nearest"
            )


        # ----------------------------------------------------
        # Column titles
        # ----------------------------------------------------

        if i == 0:

            ax.set_title(
                f"$L = {L}$",
                fontsize=14
            )


        # ----------------------------------------------------
        # Row labels
        # ----------------------------------------------------

        if j == 0:

            ax.set_ylabel(
                f"Experiment {experiment}\nActive",
                fontsize=12
            )


        # ----------------------------------------------------
        # X-axis labels
        # ----------------------------------------------------

        if i == 2:

            ax.set_xlabel(
                "$p$",
                fontsize=12
            )


        ax.tick_params(
            labelsize=9
        )


# ============================================================
# Axis ticks
# ============================================================

for ax in axes.flat:

    ax.set_xticks(
        p_values[::25]
    )

    ax.set_yticks(
        active_values[::5]
    )


# ============================================================
# Figure layout
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.88,
    bottom=0.08,
    top=0.93,
    hspace=0.10,
    wspace=0.08
)


# ============================================================
# ONE common colourbar
# ============================================================

cbar_ax = fig.add_axes(
    [0.90, 0.15, 0.02, 0.70]
)

cbar = fig.colorbar(
    mesh,
    cax=cbar_ax
)

fig.suptitle(
    "Objective Relative to $L=1$ Across Experiments",
    fontsize=16,
    y=0.98
)

cbar.set_label(
    "Objective relative to $L=1$\n"
    "(ratio: objective at $L$ / objective at $L=1$)",
    fontsize=12
)

# ============================================================
# Save
# ============================================================

plt.savefig(
    "results/experiments_objective_ratio.pdf",
    bbox_inches="tight"
)


# ============================================================
# Show
# ============================================================

plt.show()