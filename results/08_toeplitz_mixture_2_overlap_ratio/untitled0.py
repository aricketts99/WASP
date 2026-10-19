#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare objective ratios for a single experiment across specified L values.

Rows:
    Single Experiment (from out_file.csv)

Columns:
    User-defined list of L values (e.g., L = 5, 10, 20)

Each cell shows:
    J_L / J_1

where J_1 is the objective at L = 1 for the same
(active, p) within the experiment.

A single colour scale is used across all panels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

# Input file name
input_file = "out_file.csv"

# Define the specific L values you want to plot here
# Ensure L=1 exists in your data for the baseline calculation
Ls_to_plot = [2,4,8,16,32]

# If your CSV does not contain the 'active' column and needs a merge,
# provide the parameters file name here. Otherwise, set to None.
params_file = None 
# params_file = "parameters_mixture_contam_3.csv" 

# ============================================================
# Load Data
# ============================================================

df = pd.read_csv(input_file)

# Optional: Merge with parameters file if 'active' is missing in the main CSV
if params_file is not None:
    params = pd.read_csv(params_file)
    df = df.merge(
        params[["job_id", "active"]],
        on="job_id",
        how="left"
    )
    print(f"Merged with parameters file: {params_file}")
else:
    # Ensure 'active' column exists if not merging
    if "active" not in df.columns:
        raise ValueError("Column 'active' not found in data. Please provide a parameters file or ensure it is in the CSV.")

# Add experiment indicator (set to 1 for consistency with original logic)
df["experiment"] = 1

# ============================================================
# Normalise objective by active_p
# ============================================================

if "active_p" not in df.columns:
    raise ValueError("Column 'active_p' not found in data.")

df["objective"] = df["objective"] / df["active_p"]

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
# Get L = 1 baseline
# ============================================================

baseline = (
    plot_df[plot_df["L"] == 1]
    [["experiment", "active", "p", "objective"]]
    .rename(columns={"objective": "objective_L1"})
)

# ============================================================
# Merge L = 1 baseline onto the main dataframe
# ============================================================

plot_df = plot_df.merge(
    baseline,
    on=["experiment", "active", "p"],
    how="left"
)

# ============================================================
# Remove observations without an L = 1 baseline
# ============================================================

missing_baseline = plot_df[plot_df["objective_L1"].isna()]

if not missing_baseline.empty:
    print(
        "Warning: some observations do not have "
        "an L = 1 baseline and will be removed."
    )
    plot_df = plot_df[~plot_df["objective_L1"].isna()].copy()

# ============================================================
# Check positivity
# ============================================================

if (plot_df["objective"] <= 0).any():
    raise ValueError("The objective contains zero or negative values.")

if (plot_df["objective_L1"] <= 0).any():
    raise ValueError("The L = 1 objective contains zero or negative values.")

# ============================================================
# Calculate objective ratio
# ============================================================

plot_df["objective_ratio"] = plot_df["objective"] / plot_df["objective_L1"]

# ============================================================
# Filter for requested L values
# ============================================================

plot_df = plot_df[plot_df["L"].isin(Ls_to_plot)].copy()

if plot_df.empty:
    raise ValueError(f"No data found for the specified L values: {Ls_to_plot}")

# ============================================================
# Global colour scale
# ============================================================

ratio_vmin = plot_df["objective_ratio"].min()
ratio_vmax = plot_df["objective_ratio"].max()

print("\nGlobal objective ratio range:")
print(f"Minimum: {ratio_vmin:.4f}")
print(f"Maximum: {ratio_vmax:.4f}")

# ============================================================
# Grid values
# ============================================================

p_values = sorted(plot_df["p"].unique())
active_values = sorted(plot_df["active"].unique())

# ============================================================
# Create Figure
# ============================================================

n_cols = len(Ls_to_plot)
fig_width = 5 * n_cols
fig_height = 6

fig, axes = plt.subplots(
    1,
    n_cols,
    figsize=(fig_width, fig_height),
    sharex=True,
    sharey=True
)

# Ensure axes is iterable even if there is only one column
if n_cols == 1:
    axes = [axes]

# ============================================================
# Plot
# ============================================================

mesh = None # Initialize mesh variable for colorbar

for j, L in enumerate(Ls_to_plot):
    ax = axes[j]

    tmp = plot_df[
        (plot_df["experiment"] == 1)
        & (plot_df["L"] == L)
    ]

    if not tmp.empty:
        # Convert to grid
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
    else:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')

    # Column titles
    ax.set_title(f"$L = {L}$", fontsize=14)

    # Y-axis label (only on the first plot)
    if j == 0:
        ax.set_ylabel("Active", fontsize=12)

    # X-axis label
    ax.set_xlabel("$p$", fontsize=12)

    ax.tick_params(labelsize=9)

# ============================================================
# Axis ticks
# ============================================================

# Only set ticks if we have values
if len(p_values) > 0 and len(active_values) > 0:
    step_p = max(1, len(p_values) // 10) # Approx 10 ticks
    step_active = max(1, len(active_values) // 10)
    
    # Use slicing that respects list bounds
    p_ticks = p_values[::step_p] if len(p_values) > step_p else p_values
    active_ticks = active_values[::step_active] if len(active_values) > step_active else active_values

    for ax in axes:
        ax.set_xticks(p_ticks)
        ax.set_yticks(active_ticks)

# ============================================================
# Figure layout
# ============================================================

# Adjust right side to make room for colorbar
right_margin = 0.92 if n_cols > 1 else 0.85
cbar_start = 0.94 if n_cols > 1 else 0.87

fig.subplots_adjust(
    left=0.09,
    right=right_margin,
    bottom=0.15,
    top=0.90,
    wspace=0.10
)

# ============================================================
# Common colourbar
# ============================================================

if mesh is not None:
    cbar_ax = fig.add_axes([cbar_start, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label(
        "Objective relative to $L=1$\n(ratio)",
        fontsize=12
    )

fig.suptitle(
    "Objective Relative to $L=1$ (Single Experiment)",
    fontsize=16,
    y=0.98
)

# ============================================================
# Save and Show
# ============================================================

output_filename = "results/single_experiment_objective_ratio.pdf"
# Ensure directory exists or handle error if 'results' folder is missing
import os
if not os.path.exists("results"):
    os.makedirs("results")
    print("Created 'results' directory.")

plt.savefig(output_filename, bbox_inches="tight")
print(f"Plot saved to {output_filename}")

plt.show()
