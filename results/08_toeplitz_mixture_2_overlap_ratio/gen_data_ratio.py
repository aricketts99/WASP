#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:51:24 2026
Updated to vary n/p and k/n ratios.
@author: andrew
"""

import numpy as np
import pandas as pd
import itertools

# Fixed n as per your experiment script
n_fixed = 1000

# Define the range for n/p ratio (0 to 1). 
# Avoid exactly 0 to prevent division by zero. 
# Example: steps from 0.1 to 1.0. Adjust granularity as needed.
ratios_n_p = np.arange(0.1, 1.05, 0.031) 

# Define the range for k/n ratio (0 to 1). 
# k is the number of active variables.
ratios_k_n = np.arange(0.1, 1.05, 0.031)

seeds = [
    660087059,
    307242793,
    790186503,
    542001162,
    715607092,
    423526591,
    332720956,
    905313171,
    485624306,
    185424306
]

rows = []
job_id = 1

for s in seeds:
    for r_np in ratios_n_p:
        for r_kn in ratios_k_n:
            
            # Calculate p based on n/p ratio
            # p = n / ratio. Ensure p is an integer and p >= n (since ratio <= 1)
            p_calc = n_fixed / r_np
            p = int(np.round(p_calc))
            
            # Calculate k (active_p) based on k/n ratio
            # k = ratio * n
            k_calc = r_kn * n_fixed
            active_p = int(np.round(k_calc))
            
            # Feasibility Check 1: k cannot exceed p
            if active_p > p:
                continue
                
            # Feasibility Check 2: Ensure p is reasonable (optional, e.g., max limit)
            # if p > 5000: continue 

            rows.append({
                "job_id": job_id,
                "seed": s,
                "n": n_fixed,       # Explicitly save n for reference
                "p": p,
                "active_p": active_p, # Save the absolute count of active variables
                "ratio_n_p": r_np,    # Save the target ratio for plotting
                "ratio_k_n": r_kn     # Save the target ratio for plotting
            })
            job_id += 1

df = pd.DataFrame(rows)

# Sort for readability
df = df.sort_values(by=['ratio_n_p', 'ratio_k_n', 'seed'])
df.reset_index(drop=True, inplace=True)
df['job_id'] = df.index + 1

outfile = "parameters_mixture_ratios.csv"
df.to_csv(outfile, index=False)

print(f"Wrote {len(df)} jobs to {outfile}")
print(f"p range: {df['p'].min()} to {df['p'].max()}")
print(f"active_p range: {df['active_p'].min()} to {df['active_p'].max()}")
