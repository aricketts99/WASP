#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:51:24 2026

@author: andrew
"""

import numpy as np
import pandas as pd
import itertools

ps = np.arange(100, 2050, 50)
active = np.arange(0.1, 1.01, 0.04)
seed = [
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

for job_id, (s, p, a) in enumerate(itertools.product(seed, ps, active), start=1):
    rows.append({
        "job_id": job_id,
        "seed": s,
        "p": p,
        "active": a,
    })

df = pd.DataFrame(rows)

df.to_csv("parameters_mixture_contam.csv", index=False)

print(f"Wrote {len(df)} jobs.")
