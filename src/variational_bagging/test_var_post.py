#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:10:06 2026

@author: andrew
"""


import numpy as np
import pandas as pd
from src.wass_approx_gen_ham import wass_approx

samples = np.load('svb_ar1_student_t_binary_samples.npy')

p = samples.shape[1]


import itertools
Ls = list(range(1,16))
seeds = list(range(1,11))

combined = list(itertools.product(Ls, seeds))
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(wass_approx)(c[0],p,samples,1.0,c[1])
    for c in combined
)

# results = []
# for c in combined:
#     results.append(wass_approx(c[0],p,samples,1.5,c[1]))



min_per_class = {}
for key, group in itertools.groupby(results, key=lambda x: (x[-2])):
    min_per_class[key] = min(group, key=lambda x: x[0])

from figures.plots import elbow, centre_tables, voronoi_summary

elbow(min_per_class,'elbow_bagged_VI_ar_1_t_3')

df_1 = centre_tables(min_per_class,L=10)
print(df_1.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,
                  ))
df_1 = centre_tables(min_per_class,L=5)
print(df_1.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,
                  ))
voronoi_summary(min_per_class, samples, 'voronoi_bagged_VI_ar_1_t_3_L_10',L=10,threshold=0.2)
voronoi_summary(min_per_class, samples, 'voronoi_bagged_VI_ar_1_t_3_L_5',L=5,threshold=0.2)





# mu_true = X @ beta
# for l in range(1,6):
    
    
#     weights = np.asarray(min_per_class[l][1], dtype=float)
#     summaries = np.asarray(min_per_class[l][2])
    
#     if summaries.shape[0] != len(weights):
#         summaries = summaries.T
    
#     mu_pred = np.zeros_like(mu_true)
    
#     for w, gamma in zip(weights, summaries):
    
#         active = gamma.astype(bool)
    
#         if not active.any():
#             continue
    
#         Xa = X[:, active]
    
#         beta_hat = np.linalg.lstsq(
#             Xa,
#             y,
#             rcond=None,
#         )[0]
    
#         mu_pred += w * (Xa @ beta_hat)
#     # R² relative to the true mean
#     ss_res = np.sum((y - mu_pred) ** 2)
#     ss_tot = np.sum((y - np.mean(y)) ** 2)
#     r2 = 1 - ss_res / ss_tot

#     min_per_class[l] = min_per_class[l]+(r2,)

# output = {
#     "setup": {
#         "job_id": job_id,
#         "seed": seed,
#         "p": p,
#         "active": active,
#         "active_p": active_p,
#         "n": n,
#         "SNR": 1.5,
#         "rho": 0.35,
#         "sampler": "PARNI",
#         "N_iter": 1000,
#         "N_burnin": 50,
#         "n_chain": 6,
#         "n_temp": 6
#     },
#     "samples": samples,
#     "results": min_per_class,
# }

# outfile = (
#     f"results_job{job_id:06d}"
#     f"_p{p}"
#     f"_a{active:.2f}"
#     f"_seed{seed}.zst"
# )
# import pickle
# import zstandard as zstd

# cctx = zstd.ZstdCompressor(level=19)

# with open(outfile, "wb") as f:
#     with cctx.stream_writer(f) as zf:
#         pickle.dump(output, zf, protocol=pickle.HIGHEST_PROTOCOL)