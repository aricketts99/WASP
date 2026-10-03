#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:18:22 2026

@author: andrew
"""

import numpy as np
import pandas as pd
import sys

from experiments import experiment_1, experiment_2, experiment_3, experiment_4

from src.bvs import BVS_MCMC
from src.wass_approx_gen_ham import wass_approx


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

n = 1000

params = pd.read_csv("parameters_mixture_contam_3.csv")
K = 2

job_id = int(sys.argv[1])
row = params.iloc[job_id - 1]

seed = int(row.seed)
p = int(row.p)
active = float(row.active)/2.0

active_p = int(p*active)

X,y,beta,partition,epsilon = experiment_1(n,p,active_p,123456)

y  = y.reshape(-1,1)
# Set up and run

g_scale=(1.5**2)*np.log(p)/n

fit = BVS_MCMC(X, y, g=g_scale, prior_type="ind",h_exp_size=25)
fit.set_alg_par(sampler="ADS", N_iter=1000, N_burnin=1000,
                n_chain=100, n_temp=6, verbose=True,store_chains=True)
fit.sample_now()
samples = fit.chains[:, 1001:, :]

samples = samples[:, ::20, :].reshape(-1,p)

import itertools
Ls = list(range(1,16))
Ls = [1,5,10,20,50]
seeds = list(range(1,11))

combined = list(itertools.product(Ls, seeds))
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(wass_approx)(c[0],p,samples,1.0,c[1])
    for c in combined
)

# results = []
# for c in combined:
#     results.append(wass_approx(c[0],p,samples,1.0,c[1]))



min_per_class = {}
for key, group in itertools.groupby(results, key=lambda x: (x[-2])):
    min_per_class[key] = min(group, key=lambda x: x[0])




for l in Ls:
    
    
    weights = np.asarray(min_per_class[l][1], dtype=float)
    summaries = np.asarray(min_per_class[l][2])
    
    if summaries.shape[0] != len(weights):
        summaries = summaries.T
    
    mu_pred = np.zeros_like(y)
    
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
    # R² relative to the true mean
    ss_res = np.sum((y - mu_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # true binary vector
    truth = np.zeros(p, dtype=int)
    truth[:int(active_p*p)] = 1
    

    # Hamming error for each summary vector
    errors = np.sum(np.abs(summaries - truth), axis=1)/p

    # weight errors by their posterior weights
    weighted_error = np.sum(weights * errors)

    min_per_class[l] = min_per_class[l]+(r2,weighted_error,)
    
    print('R2: '+str(r2)+"   Weighted Error: "+str(weighted_error))

output = {
    "setup": {
        "job_id": job_id,
        "seed": seed,
        "p": p,
        "n": n,
        "active_p": active_p,
        "K": K,
        "sampler": "ADS",
        "N_iter": 3000,
        "N_burnin": 1000,
        "n_chain": 100,
        "n_temp": 6,
        "g": 10*p/n,
        'prior_type': 'ind'
    },
    "samples": samples,
    "results": min_per_class,
}
outfile = (f'results/experiment_1/'
    f"results_job{job_id:06d}"
    f"_p{p}"
    f"_a{active_p:.2f}"
    f"_K{K}"
    f"_seed{seed}.zst"
)
import pickle
import zstandard as zstd

cctx = zstd.ZstdCompressor(level=19)

with open(outfile, "wb") as f:
    with cctx.stream_writer(f) as zf:
        pickle.dump(output, zf, protocol=pickle.HIGHEST_PROTOCOL)
