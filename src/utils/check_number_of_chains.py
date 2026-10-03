#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:34:16 2026

@author: andrew
"""

import numpy as np
import pandas as pd

from experiments import experiment_1, experiment_2, experiment_3, experiment_4

from src.bvs import BVS_MCMC


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

n = 1000

params = pd.read_csv("parameters_mixture_contam.csv")
K = 2

job_id = 217
row = params.iloc[job_id - 1]

seed = int(row.seed)
p = 2000#int(row.p)
active = float(row.active)

active_p = int(p*active)

X,y,beta,partition,epsilon = experiment_2(n,p,active_p,123456)

y  = y.reshape(-1,1)
# Set up and run
fit = BVS_MCMC(X, y, g=p, prior_type="g",h_exp_size = active_p)
fit.set_alg_par(sampler="ADS", N_iter=3000, N_burnin=1000,
                n_chain=100, n_temp=6, verbose=True,store_chains=True)
fit.sample_now()
samples = fit.chains[:, 1001:, :].reshape(-1,p)

# ------------------------------------------------------------
# Number of unique binary vectors as a function of # chains
# ------------------------------------------------------------
active_p = 2*active_p
chains = fit.chains[:, 1001:, :active_p]
#chains = chains[:, ::10, :]
n_chains = chains.shape[0]
n_reps = 1000
def binary_to_uint64(X):
    powers = np.uint64(1) << np.arange(X.shape[1], dtype=np.uint64)
    return X.astype(np.uint64) @ powers

chain_sets = [
    set(binary_to_uint64(chains[c]))
    for c in range(n_chains)
]


rng = np.random.default_rng(12345)

mean_unique = np.empty(n_chains)
q05_unique = np.empty(n_chains)
q95_unique = np.empty(n_chains)

for k in range(1, n_chains + 1):

    unique_counts = np.empty(n_reps, dtype=np.int64)

    for r in range(n_reps):

        subset = rng.choice(n_chains, size=k, replace=False)

        union = set()

        for c in subset:
            union.update(chain_sets[c])

        unique_counts[r] = len(union)

    mean_unique[k - 1] = unique_counts.mean()
    q05_unique[k - 1] = np.quantile(unique_counts, 0.05)
    q95_unique[k - 1] = np.quantile(unique_counts, 0.95)