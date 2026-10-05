#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:41:32 2026

@author: andrew
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import sys

from src.bvs import BVS_MCMC
from src.wass_approx_gen_ham import wass_approx
from src.utils.utils import _make_beta, _scale_beta, basic, block_corr, decoy, AR_1, basic_error, t_error, laplace_error, gen_Y

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

n = 1000

params = pd.read_csv("parameters_mixture_contam.csv")
K = 2

job_id = 500#int(sys.argv[1])
row = params.iloc[job_id - 1]

seed = int(row.seed)
p = int(row.p)
active = float(row.active)/K

active_p = int(p*active)



def experiment_5(n, p, active_p, seed, n_test=1000):
    '''
    Mixture with K = 2 and equal coefficients.
    overlap_frac controls the proportion of active variables
    shared between consecutive mixture components.
    Normal errors.
    Block correlation in X.

    Returns a training set (X, y), the true coefficients (beta, K x p),
    the training component labels (partition), the training errors,
    and an independent test set (X_test, y_test) from the same DGP.
    The test set is centred/scaled using TRAINING statistics only.
    '''

    K = 2



    # ---------------------------------------------------------
    # True coefficients, shape (K, p).
    # Scaled using the TRAINING n so the SNR is defined once
    # and the same beta is used for train and test.
    # ---------------------------------------------------------
    beta = _make_beta(p, active_p, K)
    beta = _scale_beta(beta, n, SNR=2.5 * np.ones(K))

    # ---------------------------------------------------------
    # Training data
    # ---------------------------------------------------------
    X = basic(n,p,seed=seed)

    # Centre and scale X; KEEP these statistics for the test set
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1)
    X = (X - X_mean) / X_std

    epsilon = basic_error(n, seed=seed)
    y, partition = gen_Y(X, beta, epsilon, seed=seed)

    # Centre y (no intercept in the fitted model); KEEP the mean
    y_mean = y.mean(axis=0)
    y = y - y_mean

    # ---------------------------------------------------------
    # Test data: same DGP, same beta, but independent draws.
    # Every random call gets its own seed offset so nothing is a
    # copy of the training data (block_corr, basic_error and gen_Y
    # all default to seed=123456 otherwise).
    # ---------------------------------------------------------
    X_test = basic(n_test,p,seed=seed+10_00)
    X_test = (X_test - X_mean) / X_std          # training mean/std, not test's own

    epsilon_test = basic_error(n_test, seed=seed + 20_00)
    y_test, partition_test = gen_Y(X_test, beta, epsilon_test,
                                   seed=seed + 30_00)
    y_test = np.asarray(y_test).ravel() - y_mean  # training mean of y

    # ---------------------------------------------------------
    # Sanity checks (cheap, and they catch the shape/seed bugs)
    # ---------------------------------------------------------
    assert X_test.shape == (n_test, p), X_test.shape
    assert epsilon_test.shape[0] == n_test, epsilon_test.shape
    assert y_test.shape == (n_test,), y_test.shape
    assert not np.allclose(X_test[:5, :5], X[:5, :5]), "test X looks like a copy of train X"

    return X, y, beta, partition, epsilon, X_test, y_test, partition_test, epsilon_test

X,y,beta,partition,epsilon,X_test,y_test,partition_test, epsilon_test = experiment_5(n,p,active_p,seed)


labels, counts = np.unique(partition, return_counts=True)
pi = counts / len(partition)

labels, counts = np.unique(partition_test, return_counts=True)
pi_test = counts / len(partition_test)

y  = y.reshape(-1,1)
# Set up and run
g_scale = (1.5**2)*np.log(p)
fit = BVS_MCMC(X, y, g=g_scale, prior_type="g",h_exp_size=25)
fit.set_alg_par(sampler="ADS", N_iter=1000, N_burnin=1000,
                n_chain=100, n_temp=3, verbose=True,store_chains=True)
fit.sample_now()
samples = fit.chains[:, 1001:, :]

samples = samples[:, ::20, :].reshape(-1,p)

import itertools
Ls = [1,2,4,8,16,32]
seeds = list(range(1,12))

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
    y = np.asarray(y).ravel()
    weights = np.asarray(min_per_class[l][1], dtype=float)
    summaries = np.asarray(min_per_class[l][2])

    if summaries.shape[0] != len(weights):
        summaries = summaries.T

    OLS_full = {}

    shrinkage = g_scale / (1.0 + g_scale)

    # ---------------------------------------------------------
    # Fit OLS for each model
    # ---------------------------------------------------------

    for ell, (w, gamma) in enumerate(zip(weights, summaries)):

        active = gamma.astype(bool)

        # Empty model: zero coefficient vector
        if not active.any():
            OLS_full[ell + 1] = np.zeros(p)
            continue

        Xa = X[:, active]

        beta_hat = np.linalg.lstsq(
            Xa,
            y,
            rcond=None,
        )[0]

        # Embed coefficients into full p-dimensional space
        beta_full = np.zeros(p)
        beta_full[active] = beta_hat.ravel()

        OLS_full[ell + 1] = beta_full
    
    OLS_array = np.array(list(OLS_full.values()))
    beta_norms = np.linalg.norm(beta, axis=1)
    distances = np.linalg.norm(OLS_array[:, None, :] - beta[None, :, :],axis=2,)
    ### Beta metric 1, sum over mixture components and finding the closest OLS
    ### vector that matches it.  Sum over K min over L.  Weighted by partition
    ### weights.
    distances_L = distances.min(axis=0)/beta_norms
    weighted_sum_over_K = (pi*distances_L).sum()
    
    ### Beta metric 2, sum over centres and finding the closest OLS
    ### vector that matches it.  Sum over L min over K.  Weighted by partition
    ### weights.
    closest_k = distances.argmin(axis=1)
    closest_beta_norms = beta_norms[closest_k]
    distances_K = distances.min(axis=1)/closest_beta_norms
    
    weighted_sum_over_L = (weights*distances_K).sum()
    
    ### In sample R^2.
    
    Y_hat = X @ OLS_array.T
    # R2 for each component
    R2 = 1 - ((y[:, None] - Y_hat) ** 2).sum(axis=0) / ((y - y.mean()) ** 2).sum()
    
    # Weighted average component R2
    weighted_R2 = (weights * R2).sum()
    
    
    ### Out of sample R^2
    
    Y_hat_test = X_test @ OLS_array.T
    y_train_mean = y.mean()

    R2_test = 1-((y_test[:, None] - Y_hat_test) ** 2).sum(axis=0)/ ((y_test - y_train_mean) ** 2).sum()
    weighted_R2_test = (weights * R2_test).sum()
    
    
        
    print(
        weighted_sum_over_K,weighted_sum_over_L,weighted_R2,weighted_R2_test,
    )

    min_per_class[l] = min_per_class[l] + (
        weighted_sum_over_K,weighted_sum_over_L,weighted_R2,weighted_R2_test,
    )

output = {
    "setup": {
        "job_id": job_id,
        "seed": seed,
        "p": p,
        "n": n,
        "active":float(row.active),
        "active_p": active_p,
        "K": K,
        "sampler": "ADS",
        "N_iter": 1000,
        "N_burnin": 1000,
        "n_chain": 100,
        "n_temp": 3,
        "g": g_scale,
        'prior_type': 'g'
    },
    "samples": samples,
    "results": min_per_class,
}
outfile = (
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
