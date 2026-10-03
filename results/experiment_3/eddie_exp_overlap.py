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

from experiments import experiment_5

from src.bvs import BVS_MCMC
from src.wass_approx_gen_ham import wass_approx


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

n = 1000

params = pd.read_csv("parameters_mixture_contam_3.csv")
K = 5

job_id = int(sys.argv[1])
row = params.iloc[job_id - 1]

seed = int(row.seed)
p = int(row.p)
active = float(row.active)/K

active_p = int(p*active)

X,y,beta,partition,epsilon = experiment_5(n,p,active_p,0.5,seed)

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
seeds = list(range(1,21))

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

    # ---------------------------------------------------------
    # Posterior/shrunken estimates
    # ---------------------------------------------------------

    OLS_post = {
        k: v * shrinkage
        for k, v in OLS_full.items()
    }

    # ---------------------------------------------------------
    # True component proportions and coefficients
    # ---------------------------------------------------------

    beta_trues = [
        (np.mean(partition == k), beta[k])
        for k in range(K)
    ]

    # Population/aggregate true coefficient
    beta_true = np.sum(
        [p_k * beta_k for p_k, beta_k in beta_trues],
        axis=0
    )

    # ---------------------------------------------------------
    # Aggregate OLS and posterior estimates
    # ---------------------------------------------------------

    OLS_agg = sum(
        w * OLS_full[ell + 1]
        for ell, w in enumerate(weights)
    )

    post_agg = sum(
        w * OLS_post[ell + 1]
        for ell, w in enumerate(weights)
    )

    # ---------------------------------------------------------
    # OLS errors
    # ---------------------------------------------------------

    # Average component-wise error
    error_OLS_error_comp = sum(
        p_k * w * np.linalg.norm(OLS_full[ell + 1] - beta_k)
        / np.linalg.norm(beta_k)
        for p_k, beta_k in beta_trues
        for ell, w in enumerate(weights)
    )

    # Error relative to population truth
    error_OLS_error_pop = sum(
        w * np.linalg.norm(OLS_full[ell + 1] - beta_true)
        / np.linalg.norm(beta_true)
        for ell, w in enumerate(weights)
    )

    # Aggregate estimate, error relative to components
    error_OLS_agg_comp = sum(
        p_k * np.linalg.norm(OLS_agg - beta_k)
        / np.linalg.norm(beta_k)
        for p_k, beta_k in beta_trues
    )

    # Aggregate estimate, error relative to population
    error_OLS_agg_pop = (
        np.linalg.norm(OLS_agg - beta_true)
        / np.linalg.norm(beta_true)
    )

    # ---------------------------------------------------------
    # Posterior/shrunken errors
    # ---------------------------------------------------------

    # Average component-wise error
    error_post_error_comp = sum(
        p_k * w * np.linalg.norm(OLS_post[ell + 1] - beta_k)
        / np.linalg.norm(beta_k)
        for p_k, beta_k in beta_trues
        for ell, w in enumerate(weights)
    )

    # Error relative to population truth
    error_post_error_pop = sum(
        w * np.linalg.norm(OLS_post[ell + 1] - beta_true)
        / np.linalg.norm(beta_true)
        for ell, w in enumerate(weights)
    )

    # Aggregate estimate, error relative to components
    error_post_agg_comp = sum(
        p_k * np.linalg.norm(post_agg - beta_k)
        / np.linalg.norm(beta_k)
        for p_k, beta_k in beta_trues
    )

    # Aggregate estimate, error relative to population
    error_post_agg_pop = (
        np.linalg.norm(post_agg - beta_true)
        / np.linalg.norm(beta_true)
    )

    # ---------------------------------------------------------
    # R² calculations
    # ---------------------------------------------------------

    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # Individual-model R²s
    r2_OLS = {}
    r2_post = {}

    for ell, (w, gamma) in enumerate(zip(weights, summaries)):

        # This works for both ordinary and empty models.
        # Empty model has OLS_full = zero vector, hence zero prediction.
        pred_OLS = X @ OLS_full[ell + 1]
        pred_post = shrinkage * pred_OLS

        r2_OLS[ell + 1] = (
            1 - np.sum((y - pred_OLS) ** 2) / ss_tot
        )

        r2_post[ell + 1] = (
            1 - np.sum((y - pred_post) ** 2) / ss_tot
        )

    # ---------------------------------------------------------
    # 1. OLS: calculate R² for each model, then average
    # ---------------------------------------------------------

    r2_OLS_error = sum(
        w * r2_OLS[ell + 1]
        for ell, w in enumerate(weights)
    )

    # ---------------------------------------------------------
    # 2. OLS: aggregate predictions first, then calculate R²
    # ---------------------------------------------------------

    pred_OLS_agg = X @ OLS_agg

    r2_OLS_agg = (
        1 - np.sum((y - pred_OLS_agg) ** 2) / ss_tot
    )

    # ---------------------------------------------------------
    # 3. Shrunk OLS: calculate R² for each model, then average
    # ---------------------------------------------------------

    r2_post_error = sum(
        w * r2_post[ell + 1]
        for ell, w in enumerate(weights)
    )

    # ---------------------------------------------------------
    # 4. Shrunk OLS: aggregate predictions first, then calculate R²
    # ---------------------------------------------------------

    pred_post_agg = X @ post_agg

    r2_post_agg = (
        1 - np.sum((y - pred_post_agg) ** 2) / ss_tot
    )


    # ---------------------------------------------------------
    # Component-wise Hamming error
    # ---------------------------------------------------------
    
    # True binary vector for each component, recovered from beta
    beta_trues = [
        (np.mean(partition == k), (beta[k] != 0).astype(int))
        for k in range(K)
    ]
    
    # Weighted Hamming error:
    # average over posterior centres and true components
    weighted_error = sum(
        w * p_k * np.sum(np.abs(summaries[ell] - truth_k)) / p
        for ell, w in enumerate(weights)
        for p_k, truth_k in beta_trues
    )

    # ---------------------------------------------------------
    # Store results
    # ---------------------------------------------------------

    min_per_class[l] = min_per_class[l] + (
        r2_OLS_error,
        r2_OLS_agg,
        r2_post_error,
        r2_post_agg,
        weighted_error,
        error_OLS_error_comp,
        error_OLS_error_pop,
        error_OLS_agg_comp,
        error_OLS_agg_pop,
        error_post_error_comp,
        error_post_error_pop,
        error_post_agg_comp,
        error_post_agg_pop,
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print(f"\nL = {l}")
    print(f"  R² OLS:   error={r2_OLS_error:.4f}   aggregate={r2_OLS_agg:.4f}")
    print(f"  R² Post:  error={r2_post_error:.4f}   aggregate={r2_post_agg:.4f}")
    print(f"  OLS err:  component={error_OLS_error_comp:.4f}   population={error_OLS_error_pop:.4f}")
    print(f"  OLS agg:  component={error_OLS_agg_comp:.4f}   population={error_OLS_agg_pop:.4f}")
    print(f"  Post err: component={error_post_error_comp:.4f}   population={error_post_error_pop:.4f}")
    print(f"  Post agg: component={error_post_agg_comp:.4f}   population={error_post_agg_pop:.4f}")
    print(f"  Hamming:  {weighted_error:.4f}")

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
        "N_iter": 3000,
        "N_burnin": 1000,
        "n_chain": 100,
        "n_temp": 6,
        "g": g_scale,
        'prior_type': 'g'
    },
    "samples": samples,
    "results": min_per_class,
}
outfile = (f'results/experiment_overlap/'
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
