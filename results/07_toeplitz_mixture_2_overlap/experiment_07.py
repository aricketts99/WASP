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

params = pd.read_csv("parameters_mixture_contam_3.csv")
K = 5

job_id = int(sys.argv[1])
row = params.iloc[job_id - 1]

seed = int(row.seed)
p = int(row.p)
active = float(row.active)/K

active_p = int(p*active)

# def experiment_5(n, p, active_p, overlap_frac, seed):
#     '''
#     Mixture with K = 5 and equal coefficients.
#     overlap_frac controls the proportion of active variables
#     shared between consecutive mixture components.
#     Normal Errors.
#     Block correlation in X.
#     '''

#     K = 5

#     # Convert desired overlap proportion to an integer
#     # overlap. For active_p = 1, overlap is necessarily 0.
#     if active_p <= 1:
#         overlap = 0
#     else:
#         overlap = int(round(overlap_frac * active_p))
#         overlap = min(overlap, active_p - 1)

#     # Check that the K submodels fit inside p
#     required_p = active_p + (K - 1) * (active_p - overlap)

#     if required_p > p:
#         raise ValueError(
#             f"Experiment infeasible: p={p}, active_p={active_p}, "
#             f"K={K}, overlap_frac={overlap_frac}, "
#             f"overlap={overlap}. Need p >= {required_p}."
#         )

#     beta = _make_beta(p, active_p, K, overlap=overlap)
#     beta = _scale_beta(beta, n, SNR=2.5*np.ones(K))

#     X = block_corr(
#         n,
#         p,
#         block_size=active_p*np.ones(K).astype(int),
#         rho=0.8,
#         seed=seed
#     )
#     # Centre and scale X
#     X = X - X.mean(axis=0)
#     X_std = X.std(axis=0, ddof=1)
#     X = X / X_std

#     epsilon = basic_error(n, seed)

#     y, partition = gen_Y(X, beta, epsilon,seed=seed)
    
#     y = y-y.mean(axis=0)
    
#     # ---------- NEW: test set from the same DGP ----------
#     n_test = 100
#     X_test = block_corr(
#         n,
#         p,
#         block_size=active_p*np.ones(K).astype(int),
#         rho=0.8,
#         seed=seed+1
#     )
#     X_test = (X_test - X.mean(axis=0)) / X_std                # training mean/std, NOT test's own
#     epsilon_test = basic_error(n_test, seed + 1) # different seed from training
#     print("X_test:", X_test.shape)
#     print("beta:", beta.shape)
#     print("epsilon_test:", epsilon_test.shape)
#     print("n_test:", n_test)
#     y_test, partition_test = gen_Y(X_test, beta, epsilon_test,seed=seed+1)
#     y_test = np.asarray(y_test).ravel() - y.mean(axis=0)      # training y-mean

#     return X, y, beta, partition, epsilon, y_test, X_test


def experiment_5(n, p, active_p, overlap_frac, seed, n_test=1000):
    '''
    Mixture with K = 5 and equal coefficients.
    overlap_frac controls the proportion of active variables
    shared between consecutive mixture components.
    Normal errors.
    Block correlation in X.

    Returns a training set (X, y), the true coefficients (beta, K x p),
    the training component labels (partition), the training errors,
    and an independent test set (X_test, y_test) from the same DGP.
    The test set is centred/scaled using TRAINING statistics only.
    '''

    K = 5

    # ---------------------------------------------------------
    # Overlap: convert the desired proportion to an integer count.
    # For active_p = 1 the overlap is necessarily 0.
    # ---------------------------------------------------------
    if active_p <= 1:
        overlap = 0
    else:
        overlap = int(round(overlap_frac * active_p))
        overlap = min(overlap, active_p - 1)

    # Check that the K submodels fit inside p
    required_p = active_p + (K - 1) * (active_p - overlap)
    if required_p > p:
        raise ValueError(
            f"Experiment infeasible: p={p}, active_p={active_p}, "
            f"K={K}, overlap_frac={overlap_frac}, "
            f"overlap={overlap}. Need p >= {required_p}."
        )

    # ---------------------------------------------------------
    # True coefficients, shape (K, p).
    # Scaled using the TRAINING n so the SNR is defined once
    # and the same beta is used for train and test.
    # ---------------------------------------------------------
    beta = _make_beta(p, active_p, K, overlap=overlap)
    beta = _scale_beta(beta, n, SNR=2.5 * np.ones(K))

    block_size = active_p * np.ones(K).astype(int)

    # ---------------------------------------------------------
    # Training data
    # ---------------------------------------------------------
    X = block_corr(n, p, block_size=block_size, rho=0.8, seed=seed)

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
    X_test = block_corr(n_test, p, block_size=block_size, rho=0.8,
                        seed=seed + 10_000)
    X_test = (X_test - X_mean) / X_std          # training mean/std, not test's own

    epsilon_test = basic_error(n_test, seed=seed + 20_000)
    y_test, partition_test = gen_Y(X_test, beta, epsilon_test,
                                   seed=seed + 30_000)
    y_test = np.asarray(y_test).ravel() - y_mean  # training mean of y

    # ---------------------------------------------------------
    # Sanity checks (cheap, and they catch the shape/seed bugs)
    # ---------------------------------------------------------
    assert X_test.shape == (n_test, p), X_test.shape
    assert epsilon_test.shape[0] == n_test, epsilon_test.shape
    assert y_test.shape == (n_test,), y_test.shape
    assert not np.allclose(X_test[:5, :5], X[:5, :5]), "test X looks like a copy of train X"

    return X, y, beta, partition, epsilon, X_test, y_test

X,y,beta,partition,epsilon,X_test,y_test = experiment_5(n,p,active_p,0.5,seed)

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
seeds = list(range(1,102))

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
    # Out-of-sample R²
    # ---------------------------------------------------------
    # y_test is already centred by the TRAINING mean, so ss_tot_test is the
    # error of the "predict the training mean" baseline. Can be negative.
    ss_tot_test = np.sum(y_test ** 2)

    r2_OLS_test = {}
    r2_post_test = {}
    for ell in range(len(weights)):
        pred_OLS_t = X_test @ OLS_full[ell + 1]
        pred_post_t = shrinkage * pred_OLS_t
        r2_OLS_test[ell + 1] = 1 - np.sum((y_test - pred_OLS_t) ** 2) / ss_tot_test
        r2_post_test[ell + 1] = 1 - np.sum((y_test - pred_post_t) ** 2) / ss_tot_test

    # 1. OLS: R² per model, then average
    r2_OLS_error_test = sum(w * r2_OLS_test[ell + 1] for ell, w in enumerate(weights))

    # 2. OLS: aggregate coefficients/predictions first, then R²
    r2_OLS_agg_test = 1 - np.sum((y_test - X_test @ OLS_agg) ** 2) / ss_tot_test

    # 3. Shrunk OLS: R² per model, then average
    r2_post_error_test = sum(w * r2_post_test[ell + 1] for ell, w in enumerate(weights))

    # 4. Shrunk OLS: aggregate first, then R²
    r2_post_agg_test = 1 - np.sum((y_test - X_test @ post_agg) ** 2) / ss_tot_test


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
        r2_OLS_error_test, r2_OLS_agg_test,
        r2_post_error_test, r2_post_agg_test,
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
    print(f"  R² OLS (test):   error={r2_OLS_error_test:.4f}   aggregate={r2_OLS_agg_test:.4f}")
    print(f"  R² Post (test):  error={r2_post_error_test:.4f}   aggregate={r2_post_agg_test:.4f}")

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
