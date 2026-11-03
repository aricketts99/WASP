#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:41:32 2026

Single working example of experiment 8, driven by ratios with fixed n = 1000.
(Previously: one row of parameters_mixture_ratios.csv per job.)

@author: andrew
"""
import os
import numpy as np
import pandas as pd
import sys

from src.bvs import BVS_MCMC
from src.wass_approx_dss import wass_approx
from src.utils.utils import _make_beta, _scale_beta, basic, block_corr, decoy, AR_1, basic_error, t_error, laplace_error, gen_Y
from scipy.stats import wasserstein_distance_nd
import pymc as pm, pytensor.tensor as pt
import arviz as az

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

n = 1000          # fixed
K = 2

# --- Ratios (edit these) -------------------------------------
ratio_n_p = .5   # n / p            -> p = n / ratio_n_p
ratio_k_n = 0.04  # active_p / n     -> active_p = ratio_k_n * n
# -------------------------------------------------------------

p = int(round(n / ratio_n_p))
active_p = int(round(ratio_k_n * n))

job_id = 1
seed = 123456

print(f"n={n}, p={p}, active_p={active_p}, K={K}, "
      f"ratio_n_p={ratio_n_p}, ratio_k_n={ratio_k_n}")


def experiment_9(n, p, active_p, overlap_frac, seed, n_test=1000):
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
    beta = _scale_beta(beta, n, SNR=1.5 * np.ones(K))

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
                        seed=seed + 10_00)
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

X,y,beta,partition,epsilon,X_test,y_test,partition_test, epsilon_test = experiment_9(n,p,active_p,0.5,seed)


labels, counts = np.unique(partition, return_counts=True)
pi = counts / len(partition)

labels, counts = np.unique(partition_test, return_counts=True)
pi_test = counts / len(partition_test)

# y  = y.reshape(-1,1)
# # Set up and run
# g_scale = (1.5**2)*np.log(p)

D_star = int(round(1.5 * active_p))

# numpy matrices of draws
with pm.Model() as m:
    sigma = pm.HalfNormal("sigma", 2.5)
    tau = pm.HalfStudentT("tau", nu=2,
                          sigma=D_star / (p - D_star) * sigma / np.sqrt(n))
    lam = pm.HalfStudentT("lam", nu=5, sigma=1, shape=p)
    c2 = pm.InverseGamma("c2", 1, 1)
    lam_t = lam * pt.sqrt(c2 / (c2 + tau**2 * lam**2))
    z = pm.Normal("z", 0, 1, shape=p)
    beta_hs = pm.Deterministic("beta_hs", z * tau * lam_t)
    pm.Normal("obs", pt.dot(X, beta_hs), sigma, observed=y)
    idata = pm.sample(target_accept=0.95, random_seed=seed,max_treedepth=8,chains=8,cores=8)


# p x M matrix of posterior draws (columns are draws)
samples = idata.posterior["beta_hs"].values.reshape(-1, p)   # (p, M)


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X_s = samples                                  # (M, p): one row per posterior draw



ks = range(1, 51)
inertia = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_s)
    inertia.append(km.inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(ks), inertia, "o-")
ax.set_xlabel("number of clusters k")
ax.set_ylabel("within-cluster sum of squares (inertia)")
ax.set_title("Elbow plot")
ax.set_xticks(list(ks))
fig.tight_layout()
plt.show()

# cluster labels for a chosen k
k = 2
labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X_s)
print(np.bincount(labels) / len(labels))         # cluster proportions

# test prediction: X_test and y_test already use the training centring
# pred = X_test @ beta_samples                                        # (n_test, M)
# test_mse = np.mean((pred.mean(axis=1) - y_test) ** 2)
# samples = fit.chains[:, 1001:, :]

# samples = samples[:, ::20, :].reshape(-1,p)

import itertools
Ls = [1,2,4,8,16]#,32]
seeds = list(range(1,22))

combined = list(itertools.product(Ls, seeds))
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(wass_approx)(c[0],p,samples,0.5,c[1],X)
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
    
    
    
    wass_beta = wasserstein_distance_nd(beta,OLS_array,pi,weights)
        
    print(f"L={l}: wass_beta={wass_beta:.4f}, "
          f"R2={weighted_R2:.4f}, R2_test={weighted_R2_test:.4f}")

    min_per_class[l] = min_per_class[l] + (
        weighted_sum_over_K,weighted_sum_over_L,weighted_R2,weighted_R2_test,wass_beta
    )

# output = {
#     "setup": {
#         "job_id": job_id,
#         "seed": seed,
#         "p": p,
#         "n": n,
#         "K": K,
#         "active_p": active_p,
#         "sampler": "ADS",
#         "N_iter": 1000,
#         "N_burnin": 1000,
#         "n_chain": 100,
#         "n_temp": 3,
#         "g": g_scale,
#         'prior_type': 'g',
#         'ratio_k_n':ratio_k_n,
#         'ratio_n_p':ratio_n_p
#     },
#     "samples": samples,
#     "results": min_per_class,
# }

# os.makedirs("results_08", exist_ok=True)
# outfile = (f'results_08/'
#     f"example_n{n}"
#     f"_rnp{ratio_n_p}"
#     f"_rkn{ratio_k_n}"
#     f"_p{p}"
#     f"_a{active_p}"
#     f"_K{K}"
#     f"_seed{seed}.zst"
# )
# import pickle
# import zstandard as zstd

# cctx = zstd.ZstdCompressor(level=19)

# with open(outfile, "wb") as f:
#     with cctx.stream_writer(f) as zf:
#         pickle.dump(output, zf, protocol=pickle.HIGHEST_PROTOCOL)

# print(f"Saved to {outfile}")


