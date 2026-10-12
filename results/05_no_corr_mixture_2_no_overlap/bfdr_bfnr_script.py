#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:24:30 2026

@author: andrew
"""

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

job_id = 100#int(sys.argv[1])
row = params.iloc[job_id - 1]

seed = int(row.seed)
p = int(row.p)
active = float(row.active)/K

active_p = int(p*active)



def experiment_7(n, p, active_p, overlap_frac, seed, n_test=1000):
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

X,y,beta,partition,epsilon,X_test,y_test,partition_test, epsilon_test = experiment_7(n,p,active_p,0.5,seed)


#### WE ARE GOING TO VARY ALPHA TO EXAMINE THE RELATIONSHIP BETWEEN IT AND BFDR/BFNR TRADEOFF OVER L




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
alphas = np.linspace(0.1, 1.9)
Ls = [1,2,4,8,16,32,64]
seeds = list(range(1,12))

combined = list(itertools.product(Ls, seeds,alphas))
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(wass_approx)(c[0],p,samples,c[2],c[1])
    for c in combined
)

# results = []
# for c in combined:
#     results.append(wass_approx(c[0],p,samples,1.0,c[1]))



min_per_class = {}
for key, group in itertools.groupby(results, key=lambda x: (x[-2],x[-1])):
    min_per_class[key] = min(group, key=lambda x: x[0])
    

def compute_bfdr_bfnr(samples, weights, neighbourhoods, alpha):
    """
    Compute posterior BFDR and BFNR for a Wasserstein posterior approximation.

    Parameters
    ----------
    samples : ndarray (n_samples, p)
        Original posterior binary samples.

    weights : ndarray (L,)
        Wasserstein cluster weights.

    neighbourhoods : ndarray (n_samples,)
        Cluster assignment for each posterior sample.

    alpha : float
        Hamming loss parameter.

    Returns
    -------
    bfdr : float
    bfnr : float
    """

    fp = 0.0
    tp = 0.0
    fn = 0.0

    L = len(weights)

    for k in range(L):

        # samples assigned to cluster k
        cluster = samples[neighbourhoods == k]

        if cluster.shape[0] == 0:
            continue

        # cluster-specific posterior inclusion probabilities
        pi_k = cluster.mean(axis=0)

        # Bayes decision under weighted Hamming loss
        selected = pi_k >= alpha / 2


        # expected false discoveries
        fp_k = np.sum(1 - pi_k[selected])

        # expected true discoveries
        tp_k = np.sum(pi_k[selected])

        # expected false non-discoveries
        fn_k = np.sum(pi_k[~selected])


        # weight by Wasserstein posterior mass
        fp += weights[k] * fp_k
        tp += weights[k] * tp_k
        fn += weights[k] * fn_k


    bfdr = fp / max(fp + tp, 1e-12)

    bfnr = fn / max(fn + tp, 1e-12)

    return np.array([bfdr, bfnr])


metrics = np.empty((len(Ls), len(alphas), 2))

alpha_idx = {alpha: j for j, alpha in enumerate(alphas)}
L_idx = {L: i for i, L in enumerate(Ls)}

for (L, alpha), values in min_per_class.items():
    i = L_idx[L]
    j = alpha_idx[alpha]

    loss, weights, centres, neighbourhoods, B, l, _ = values

    metrics[i, j] = compute_bfdr_bfnr(
        samples, weights, neighbourhoods, alpha
    )

import matplotlib.pyplot as plt

for i, L in enumerate(Ls):
    plt.plot(
        metrics[i, :, 0],
        metrics[i, :, 1],
        marker="o",
        markersize=3,
        label=f"L={L}"
    )
    
plt.xlabel("BFDR")
plt.ylabel("BFNR")
plt.legend()
plt.show()

