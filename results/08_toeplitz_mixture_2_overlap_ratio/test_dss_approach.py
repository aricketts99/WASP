#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:41:32 2026

@author: andrew
"""
import os
import numpy as np
import pandas as pd
import sys


import argparse
import os
import time

import numpy as np
from scipy.special import expit

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro.diagnostics import summary
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from src.wass_approx_dss import wass_approx
from src.bvs import BVS_MCMC

from src.utils.utils import _make_beta, _scale_beta, basic, block_corr, decoy, AR_1, basic_error, t_error, laplace_error, gen_Y
from scipy.stats import wasserstein_distance_nd

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

n = 1000

params = pd.read_csv("parameters_mixture_ratios.csv")
K = 2

job_id = 9348#int(sys.argv[1])
row = params.iloc[job_id - 1]

# seed = int(row.seed)
# p = int(row.p)
# active = float(row.active)/K

# active_p = int(p*active)

seed = int(row.seed)
p = int(row.p)
# Read active_p directly from the new CSV
active_p = int(row.active_p) 

# Optional: Verify n matches if you want to be strict
n = int(row.n)

ratio_n_p = row.ratio_n_p
ratio_k_n = row.ratio_k_n



def experiment_8(n, p, active_p, overlap_frac, seed, n_test=1000):
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
    beta = _scale_beta(beta, n, SNR=5.5 * np.ones(K))

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






# regression model with continuous-valued outputs/responses
def model_normal_likelihood(X, Y):
    D_X = X.shape[1]

    # sample from horseshoe prior
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(D_X)))
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.ones(1)))

    # note that in practice for a normal likelihood we would probably want to
    # integrate out the coefficients (as is done for example in sparse_regression.py).
    # however, this trick wouldn't be applicable to other likelihoods
    # (e.g. bernoulli, see below) so we don't make use of it here.
    unscaled_betas = numpyro.sample("unscaled_betas", dist.Normal(0.0, jnp.ones(D_X)))
    scaled_betas = numpyro.deterministic("betas", tau * lambdas * unscaled_betas)

    # compute mean function using linear coefficients
    mean_function = jnp.dot(X, scaled_betas)

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    numpyro.sample("Y", dist.Normal(mean_function, sigma_obs), obs=Y)


# helper function for HMC inference
def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )

    mcmc.run(rng_key, X, Y)
    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()
    summary_dict = summary(samples, group_by_chain=False)

    print("\nMCMC elapsed time:", time.time() - start)

    return samples



def main(args):
    N, D_X = args.num_data, 32

    print("[Experiment with continuous-valued responses]")
    # first generate and analyze data with continuous-valued responses
    X,y,beta,partition,epsilon,X_test,y_test,partition_test, epsilon_test = experiment_8(n,p,active_p,0.5,seed)

    # do inference
    rng_key, rng_key_predict = random.split(random.key(0))
    samples = run_inference(model_normal_likelihood, args, rng_key, X, y)

    # print(samples.shape)
    # y  = y.reshape(-1,1)
    # samples = fit.chains[:, 11:, :].reshape(-1,p)
    loss,weights,centres,neighbourhoods,B,l,lambda_ = wass_approx(5,p,np.array(samples['betas']),10.0,123,X)

    return samples,loss,weights,centres,neighbourhoods,B,l,lambda_


def generate_synthetic_data(N, p, M, L=4, noise_level=0.1, random_state=42):
    """
    Generates synthetic X, samples, and ground truth labels.
    """
    np.random.seed(random_state)
    
    # 1. Generate Ground Truth Centres (Binary, Non-overlapping support)
    # Each centre gets p/L unique indices
    gt_centres = np.zeros((L, p))
    indices_per_cluster = p // L
    
    for k in range(L):
        start_idx = k * indices_per_cluster
        end_idx = start_idx + indices_per_cluster
        # Set a block of indices to 1 for each cluster
        gt_centres[k, start_idx:end_idx] = 1
    
    # 2. Generate Samples around these centres
    true_labels = np.random.choice(L, size=M)
    samples = np.zeros((M, p))
    
    for i in range(M):
        cluster_idx = true_labels[i]
        # Base coefficient vector
        base_vector = gt_centres[cluster_idx]
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, size=p)
        samples[i] = base_vector + noise
        
    # 3. Generate Design Matrix X
    X = np.random.normal(0, 1, size=(N, p))
    
    return X, samples, true_labels, gt_centres

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horseshoe regression example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=10, type=int)
    parser.add_argument("--num-data", nargs="?", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    #samples,loss,weights,centres,neighbourhoods,B,l,lambda_ = main(args)
    

    # Import wass_approx from your script
    # from your_script import wass_approx 
    

    
    # --- Parameters ---
    N = 1000   # Number of rows in X
    p = 800    # Number of features (coefficients)
    M = 2000   # Number of samples
    L = 8     # Number of clusters (must match ground truth)
    lambda_ = 0.5
    r_state = 42
    
    # --- Generate Data ---
    X, samples, true_labels, gt_centres = generate_synthetic_data(N, p, M, L)
    
    # --- Run Clustering ---
    loss,weights,centres,neighbourhoods,B,l,lambda_ = wass_approx(8, p, samples, lambda_, r_state, X)
    
    
    # # --- Validation ---
    # ari = adjusted_rand_score(true_labels, centres.T)
    # print(f"Adjusted Rand Score: {ari:.4f}")
    # print(f"Ground Truth Centres Support: {np.sum(gt_centres, axis=1)}")

    



