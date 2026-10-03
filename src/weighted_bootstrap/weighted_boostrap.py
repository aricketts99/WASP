#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:14:46 2026

@author: andrew
"""


import numpy as np
import pandas as pd
import sys
from src.utils import lrrsg
from src.wass_approx_gen_ham import wass_approx
from sklearn import linear_model
from skglm import WeightedLasso


seed = 34567890
p = 200
active = 0.1

active_p = int(p*active)
df = 5

n = 100

prng = np.random.default_rng(12345)
beta0 = prng.choice([-3,3],size=active_p)

# Generate simulated data
beta, y, X = lrrsg(n=n, p=p, beta0=beta0,SNR=3.5,rho=0.5,seed=seed)

# Student-t errors, scaled to have variance sigma2
epsilon = (
    np.sqrt(1)
    * np.sqrt((df - 2) / df)
    * np.random.standard_t(df, size=(n, 1))
)

X = (X-X.mean())/X.std()
# responses
y = X.dot(beta) + epsilon

y = y-y.mean()


prng = np.random.default_rng(12345)
B = 100_000

#Random weighting 1 has W_{0,j} = 1 for all j and is of shape Bx(n+1)
RW1 = prng.exponential(1.0,size=(B,(n)))
RW1[:, 0] = 1

#Random weighting 1 has W_{0,j} = W_0 ~ Exp(1) for all j and is of shape Bx(n+1)
RW2 = prng.exponential(1.0,size=(B,(n+1)))
RW2[:,0] = prng.exponential(1.0,size=1)

RW3 = prng.exponential(1.0,size=(B,(n+p)))



def sample_RW_1_one_step(index,lambda_=0.5):
    clf = linear_model.Lasso(alpha=lambda_)
    scale = np.sqrt(RW1[index])
    wX = X*scale[:,None]
    wy = y.flatten()*scale
    clf.fit(wX, wy)
    return clf.coef_>0

def sample_RW_2_one_step(index,lambda_=4):
    model = WeightedLasso(lambda_,np.tile(RW2[0][0],p))
    scale = np.sqrt(RW2[index][1:])
    wX = X*scale[:,None]
    wy = y.flatten()*scale
    model.fit(wX, wy)
    return model.coef_>0


def sample_RW_3_one_step(index,lambda_=4):
    model = WeightedLasso(lambda_,RW3[index][100:])
    scale = np.sqrt(RW3[index][:100])
    wX = X*scale[:,None]
    wy = y.flatten()*scale
    model.fit(wX, wy)
    return model.coef_>0


from joblib import Parallel, delayed

samples = np.array(Parallel(
    n_jobs=8,
    verbose=10,
    max_nbytes="10M",
    mmap_mode="r"
)(
    delayed(sample_RW_3_one_step)(i)
    for i in range(B)
)).astype('float32')

import itertools
Ls = list(range(1,9))
seeds = list(range(1,101))

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

mu_true = X @ beta
for l in range(1,16):
    
    
    weights = np.asarray(min_per_class[l][1], dtype=float)
    summaries = np.asarray(min_per_class[l][2])
    
    if summaries.shape[0] != len(weights):
        summaries = summaries.T
    
    mu_pred = np.zeros_like(mu_true)
    
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