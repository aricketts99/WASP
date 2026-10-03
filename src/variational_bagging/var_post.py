#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:52:40 2026

@author: andrew
"""

import numpy as np
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import sys
from src.utils import lrrsg
from src.bvs import BVS_MCMC
from src.wass_approx_gen_ham import wass_approx



job_id = 959#int(sys.argv[1])

params = pd.read_csv("parameters.csv")

row = params.iloc[job_id - 1]

seed = int(row.seed)
p = int(row.p)
active = float(row.active)

active_p = int(p*active)

n = 1000

prng = np.random.default_rng(12345)
beta0 = prng.choice([-3,3],size=active_p)

# Generate simulated data
beta, Y, X = lrrsg(n=n, p=p, beta0=beta0,SNR=3.5,rho=0.1,seed=seed)


import numpy as np
from joblib import Parallel, delayed


_R_X = None
_R_Y = None
_SVB_FIT = None


def initialise_worker(X, Y):
    global _R_X, _R_Y, _SVB_FIT

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()

    importr("sparsevb")

    _R_X = numpy2ri.py2rpy(np.asarray(X, dtype=float))
    _R_Y = numpy2ri.py2rpy(np.asarray(Y, dtype=float))

    _SVB_FIT = ro.r("sparsevb::svb.fit")

    print("R worker initialised", flush=True)


def run_batch(seeds):
    global _R_X, _R_Y, _SVB_FIT

    results = []

    for seed in seeds:
        rng = np.random.default_rng(int(seed))

        # Equivalent to sample(1:ncol(X)) in R
        update_order = rng.permutation(_R_X.ncol) + 1

        result = _SVB_FIT(
            X=_R_X,
            Y=_R_Y,
            update_order=update_order,
        )

        results.append({
            "mu": np.asarray(result.rx2("mu")),
            "sigma": np.asarray(result.rx2("sigma")),
            "gamma": np.asarray(result.rx2("gamma")),
            "intercept": np.asarray(result.rx2("intercept")),
            "seed": int(seed),
            "update_order": update_order,
        })

    return results
initialise_worker(X, Y)

seeds = np.arange(10, dtype=np.uint32)

results = run_batch(seeds)

print("Finished:", len(results))

# ------------------------------------------------------------
# TEST ONLY
# ------------------------------------------------------------

seeds = np.arange(10, dtype=np.uint32)

results_by_worker = Parallel(
    n_jobs=1,
    backend="loky",
    initializer=initialise_worker,
    initargs=(X, Y),
    verbose=10,
)(
    delayed(run_batch)(seeds),
)

results = results_by_worker[0]

print("Finished:", len(results))