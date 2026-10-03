#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:46:16 2026

@author: andrew
"""
import numpy as np
from src.utils.utils import _make_beta, _scale_beta, basic, block_corr, decoy, AR_1, basic_error, t_error, laplace_error, gen_Y


def experiment_1(n,p,active_p,seed):
    '''
    Basic Mixture with K = 2, no overlap and equal coefficients. Normal Errors.

    '''
    
    beta = _make_beta(p,active_p,2)
    beta = _scale_beta(beta,n,SNR=1.5*np.ones(2))
    X = basic(n,p,seed)
    epsilon = basic_error(n,seed)
    
    y,partition = gen_Y(X, beta, epsilon)
    
    return X,y,beta,partition, epsilon

def experiment_2(n,p,active_p,seed):
    '''
    Basic Mixture with K = 2, no overlap and equal coefficients. Normal Errors.
    AR_1 correlation

    '''
    
    beta = _make_beta(p,active_p,2)
    beta = _scale_beta(beta,n,SNR=1.5*np.ones(2))
    X = AR_1(n, p, active_p,rho=0.8)
    epsilon = basic_error(n,seed)
    
    y,partition = gen_Y(X, beta, epsilon)
    
    return X,y,beta,partition, epsilon


def experiment_3(n,p,active_p,seed):
    '''
    Contamination example, mixture with two components no overlap
    but one component has a lot less observations but much higher beta.

    '''
    beta = _make_beta(p,active_p,2)
    beta = _scale_beta(beta,n,SNR=np.array([5.0,125.0]))
    X = basic(n,p,seed)
    epsilon = basic_error(n,seed)
    
    
    
    y,partition = gen_Y(X, beta, epsilon,weights=np.array([0.9,0.1]))
    
    return X,y,beta,partition, epsilon

def experiment_4(n,p,active_p,seed):
    '''
    Contamination example, mixture with two components no overlap
    but one component has a lot less observations but much higher beta.
    AR_1 correlation.
    '''
    beta = _make_beta(p,active_p,2)
    beta = _scale_beta(beta,n,SNR=np.array([5.0,1250.0]))
    X = AR_1(n, p, active_p,rho=0.3)
    epsilon = basic_error(n,seed)
    
    
    
    y,partition = gen_Y(X, beta, epsilon,weights=np.array([0.9,0.1]))
    
    return X,y,beta,partition, epsilon



def experiment_5(n, p, active_p, overlap_frac, seed):
    '''
    Mixture with K = 5 and equal coefficients.
    overlap_frac controls the proportion of active variables
    shared between consecutive mixture components.
    Normal Errors.
    Block correlation in X.
    '''

    K = 5

    # Convert desired overlap proportion to an integer
    # overlap. For active_p = 1, overlap is necessarily 0.
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

    beta = _make_beta(p, active_p, K, overlap=overlap)
    beta = _scale_beta(beta, n, SNR=2.5*np.ones(K))

    X = block_corr(
        n,
        p,
        block_size=active_p*np.ones(K).astype(int),
        rho=0.9
    )

    epsilon = basic_error(n, seed)

    y, partition = gen_Y(X, beta, epsilon)

    return X, y, beta, partition, epsilon


def experiment_6(n, p, active_p, overlap_frac, seed):
    '''
    Mixture with K = 5 and equal coefficients.
    overlap_frac controls the proportion of active variables
    shared between consecutive mixture components.
    Normal Errors.
    Block correlation in X.
    '''

    K = 5

    # Convert desired overlap proportion to an integer
    # overlap. For active_p = 1, overlap is necessarily 0.
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

    beta = _make_beta(p, active_p, K, overlap=overlap)
    beta = _scale_beta(beta, n, SNR=1.5*np.ones(K))
    
    X = AR_1(n, p, active_p,0.9,seed)

    epsilon = basic_error(n, seed)

    y, partition = gen_Y(X, beta, epsilon)

    return X, y, beta, partition, epsilon








