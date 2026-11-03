#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:52:11 2026

@author: andrew
"""
import numpy as np
from numba import njit
from sklearn.cluster import AgglomerativeClustering,KMeans, MiniBatchKMeans

@njit(cache=True)
def cluster_sum(samples, labels, L):
    m, d = samples.shape

    sums = np.zeros((L, d), dtype=np.float32)
    counts = np.zeros(L, dtype=np.int32)

    for i in range(m):
        k = labels[i]
        counts[k] += 1
        for j in range(d):
            sums[k, j] += samples[i, j]

    return sums, counts

def generalized_hamming(X, alpha):
    """
    Compute generalized asymmetric Hamming distances between rows of X.
    X: binary matrix (n_samples, n_features)
    alpha: weight for 0->1, 2-alpha is weight for 1->0
    Returns: D, shape (n_samples, n_samples), where D[i,j] = distance from X[i] to X[j]
    """
    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        xi = X[i]
        # 0->1 mismatches
        mask01 = (xi == 0) & (X == 1)
        # 1->0 mismatches
        mask10 = (xi == 1) & (X == 0)
        D[i] = alpha * mask01.sum(axis=1) + (2 - alpha) * mask10.sum(axis=1)

    return D


def wass_approx(l,p,samples,alpha,r_state):
    m = samples.shape[0]

    kmeans = KMeans(n_clusters=l, init='k-means++', random_state=r_state,n_init=10)
    labelling = kmeans.fit_predict(samples)

    labels,counts = np.unique(labelling, return_counts=True)
    weights = counts/m
    centres = np.zeros((l,p))
    for voronoi_cell in labels:
        cell = samples[labelling == voronoi_cell]
        mean_vector = cell.mean(axis=0)
        centres[voronoi_cell] = (mean_vector >= alpha/2.0).astype(int)

    samples = samples.astype(np.uint8, copy=False)
    centres = centres.astype(np.uint8, copy=False)

    neg = (centres[:, None, :] & (1 - samples)).sum(axis=2)
    

    pos = ((1 - centres[:, None, :]) & samples).sum(axis=2)
    
    distances = alpha * neg + (2 - alpha) * pos

    neighbourhoods = np.argmin(distances,axis=0)
    result = distances[neighbourhoods, np.arange(distances.shape[1])]

    loss_sums = np.bincount(
        neighbourhoods,
        weights=result,
        minlength=l
    )
    
    B = loss_sums / counts
    
    iters = 0
    new_loss = weights@B
    old_loss = np.inf
    epsilon = 1e-8
    best_sol = []

    while old_loss-new_loss > epsilon and iters < 100:
        
        #Update alpha
        

        
        old_loss=new_loss

        neg = (centres[:, None, :] & (1 - samples)).sum(axis=2)
        

        pos = ((1 - centres[:, None, :]) & samples).sum(axis=2)
        
        distances = alpha * neg + (2 - alpha) * pos
        neighbourhoods = np.argmin(distances,axis=0)

        counts = np.array([np.count_nonzero(neighbourhoods == i) for i in range(0, l)])
        idx = np.where(counts==0)[0]
        neighbourhoods_size_1 = np.where(counts==1)[0]
        ids_in_sample = np.where(np.isin(neighbourhoods,neighbourhoods_size_1))[0]
        distances.T[ids_in_sample] = 0
        #norm_distance = distances / distances.sum(axis=1)[:, np.newaxis]
        
        row_sums = distances.sum(axis=1, keepdims=True)
        
        norm_distance = np.divide(
            distances,
            row_sums,
            out=np.zeros_like(distances, dtype=float),
            where=row_sums != 0
        )
        
        # Replace zero-sum rows with uniform probabilities
        zero_rows = (row_sums[:, 0] == 0)
        norm_distance[zero_rows] = 1.0 / distances.shape[1]
        for centre in idx:
            index = np.random.choice(m,p=norm_distance[centre])
            neighbourhoods[index] = centre

        counts = np.bincount(neighbourhoods, minlength=l)
        
        cluster_sums = np.zeros(
            (l, samples.shape[1]),
            dtype=np.float32
        )
        

        
        cluster_sums, counts = cluster_sum(samples, neighbourhoods, l)
        
        A = cluster_sums / counts[:, None]

        centres = (A >= (alpha / 2)).astype(np.uint8)
        weights = counts/m

        neg = (centres[:, None, :] & (1 - samples)).sum(axis=2)
        

        pos = ((1 - centres[:, None, :]) & samples).sum(axis=2)
        
        distances = alpha * neg + (2 - alpha) * pos
        
        result = distances[neighbourhoods, np.arange(distances.shape[1])]
        

        loss_sums = np.bincount(
            neighbourhoods,
            weights=result,
            minlength=l
        )
        
        B = loss_sums / counts
        new_loss = weights@B
        best_sol.append(weights@B)
        iters += 1
    return weights@B,weights,centres,neighbourhoods,B,l,alpha