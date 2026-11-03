#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:27:52 2026

@author: andrew
"""

# 1. Initialize L binary centers using Kmeans
# 2. REPEAT until convergence:
#       a. ASSIGNMENT: Assign each sample to the nearest center (min DSS loss)
#       b. UPDATE: For each center, compute cluster mean, then find optimal binary center via LARS/IHT
#       c. CHECK: Calculate total loss, stop if change is small
# 3. RETURN best centers

import numpy as np
from sklearn.cluster import AgglomerativeClustering,KMeans, MiniBatchKMeans
from sklearn.linear_model import lars_path

def wass_approx(l,p,samples,lambda_,r_state,X):
    '''
    Parameters
    ----------
    l : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    samples : shape (m,p)
        DESCRIPTION.
    lambda_ : TYPE
        DESCRIPTION.
    r_state : TYPE
        DESCRIPTION.
    X : shape (n,p)
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    ##### PRE-COMPUTE PHASE FOR Xs AND betas
    ##### X is of shape (n,p) and samples(betas) are of shape (m,p)
    ##### I can thus make a matrix of shape (n,m) which we then can compute
    ##### the norm column wise to reduce to an array of m
    interaction_matrix = X @ samples.T
    norms = np.power(np.linalg.norm(interaction_matrix, axis=0),2)
    
    
    m = samples.shape[0]
    n = float(X.shape[0])
    inv_n = np.power(n,-1)
    
    
    #### Compute Gram matrix since this is fixed for all iterations
    G = X.T @ X
    print(m)
    
    kmeans = KMeans(n_clusters=l, init='k-means++', random_state=r_state,n_init=10)
    labelling = kmeans.fit_predict(samples)

    labels,counts = np.unique(labelling, return_counts=True)
    weights = counts/m
    centres = np.zeros((l,p))
    
    
    #Loop over all Kmeans clusters to create voronoi cells of samples
    #and compute the centre.
    #Centre computation is done by calculating the mean vector in each
    #cell.  Then using X@mean_vector as the Y data, compute the LARS path.
    #For every model in the LARS path, we choose the model which minimses
    #the cluster wise loss.
    for voronoi_cell in labels:
        
        #Get all the samples in the cell,calculate the mean vector and
        #get the predicitions.
        cell = samples[labelling == voronoi_cell]
        mean_vector = cell.mean(axis=0)
        y = X@mean_vector
        
        #Compute the LARS path, get the models
        _,_,path = lars_path(X,y,return_path=True,max_iter=1000)
        models_in_path = (path>0).astype('int')
        print('Models in Path Shape: '+str(models_in_path.shape))
        
        
        #For each model in LARS path compute the objective in the cluster.
        #That is compute \lambda||model||_0+n^{-1}*\sum_{\beta\in cell} ||X\gamma-X\beta||^2_2
        #The second half gives, \sum_{\beta\in cell} ||X\gamma-X\beta||^2_2 = 
        #\sum_{\beta\in cell} \gamma^T(X^TX)\gamma-2\gamma^T(X^TX)\beta_i+\beta_i^T(X^TX)\beta_i
        # = m*\gamma^TG\gamma-2\gamma^TG\sum_{i\incell}\beta_i+norms
        counting_norm = lambda_*np.sum(models_in_path,axis=0) #this is of shape path length
        
        #This does \gamma^TG\gamma for each model in models_in_path
        norm_1 = counts[voronoi_cell]*np.sum(models_in_path * (G @ models_in_path), axis=0)
        norm_2 = -2*models_in_path.T@G@np.sum(cell,axis=0)
        norm_3 = np.sum(norms[labelling == voronoi_cell])
        norm = counting_norm+inv_n*(norm_1+norm_2+norm_3)
        centre = models_in_path.T[np.argmin(norm)]
        centres[voronoi_cell] = centre
        
        
        # model_projection = X@path #this is of shape (n,pathlenghth)
        # print('Model Projection Shape: '+str(model_projection.shape))
        # norm_1 = counts[voronoi_cell]*np.power(np.linalg.norm(model_projection,axis=0),2) #this is of shape (path_length)
        # print('Norm 1 Shape: '+str(norm_1.shape))
        # norm_2 = np.sum(norms[labelling == voronoi_cell]) #(this of of shape 1)
        # print('Norm 2 Shape: '+str(norm_2.shape))
        # sum_X_betas = np.sum(interaction_matrix[:,labelling == voronoi_cell],axis=1)
        # norm_3 = -2*model_projection.T@sum_X_betas
        # print('Norm 3 Shape: '+str(norm_3.shape))
        # norm = counting_norm+inv_n*(norm_1+norm_2+norm_3)
        # centre = models_in_path.T[np.argmin(norm)]
        # print(centre.shape)
        # print(centres.shape)
        # centres[voronoi_cell] = centre
    
    ##### Calculate distance between each centre and each sample in each cell
    centres_G = centres @ G
    samples_G = samples @ G
    
    # Compute the squared norms for each vector
    # norm_centres shape: (L,), norm_samples shape: (M,)
    norm_centres = np.sum(centres * centres_G, axis=1)
    norm_samples = np.sum(samples * samples_G, axis=1)
    
    # Compute the cross term (centres @ G @ samples.T)
    # Shape: (L, M)
    cross_term = centres_G @ samples.T
    
    # Combine terms using broadcasting
    # D_sq[l, i] = norm_centres[l] + norm_samples[i] - 2 * cross_term[l, i]
    D_sq = norm_centres[:, np.newaxis] + norm_samples[np.newaxis, :] - 2 * cross_term
    
    # Ensure non-negative due to numerical precision errors and take sqrt
    distances = np.maximum(D_sq, 0.0)
    neighbourhoods = np.argmin(distances,axis=0)
    result = distances[neighbourhoods, np.arange(distances.shape[1])]

    loss_sums = np.bincount(
        neighbourhoods,
        weights=result,
        minlength=l
    )
    
    size_of_centres = lambda_*np.sum(centres,axis=1)
    print(size_of_centres.shape)
    
    print(loss_sums)
    
    loss_sums = inv_n*loss_sums+size_of_centres
    print(loss_sums)
    
    B = loss_sums / counts
    print(B)
    print(weights)
    
    iters = 0
    new_loss = weights@B
    old_loss = np.inf
    epsilon = 1e-8
    best_sol = []
    
    while old_loss-new_loss > epsilon and iters < 100:
        
        old_loss=new_loss
        ##### Calculate distance between each centre and each sample in each cell
        centres_G = centres @ G
        samples_G = samples @ G
        
        #squared norms for each vector, norm_centres shape: (L,), norm_samples shape: (M,)
        norm_centres = np.sum(centres * centres_G, axis=1)
        norm_samples = np.sum(samples * samples_G, axis=1)
        
        #cross term (centres @ G @ samples.T) shape: (L, M)
        cross_term = centres_G @ samples.T
        
        # Combine terms using broadcasting
        # D_sq[l, i] = norm_centres[l] + norm_samples[i] - 2 * cross_term[l, i]
        D_sq = norm_centres[:, np.newaxis] + norm_samples[np.newaxis, :] - 2 * cross_term
        
        # Ensure non-negative due to numerical precision errors and take sqrt
        distances = np.maximum(D_sq, 0.0)
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
        
        print(counts)
        weights = counts/m

        new_centres = np.zeros((l, p))
        
        # Pre-compute Gram matrix is already available as G
        # Iterate over each cluster to find the optimal sparse centre
        for k in range(l):
            # Get indices of samples assigned to cluster k
            cluster_indices = np.where(neighbourhoods == k)[0]
            count_k = len(cluster_indices)
            
            # Avoid division by zero if cluster is empty
            if count_k == 0:
                continue
            
            #get mean vector and compute LARS path
            mean_vector = samples[cluster_indices].mean(axis=0)
            y_k = X @ mean_vector
            _, _, path = lars_path(X, y_k, Gram=G, return_path=True)
            
            models_in_path = (path > 0).astype('int')
            
            #size penalty
            counting_norm = lambda_ * np.sum(models_in_path, axis=0)
            
            #For each model compute m_k * gamma^T G gamma (Binary)
            G_models = G @ models_in_path
            norm_1 = count_k * np.sum(models_in_path * G_models, axis=0)
            

            sum_beta = np.sum(samples[cluster_indices], axis=0)
            G_sum_beta = G @ sum_beta
            norm_2 = -2 * models_in_path.T @ G_sum_beta
            norm_3 = np.sum(norms[cluster_indices])
            norm = counting_norm + inv_n * (norm_1 + norm_2 + norm_3)
            
            # Select Best Binary Centre
            best_idx = np.argmin(norm)
            new_centres[k] = models_in_path[:, best_idx]

        
        centres = new_centres

        ##### Calculate distance between each centre and each sample in each cell
        centres_G = centres @ G
        samples_G = samples @ G
        
        # Compute the squared norms for each vector
        # norm_centres shape: (L,), norm_samples shape: (M,)
        norm_centres = np.sum(centres * centres_G, axis=1)
        norm_samples = np.sum(samples * samples_G, axis=1)
        
        # Compute the cross term (centres @ G @ samples.T)
        # Shape: (L, M)
        cross_term = centres_G @ samples.T
        
        # Combine terms using broadcasting
        # D_sq[l, i] = norm_centres[l] + norm_samples[i] - 2 * cross_term[l, i]
        D_sq = norm_centres[:, np.newaxis] + norm_samples[np.newaxis, :] - 2 * cross_term
        
        # Ensure non-negative due to numerical precision errors and take sqrt
        distances = np.maximum(D_sq, 0.0)
        neighbourhoods = np.argmin(distances,axis=0)
        result = distances[neighbourhoods, np.arange(distances.shape[1])]

        loss_sums = np.bincount(
            neighbourhoods,
            weights=result,
            minlength=l
        )
        
        size_of_centres = lambda_*np.sum(centres,axis=1)
        print(size_of_centres.shape)
        
        print(loss_sums)
        
        loss_sums = inv_n*loss_sums+size_of_centres
        print(loss_sums)
        
        B = loss_sums / counts
        print(B)
        print(weights)
        new_loss = weights@B
        best_sol.append(weights@B)
        iters += 1
    return weights@B,weights,centres,neighbourhoods,B,l,lambda_
        
        
        
        
        
        