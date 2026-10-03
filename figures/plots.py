#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:02:17 2026

@author: andrew
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def elbow(min_per_class,filename):
    '''
    Takes dict of tuples over L to plot loss
    '''
    
    # fig,ax = plt.subplots(figsize=(10,10))
    
    # ax.plot([v[0] for v in min_per_class.values()])

    # ax.set_xlabel('L')
    # ax.set_ylabel('Loss')
    # ax.set_title('Elbow Plot: Loss vs L (number of centres/neighbourhoods)')
    # ax.legend()
    # ax.grid(True)
    # fig.savefig('figures/'+str(filename),format='pdf',bbox_inches='tight')
    
    # Build a DataFrame from your dict
    df = pd.DataFrame({
        "L": list(min_per_class.keys()),
        "Loss": [v[0] for v in min_per_class.values()]
    }).sort_values("L")
    
    sns.set_theme(style="whitegrid", context="talk")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x="L",
        y="Loss",
        marker="o",
        ax=ax
    )
    
    ax.set_title("Elbow Plot: Loss vs L (number of centres/neighbourhoods)")
    ax.set_xlabel("L")
    ax.set_ylabel("Loss")
    
    plt.tight_layout()
    
    fig.savefig(f"figures/{filename}.pdf", format="pdf", bbox_inches="tight")
    return

def centre_tables(min_per_class,L=1):
    centres = min_per_class[L][2]
    weights = min_per_class[L][1]
    indices_per_row = [np.where(row == 1)[0]+1 for row in centres]
    df = pd.DataFrame({"Indices of Active Variables of Centres": indices_per_row,"Weights": weights})
    df = df.sort_values(by=["Weights"],ascending=False)
    df["Indices of Active Variables of Centres"] = df[
    "Indices of Active Variables of Centres"
    ].apply(
        lambda x: r"$\{" + ",".join(map(str, x)) + r"\}$"
    )
    latex = df.to_latex(index=False)
    return df

def voronoi_summary(min_per_class,samples,filename,threshold=0.1,L=1):
    fig, axs = plt.subplots(figsize=(15+L,8+L))
    centres = min_per_class[L][2]
    neighbourhoods = min_per_class[L][3]
    weights = min_per_class[L][1]
    
    sample_dataframe = pd.DataFrame(samples)
    sample_dataframe['neighbourhoods'] = neighbourhoods
    row_weights = sample_dataframe['neighbourhoods'].map(lambda x: weights[x])
    sample_dataframe = sample_dataframe.iloc[row_weights.argsort()[::-1]].reset_index(drop=True)
    grouped_mean = sample_dataframe.groupby('neighbourhoods').mean().iloc[np.argsort(weights)[::-1]].reset_index(drop=True)
    centres = centres[np.argsort(weights)[::-1]]
    weights = np.sort(weights)[::-1]
    # Convert grouped DataFrame to a NumPy array
    data_array = grouped_mean.values
    # Mean per index
    means = data_array
    n_sets, N = means.shape

    
    mask = np.any(means >= threshold, axis=0)
    filtered_rows = means[:, mask]
    sns.heatmap(
        filtered_rows,
        cmap="Blues",
        annot=True,
        cbar=True,
        linewidths=1.5,
        linecolor="black",
        xticklabels=np.where(mask)[0]+1,
        ax=axs,
        fmt='.2f'
    )
    indices_per_row = [np.where(row == 1)[0] for row in centres]
    axs.set_title('Mean Vector for Each Centre', fontsize=12)
    fig.savefig('figures/'+str(filename)+'.pdf',format='pdf',bbox_inches='tight')
    return

def log_posts(fit,filename):
    fig, axs = plt.subplots(fit.n_temp, 1, figsize=(20, 18), sharex=True)

    for temp in range(fit.n_temp):
        for chain in range(fit.n_chain):
            axs[temp].plot(fit.log_posts[temp, chain, :], alpha=0.5)

        axs[temp].set_ylabel(f"T={temp}")

    axs[-1].set_xlabel("Iteration")
    fig.savefig('figures/'+str(filename),format='pdf',bbox_inches='tight')
    return

def model_size(fit,filename):
    fig, axs = plt.subplots(fit.n_temp, 1,
                            figsize=(10,8),
                            sharex=True)
    
    for ax, t in zip(axs, range(fit.n_temp)):
    
        for chain in range(fit.n_chain):
            ax.plot(fit.model_sizes[t, chain],
                    alpha=0.3)
    
        ax.set_title(f"Temp {t}")
    
    fig.savefig('figures/'+str(filename),format='pdf',bbox_inches='tight')
    return