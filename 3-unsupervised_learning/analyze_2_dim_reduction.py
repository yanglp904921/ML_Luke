
import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt


import utils_load as load
import utils_plot as uplt


data_list = ['mnist', 'phish']
for data in data_list:
    if data == 'phish':
        x, y, _ = load.load_phish_data()
    elif data == 'mnist':
        x, y, _ = pkl.load(file=open('data/{}.pkl'.format(data), 'rb'))
        x, y = x[0:20000], y[0:20000]
    with open('results/dim_reduction_{}.pkl'.format(data), 'rb') as f:
        df_pca, df_ica, df_rca, df_rfc, x_pca, x_ica, x_rca, x_rfc = pkl.load(f)

    # PCA ===================================================================
    fig = plt.figure(figsize=(8, 6))
    axs = fig.subplots(2, 2, sharex=False).flatten()
    # -----------------------------------------------------------------------
    df_pca.plot(y='variance', style='-o', grid=True, ax=axs[0])
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs[0].set_title('Variance Distribution ({})'.format(data))
    axs[0].set_xlabel('Feature ID')
    axs[0].set_ylabel('Variance')
    # ------------------------------------------------------------------------
    df_pca.plot(y='eigenvalue', style='-o', grid=True, ax=axs[1])
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs[1].set_title('Eigenvalue Distribution ({})'.format(data))
    axs[1].set_xlabel('Feature ID')
    axs[1].set_ylabel('Eigenvalue')
    # ------------------------------------------------------------------------
    df_pca.plot(y='cumulative_variance', style='-o', grid=True, ax=axs[2])
    axs[2].set_title('Cumulative Variance ({})'.format(data))
    axs[2].set_xlabel('Dimension')
    axs[2].set_ylabel('Cum Variance')
    # ------------------------------------------------------------------------
    uplt.plot_tsne(x_pca[:, 0:2], y, axs[3])
    axs[3].set_title('Scatter Plot ({})'.format(data))
    # ------------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/2_{}_1_PCA.png'.format(data))

    # ICA ===================================================================
    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2, sharex=False).flatten()
    # -----------------------------------------------------------------------
    df_ica.plot(y='kurtosis', style='-o', grid=True, ax=axs[0])
    axs[0].set_title('Kurtosis Change ({})'.format(data))
    axs[0].set_xlabel('Dimension')
    axs[0].set_ylabel('Kurtosis')
    # ------------------------------------------------------------------------
    uplt.plot_tsne(x_ica[:, 0:2], y, axs[1])
    axs[1].set_title('Scatter Plot ({})'.format(data))
    # ------------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/2_{}_2_ICA.png'.format(data))

    # RCA ===================================================================
    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2, sharex=False).flatten()
    # -----------------------------------------------------------------------
    col = 'reconstruction_err_mn'
    df_rca.plot(y=col, style='-o', grid=True, label='Error', ax=axs[0])
    ix = np.array(df_rca.index)
    mn = df_rca['reconstruction_err_mn'].values
    st = df_rca['reconstruction_err_st'].values
    axs[0].fill_between(ix, mn-2*st, mn+2*st, alpha=0.1, color="b")
    axs[0].set_title('Reconstruction Error ({})'.format(data))
    axs[0].set_xlabel('Dimension')
    axs[0].set_ylabel('Error')
    # ------------------------------------------------------------------------
    uplt.plot_tsne(x_rca[:, 0:2], y, axs[1])
    axs[1].set_title('Scatter Plot ({})'.format(data))
    # ------------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/2_{}_3_RCA.png'.format(data))

    # RFC ===================================================================
    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2, sharex=False).flatten()
    # -----------------------------------------------------------------------
    col = 'feature_importance'
    df_rfc2 = df_rfc.iloc[0:15, :]
    df_rfc2.plot.bar(x='feature_index', y=col, label=None, grid=True, ax=axs[0])
    axs[0].set_title('Feature Importance ({})'.format(data))
    axs[0].set_xlabel('Feature ID')
    axs[0].set_ylabel('Importance (%)')
    # ------------------------------------------------------------------------
    uplt.plot_tsne(x_rfc[:, 0:2], y, axs[1])
    axs[1].set_title('Scatter Plot ({})'.format(data))
    # ------------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/2_{}_4_RFC.png'.format(data))

