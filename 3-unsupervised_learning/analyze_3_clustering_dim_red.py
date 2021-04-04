
import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt


import utils_load as load
import utils_plot as uplt


data_list = ['phish', 'mnist']
k_list = [20, 30]
for ii in range(len(k_list)):
    data, k = data_list[ii], k_list[ii]
    if data == 'phish':
        x, y, _ = load.load_phish_data()
    elif data == 'mnist':
        x, y, _ = pkl.load(file=open('data/{}.pkl'.format(data), 'rb'))
        x, y = x[0:20000], y[0:20000]
    with open('results/dim_reduction_{}.pkl'.format(data), 'rb') as f:
        df_pca, df_ica, df_rca, df_rfc, x_pca, x_ica, x_rca, x_rfc = pkl.load(f)

    x_dim = (x_pca.shape[1], x_ica.shape[1], x_rca.shape[1], x_rfc.shape[1])
    print('-------- X dim ({}) ----------'.format(data))
    print('x_pca, x_ica, x_rca, x_rfc: '+str(x_dim))
    uplt.plot_scatter_dim_red(x, x_pca, y, k, data, '1_PCA')
    uplt.plot_scatter_dim_red(x, x_ica, y, k, data, '2_ICA')
    uplt.plot_scatter_dim_red(x, x_rca, y, k, data, '3_RCA')
    uplt.plot_scatter_dim_red(x, x_rfc, y, k, data, '4_RFC')


# -------- X dim (phish) ----------
# x_pca, x_ica, x_rca, x_rfc: (26, 38, 18, 14)
# -------- X dim (mnist) ----------
# x_pca, x_ica, x_rca, x_rfc: (152, 100, 78, 243)

