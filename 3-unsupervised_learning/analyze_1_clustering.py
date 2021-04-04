
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import utils_load as load
import utils_plot as uplt


data_list = ['phish', 'mnist']
k_list = [20, 30]
for ii in range(len(k_list)):
    data, k = data_list[ii], k_list[ii]
    print('---------- {} -----------'.format(data))
    df, = pkl.load(file=open('results/clustering_{}.pkl'.format(data), 'rb'))
    fig = plt.figure(figsize=(8, 6))
    ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2, sharex=True)
    # ---------------------------------------------
    df1 = df[[('SSE', 'K-Means'), ('log-likely', 'EM')]].copy()
    df1.columns = ['K-Means SSE', 'EM Log-Likelihood']
    df1['K-Means SSE'] = df1['K-Means SSE']/(df1['K-Means SSE'].max())
    df1['EM Log-Likelihood'] = df1['EM Log-Likelihood'] / (df1['EM Log-Likelihood'].max())
    df1.plot.line(style='-o', ax=ax1, grid=True)
    ax1.set_title('Similarity Measure ({})'.format(data))
    ax1.set_ylabel('Similarity Measure')
    # ---------------------------------------------
    df.plot.line(y='homogeneity score', ax=ax2, grid=True, marker='o')
    ax2.set_title('Homogeneity Score ({})'.format(data))
    ax2.set_ylabel('Homogeneity Score')
    # ---------------------------------------------
    df.plot.line(y='adj mutual info', ax=ax3, grid=True, marker='o')
    ax3.set_title('Adj Mutual Info ({})'.format(data))
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Adj Mutual Info')
    # ---------------------------------------------
    df.plot.line(y='accuracy score', ax=ax4, grid=True, marker='o')
    ax4.set_title('accuracy score ({})'.format(data))
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Accuracy Score')
    # ---------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/1_{}_1_Search.png'.format(data))

    # plot scatter
    if data == 'bank':
        x, y, _ = load.load_bank_data()
        x, y = x[0:5000], y[0:5000]
    elif data == 'phish':
        x, y, _ = load.load_phish_data()
        x, y = x[0:5000], y[0:5000]
    elif data == 'mnist':
        x, y, _ = pkl.load(file=open('data/{}.pkl'.format(data), 'rb'))
        x, y = x[0:10000], y[0:10000]
    uplt.plot_scatter(x, y, k, data, '2_Scater')

