

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from utils_cal import *

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans as KMeans
from sklearn.mixture import GaussianMixture as GMixture


sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})


def plot_tsne(x, y, ax):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(y))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    c = palette[y.astype(np.int)]
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=c)
    ax.set_xticks([])
    ax.set_yticks([])

    # add the labels for each digit
    for i in range(num_classes):
        xtext, ytext = np.median(x[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=14)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=3, foreground="w"),
            PathEffects.Normal()])


def plot_scatter(x, y, k, data_name, step_name, seed=1):
    km = KMeans(random_state=seed)
    gm = GMixture(random_state=seed, covariance_type='diag')
    km.set_params(n_clusters=k)
    gm.set_params(n_components=k)
    km.fit(x)
    gm.fit(x)
    l_km = km.predict(x)
    l_gm = gm.predict(x)
    y_km = cal_cluster_pred(y, l_km)
    y_gm = cal_cluster_pred(y, l_gm)

    x_tsne = TSNE(random_state=seed).fit_transform(x)
    y_list = [y, y, l_km, y_km, l_gm, y_gm]
    titles = ['', 'Original Label',
              'K-Means Clusters', 'K-Means Majority Vote',
              'EM Clusters', 'EM Majority Vote']

    fig = plt.figure(figsize=(8, 10))
    axs = fig.subplots(3, 2)
    axs = axs.flatten()
    for ii in range(len(y_list)):
        if ii == 0:
            axs[ii].set_xticks([])
            axs[ii].set_yticks([])
            continue
        yii, axii = y_list[ii], axs[ii]
        plot_tsne(x_tsne, yii, axii)
        axii.set_title('{} ({})'.format(titles[ii], data_name))
    fig.tight_layout()
    fig.savefig('figures/1_{}_{}.png'.format(data_name, step_name))


def cal_cluster_labels(x, y, k, seed):
    km = KMeans(random_state=seed)
    gm = GMixture(random_state=seed, covariance_type='diag')
    km.set_params(n_clusters=k)
    gm.set_params(n_components=k)
    km.fit(x)
    gm.fit(x)
    l_km = km.predict(x)
    l_gm = gm.predict(x)
    y_km = cal_cluster_pred(y, l_km)
    y_gm = cal_cluster_pred(y, l_gm)
    return l_km, l_gm, y_km, y_gm


def plot_scatter_dim_red(x_org, x_dim, y, k, data_name, mdl_name, seed=1):
    print('----- {}: {} -----'.format(data_name, mdl_name))
    l_km_org, l_gm_org, y_km_org, y_gm_org = cal_cluster_labels(x_org, y, k, seed)
    l_km_dim, l_gm_dim, y_km_dim, y_gm_dim = cal_cluster_labels(x_dim, y, k, seed)
    x_tsne_org = TSNE(random_state=seed).fit_transform(x_org)
    x_tsne_dim = TSNE(random_state=seed).fit_transform(x_dim)
    x_list = [x_tsne_org, x_tsne_dim, x_tsne_org, x_tsne_dim,
              x_tsne_org, x_tsne_dim, x_tsne_org, x_tsne_dim]
    y_list = [l_km_org, l_km_dim, y_km_org, y_km_dim,
              l_gm_org, l_gm_dim, y_gm_org, y_gm_dim]
    titles = ['K-Means Clusters (Original Features)',
              'K-Means Clusters (Dim-Reduced Features)',
              'K-Means Majority Vote (Original Features)',
              'K-Means Majority Vote (Dim-Reduced Features)',
              'EM Clusters (Original Features)',
              'EM Clusters (Dim-Reduced Features)',
              'EM Majority Vote (Original Features)',
              'EM Majority Vote (Dim-Reduced Features)']
    fig = plt.figure(figsize=(8, 12))
    axs = fig.subplots(4, 2)
    axs = axs.flatten()
    for ii in range(len(y_list)):
        xii, yii, axii = x_list[ii], y_list[ii], axs[ii]
        plot_tsne(xii, yii, axii)
        axii.set_title(titles[ii])
    fig.tight_layout()
    fig.savefig('figures/3_{}_{}.png'.format(data_name, mdl_name))





'''
def plot_scatter_dim_red(x_org, x_dim, y, k, data_name, mdl_name, seed=1):
    print('----- {}: {} -----'.format(data_name, mdl_name))
    km = KMeans(random_state=seed)
    gm = GMixture(random_state=seed, covariance_type='diag')
    km.set_params(n_clusters=k)
    gm.set_params(n_components=k)
    km.fit(x_dim)
    gm.fit(x_dim)
    l_km = km.predict(x_dim)
    l_gm = gm.predict(x_dim)
    y_km = cal_cluster_pred(y, l_km)
    y_gm = cal_cluster_pred(y, l_gm)

    x_tsne_org = TSNE(random_state=seed).fit_transform(x_org)
    x_tsne_dim = TSNE(random_state=seed).fit_transform(x_dim)
    y_list = [y, y, l_km, y_km, l_gm, y_gm]
    titles = ['Original Label (Original Features)',
              'Original Label (Dim-Reduced Features)',
              'K-Means Clusters (Dim-Reduced Features)',
              'K-Means Majority Vote (Dim-Reduced Features)',
              'EM Clusters (Dim-Reduced Features)',
              'EM Majority Vote (Dim-Reduced Features)']

    fig = plt.figure(figsize=(8, 10))
    axs = fig.subplots(3, 2)
    axs = axs.flatten()
    for ii in range(len(y_list)):
        yii, axii = y_list[ii], axs[ii]
        if ii == 0:
            plot_tsne(x_tsne_org, yii, axii)
        else:
            plot_tsne(x_tsne_dim, yii, axii)
        axii.set_title(titles[ii])
    fig.tight_layout()
    fig.savefig('figures/3_{}_{}.png'.format(data_name, mdl_name))


def plot_cluster_search(df, data):
    fig = plt.figure(figsize=(8, 6))
    ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2, sharex=True)
    # ---------------------------------------------
    df.plot.line(y=('SSE', 'K-Means'), style='-bo', ax=ax1, label='K-Means')
    ax1.set_title('Similarity Measure ({})'.format(data))
    # ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Sum of Squared Error')
    ax5 = ax1.twinx()
    df.plot.line(y=('log-likely', 'EM'), ax=ax5, style='-ro', label='EM')
    ax5.set_ylabel('Log-Likelihood')
    ax5.tick_params(axis='y', colors='red')
    ax5.yaxis.label.set_color('red')
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
    fig.savefig('figures/1_Cluster_{}.png'.format(data))
'''


