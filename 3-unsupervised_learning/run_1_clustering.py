
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.cluster import KMeans as KMeans
from sklearn.mixture import GaussianMixture as GMixture
from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_score, \
    homogeneity_score, adjusted_mutual_info_score

import utils_load as load
import utils_cal as ucal

# data = 'bank'
# data = 'phish'
data = 'mnist'
if data == 'bank':
    x, y, _ = load.load_bank_data()
elif data == 'phish':
    x, y, _ = load.load_phish_data()
elif data == 'mnist':
    x, y, _ = pkl.load(file=open('data/{}.pkl'.format(data), 'rb'))
    x, y = x[0:20000], y[0:20000]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


seed = 0
num_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 100]
# num_clusters = [3, 5, 7, 10, 15]


models = ['K-Means', 'EM']
metrics = ['SSE', 'log-likely', 'silhouette score', 'adj mutual info',
           'homogeneity score', 'accuracy score', 'f1 score']
cols = pd.MultiIndex.from_product([metrics, models], names=["model", "metric"])
df = pd.DataFrame(index=num_clusters, columns=cols)


km = KMeans(random_state=seed)
gm = GMixture(random_state=seed, covariance_type='diag')
for k in num_clusters:
    print('k={}'.format(k))
    km.set_params(n_clusters=k)
    gm.set_params(n_components=k)
    km.fit(x_train)
    gm.fit(x_train)

    labels_km = km.predict(x_train)
    labels_gm = gm.predict(x_train)
    y_pred_km = ucal.cal_cluster_pred(y_train, labels_km)
    y_pred_gm = ucal.cal_cluster_pred(y_train, labels_gm)

    df.loc[k, ('SSE', 'K-Means')] = -km.score(x_train)
    df.loc[k, ('log-likely', 'EM')] = gm.score(x_train)
    df.loc[k, ('silhouette score', 'K-Means')] = silhouette_score(x_train, labels_km)
    df.loc[k, ('silhouette score', 'EM')] = silhouette_score(x_train, labels_gm)

    df.loc[k, ('adj mutual info', 'K-Means')] = adjusted_mutual_info_score(y_train, labels_km)
    df.loc[k, ('adj mutual info', 'EM')] = adjusted_mutual_info_score(y_train, labels_gm)
    df.loc[k, ('homogeneity score', 'K-Means')] = homogeneity_score(y_train, labels_km)
    df.loc[k, ('homogeneity score', 'EM')] = homogeneity_score(y_train, labels_gm)

    df.loc[k, ('accuracy score', 'K-Means')] = ucal.cal_cluster_accuracy(y_train, y_pred_km)
    df.loc[k, ('accuracy score', 'EM')] = ucal.cal_cluster_accuracy(y_train, y_pred_gm)
    # df.loc[k, ('f1 score', 'K-Means')] = ucal.cal_f1_score(y_train, y_pred_km)
    # df.loc[k, ('f1 score', 'EM')] = ucal.cal_f1_score(y_train, y_pred_gm)
pkl.dump([df], open('results/{}.pkl'.format(data), 'wb'))


