
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.linalg import pinv


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans as KMeans
from sklearn.mixture import GaussianMixture as GMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC


def search_pca(x, seed=1):
    print(10 * '-' + ' PCA ' + 10 * '-')
    col = 'cumulative_variance'
    pca = PCA(random_state=seed)
    pca.fit(x)
    df = pd.DataFrame(index=np.arange(1, x.shape[1] + 1, 1))
    df[col] = np.cumsum(pca.explained_variance_ratio_)
    df['variance'] = pca.explained_variance_
    df['eigenvalue'] = pca.singular_values_
    return df


def search_ica(x, seed=1):
    col = 'kurtosis'
    if x.shape[1] < 100:
        dims = list(np.arange(2, (x.shape[1] - 1), 3)) + [x.shape[1]]
    else:
        dims = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, x.shape[1]]
    df = pd.DataFrame(index=dims)
    df[col] = np.nan
    ica = ICA(random_state=seed)
    for k in dims:
        print(10*'-'+' ICA (dim={})'.format(k)+10*'-')
        ica.set_params(n_components=k)
        x_ = ica.fit_transform(x)
        kurt = kurtosis(x_)
        df.loc[k, col] = np.mean(np.abs(kurt))
    return df


def search_rca(x, n_per_k=10):
    col1 = 'reconstruction_err_mn'
    col2 = 'reconstruction_err_st'
    if x.shape[1] < 100:
        dims = list(np.arange(2, (x.shape[1] - 1), 3)) + [x.shape[1]]
    else:
        dims = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, x.shape[1]]
    df = pd.DataFrame(index=dims)
    df[col1], df[col2] = np.nan, np.nan
    err_mat = np.zeros(shape=(len(dims), n_per_k))
    for i in range(len(dims)):
        k = dims[i]
        print(10 * '-' + ' RCA (dim={})'.format(k) + 10 * '-')
        for j in range(n_per_k):
            rca = RCA(n_components=k, random_state=j)
            rca.fit(x)
            w = rca.components_.todense()
            p = pinv(w)
            x_reconstruct = ((p @ w) @ (x.T)).T
            err_mat[i, j] = np.nanmean(np.square(x-x_reconstruct))
    df[col1] = err_mat.mean(axis=1, keepdims=True)
    df[col2] = err_mat.std(axis=1, keepdims=True)
    return df


def search_rca_cor(x, n_per_k=10, seed=1):
    col1 = 'pairwise_distance_cor_mn'
    col2 = 'pairwise_distance_cor_st'
    if x.shape[1] < 100:
        dims = list(np.arange(2, (x.shape[1] - 1), 3)) + [x.shape[1]]
    else:
        dims = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, x.shape[1]]
    df = pd.DataFrame(index=dims)
    df[col1], df[col2] = np.nan, np.nan
    dis0 = pairwise_distances(x)
    dis_mat = np.zeros(shape=(len(dims), n_per_k))
    for i in range(len(dims)):
        k = dims[i]
        for j in range(n_per_k):
            print(10 * '-' + ' RCA (dim={}, i={})'.format(k, j) + 10 * '-')
            rca = RCA(n_components=k, random_state=j)
            x_ = rca.fit_transform(x)
            dis1 = pairwise_distances(x_)
            dis_mat[i, j] = np.corrcoef(dis0.ravel(), dis1.ravel())[0, 1]
    df[col1] = dis_mat.mean(axis=1, keepdims=True)
    df[col2] = dis_mat.std(axis=1, keepdims=True)
    return df


def search_rfc(x, y, seed=1):
    print(10 * '-' + ' RFC ' + 10 * '-')
    min_samples_leaf = np.int32(x.shape[0]/100)
    rfc = RFC(min_samples_leaf=min_samples_leaf, n_estimators=400,
              random_state=seed, n_jobs=-1)
    rfc.fit(x, y)
    df = pd.DataFrame()
    df['feature_index'] = np.arange(0, len(rfc.feature_importances_))
    df['feature_importance'] = rfc.feature_importances_
    df.sort_values(by=['feature_importance'], inplace=True, ascending=False)
    df['cumulative_score'] = df['feature_importance'].cumsum()
    df.index = np.arange(0, len(rfc.feature_importances_))
    return df


def transform_x_pca(x, df, percent=0.95, seed=1):
    col = 'cumulative_variance'
    k = np.min(df.index[df[col]>=percent])
    model = PCA(random_state=seed)
    x_pca = model.fit_transform(x)[:, 0:k]
    return x_pca


def transform_x_ica(x, df, seed=1):
    col = 'kurtosis'
    k = df.index[df[col].argmax()]
    if k == df.index.max():
        k = np.floor(0.85 * df.index.max())
    if k > 100:
        k = 100
    model = ICA(random_state=seed, n_components=k)
    x_ica = model.fit_transform(x)
    return x_ica


def transform_x_rca(x, seed=1):
    col = 'reconstruction_err_mn'
    # k = df.index[df[col].argmax()]
    # if k == df.index.max():
    #     k = np.floor(0.9 * df.index.max())
    n = x.shape[1]
    if n < 50:
        k = np.floor(0.4 * n).astype(int)
    elif n < 500:
        k = np.floor(0.1 * n).astype(int)
    elif n:
        k = np.floor(0.1 * n).astype(int)
    model = RCA(random_state=seed, n_components=k)
    x_rca = model.fit_transform(x)
    return x_rca


def transform_x_rfc(x, df, percent=0.95):
    col = 'cumulative_score'
    index_max = np.min(df.index[df[col]>=percent])+1
    feature_id = df['feature_index'].values[:index_max]
    x_rfc = x[:, feature_id]
    return x_rfc


def hstach_kmeans_label(x, k, seed=1):
    km = KMeans(random_state=seed)
    km.set_params(n_clusters=k)
    km.fit(x)
    l = km.predict(x)
    l_sd = np.reshape((l - l.mean()) / l.std(), (len(l), 1))
    x_new = np.hstack([x, l_sd])
    return x_new


def hstack_em_label(x, k, seed=1):
    gm = GMixture(random_state=seed, covariance_type='diag')
    gm.set_params(n_components=k)
    gm.fit(x)
    l = gm.predict(x)
    l_sd = np.reshape((l - l.mean()) / l.std(), (len(l), 1))
    x_new = np.hstack([x, l_sd])
    return x_new


def cal_cluster_pred(y_train, cluster_labels):
    y_predict = np.zeros(y_train.shape)
    for label in set(cluster_labels):
        index = cluster_labels == label
        y_uniques, counts = np.unique(y_train[index], return_counts=True)
        v_most_count = y_uniques[np.argmax(counts)]
        y_predict[index] = v_most_count
    return y_predict


def cal_cluster_accuracy(y_train, cluster_labels):
    y_pred = cal_cluster_pred(y_train, cluster_labels)
    return accuracy_score(y_train, y_pred)


def cal_f1_score(y_train, cluster_labels):
    y_pred = cal_cluster_pred(y_train, cluster_labels)
    return f1_score(y_train, y_pred)


