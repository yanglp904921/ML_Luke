
import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as ANN
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, confusion_matrix

import utils_load as load
import utils_cal as ucal
import utils_plot as uplt


k = 20
data = 'phish'
x, y, _ = load.load_phish_data()
with open('results/dim_reduction_{}.pkl'.format(data), 'rb') as f:
    _, _, _, _, x_pca, x_ica, x_rca, x_rfc = pkl.load(f)


x_names = ['Original', 'PCA', 'ICA', 'RCA', 'RFC']
models = ['KM', 'EM']
rows = ['{}+{}'.format(s, t) for t in models for s in x_names]
x_temp = [x, x_pca, x_ica, x_rca, x_rfc]
x_km, x_em = [], []
for xi in x_temp:
    x_km.append(ucal.hstach_kmeans_label(xi, k=20))
    x_em.append(ucal.hstack_em_label(xi, k=20))
x_list = x_km + x_em


cols = ['f1_train', 'f1_test', 'accuracy_train', 'accuracy_test', 'time']
df = pd.DataFrame(index=rows, columns=cols)
for ii in range(len(x_list)):
    row, xi, yi = rows[ii], x_list[ii], y
    print('---------- {}: {} ----------'.format(ii, row))
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(xi, yi, test_size=0.20)
    ann = ANN(hidden_layer_sizes=5, learning_rate_init=0.01, activation='logistic',
              solver='adam', random_state=0, max_iter=10000)
    ann.fit(x_train, y_train)
    yp_train = ann.predict(x_train)
    yp_test = ann.predict(x_test)
    df.loc[row, 'f1_train'] = f1_score(y_train, yp_train)
    df.loc[row, 'f1_test'] = f1_score(y_test, yp_test)
    df.loc[row, 'accuracy_train'] = accuracy_score(y_train, yp_train)
    df.loc[row, 'accuracy_test'] = accuracy_score(y_test, yp_test)
    df.loc[row, 'time'] = (time.time() - start_time)
pkl.dump([df], file=open('results/ann_clusters_{}.pkl'.format(data), 'wb'))


