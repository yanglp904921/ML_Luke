
# import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt


def plot_model_performance(clf):
    dfc = clf.model_complexity
    dfl = clf.learning_curve
    model_name = clf.get_model_name()
    data_name = clf.get_data_name()
    param_name = dfc['param_name'][0].replace('_', ' ').title()
    cols = ['f1_train', 'f1_test', 'accuracy_train', 'accuracy_test']
    if (np.min(dfc[cols].values) < 0.5) or (np.min(dfl[cols].values) < 0.5):
        y_lim = [0, 1.05]
    else:
        y_lim = [0.5, 1.05]

    fig = plt.figure(figsize=(5, 7))
    ax1, ax2 = fig.subplots(2, 1, sharex=False)
    # ---------------------------------------------
    # style=['bo--', 'bo-', 'ro--', 'ro-']
    dfc.plot.line(x='param', y=cols, ax=ax1, grid=True, marker='o')
    ax1.set_ylim(y_lim)
    ax1.set_title('Model Complexity of {} ({})'.format(model_name, data_name))
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Scores')
    # ---------------------------------------------
    dfl.plot.line(x='train_size', y=cols, ax=ax2, grid=True, marker='o')
    ax2.set_ylim(y_lim)
    ax2.set_title('Learning Curve of {} ({})'.format(model_name, data_name))
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Scores')
    fig.tight_layout()
    fig.savefig('figures/Single_{}_{}.png'.format(data_name, model_name))


def plot_decision_boundary(model, X, y):
    # extra margen
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # grilla de puntos a una distancia h entre ellos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # predice el valor de la funcion para toda la grilla
    x_test = np.c_[xx.ravel(), yy.ravel()]
    Z = model(x_test)
    Z = Z.reshape(xx.shape)
    # grafica el contorno y los ejemplos de entrenamiento
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    # plt.ylabel('x2')
    # plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

