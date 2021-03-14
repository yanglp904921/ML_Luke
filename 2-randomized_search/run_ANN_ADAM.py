
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import pickle as pkl
import mlrose_hiive as mlh
from mlrose_hiive.algorithms.decay import ExpDecay
from sklearn.neural_network import MLPClassifier as ANN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, \
    roc_auc_score, accuracy_score, f1_score, confusion_matrix


x, y = pkl.load(file=open('data/planar_data.pkl', 'rb'))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


hidden_nodes = [5]
learning_rate = 0.01
algo = 'adam'
train_start = timer()
ann_ADAM = ANN(solver='adam', hidden_layer_sizes=hidden_nodes,
               activation='logistic', learning_rate_init=0.05,)
ann_ADAM.fit(x_train, y_train)
train_end = timer()
duration = train_end - train_start
y_train_head = ann_ADAM.predict(x_train)
y_test_head = ann_ADAM.predict(x_test)
stats = {'algo': 'ADAM',
         'time': duration,
         'iterations': len(ann_ADAM.loss_curve_),
         'accuracy_train': accuracy_score(y_train, y_train_head),
         'accuracy_test': accuracy_score(y_test, y_test_head),
         'model': ann_ADAM}
df_ADAM = pd.DataFrame([stats])
pkl.dump([df_ADAM], file=open('results/Neural_Net_ADAM.pkl', 'wb'))


hidden_nodes = [5]
max_iter = 100000
learning_rate = 0.001
activation = 'sigmoid'
algo = 'gradient_descent'
train_start = timer()
ann_GD = mlh.NeuralNetwork(algorithm=algo, activation=activation,
                           hidden_nodes=hidden_nodes, learning_rate=learning_rate,
                           max_attempts=100, max_iters=max_iter, early_stopping=False,
                           is_classifier=True, bias=True, curve=True)
ann_GD.fit(x_train, y_train)
train_end = timer()
duration = train_end - train_start
y_train_head = ann_GD.predict(x_train)
y_test_head = ann_GD.predict(x_test)
stats = {'algo': 'GD',
         'time': duration,
         'iterations': ann_GD.fitness_curve.shape[0],
         'accuracy_train': accuracy_score(y_train, y_train_head),
         'accuracy_test': accuracy_score(y_test, y_test_head),
         'model': ann_GD}
df_GD = pd.DataFrame([stats])
pkl.dump([df_GD], file=open('results/Neural_Net_GD.pkl', 'wb'))

