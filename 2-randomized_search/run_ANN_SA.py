
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import pickle as pkl
import mlrose_hiive as mlh
from mlrose_hiive.algorithms.decay import ExpDecay
from sklearn.neural_network import MLPClassifier as ANN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report, \
    roc_auc_score, accuracy_score, f1_score, confusion_matrix


# Load Data
x, y = pkl.load(file=open('data/planar_data.pkl', 'rb'))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

seed = 1
np.random.seed(seed)

hidden_nodes = [5]
max_iter = 100000
learning_rate = 0.01
activation = 'sigmoid'


algo = 'simulated_annealing'
init_temperatures = [1, 10, 100, 500, 1000]
df_SA = []
for temp in init_temperatures:
    print('{} init_temperature: {}'.format(algo, temp))
    train_start = timer()
    exp_decay = ExpDecay(init_temp=temp, exp_const=0.005, min_temp=0.001)
    ann_SA = mlh.NeuralNetwork(algorithm=algo, activation=activation, schedule=exp_decay,
                               hidden_nodes=hidden_nodes, learning_rate=learning_rate,
                               max_attempts=100, max_iters=max_iter, early_stopping=True,
                               is_classifier=True, bias=True, curve=True)
    ann_SA.fit(x_train, y_train)
    train_end = timer()
    duration = train_end - train_start
    y_train_head = ann_SA.predict(x_train)
    y_test_head = ann_SA.predict(x_test)
    stats = {'algo': 'SA',
             'init_temperature': temp,
             'time': duration,
             'iterations': ann_SA.fitness_curve.shape[0],
             'accuracy_train': accuracy_score(y_train, y_train_head),
             'accuracy_test': accuracy_score(y_test, y_test_head),
             'model': ann_SA}
    df_SA.append(stats)
df_SA = pd.DataFrame(df_SA)
pkl.dump([df_SA], file=open('results/Neural_Net_SA.pkl', 'wb'))

