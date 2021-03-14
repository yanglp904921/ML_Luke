
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
algo = 'random_hill_climb'
restarts = [0, 5, 10, 25, 50,  100]
df_RHC = []
for restart in restarts:
    print('{} restart: {}'.format(algo, restart))
    train_start = timer()
    ann_SA = mlh.NeuralNetwork(algorithm=algo, activation=activation, restarts=restart,
                               hidden_nodes=hidden_nodes, learning_rate=learning_rate,
                               max_attempts=100, max_iters=max_iter, early_stopping=False,
                               is_classifier=True, bias=True, curve=True)
    ann_SA.fit(x_train, y_train)
    train_end = timer()
    duration = train_end - train_start
    y_train_head = ann_SA.predict(x_train)
    y_test_head = ann_SA.predict(x_test)
    stats = {'algo': 'RHC',
             'restart': restart,
             'time': duration,
             'iterations': ann_SA.fitness_curve.shape[0],
             'accuracy_train': accuracy_score(y_train, y_train_head),
             'accuracy_test': accuracy_score(y_test, y_test_head),
             'model': ann_SA}
    df_RHC.append(stats)
df_RHC = pd.DataFrame(df_RHC)
pkl.dump([df_RHC], file=open('results/Neural_Net_RHC.pkl', 'wb'))

