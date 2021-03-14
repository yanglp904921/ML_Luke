
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
max_iter = 10000
learning_rate = 0.01
activation = 'sigmoid'


algo = 'genetic_alg'
pop_sizes = [100, 500, 1000]
# pop_sizes = [100, 200, 500, 1000, 1500, 2000]
mutation_probs = [0.1, 0.3, 0.6]
df_GA = []
n = 1
for pop in pop_sizes:
    for mutate in mutation_probs:
        print('{} : population={}, mutation={}'.format(algo, pop, mutate))
        train_start = timer()
        ann_GA = mlh.NeuralNetwork(algorithm=algo, activation=activation,
                                   pop_size=pop, mutation_prob=mutate,
                                   hidden_nodes=hidden_nodes, learning_rate=learning_rate,
                                   max_attempts=100, max_iters=max_iter, early_stopping=False,
                                   is_classifier=True, bias=True, curve=True)
        ann_GA.fit(x_train, y_train)
        train_end = timer()
        duration = train_end - train_start
        y_train_head = ann_GA.predict(x_train)
        y_test_head = ann_GA.predict(x_test)
        stats = {'algo': 'GA',
                 'pop_size': pop,
                 'mutation_probs': mutate,
                 'time': duration,
                 'iterations': ann_GA.fitness_curve.shape[0],
                 'accuracy_train': accuracy_score(y_train, y_train_head),
                 'accuracy_test': accuracy_score(y_test, y_test_head),
                 'model': ann_GA}
        df_stats = pd.DataFrame([stats])
        pkl.dump([df_stats], file=open('results/Neural_Net_GA_{}.pkl'.format(n), 'wb'))
        df_GA.append(stats)
        n = n+1
df_GA = pd.DataFrame(df_GA)
pkl.dump([df_GA], file=open('results/Neural_Net_GA.pkl', 'wb'))

