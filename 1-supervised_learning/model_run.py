
from model_class import Classifier
from utils_load import *

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DTS
from sklearn.ensemble import GradientBoostingClassifier as BST
from sklearn.neural_network import MLPClassifier as ANN


# Load Data
# data_name = "Planar"
data_name = "Phish"
if data_name == 'Planar':
    x, y = load_planar_data()
elif data_name == "Phish":
    x, y = load_phish_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

leaf_min = round(0.005*len(y_train))
leaf_max = round(0.05*len(y_train))
tree_leaf = np.linspace(leaf_min, leaf_max, 10).astype(int)
tree_depth = np.linspace(1, 21, 10).astype(int)

bst_leaf = np.linspace(leaf_min, leaf_max, 5).astype(int)
bst_depth = np.arange(3)+1
bst_estimator = np.linspace(10, 100, 5).astype(int)
bst_learn_rate = np.linspace(0.001, 0.1, 4)


run = {"KNN": {'data_name': data_name,
               'model_name': 'KNN',
               'model_fit': None,
               'model_init': KNN(n_neighbors=5, weights='uniform'),
               'param_grid': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 25],
                              'weights': ['uniform', 'distance']},
               'param_complexity': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 25]}},
       "DTS": {'data_name': data_name,
               'model_name': 'DTS',
               'model_fit': None,
               'model_init': DTS(criterion='entropy'),
               'param_grid': {'min_samples_leaf': tree_leaf,
                              'max_depth': tree_depth},
               'param_complexity': {'max_depth': np.arange(29)+1}},
       "BST": {'data_name': data_name,
               'model_name': 'BST',
               'model_fit': None,
               'model_init': BST(),
               'param_grid': {'min_samples_leaf': tree_leaf,
                              'max_depth': np.arange(4)+1,
                              'learning_rate': bst_learn_rate,
                              'n_estimators': bst_estimator},
               'param_complexity': {'n_estimators': bst_estimator}},
       "SVM": {'data_name': data_name,
               'model_name': 'SVM',
               'model_fit': None,
               'model_init': SVM(random_state=0),
               'param_grid': {'C': [1e-3, 1e-2, 1e01, 1],
                              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
               'param_complexity': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}},
       "ANN": {'data_name': data_name,
               'model_name': 'ANN',
               'model_fit': None,
               'model_init': ANN(solver='adam', random_state=0, max_iter=1000),
               'param_grid': {'hidden_layer_sizes': [5, 10, 15],
                              'activation': ['logistic'],
                              'learning_rate_init': [0.05]},
               'param_complexity': {'hidden_layer_sizes': [1, 2, 3, 4, 5, 10, 20, 50]}},
       }

run = {"BST": {'data_name': data_name,
               'model_name': 'BST',
               'model_fit': None,
               'model_init': BST(),
               'param_grid': {'min_samples_leaf': bst_leaf,
                              'max_depth': bst_depth,
                              'learning_rate': bst_learn_rate,
                              'n_estimators': bst_estimator},
               'param_complexity': {'n_estimators': bst_estimator}},
       }

for key, model in run.items():
    model_init = model['model_init']
    model_name = model['model_name']
    data_name = model['data_name']
    param_grid = model['param_grid']
    param_complexity = model['param_complexity']
    clf = Classifier(model=model_init, model_name=model_name, data_name=data_name)
    clf.search_hyper_params(x_train, y_train, **param_grid)
    clf.cal_model_complexity(x_train, y_train, x_test, y_test, param_complexity)
    clf.cal_learning_curve(x_train, y_train)
    clf.cal_model_evaluation(x_train, y_train, x_test, y_test)
    print('=============== data={}, model={} =============='.format(data_name, model_name))
    print(clf.model_evaluation)
    # clf.model_complexity.plot(x='param', y=['f1_train', 'f1_test'])
    model['model_fit'] = clf
    pkl.dump([model], file=open('models/{}_{}.pkl'.format(data_name, model_name), 'wb'))

