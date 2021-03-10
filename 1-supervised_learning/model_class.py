
import numpy as np
import pandas as pd

from sklearn.utils._testing import ignore_warnings
# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone as sk_clone


from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, confusion_matrix


class Classifier:
    def __init__(self, model_name="", data_name="", model=None,  score="f1", verbose=1):
        # model name
        model_set = {'SVM', 'KNN', 'DTS', 'BST', 'ANN'}
        if model_name in model_set:
            self._model_name = model_name
        else:
            raise Exception("model name should be: {}".format(model_set))
        # model for classification
        self._model = model
        self._data_name = data_name
        # score used for GridSearch
        self._score = score
        # Controls the verbosity
        self._verbose = verbose
        # Results of GridSearch
        self._grid_search = None
        # learn curve
        self.learning_curve = None
        # model complexity
        self.model_complexity = None
        # model complexity
        self.model_evaluation = None
        # confusion matrix
        self.confusion_matrix = None


    def model(self):
        return self._model

    def set_params(self, **params):
        return self.model().set_params(**params)

    def get_params(self, deep=False):
        return self.model().get_params(deep=deep)

    def get_model_name(self):
        return self._model_name

    def get_data_name(self):
        return self._data_name

    def copy_model(self):
        model_new = sk_clone(self._model)
        return model_new

    def fit(self, x_train, y_train):
        if self._model is None:
            return None
        else:
            return self.model().fit(x_train, y_train)

    def predict(self, x_test):
        if self._model is None:
            return None
        else:
            return self.model().predict(x_test)

    @ignore_warnings(category=ConvergenceWarning)
    def search_hyper_params(self, x_train, y_train, **param_grid):
        print("========== Grid Search Started ==========")
        grid_search = GridSearchCV(estimator=self._model,
                                   param_grid=param_grid,
                                   scoring=self._score,
                                   verbose=self._verbose,
                                   cv=5)
        grid_search.fit(x_train, y_train)
        print("========== Grid Search Done ==========")
        # self._grid_search = grid_search
        self._model = grid_search.best_estimator_

    @ignore_warnings(category=ConvergenceWarning)
    def cal_model_complexity(self, x_train, y_train, x_test, y_test,
                             param_complexity, is_print_params=False):
        print("========== Model Complexity Started ==========")
        cols = ['param', 'f1_train', 'f1_test', 'accuracy_train', 'accuracy_test']
        param_list = list(param_complexity.values())[0]
        param_name = list(param_complexity.keys())[0]
        df = pd.DataFrame(data=np.full([len(param_list), len(cols)], np.nan), columns=cols)
        df['param'] = param_list
        df['param_name'] = param_name

        clf = self.copy_model()
        if is_print_params:
            print("--- model parameters: {} -----".format(clf.get_params()))
        for ii in range(len(param_list)):
            print("--- run={}, {}={} -----".format(ii, param_name, param_list[ii]))
            param = {param_name: param_list[ii]}
            clf.set_params(**param)
            clf.fit(x_train, y_train)
            y_pred_train = clf.predict(x_train)
            y_pred_test = clf.predict(x_test)
            df.loc[ii, 'f1_train'] = f1_score(y_train, y_pred_train)
            df.loc[ii, 'f1_test'] = f1_score(y_test, y_pred_test)
            df.loc[ii, 'accuracy_train'] = accuracy_score(y_train, y_pred_train)
            df.loc[ii, 'accuracy_test'] = accuracy_score(y_test, y_pred_test)
        self.model_complexity = df
        print("========== Model Complexity Done ==========")

    @ignore_warnings(category=ConvergenceWarning)
    def cal_learning_curve(self, x, y, is_print_params=False):
        print("========== Learning Curve Started ==========")
        num_samples = x.shape[0]
        cols = ['train_size', 'f1_train', 'f1_test', 'accuracy_train',
                'accuracy_test', 'time_train', 'time_test']
        train_sizes = (num_samples * np.linspace(0.05, 1, 20)).astype('int')
        df = pd.DataFrame(data=np.full([len(train_sizes), len(cols)], np.nan), columns=cols)
        df['train_size'] = train_sizes

        clf = self.copy_model()
        if is_print_params:
            print("--- model parameters: {} -----".format(clf.get_params()))
        for ii in range(len(train_sizes)):
            print("--- run={}, train_size={} -----".format(ii, train_sizes[ii]))
            ind = np.random.randint(num_samples, size=train_sizes[ii])
            x_sample, y_sample = x[ind, :], y[ind,]
            scores = cross_validate(clf, x_sample, y_sample, cv=10,
                                    scoring=['f1', 'accuracy'],
                                    return_train_score=True)
            df.loc[ii, 'f1_train'] = np.mean(scores['train_f1'])
            df.loc[ii, 'f1_test'] = np.mean(scores['test_f1'])
            df.loc[ii, 'accuracy_train'] = np.mean(scores['train_accuracy'])
            df.loc[ii, 'accuracy_test'] = np.mean(scores['test_accuracy'])
            df.loc[ii, 'time_train'] = np.mean(scores['fit_time'])
            df.loc[ii, 'time_test'] = np.mean(scores['score_time'])
        self.learning_curve = df
        print("========== Learning Curve Done ==========")

    def cal_model_evaluation(self, x_train, y_train, x_test, y_test,
                             is_print_params=False):
        print("========== Model Evaluation Started ==========")
        cols = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        clf = self.copy_model()
        if is_print_params:
            print("--- model parameters: {} -----".format(clf.get_param()))
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        scores = [accuracy_score(y_test, y_pred),
                  roc_auc_score(y_test, y_pred),
                  precision_score(y_test, y_pred),
                  recall_score(y_test, y_pred),
                  f1_score(y_test, y_pred)]
        scores = np.array(scores).reshape((1, len(cols)))
        df = pd.DataFrame(data=scores, columns=cols)
        self.model_evaluation = df
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        print("========== Model Evaluation Done ==========")






