
from model_class import Classifier
from utils_plot import *
from utils_load import *
from sklearn.model_selection import train_test_split


data_name = "Planar"
x, y = load_planar_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.60)
model_set = ['KNN', 'DTS', 'BST', 'SVM', 'ANN']


fig1 = plt.figure(figsize=(8, 16))
hidden_layer_sizes = [1, 2, 4, 5, 20, 100]

clf_load, = pkl.load(file=open('models/{}_{}.pkl'.format(data_name, 'ANN'), 'rb'))
model_fit = clf_load['model_fit']

for ii, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(3, 2, ii + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    clf = Classifier(model=model_fit, model_name='ANN', data_name=data_name)
    param = {'hidden_layer_sizes': hidden_layer_sizes[ii]}
    clf.set_params(**param)
    clf.fit(x_train, y_train)
    plot_decision_boundary(clf.predict, x_test, y_test)
# fig1.tight_layout()
plt.show()

