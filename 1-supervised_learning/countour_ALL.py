
from model_class import Classifier
from utils_plot import *
from utils_load import *
from sklearn.model_selection import train_test_split


data_name = "Planar"
x, y = load_planar_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.60)
model_set = ['KNN', 'DTS', 'BST', 'SVM', 'ANN']


fig1 = plt.figure(figsize=(8, 16))
for ii in range(len(model_set)):
    model_name = model_set[ii]
    clf_load, = pkl.load(file=open('models/{}_{}.pkl'.format(data_name, model_name), 'rb'))
    model_fit = clf_load['model_fit']
    plt.subplot(3, 2, ii + 1)
    plt.title('Model: {} (subplot {})'.format(model_name, ii+1))
    clf = Classifier(model=model_fit, model_name=model_name, data_name=data_name)
    # param = {'hidden_layer_sizes': hidden_layer_sizes[ii]}
    # clf.set_params(**param)
    clf.fit(x_train, y_train)
    plot_decision_boundary(clf.predict, x_test, y_test)
# fig1.tight_layout()
plt.show()

a=1