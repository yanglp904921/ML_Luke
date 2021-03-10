
from utils_plot import *


data_name = "Planar"
# data_name = "Phish"


is_print_model = False
if is_print_model:
    model_set = ['KNN', 'DTS', 'BST', 'SVM', 'ANN']
    model_name = model_set[4]
    clf, = pkl.load(file=open('models/{}_{}.pkl'.format(data_name, model_name), 'rb'))
    model_fit = clf['model_fit']
    model_best = model_fit._model
    print(model_best)


df_list = []
tm_list = []
model_set = ['KNN', 'SVM', 'DTS', 'BST', 'ANN']
for ii in range(len(model_set)):
    model_name = model_set[ii]
    clf, = pkl.load(file=open('models/{}_{}.pkl'.format(data_name, model_name), 'rb'))
    model_fit = clf['model_fit']

    time = model_fit.learning_curve[['time_train']].copy()
    time.rename(columns={"time_train": model_name}, inplace=True)
    time.index = model_fit.learning_curve['train_size']
    tm_list.append(time)

    evaluatoin = model_fit.model_evaluation
    evaluatoin.index = [model_name]
    df_list.append(evaluatoin)
    # plot_model_performance(clf=clf['model_fit'])
# plt.close('all')


df_time = pd.concat(tm_list, axis=1)
df_models = pd.concat(df_list)
df_models = df_models.transpose()


fig1 = plt.figure(figsize=(6, 3))
ax1 = fig1.add_subplot(111)
df_models.plot.bar(ax=ax1)
plt.xticks(rotation='horizontal')
ax1.legend(loc='upper center', ncol=5)
ax1.set_ylim([0, 1.2])
ax1.set_title("Comparison of Model Performance ({})".format(data_name))
ax1.set_ylabel('Scores')
fig1.tight_layout()
fig1.savefig('figures/Comparison_{}.png'.format(data_name))


fig1 = plt.figure(figsize=(6, 3))
ax1 = fig1.add_subplot(111)
df_time.plot(ax=ax1, marker='o')
plt.grid()
ax1.set_title("Comparison of Training Time ({})".format(data_name))
ax1.set_ylabel('Train Time (s)')
ax1.set_ylabel('Sample Size')
fig1.tight_layout()
fig1.savefig('figures/Time_{}.png'.format(data_name))


