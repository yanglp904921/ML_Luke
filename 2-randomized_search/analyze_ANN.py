
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import functions as fn

df_ADAM, = pkl.load(file=open('results/Neural_Net_ADAM.pkl', 'rb'))
# df_GD,   = pkl.load(file=open('results/Neural_Net_GD.pkl', 'rb'))
df_RHC,  = pkl.load(file=open('results/Neural_Net_RHC.pkl', 'rb'))
df_SA,   = pkl.load(file=open('results/Neural_Net_SA.pkl', 'rb'))
df_GA,   = pkl.load(file=open('results/Neural_Net_GA.pkl', 'rb'))


# RHC
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
df_RHC.plot.line(x='restart', y=['accuracy_train'], ax=ax, marker='o')
ax.set_title('{} Hyper-Parameter ({})'.format('RHC', 'Restarts'))
ax.set_xlabel('Restart')
ax.set_ylabel('Accuracy')
fig.tight_layout()
fig.savefig('figures/{}_{}_1_{}.png.png'.format(5, 'ANN', 'RHC'))


# SA
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
df_SA.plot.line(x='init_temperature', y=['accuracy_train'], ax=ax, marker='o')
ax.set_title('{} Hyper-Parameter ({})'.format('SA', 'init_temperature'))
ax.set_xlabel('init_temperature')
ax.set_ylabel('Accuracy')
fig.tight_layout()
fig.savefig('figures/{}_{}_2_{}.png.png'.format(5, 'ANN', 'SA'))


# GA Population Size
df_GA_pop = df_GA[df_GA['mutation_probs'] == 0.1]
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
df_GA_pop.plot.line(x='pop_size', y=['accuracy_train'], ax=ax, marker='o')
ax.set_title('{} Hyper-Parameter ({})'.format('GA', 'Population Size'))
ax.set_xlabel('Population Size')
ax.set_ylabel('Accuracy')
fig.tight_layout()
fig.savefig('figures/{}_{}_3_{}.png.png'.format(5, 'ANN', 'GA'))


# GA Mutation Rate
df_GA_mut = df_GA[df_GA['pop_size'] == 100]
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
df_GA_mut.plot.line(x='mutation_probs', y=['accuracy_train'], ax=ax, marker='o')
ax.set_title('{} Hyper-Parameter ({})'.format('GA', 'Mutation Rate'))
ax.set_xlabel('Mutation Rate')
ax.set_ylabel('Accuracy')
fig.tight_layout()
fig.savefig('figures/{}_{}_4_{}.png.png'.format(5, 'ANN', 'GA'))


# Model Comparison
cols_sel = ['algo', 'time', 'iterations', 'accuracy_train', 'accuracy_test', 'model']
df_best = pd.concat([df_ADAM.loc[[0], cols_sel],
                     df_RHC.loc[[df_RHC['accuracy_train'].idxmax()], cols_sel],
                     df_SA.loc[[df_SA['accuracy_train'].idxmax()], cols_sel],
                     df_GA.loc[[df_GA['accuracy_train'].idxmax()], cols_sel]],
                    axis=0, ignore_index=True)
fig1 = plt.figure(figsize=(4, 3))
ax1 = fig1.add_subplot(111)
df_accuracy = df_best[['accuracy_train', 'accuracy_test']]
df_accuracy.index = list(df_best['algo'])
df_accuracy.transpose().plot.bar(ax=ax1)
plt.xticks(rotation='horizontal')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=df_accuracy.shape[0])
ax1.set_title("Accuracy of Optimization Algos")
fig1.tight_layout()
ax1.set_ylabel('Accuracy')
ll, bb, ww, hh = ax1.get_position().bounds
ax1.set_position([1.2 * ll, 0.85 * bb, ww, 1.1 * hh])
fig1.savefig('figures/{}_{}_5_{}.png'.format(5, 'ANN', 'Accuracy'))


fig1 = plt.figure(figsize=(4, 3))
ax1 = fig1.add_subplot(111)
df_time = df_best[['time']]
df_time.index = list(df_best['algo'])
df_time.transpose().plot.bar(ax=ax1)
plt.xticks(rotation='horizontal')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=df_time.shape[0])
ax1.set_title("Time of Optimization Algos")
ax1.set_yscale('log')
fig1.tight_layout()
ax1.set_ylabel('Time (s)')
ll, bb, ww, hh = ax1.get_position().bounds
ax1.set_position([1.2 * ll, 0.85 * bb, ww, 1.1 * hh])
fig1.savefig('figures/{}_{}_6_{}.png'.format(5, 'ANN', 'Time'))

