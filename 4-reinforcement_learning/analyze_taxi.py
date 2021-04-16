
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from mdp_plot import *


# '''
name = 'Taxi-v3'
file = 'results/vi_pi_{}.pkl'.format(name)
df_vi, df_pi, env_dict = pkl.load(file=open(file, 'rb'))
num_row = df_vi.shape[0]

# --------------------------------------------------------------------------
key = 'Policy'
df = pd.DataFrame(index=np.arange(0, len(df_vi['policy'][0])))
df['Value Iteration'] = df_vi[key.lower()][num_row-1]
df['Policy Iteration'] = df_pi[key.lower()][num_row-1]
fig = plt.figure(figsize=(5, 4))
axs = fig.subplots(1, 1, sharex=False)
df.plot.line(y='Value Iteration', ax=axs, style='o', grid=False)
df.plot.line(y='Policy Iteration', ax=axs, style='x', grid=False)
axs.set_title('Comparison of Policy')
axs.set_xlabel('State ID')
axs.set_ylabel('Selected Action')
fig.savefig('figures/2_{}_{}_{}.png.png'.format(name, 0, key.lower()))


# --------------------------------------------------------------------------
key = 'Iterations'
df = pd.DataFrame(index=np.arange(0, df_vi.shape[0]))
df['Value Iteration'] = df_vi[key.lower()]
df['Policy Iteration'] = df_pi[key.lower()]
fig = plt.figure(figsize=(5, 4))
axs = fig.subplots(1, 1, sharex=False)
df.plot(ax=axs, style='-o', grid=True)
axs.set_title('Comparison of Total Iterations')
axs.set_xlabel('Gamma')
axs.set_ylabel('Number of Iterations')
axs.set_xticks(df.index)
axs.set_xticklabels(df_vi['gamma'].astype(str))
axs.set_ylim([0, 18])
fig.savefig('figures/2_{}_{}_{}.png.png'.format(name, 1, key.lower()))

# --------------------------------------------------------------------------
key = 'Time'
df = pd.DataFrame(index=np.arange(0, df_vi.shape[0]))
df['Value Iteration'] = df_vi[key.lower()]
df['Policy Iteration'] = df_pi[key.lower()]
fig = plt.figure(figsize=(5, 4))
axs = fig.subplots(1, 1, sharex=False)
df.plot(ax=axs, style='-o', grid=True)
axs.set_title('Comparison of Time')
axs.set_xlabel('Gamma')
axs.set_ylabel('Time (s)')
axs.set_xticks(df.index)
axs.set_xticklabels(df_vi['gamma'].astype(str))
axs.set_yscale('log')
fig.savefig('figures/2_{}_{}_{}.png.png'.format(name, 2, key.lower()))

# --------------------------------------------------------------------------
key = 'Rewards'
df = pd.DataFrame(index=np.arange(0, df_vi.shape[0]))
df['Value Iteration'] = df_vi[key.lower()]
df['Policy Iteration'] = df_pi[key.lower()]
fig = plt.figure(figsize=(5, 4))
axs = fig.subplots(1, 1, sharex=False)
df.plot(ax=axs, style='-o', grid=True)
axs.set_title('Comparison of Rewards')
axs.set_xlabel('Gamma')
axs.set_ylabel('Average Rewards')
axs.set_xticks(df.index)
axs.set_xticklabels(df_vi['gamma'].astype(str))
axs.set_ylim([0, 9])
fig.savefig('figures/2_{}_{}_{}.png.png'.format(name, 3, key.lower()))
# '''


name = 'Taxi-v3'
df_ql, = pkl.load(file=open('results/qlearn_{}.pkl'.format(name), 'rb'))
df_sa, = pkl.load(file=open('results/sarsa_{}.pkl'.format(name), 'rb'))


# gamma=0.99, alpha=0.3, epsilon_init=1, epsilon_min=0, epsilon_decay=0.001
window = 100
index = 805
df_sel_ql = df_ql.loc[index, :]
df_sel_sa = df_sa.loc[index, :]
# index_best = get_best_parameters(df_ql)

para_list = ['gamma', 'alpha']
for ii, para in enumerate(para_list):
    df_rewards_ql, df_epsilon_ql = get_df_one_parameter(df_ql, df_sel_ql, para)
    df_rewards_sa, df_epsilon_sa = get_df_one_parameter(df_sa, df_sel_sa, para)
    df_avg_ql = df_rewards_ql.rolling(window=window).mean()
    df_avg_sa = df_rewards_sa.rolling(window=window).mean()
    # ---------------------------------------------------------------------------------
    fig = plt.figure(figsize=(5, 8))
    axs = fig.subplots(2, 1, sharex=False)
    df_avg_ql.plot.line(ax=axs[0], grid=True)
    axs[0].set_title('Impact of {} (Q-Learning)'.format(para.title()))
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Average Rewards')
    # ---------------------------------------------------------------------------------
    df_avg_sa.plot.line(ax=axs[1], grid=True)
    axs[1].set_title('Impact of {} (SARSA)'.format(para.title()))
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Average Rewards')
    # ---------------------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/2_{}_{}_{}.png.png'.format(name, 4+ii, para))


para_list = ['epsilon_init', 'epsilon_min', 'epsilon_decay']
for ii, para in enumerate(para_list):
    df_rewards_ql, df_epsilon_ql = get_df_one_parameter(df_ql, df_sel_ql, para)
    df_rewards_sa, df_epsilon_sa = get_df_one_parameter(df_sa, df_sel_sa, para)
    df_avg_ql = df_rewards_ql.rolling(window=window).mean()
    df_avg_sa = df_rewards_sa.rolling(window=window).mean()
    # ---------------------------------------------------------------------------------
    fig = plt.figure(figsize=(5, 9))
    axs = fig.subplots(3, 1, sharex=False)
    df_avg_ql.plot.line(ax=axs[0], grid=True)
    axs[0].set_title('Impact of {} (Q-Learning)'.format(para.title()))
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Average Rewards')
    # ---------------------------------------------------------------------------------
    df_avg_sa.plot.line(ax=axs[1], grid=True)
    axs[1].set_title('Impact of {} (SARSA)'.format(para.title()))
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Average Rewards')
    # ---------------------------------------------------------------------------------
    df_epsilon_ql.plot.line(ax=axs[2], grid=True)
    axs[2].set_title('Impact of {} (SARSA)'.format(para.title()))
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Epsilon')
    # ---------------------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/2_{}_{}_{}.png.png'.format(name, 6+ii, para))


