
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from mdp_plot import *


# '''
name = 'FrozenLake-v0'
file = 'results/vi_pi_{}.pkl'.format(name)
df_vi0, df_pi0, env_dict = pkl.load(file=open(file, 'rb'))
dict_vi = pivot_dataframe(df_vi0)
dict_pi = pivot_dataframe(df_pi0)

policy_vi, policy_pi = dict_vi['policy'][0.999999], dict_pi['policy'][0.999999]
fig = plt.figure(figsize=(6, 11))
axs = fig.subplots(4, 2)
for row in range(4):
    env_name = policy_vi.index[row]
    env = env_dict[env_name]
    display_policy(env, policy_vi[row], axs[row, 0])
    display_policy(env, policy_pi[row], axs[row, 1])
    print('Policy Same: {}'.format(np.all(policy_vi[row] == policy_pi[row])))
fig.tight_layout()
fig.savefig('figures/1_{}_{}_{}.png.png'.format(name, 0, 'policy'))


keys = ['Iterations', 'Time', 'Rewards']
ylabels = ['Number of Iterations', 'Run Time (s)',
           'Average Total Rewards', 'Average Steps']
short_names = ['4x4', '8x8', '12x12', '16x16']
for ii in range(len(keys)):
    key, ylabel = keys[ii], ylabels[ii]
    df_vi, df_pi = dict_vi[key.lower()], dict_pi[key.lower()]
    df_vi.index, df_pi.index = short_names, short_names
    fig = plt.figure(figsize=(4, 7))
    ax1, ax2 = fig.subplots(2, 1, sharex=False)
    # --------------------------------------------------------------------------
    df_vi.plot.bar(ax=ax1)
    if key=='Iterations':
        ax1.set_yscale('log')
    ax1.tick_params(axis='x', labelrotation=0)
    ax1.get_legend().remove()
    ax1.set_title('{} ({})'.format(key, 'Value Iteration'))
    ax1.set_xlabel('Map Size')
    ax1.set_ylabel(ylabel)
    # --------------------------------------------------------------------------
    df_pi.plot.bar(ax=ax2)
    if key=='Iterations':
        ax2.set_yscale('log')
    ax2.tick_params(axis='x', labelrotation=0)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.4, -0.28), ncol=3)
    ax2.set_title('{} ({})'.format(key, 'Policy Iteration'))
    ax2.set_xlabel('Map Size')
    ax2.set_ylabel(ylabel)
    # --------------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig('figures/1_{}_{}_{}.png.png'.format(name, ii+1, key.lower()))
# '''


name = 'FrozenLake-v0'
df_ql, = pkl.load(file=open('results/qlearn_{}.pkl'.format(name), 'rb'))
df_sa, = pkl.load(file=open('results/sarsa_{}.pkl'.format(name), 'rb'))


# gamma=0.99, alpha=0.3, epsilon_init=1, epsilon_min=0, epsilon_decay=0.001
window = 100
index = 1141
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
    fig.savefig('figures/1_{}_{}_{}.png.png'.format(name, 4+ii, para))


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
    fig.savefig('figures/1_{}_{}_{}.png.png'.format(name, 6+ii, para))

