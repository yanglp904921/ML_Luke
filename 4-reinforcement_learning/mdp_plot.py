
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pivot_dataframe(df_in):
    keys = ['iterations', 'time', 'rewards', 'steps', 'policy']
    dict_df = {}
    for key in keys:
        short_names = ['4x4', '8x8', '12x12', '16x16']
        cols = ['FrozenLake-v0_' + s for s in short_names]
        df = df_in.pivot(index='gamma', columns='env', values=key)
        df = df[cols]
        # df.columns = short_names
        dict_df[key] = df.transpose()
    return dict_df


def get_best_parameters(df_in):
    num_runs = df_in.shape[0]
    rewards = np.zeros((num_runs,))
    for row in range(num_runs):
        df = df_in['info'][row]
        r_temp = np.array(df['rewards'][-500:].values, dtype=float)
        rewards[row] = np.nanmean(r_temp)
    index = np.argmax(rewards)
    return index


def get_df_one_parameter(df_in, df_para, para_in):
    num_runs = df_in.shape[0]
    para_list = ['gamma', 'alpha', 'epsilon_init', 'epsilon_min', 'epsilon_decay']
    para_list.remove(para_in)
    tf = np.zeros((num_runs, len(para_list)), dtype=bool)
    for ii, para in enumerate(para_list):
        tf[:, ii] = df_in[para] == df_para[para]
    tf_sel = np.all(tf, axis=1)
    df_sel = df_in[tf_sel]
    df_sel.index = np.arange(0, df_sel.shape[0])

    num_rows = df_para['info'].shape[0]
    # col_name = list(df_sel[para_in])
    col_name = df_sel[para_in]
    df_rewards = pd.DataFrame(index=np.arange(num_rows), columns=col_name)
    df_epsilon = pd.DataFrame(index=np.arange(num_rows), columns=col_name)
    for ii, col in enumerate(df_sel[para_in]):
        df = df_sel['info'][ii]
        df_rewards[col] = df['rewards']
        df_epsilon[col] = df['epsilon']
    return df_rewards, df_epsilon


def map_to_number(map_str):
    map_num = np.zeros(map_str.shape, dtype=int)
    for i, row in enumerate(map_str):
        for j, loc in enumerate(row):
            if loc == b'S':
                map_num[i, j] = 1
            elif loc == b'F':
                map_num[i, j] = 0
            elif loc == b'H':
                map_num[i, j] = -1
            elif loc == b'G':
                map_num[i, j] = 2
    return map_num


def display_policy(env, policy, axs):
    map_str = env.desc
    nrow, ncol = map_str.shape
    map_num = map_to_number(map_str)
    axs.imshow(map_num, interpolation="nearest")
    pi_mat = np.reshape(policy, map_str.shape)
    for i in range(pi_mat[0].size):
        for j in range(pi_mat[0].size):
            text = '\u2190'
            if pi_mat[i, j] == 1:
                text = '\u2193'
            elif pi_mat[i, j] == 2:
                text = '\u2192'
            elif pi_mat[i, j] == 3:
                text = '\u2191'
            if map_num[i, j] == 1:
                text = 'S'
            if map_num[i, j] == 2:
                text = 'G'
            if map_num[i, j] == -1:
                text = ''
            axs.text(j, i, text, ha="center", va="center", color="w")
    axs.set_xticks(np.arange(0, ncol))
    axs.set_xticklabels(np.arange(1, ncol + 1).astype(str))
    axs.set_yticks(np.arange(0, nrow))
    axs.set_yticklabels(np.arange(1, nrow + 1).astype(str))

