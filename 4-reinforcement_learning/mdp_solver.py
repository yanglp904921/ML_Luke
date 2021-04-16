
import time
from itertools import product
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt


def solve_value_iteration(env, gamma=0.9, diff_threshold=1e-12, verbose=False):
    run_start = time.time()
    v_s,  pi_s = np.zeros((env.nS,)), np.zeros((env.nS,), dtype=int)
    iteration = 0
    diff = np.inf
    while diff > diff_threshold:
        iteration += 1
        v_s0 = v_s.copy()
        for s in range(env.nS):
            v_s[s], pi_s[s] = cal_value_best_action(env, s, v_s, gamma)
        diff = np.max(np.abs(v_s-v_s0))
        if verbose:
            print('value iteration: episode={}, v[14]={}'.format(iteration, v_s[14]))
    run_time = time.time() - run_start
    return v_s, pi_s, iteration, run_time


def policy_evaluate(env, values, policy, gamma, diff_threshold):
    v_s = values.copy()
    diff = np.inf
    while diff > diff_threshold:
        v_s0 = v_s.copy()
        for s0 in range(env.nS):
            v0, a0 = v_s[s0], policy[s0]
            v_s[s0] = cal_value_one_action(env, s0, a0, v_s, gamma)
        diff = np.max(np.abs(v_s - v_s0))
    return v_s


def policy_improve(env, values, policy, gamma):
    policy_new = policy.copy()
    for s in range(env.nS):
        v_max, a_max = cal_value_best_action(env, s, values, gamma)
        policy_new[s] = a_max
    return policy_new


def solve_policy_iteration(env, gamma=0.9, diff_threshold=1e-3, verbose=False):
    run_start = time.time()
    pi0 = np.random.randint(env.nA, size=env.nS)
    v0, v1 = np.zeros((env.nS,)), np.zeros((env.nS,))
    iteration = 0
    policy_converge = False
    while not policy_converge:
        iteration += 1
        if verbose:
            print('policy iteration: episode={}'.format(iteration))
        v1 = policy_evaluate(env, v0, pi0, gamma, diff_threshold)
        pi1 = policy_improve(env, v1, pi0, gamma)
        policy_converge = np.all(pi0 == pi1)
        v0, pi0 = v1, pi1
    run_time = time.time() - run_start
    return v0, pi0, iteration, run_time


def solve_sarsa(env, gamma=0.9, alpha=0.1, num_episodes=1e4,
                epsilon_init=1, epsilon_min=0.01, epsilon_decay=0.001,
                verbose=False, is_plot=False):
    df = pd.DataFrame(index=np.arange(0, num_episodes),
                      columns=['episode', 'epsilon', 'rewards'])

    epsilon = epsilon_init
    nS, nA = env.nS, env.nA
    q_s_a = np.zeros((nS, nA))
    rewards_best = -np.inf
    for ii in range(1, int(num_episodes)+1):
        s0 = env.reset()
        rewards_total = 0
        done = False
        step = 0
        while (not done) and step<=1000:
            step += 1
            a0 = epsilon_greedy_action(env, s0, q_s_a, epsilon)
            s1, r0, done, info = env.step(a0)
            a1 = epsilon_greedy_action(env, s1, q_s_a, epsilon)
            delta = r0 + gamma * q_s_a[s1, a1] - q_s_a[s0, a0]
            q_s_a[s0, a0] = q_s_a[s0, a0] + alpha * delta
            s0 = s1
            rewards_total += r0

        if verbose and (ii % 500 == 0):
            print((ii, epsilon, step, rewards_total))
        df.loc[ii-1, :] = [ii, epsilon, rewards_total]
        if rewards_total > rewards_best:
            q_best = q_s_a.copy()
        epsilon = max(epsilon_init - epsilon_decay * ii, epsilon_min)

    if is_plot:
        df2 = df.copy()
        df2['avg_rewards'] = df2['rewards'].rolling(window=100).mean()
        df2.plot(x='episode', y='avg_rewards')
        plt.show()
    v = np.max(q_best, axis=1, keepdims=False)
    pi = np.argmax(q_best, axis=1)
    return v, pi, df


def solve_q_learning(env, gamma=0.9, alpha=0.1, num_episodes=1e4,
                     epsilon_init=1, epsilon_min=0.01, epsilon_decay=0.001,
                     verbose=False, is_plot=False):
    df = pd.DataFrame(index=np.arange(0, num_episodes),
                      columns=['episode', 'epsilon', 'rewards'])

    epsilon = epsilon_init
    nS, nA = env.nS, env.nA
    q_s_a = np.zeros((nS, nA))
    rewards_best = -np.inf
    for ii in range(1, int(num_episodes)+1):
        s0 = env.reset()
        rewards_total = 0
        done = False
        while not done:
            a0 = epsilon_greedy_action(env, s0, q_s_a, epsilon)
            s1, r0, done, info = env.step(a0)
            delta = r0 + gamma * np.max(q_s_a[s1, :]) - q_s_a[s0, a0]
            q_s_a[s0, a0] = q_s_a[s0, a0] + alpha * delta
            s0 = s1
            rewards_total += r0

        if verbose and (ii % 500 == 0):
            print((ii, epsilon, np.max(q_s_a, axis=1, keepdims=False)[14], rewards_total))
        if rewards_total > rewards_best:
            q_best = q_s_a.copy()
        df.loc[ii-1, :] = [ii, epsilon, rewards_total]
        epsilon = max(epsilon_init - epsilon_decay * ii, epsilon_min)

    if is_plot:
        df2 = df.copy()
        df2['avg_rewards'] = df2['rewards'].rolling(window=100).mean()
        df2.plot(x='episode', y='avg_rewards')
        plt.show()
    v = np.max(q_best, axis=1, keepdims=False)
    pi = np.argmax(q_best, axis=1)
    return v, pi, df


def cal_value_one_action(env, s0, a0, values, gamma):
    info = np.array(env.P[s0][a0])
    prob, s1, r0, done = info[:, 0], info[:, 1], info[:, 2], info[:, 3]
    s1 = s1.astype(int)
    done = done.astype(bool)
    v1 = values[s1]
    v1[done] = 0 # ensure the value of terminate state is 0
    v1_a0 = np.sum(prob * (r0 + gamma * v1))
    return v1_a0


def cal_value_best_action(env, s0, values, gamma):
    dict_actions = env.P[s0]
    actions = list(dict_actions.keys())
    nA = len(actions)
    v1_actions = np.zeros((nA,))
    for ii, a0 in zip(range(nA), actions):
        v1_actions[ii] = cal_value_one_action(env, s0, a0, values, gamma)
    index_max = np.argmax(v1_actions)
    v1_max, a0_max = v1_actions[index_max], actions[index_max]
    return v1_max, a0_max


def epsilon_greedy_action(env, s0, q_s_a, epsilon):
    if np.random.uniform() <= epsilon:
        # select from action set uniformly
        a0 = env.action_space.sample()
    else:
        # select the optimal action based on Q_table
        a0 = np.argmax(q_s_a[s0, :])
    return a0


def play_mdp(env, policy, num_episodes=1000):
    rewards = np.zeros((num_episodes,))
    steps = np.zeros((num_episodes,))
    for ii in range(num_episodes):
        s0 = env.reset()
        step, reward, done = 0, 0, False
        while not done:
            step += 1
            a0 = policy[s0]
            s1, r0, done, _ = env.step(a0)
            reward += r0
            s0 = s1
        rewards[ii], steps[ii] = reward, step
    return rewards.mean(), steps.mean()


def run_vi_pi(env_dict, gammas):
    cols = ['env', 'gamma', 'iterations', 'time', 'values',
            'policy', 'rewards', 'steps']
    ii = 0
    df_vi = pd.DataFrame(columns=cols)
    for name, env in env_dict.items():
        for gamma in gammas:
            print('value iteration: name={}, gamma={}'.format(name, gamma))
            v, pi, n, t = solve_value_iteration(env=env, gamma=gamma, verbose=False)
            mean_rewards, mean_steps = play_mdp(env, pi)
            df_vi.loc[ii, :] = [name, gamma, n, t, v, pi, mean_rewards, mean_steps]
            ii += 1

    ii = 0
    df_pi = pd.DataFrame(columns=cols)
    for name, env in env_dict.items():
        for gamma in gammas:
            print('policy iteration: name={}, gamma={}'.format(name, gamma))
            v, pi, n, t = solve_policy_iteration(env=env, gamma=gamma, verbose=False)
            mean_rewards, mean_steps = play_mdp(env, pi)
            df_pi.loc[ii, :] = [name, gamma, n, t, v, pi, mean_rewards, mean_steps]
            ii += 1

    file_dir = 'results/vi_pi_{}.pkl'.format(name.split(sep='_')[0])
    pkl.dump([df_vi, df_pi, env_dict], file=open(file_dir, 'wb'))
    return df_vi, df_pi


def run_sarsa(env, name_env, gammas, alphas, epsilon_inits, epsilon_mins,
              epsilon_decays, num_episodes):
    cols = ['gamma', 'alpha', 'epsilon_init', 'epsilon_min', 'epsilon_decay',
            'policy', 'info']
    df_sa = pd.DataFrame(columns=cols)

    ii = 0
    paras = product(gammas, alphas, epsilon_inits, epsilon_mins, epsilon_decays)
    for gamma, alpha, epsilon_init, epsilon_min, epsilon_decay in paras:
        print((gamma, alpha, epsilon_init, epsilon_min, epsilon_decay))
        _, pi, df = solve_sarsa(env=env, gamma=gamma, alpha=alpha,
                                epsilon_init=epsilon_init,
                                epsilon_min=epsilon_min,
                                epsilon_decay=epsilon_decay,
                                num_episodes=num_episodes,
                                verbose=True)
        df_sa.loc[ii, :] = (gamma, alpha, epsilon_init, epsilon_min, epsilon_decay, pi, df)
        ii += 1
    file_dir = 'results/sarsa_{}.pkl'.format(name_env)
    pkl.dump([df_sa], file=open(file_dir, 'wb'))
    return df_sa


def run_q_learning(env, name_env, gammas, alphas, epsilon_inits, epsilon_mins,
                   epsilon_decays, num_episodes):
    cols = ['gamma', 'alpha', 'epsilon_init', 'epsilon_min', 'epsilon_decay',
            'policy', 'info']
    df_ql = pd.DataFrame(columns=cols)

    ii = 0
    paras = product(gammas, alphas, epsilon_inits, epsilon_mins, epsilon_decays)
    for gamma, alpha, epsilon_init, epsilon_min, epsilon_decay in paras:
        print((gamma, alpha, epsilon_init, epsilon_min, epsilon_decay))
        _, pi, df = solve_q_learning(env=env, gamma=gamma, alpha=alpha,
                                     epsilon_init=epsilon_init,
                                     epsilon_min=epsilon_min,
                                     epsilon_decay=epsilon_decay,
                                     num_episodes=num_episodes,
                                     verbose=False)
        df_ql.loc[ii, :] = (gamma, alpha, epsilon_init, epsilon_min, epsilon_decay, pi, df)
        ii += 1
    file_dir = 'results/qlearn_{}.pkl'.format(name_env)
    pkl.dump([df_ql], file=open(file_dir, 'wb'))
    return df_ql



