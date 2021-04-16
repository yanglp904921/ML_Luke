
import gym
from mdp_solver import *
from gym.envs.toy_text.frozen_lake import generate_random_map


np.random.seed(0)
name = 'FrozenLake-v0'
size = [4, 8, 12, 16]
envs = [gym.make(name, desc=generate_random_map(s)).unwrapped for s in size]
env_dict = {name+'_{}x{}'.format(s, s): e for s, e in zip(size, envs)}
gammas = [0.8, 0.9, 0.99, 0.9999, 0.999999, 0.99999999]
df_vi, df_pi = run_vi_pi(env_dict, gammas)


env = env_dict['FrozenLake-v0_4x4']
gammas = [0.5, 0.7, 0.9, 0.99, 0.9999, 0.999999]
alphas = [0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95]
epsilon_inits = [0.4, 0.5, 0.8, 1]
epsilon_mins = [0, 0.001, 0.01, 0.1]
epsilon_decays = [0.0001, 0.001, 0.01]
num_episodes = 2000
df_ql = run_q_learning(env, name, gammas, alphas, epsilon_inits, epsilon_mins,
                       epsilon_decays, num_episodes)
df_sa = run_sarsa(env, name, gammas, alphas, epsilon_inits, epsilon_mins,
                  epsilon_decays, num_episodes)

