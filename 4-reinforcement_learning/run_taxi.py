
import gym
from mdp_solver import *

np.random.seed(0)
name = 'Taxi-v3'
env_dict = {name+'_5x5': gym.make("Taxi-v3").env}
gammas = [0.8, 0.9, 0.99, 0.9999]
df_vi, df_pi = run_vi_pi(env_dict, gammas)


env = env_dict['Taxi-v3_5x5']
gammas = [0.7, 0.9, 0.99, 0.9999]
alphas = [0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95]
epsilon_inits = [0.4, 0.5, 0.8, 1]
epsilon_mins = [0, 0.001, 0.01, 0.1]
epsilon_decays = [0.0001, 0.001, 0.01]
num_episodes = 2000
df_ql = run_q_learning(env, name, gammas, alphas, epsilon_inits, epsilon_mins,
                       epsilon_decays, num_episodes)
df_sa = run_sarsa(env, name, gammas, alphas, epsilon_inits, epsilon_mins,
                  epsilon_decays, num_episodes)

