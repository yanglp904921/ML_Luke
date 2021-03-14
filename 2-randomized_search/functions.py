
import numpy as np
import pandas as pd
import pickle as pkl
import mlrose_hiive as mlh
import matplotlib.pyplot as plt


def run_optimizers(problem, para, name, seed=1, is_save=True):
    np.random.seed(seed)
    results = dict()
    results['problem'] = name
    results['stats_RHC'], results['curve_RHC'] = run_random_hill_climbing(problem, para['RHC'])
    results['stats_SA'],  results['curve_SA'] = run_simulated_annealing(problem, para['SA'])
    results['stats_GA'],  results['curve_GA'] = run_genetic_algo(problem, para['GA'])
    results['stats_MIMIC'],  results['curve_MIMIC'] = run_mimic(problem, para['MIMIC'])
    if is_save:
        pkl.dump([results], file=open('results/Hyper_Paras_{}.pkl'.format(name.replace(' ', '_')), 'wb'))
    return results


def run_random_hill_climbing(problem, para, seed=1):
    print('running RHC')
    optimizer = mlh.RHCRunner(problem=problem,
                              seed=seed,
                              experiment_name=para['experiment_name'],
                              iteration_list=para['iteration_list'],
                              restart_list=para['restart_list'],
                              max_attempts=para['max_attempts']
                              )
    run_stats, run_curve = optimizer.run()
    return run_stats, run_curve


def run_simulated_annealing(problem, para, seed=1):
    print('running SA')
    optimizer = mlh.SARunner(problem=problem,
                             seed=seed,
                             experiment_name=para['experiment_name'],
                             iteration_list=para['iteration_list'],
                             max_attempts=para['max_attempts'],
                             temperature_list=para['temperature_list'],
                             decay_list=para['decay_list']
                             )
    run_stats, run_curve = optimizer.run()
    return run_stats, run_curve


def run_genetic_algo(problem, para, seed=1):
    print('running GA')
    optimizer = mlh.GARunner(problem=problem,
                             seed=seed,
                             experiment_name=para['experiment_name'],
                             iteration_list=para['iteration_list'],
                             population_sizes=para['population_sizes'],
                             mutation_rates=para['mutation_rates'],
                             max_attempts=para['max_attempts']
                             )
    run_stats, run_curve = optimizer.run()
    return run_stats, run_curve


def run_mimic(problem, para, seed=1):
    print('running MIMIC')
    optimizer = mlh.MIMICRunner(problem=problem,
                                seed=seed,
                                experiment_name=para['experiment_name'],
                                iteration_list=para['iteration_list'],
                                population_sizes=para['population_sizes'],
                                keep_percent_list=para['keep_percent_list'],
                                max_attempts=para['max_attempts'],
                                use_fast_mimic=True)
    run_stats, run_curve = optimizer.run()
    return run_stats, run_curve


# code from: https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
# define alternative N-Queens fitness function for maximization prob.
def queens_max(state):
    fitness_cnt = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            # check for attacking pairs
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                # if noa attack
                fitness_cnt += 1
    return fitness_cnt


def find_best_curve(curves, stats):
    cols_common = ['Iteration', 'Time', 'Fitness', 'FEvals', 'max_iters']
    cols_all = list(curves.columns)
    cols_ids = list(set(cols_all).difference(cols_common))

    ind_best_state = stats['Fitness'].idxmax()
    ind_best_curve = np.full(shape=(curves.shape[0], len(cols_ids)), fill_value=True)
    for ii in range(0, len(cols_ids)):
        ind_best_curve[:, ii] = curves[cols_ids[ii]] == stats.loc[ind_best_state, cols_ids[ii]]
    ind_best_curve2 = ind_best_curve.all(axis=1, keepdims=False)
    curve_best = curves[ind_best_curve2].copy()
    return curve_best


def plot_hyper_params_SA(stats, algorithm, ii, problem):
    if algorithm != 'SA':
        return None
    stats_expo = stats[stats['schedule_type']=='exponential']
    stats_geom = stats[stats['schedule_type']=='geometric']
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    stats_expo.plot.line(x='Temperature', y='Fitness', ax=ax, marker='o', label='Exponential')
    stats_geom.plot.line(x='Temperature', y='Fitness', ax=ax, marker='o', label='Geometric')
    ax.set_title('{} Hyper-Parameter ({})'.format(algorithm, 'Temperature'))
    ax.set_xlabel('Initial Temperature')
    ax.set_ylabel('Fitness Score')
    # ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig('figures/{}_{}_1_{}.png.png'.format(ii, problem, algorithm))


def plot_hyper_params_GA(stats, algorithm, ii, problem):
    if algorithm != 'GA':
        return None
    stats_best = stats.loc[stats['Fitness'].idxmax(), :].copy()
    stats_population = stats[stats['Mutation Rate'] == stats_best['Mutation Rate']]
    stats_mutation = stats[stats['Population Size'] == stats_best['Population Size']]

    fig = plt.figure(figsize=(4, 6))
    ax1, ax2 = fig.subplots(2, 1, sharex=False)

    stats_population.plot.line(x='Population Size', y='Fitness', ax=ax1, marker='o', grid=True)
    ax1.set_title('{} Hyper-Parameter ({})'.format(algorithm, 'Population Size'))
    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Fitness Score')

    stats_mutation.plot.line(x='Mutation Rate', y='Fitness', ax=ax2, marker='o', grid=True)
    ax2.set_title('{} Hyper-Parameter ({})'.format(algorithm, 'Mutation Rate'))
    ax2.set_xlabel('Mutation Rate')
    ax2.set_ylabel('Fitness Score')

    fig.tight_layout()
    fig.savefig('figures/{}_{}_2_{}.png.png'.format(ii, problem, algorithm))


def plot_hyper_params_MIMIC(stats, algorithm, ii, problem):
    if algorithm != 'MIMIC':
        return None
    stats_best = stats.loc[stats['Fitness'].idxmax(), :].copy()
    stats_population = stats[stats['Keep Percent'] == stats_best['Keep Percent']]
    stats_keep_rate = stats[stats['Population Size'] == stats_best['Population Size']]

    fig = plt.figure(figsize=(4, 6))
    ax1, ax2 = fig.subplots(2, 1, sharex=False)

    stats_population.plot.line(x='Population Size', y='Fitness', ax=ax1, marker='o', grid=True)
    ax1.set_title('{} Hyper-Parameter ({})'.format(algorithm, 'Population Size'))
    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Fitness Score')

    stats_keep_rate.plot.line(x='Keep Percent', y='Fitness', ax=ax2, marker='o', grid=True)
    ax2.set_title('{} Hyper-Parameter ({})'.format(algorithm, 'Keep Percent'))
    ax2.set_xlabel('Keep Percent')
    ax2.set_ylabel('Fitness Score')

    fig.tight_layout()
    fig.savefig('figures/{}_{}_3_{}.png.png'.format(ii, problem, algorithm))


