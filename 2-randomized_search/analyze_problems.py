
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import functions as fn


# algos = ['SA']
probs = ['Four Peaks', 'Flip Flop',  'Knapsack']
# probs = ['Four Peaks', 'N-Queens', 'Flip Flop', 'Knapsack']
algos = ['RHC', 'GA', 'SA', 'MIMIC']
cols_common = ['Iteration', 'Time', 'Fitness', 'FEvals', 'max_iters']
cols_iterations = ['Max Score', 'Mean', 'Min', 'Max']
cols_time = ['Max Score', 'Mean', 'Min', 'Max']
ii_probs = np.arange(1, len(probs)+1, 1)
jj_algos = np.arange(1, len(algos)+1, 1)

for ii, prob in zip(ii_probs, probs):
    file_name = 'results/Hyper_Paras_{}.pkl'.format(prob.replace(' ', '_'))
    results, = pkl.load(file=open(file_name, 'rb'))

    df_fitness = pd.DataFrame(index=algos, columns=['Max Score', 'Mean Score'])
    df_iteration = pd.DataFrame(index=algos, columns=cols_iterations)
    df_time = pd.DataFrame(index=algos, columns=cols_time)


    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)

    for jj, algo in zip(jj_algos, algos):
        cols_all = list(results['curve_{}'.format(algo)].columns)
        cols_ids = list(set(cols_all).difference(cols_common))

        stats = results['stats_{}'.format(algo)].copy()
        stats = stats[stats['Iteration'] != 0].reindex()
        stats_best = stats.loc[stats['Fitness'].idxmax(), :].copy()

        curves = results['curve_{}'.format(algo)].copy()
        curve_best = fn.find_best_curve(curves, stats)
        curve_best.plot.line(x='Iteration', y='Fitness', ax=ax, label=algo)
        if 'Temperature' in cols_ids:
            id_temperature = [x.__dict__['init_temp'] for x in curves['Temperature']]
            curve_iterations = curves.groupby(id_temperature)['Iteration'].max()
        else:
            curve_iterations = curves.groupby(cols_ids)['Iteration'].max()

        df_fitness.loc[algo, :] = [stats['Fitness'].max(), stats['Fitness'].mean()]
        df_iteration.loc[algo, :] = [curve_best['Iteration'].max(), curve_iterations.mean(),
                                     curve_iterations.min(), curve_iterations.max()]
        df_time.loc[algo, :] = [stats_best['Time'], stats['Time'].mean(),
                                stats['Time'].min(), stats['Time'].max()]

        fn.plot_hyper_params_SA(stats, algo, ii, prob)
        fn.plot_hyper_params_GA(stats, algo, ii, prob)
        fn.plot_hyper_params_MIMIC(stats, algo, ii, prob)


    ax.set_title('Fitting Curve ({})'.format(prob))
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Fitness Score')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig('figures/{}_{}_4_{}.png.png'.format(ii, prob, 'Curves'))

    fig1 = plt.figure(figsize=(4, 3))
    ax1 = fig1.add_subplot(111)
    df_fitness.transpose().plot.bar(ax=ax1)
    plt.xticks(rotation='horizontal')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(algos))
    ax1.set_title("Fitness Score ({})".format(prob))
    fig1.tight_layout()
    ax1.set_ylabel('Score')
    ll, bb, ww, hh = ax1.get_position().bounds
    ax1.set_position([1.2*ll, 0.85*bb, ww, 1.1*hh])
    fig1.savefig('figures/{}_{}_5_{}.png'.format(ii, prob, 'Fitness'))

    fig1 = plt.figure(figsize=(4, 3))
    ax1 = fig1.add_subplot(111)
    df_iteration.transpose().plot.bar(ax=ax1)
    plt.xticks(rotation=0)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(algos))
    ax1.set_title("Number of Iterations ({})".format(prob))
    ax1.set_yscale('log')
    fig1.tight_layout()
    ax1.set_ylabel('Iterations')
    ll, bb, ww, hh = ax1.get_position().bounds
    ax1.set_position([1.2*ll, 0.85*bb, ww, 1.1*hh])
    fig1.savefig('figures/{}_{}_6_{}.png'.format(ii, prob, 'Iteration'))

    fig1 = plt.figure(figsize=(4, 3))
    ax1 = fig1.add_subplot(111)
    df_time.transpose().plot.bar(ax=ax1)
    plt.xticks(rotation=0)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(algos))
    ax1.set_title("Computation Time ({})".format(prob))
    ax1.set_yscale('log')
    fig1.tight_layout()
    ax1.set_ylabel('Time (s)')
    ll, bb, ww, hh = ax1.get_position().bounds
    ax1.set_position([1.15*ll, 0.85*bb, ww, 1.1*hh])
    fig1.savefig('figures/{}_{}_7_{}.png'.format(ii, prob, 'Time'))

# plt.close('all')


