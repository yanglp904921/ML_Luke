
import numpy as np
import pandas as pd
import pickle as pkl
import mlrose_hiive as mlh
import functions as fn

prob_name = 'Flip Flop'
para = {'RHC': {'experiment_name': prob_name,
                'iteration_list': [10000],
                'restart_list': [100],
                'max_attempts': 100
                },
        'SA': {'experiment_name': prob_name,
               'iteration_list': [10000],
               'temperature_list': [1, 5, 10, 50, 100, 200],
               'decay_list': [mlh.ExpDecay, mlh.GeomDecay],
               'max_attempts': 100,
               },
        'GA': {'experiment_name': prob_name,
               'iteration_list': [10000],
               'population_sizes': [100, 200, 500, 1000, 1500, 2000],
               'mutation_rates': [0.1, 0.2, 0.4, 0.6, 0.8],
               'max_attempts': 100,
               },
        'MIMIC': {'experiment_name': prob_name,
                  'iteration_list': [10000],
                  'population_sizes': [100, 200, 500, 1000, 1500, 2000],
                  'keep_percent_list': [0.1, 0.2, 0.4, 0.6, 0.8],
                  'max_attempts': 100,
                  },
        }

problem = mlh.FlipFlopOpt(length=300)
fn.run_optimizers(problem, para, name=prob_name)

