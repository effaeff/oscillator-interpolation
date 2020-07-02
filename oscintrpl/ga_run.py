import numpy as np
import random
import os
import misc
from oscillator import calc_frf
from plot_utils import InteractivePlotter
from processing import processing
from ga_oscifit import Individual, Population
from properties import (
    data_dir,
    processed_dir,
    plot_dir,
    results_dir
)

from joblib import Parallel, delayed

GENERATIONS = 1000
POPULATION_SIZE = 100
#HELLER
# initial_oscis_XX = np.array(
#     [
#         1057.17, 426.137, 0.520891,
#         1470.49, 596.826, 0.186868,
#         2053.32, 562.724, 0.766667,
#         2332.95, 769.625, 0.326219,
#         #2198.73, 351.788, 3.78851,
#         #2677.06, 27.3344, 5282.94,
#         4358.78, 3310.14, 0.166609
#     ]
# )

# initial_oscis_YY = np.array(
#     [
#         1028.42, 386.892, 0.536729,
#         1398.41, 698.039, 0.204091,
#         #1544.62, 90.4753, 6.70184,
#         2135.79, 515.363, 0.708946,
#         #2456.69, 308.889, 0.685429,
#         #2676.12, 33.9041, 5209.01,
#         4358.78, 3310.14, 0.166609
#     ]
# )


#HSC
initial_oscis_XX = np.array(
    [
        1476.97, 546.095, 0.163119,
        2159.05, 985.939, 0.515099,
        2402.26, 637.446, 0.335579,
        2786.64, 218.469, 12.2829
    ]
)

initial_oscis_YY = np.array(
    [
		1466.71, 1171.24, 0.120436,   
        1526.71, 1171.24, 0.120436,
		2375.03, 871.08, 0.203835
    ]
)


def gen_dirs():
    """Generate needed file structure"""
    for directory in [data_dir, processed_dir, plot_dir, results_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print(f"Error: Creation of directory {directory} failed.")


def fit(d):
    """Start fitting of one dataset"""
    axis = ['XX', 'YY']
    for idx in range(2):
        target_data = [d[:, 4+idx], d[:, 6+idx]]
        x_data = d[:, 3]
        if idx == 0:
            initial_osci = initial_oscis_XX
        else:
            initial_osci = initial_oscis_YY
        population = Population(POPULATION_SIZE, x_data, target_data, initial_osci)
        population.evaluate()
        population.sort()
        g_idx = 0
        while g_idx < GENERATIONS and population.enhance():
            population.updatePlot(g_idx)
            print('Generation: {} Individuals: {} Best Fitness: {:4.6f} Max Fitness: {:4.6f}'.format(g_idx, len(population.individuals) ,population.best[-1].fitness, population.individuals[-1].fitness))
            g_idx += 1
        population.best[-1].print_params()
        population.best[-1].write_params(results_dir + '/oscis_{}_{:3.2f}_{:3.2f}_{:3.2f}.osci'.format(axis[idx], d[0, 0], d[0, 1], d[0, 2]))
        population.savePlot(plot_dir + '/oscis_{}_{:3.2f}_{:3.2f}_{:3.2f}.png'.format(axis[idx], d[0, 0], d[0, 1], d[0, 2]))


def main():
    """Main method to start parallel fitting"""
    misc.to_local_dir(__file__)
    gen_dirs()
    data = processing(store=True, plot=True)   
    Parallel(n_jobs=12)(delayed(fit)(d) for d in data)


if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()