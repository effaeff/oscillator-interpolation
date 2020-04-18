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

GENERATIONS = 200
POPULATION_SIZE = 100

#TODO: add YY oscis
initial_osci = np.array(
        [
            1072.74, 409.854, 0.644504,
            1464.13, 520.525, 0.436132,
            1522.53, 198.228, 0.52103,
            2123.51, 533.912, 0.992652,
            2425.83, 629.189, 0.251646,
            4358.78, 3310.14, 0.166609,
            4484.15, 207.088, 2.80809,
            5062.56, 626.647, 0.35843,
            5966.04, 1027.73, 0.0954085,
            7427.76, 10.2595, 0.0183696
        ]
    )   

def gen_dirs():
    for directory in [data_dir, processed_dir, plot_dir, results_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print(f"Error: Creation of directory {directory} failed.")

def main():
    """Main method"""
    misc.to_local_dir(__file__)
    gen_dirs()
    data = processing(store=True, plot=True) 
    axis = ['XX', 'YY']
    for d in data:
        for idx in range(2):
            target_data = [d[:, 4+idx], d[:, 6+idx]]
            x_data = d[:, 3]
            population = Population(POPULATION_SIZE, x_data, target_data, initial_osci)
            population.evaluate()
            population.sort()
            g_idx = 0
            while g_idx < GENERATIONS and population.enhance():
                population.updatePlot(g_idx)
                print('Generation: {} Individuals: {} Best Fitness: {:4.6f} Max Fitness: {:4.6f}'.format(g_idx, len(population.individuals) ,population.best[-1].fitness, population.individuals[-1].fitness))
                g_idx += 1
            population.best[-1].print_params()
            population.best[-1].write_params(results_dir + '/oscis_{}_{:3.2f}_{:3.2f}_{:3.2f}.txt'.format(axis[idx], d[0, 0], d[0, 1], d[0, 2]))
            population.savePlot(plot_dir + '/oscis_{}_{:3.2f}_{:3.2f}_{:3.2f}.png'.format(axis[idx], d[0, 0], d[0, 1], d[0, 2]))
            

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()