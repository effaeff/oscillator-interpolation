import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from oscillator import calc_frf
from plot_utils import InteractivePlotter
from properties import (
    dark2,
    figsize,
    fontsize,
    x_range,
    HELLER
)
from sklearn import preprocessing

GAMMA_MIN, GAMMA_MAX = 0.1, 5000.0
MASS_MIN, MASS_MAX = 0.000001, 100.0
MUTATE_OSCIS = False
BEST_CHOICE = 0.3
MUTATION_RATE = 0.7
CROSSOVER = 0.75
FIX_XX_OSCI = [3807.51, 2236.61, 0.04] if HELLER else [3807.51, 2236.61, 0.04]#ToDo: check osci for Heller
FIX_YY_OSCI = [3582.16, 1527.42, 0.09] if HELLER else [3582.16, 1527.42, 0.09]#ToDo: check osci for Heller

class Individual:

    def __init__(self, params, mutate=True):
        """Initialize individual"""
        self.fitness = None
        self.params = []
        self.num_oscis = int(len(params) / 3) 
        if mutate:
            for i in range(len(params)):
                self.params.append(params[i] * (1 + random.choice([-0.01, 0.01])))
        else:
            self.params = params
  

    def evaluate(self, x_data, target_data, fit_reference):
        """Calculates fitness value for individual"""
        self.fitness = 0
        if self.num_oscis == 4:
            fix_osci = np.array(FIX_XX_OSCI)
        else:
            fix_osci = np.array(FIX_YY_OSCI)
        tmp_params = np.concatenate([self.params, fix_osci])
        amp, phase = calc_frf(x_data, tmp_params)
        amp_fit = np.sum((target_data[0] - amp)**2)
        ph_fit = np.sum((target_data[1] - phase)**2) / 100000
        self.fitness = amp_fit  + ph_fit


    def mutate(self, rate):
        """Mutate individual"""
        if MUTATE_OSCIS:
            #mutate per osci
            for i in range(self.num_oscis):
                if (random.choice([0.0, 1.0]) < rate):
                    self.params[i*3] = max(x_range[0] ,min(x_range[1], random.uniform(-2.0, 2.0)  + self.params[i*3]))
                    self.params[(i*3)+1] = max(GAMMA_MIN, min(GAMMA_MAX, random.uniform(0.98, 1.02)  * self.params[(i*3)+1]))
                    self.params[(i*3)+2] = max(MASS_MIN, min(MASS_MAX, random.uniform(0.98, 1.02) * self.params[(i*3)+2]))
        else:
            #mutate each param seperate
            for i in range(len(self.params)):
                if (random.choice([0.0, 1.0]) < rate):
                    if i % 3 == 0:#omega +/- max 0.5 Hz
                        self.params[i] = max(x_range[0], min(x_range[1], random.uniform(-0.5, 0.5)  + self.params[i])) 
                    if i % 3 == 1:#gamma +/- 2%
                        self.params[i] = max(GAMMA_MIN, min(GAMMA_MAX, random.choice([1.02, 0.98])  * self.params[i]))
                    if i % 3 == 2:#mass +/- 2%
                        self.params[i] = max(MASS_MIN, min(MASS_MAX, random.choice([1.02, 0.98]) * self.params[i]))


    def crossover(self, ind2):
        """Cross two individuals"""
        new_params = []
        for i in range(self.num_oscis):
            if (random.choice([0.0, 1.0]) < CROSSOVER):
                new_params.append(self.params[i*3])
                new_params.append(self.params[(i*3)+1])
                new_params.append(self.params[(i*3)+2])
            else:
                new_params.append(ind2.params[(i*3)])
                new_params.append(ind2.params[(i*3)+1])
                new_params.append(ind2.params[(i*3)+2]) 
        return Individual(new_params, mutate=False)


    def write_params(self, filename):
        """Write oscillator parameter in file"""
        with open (filename, 'w') as out:
            out.write('XX:\n')
            for i in range(self.num_oscis):
                out.write('  - {{ freq: {:4.2f}, gamma: {:4.3f}, mass: {:4.6f} }}\n'.format(self.params[i*3], self.params[(i*3)+1], self.params[(i*3)+2]))
            out.write('  - {{ freq: {:4.2f}, gamma: {:4.3f}, mass: {:4.6f} }}\n'.format(FIX_XX_OSCI[0], FIX_XX_OSCI[1], FIX_XX_OSCI[2]))
            out.write('YY:\n')
            for i in range(self.num_oscis):
                out.write('  - {{ freq: {:4.2f}, gamma: {:4.3f}, mass: {:4.6f} }}\n'.format(self.params[i*3], self.params[(i*3)+1], self.params[(i*3)+2]))
            out.write('  - {{ freq: {:4.2f}, gamma: {:4.3f}, mass: {:4.6f} }}\n'.format(FIX_YY_OSCI[0], FIX_YY_OSCI[1], FIX_YY_OSCI[2]))


    def print_params(self):
        """Print individual parameters"""
        print('Individual: ')
        for i in range(self.num_oscis):
            print ('omega: {:4.4f}, gamma: {:4.4f}, mass: {:3.6f}'.format(self.params[(i*3)], self.params[(i*3)+1], self.params[(i*3)+2]))   


class Population:

    def __init__(self, size, x_data, target_data, initial_osci):
        """initialize population"""
        self.num_oscis = int(len(initial_osci)/3)
        self.fit_reference = initial_osci
        self.size = size
        self.individuals = [Individual(initial_osci) for _ in range(size-1)]
        initial = Individual(initial_osci, mutate=False)
        initial.evaluate(x_data, target_data, self.fit_reference)
        self.individuals.append(initial)
        self.best = []
        self.best.append(initial)
        self.mutation_rate = MUTATION_RATE
        self.crossover_value = CROSSOVER
        self.x_data = x_data
        self.target_data = target_data
        self.plotter = InteractivePlotter(len(self.target_data), figsize, fontsize, dark2)
        self.plotter.init_plot(
            np.array([*calc_frf(self.x_data, initial_osci)]), self.target_data
        )

    
    def sort(self):
        """sorte individuals"""
        self.individuals = sorted(self.individuals, key=lambda ind: ind.fitness)


    def evaluate(self):
        """evaluate fitness values"""
        for ind in self.individuals:
            ind.evaluate(self.x_data, self.target_data, self.fit_reference)


    def choice_best(self):
        """choose individuals by best fitness"""
        return deepcopy(self.individuals[random.randint(0, int(len(self.individuals) * BEST_CHOICE))]) 


    def enhance(self):
        """get new generation"""
        next_generation = []
        #for i in range(int(self.size*0.5) - 4):
        for i in range(int(self.size) - 8):
            choice = self.choice_best()
            new_individual = choice
            new_individual.mutate(self.mutation_rate)
            next_generation.append(new_individual)
        # for i in range(int(self.size*0.5) - 4):
        #     first_choice = self.choice_best()
        #     second_choice = self.choice_best()
        #     new_individual = first_choice.crossover(second_choice)
        #     next_generation.append(new_individual)
        for i in range(3):
            next_generation.append(deepcopy(self.individuals[i]))
        for i in range(5):
            indi = deepcopy(self.individuals[i])
            indi.mutate(self.mutation_rate)
            next_generation.append(indi)
        self.individuals = next_generation
        self.evaluate()
        self.sort()
        self.best.append(self.individuals[0])
        if self.best[-1].fitness == self.best[-2].fitness:
            self.mutation_rate += 0.05
        else:
            self.mutation_rate = MUTATION_RATE
        #early out when best_fitness doesnt changed in last 20 generations
        if len(self.best) > 21 and self.best[-20].fitness - self.best[-1].fitness < 0.00001:
            return False
        return True

    
    def updatePlot(self, generation):
        """update InteractivePlotter"""
        if self.num_oscis == 4:
            fix_osci = np.array(FIX_XX_OSCI)
        else:
            fix_osci = np.array(FIX_YY_OSCI)
        tmp_params = np.concatenate([self.best[-1].params, fix_osci])
        amp, phase = calc_frf(self.x_data, tmp_params)
        self.plotter.update_plot(np.array([amp, phase]), self.target_data)


    def savePlot(self, filename):
        """save plot"""
        self.plotter.save_plot(filename)
        
