import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import pi, sqrt, atan
from scipy import optimize
from scipy.optimize import curve_fit
import misc
import os

#cost function
def cost(parameters):
    sum = 0
    for i in range(int(len(parameters)/3)):
        sum += amp(x, *parameters[i*3:(i+1)*3])
    return np.sum(np.power(sum - y, 2)) / len(x)    


random_seed = 1234
test_size = 0.1
num_oscis = 4
data_dir = './data/02_processed'
data = np.load('{}/processed_data.npy'.format(data_dir))
#osci_init_freq= np.asarray([1070, 1500, 2100, 2350])

train, test = train_test_split(data, test_size=test_size, random_state=random_seed)

#only x data used at the moment
for data in train:
    x = data[:, 3]
    y = data[:, 4]
    #data from evo osci // xx data from file 128_Heller_XX_A_B-120_0__0__Signal 2_Signal 1
    initial_guess = [1073.31, 604.333, 0.440025, 1519.34, 312.594, 0.262421, 2379.1, 431.59, 0.514127, 2089.42, 635.301, 0.772651, 2259.18, 482.119, 0.813576]
    #initial_guess = [1074.33,	643.203, 0.421481, 1517.06, 348.865, 0.24052, 2353.21, 651.648, 0.30505, 2111.69, 790.514, 0.625564]
    #initial_guess = [1070, 10, 0.5, 1500, 10, 0.5, 2100, 10, 0.5, 2350, 10, 0.5]
    result = optimize.minimize(cost, initial_guess, method='L-BFGS-B')
    print('steps', result.nit, result.fun)
    for i in range(int(len(initial_guess)/3)):
        print(f'g_{i}: amplitude: {result.x[i*3]:3.6f} mean: {result.x[(i*3)+1]:3.6f} sigma: {result.x[(i*3)+2]:3.6f}')
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=1)
    sum = np.zeros(len(x))
    for i in range(int(len(initial_guess)/3)):
        #ax.plot(x, amp(x, *result.x[i*3:(i+1)*3]))
        sum += amp(x, *result.x[i*3:(i+1)*3])
    ax.plot(x, sum)
    plt.show()
    quit()
