import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import pi, sqrt, atan

class Osci:
    def __init__(self, omega, gamma, mass):
        self.omega = omega * 2.0 * pi
        self.gamma = gamma
        self.mass = mass
    
    def get_amplitude(self, freq):
        fr = freq * 2.0 * pi
        return (1 / self.mass) / (sqrt((self.omega**2 - fr**2)**2 + 4 * self.gamma**2 * self.omega**2))
    
    def get_phase(self, freq):
        fr = freq * 2.0 * pi
        if abs(fr - self.omega) < 0.001:
            return pi / 2.0
        ph = atan(2* self.omega * fr / (self.omega**2 - fr**2))
        if ph < 0:
            ph += pi
        return ph


random_seed = 1234
test_size = 0.1
num_oscis = 4
data_dir = '../data/02_processed'
data = np.load('{}/processed_data.npy'.format(data_dir))
osci_init_freq={1070, 1500, 2100, 2350}

train, test = train_test_split(data, test_size=test_size, random_state=random_seed)

#initialize oscis
oscis = []
for i in range(num_oscis):
    oscis.append(Osci(osci_init_freq[i], 10.0, 1.0))

