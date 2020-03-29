"""Gradient-based optimization of oscillator parameter values given a measured FRF"""

import math
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
import misc
from oscillator import calc_frf
from plot_utils import InteractivePlotter
from properties import (
    x_range,
    freq_steps_aggreg,
    freq_step,
    delimiter,
    dark2,
    figsize,
    fontsize
)


class OscillatorOptimizer:
    def __init__(self, target_data, start_params, param_bounds):
        self.target_data = target_data
        self.x_data = np.arange(x_range[0], x_range[1] - 1, freq_steps_aggreg)
        self.start_params = start_params
        self.param_bounds = param_bounds

    def cost_fn(self, params):
        """Cost function for optimization"""
        amp, phase = calc_frf(self.x_data, params)

        self.plotter.update_plot(np.array([amp, phase]), self.target_data)

        error = (
            np.sum((self.target_data[0] - amp)**2) +
            np.sum((self.target_data[1] - phase)**2)
        )
        return error

    def run(self):
        """Run the optimization task"""
        self.plotter = InteractivePlotter(len(self.target_data), figsize, fontsize, dark2)
        self.plotter.init_plot(
            np.array([*calc_frf(self.x_data, self.start_params)]), self.target_data
        )

        np.set_printoptions(suppress=True)

        opt_res = opt.minimize(
            self.cost_fn,
            self.start_params,
            # bounds=self.param_bounds,
            method='TNC'
        )
        print(np.reshape(opt_res.x, (-1, 3)))
        self.plotter.show_plot()

def main():
    """Main method"""
    test_file = 'XX_G_B-60.txt'
    frf = np.loadtxt(test_file, delimiter=delimiter)
    target_data = frf[(np.where((frf[:, 0] > x_range[0]) & (frf[:, 0] < x_range[1])))]

    n_rows = int((x_range[1] - x_range[0]) / freq_step - 1)
    aggreg = int(freq_steps_aggreg / freq_step)
    n_rows = int(n_rows / aggreg)

    frf_aggreg = []

    for i in range(int(len(target_data) / aggreg)):
        frf_aggreg.append(
            (
                target_data[(i * aggreg) + int(aggreg / 2) - 1, 0],
                np.mean(target_data[i * aggreg : (i + 1) * aggreg, 1]),
                np.mean(target_data[i * aggreg : (i + 1) * aggreg, 2])
            )
        )

    target_data = np.transpose(np.asarray(frf_aggreg))[1:]
    test_osc = np.array(
        [
            [4358.78, 3310.14, 0.166609],
            [4484.15, 207.088, 2.80809],
            [2425.83, 629.189, 0.251646],
            [5062.56, 626.647, 0.35843],
            [2123.51, 533.912, 0.992652],
            [1464.13, 520.525, 0.436132],
            [1522.53, 198.228, 0.52103],
            [1072.74, 409.854, 0.644504],
            [5966.04, 1027.73, 0.0954085],
            [7427.76, 10.2595, 0.0183696]
        ]
    )    
    bounds = (
        (
            (x_range[0], x_range[1]),
            (100, 2000),
            (0.1, 100)
        ) * len(test_osc)
    )

    osc_opt = OscillatorOptimizer(target_data, np.reshape(test_osc, (-1,)), bounds)

    # amp, phase = calc_frf(
        # np.arange(x_range[0], x_range[1] - 1, freq_steps_aggreg),
        # np.reshape(test_osc, (-1,))
    # )
    # __, axs = plt.subplots(2, 1)
    # axs[0].plot(amp)
    # axs[0].plot(target_data[0])
    # axs[1].plot(phase)
    # axs[1].plot(target_data[1])
    # plt.show()

    osc_opt.run()

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()

