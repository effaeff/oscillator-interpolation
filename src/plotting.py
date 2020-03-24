"""Script for FRF plotting"""

import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
from plotter import modify_axis


rc('font', family='Arial')
matplotlib.rcParams.update({'font.size': 14})

DARK2 = [(217 / 255, 95 / 255, 2 / 255),
         (27 / 255, 158 / 255, 119 / 255),
         (117 / 255, 112 / 255, 179 / 255),
         (231 / 255, 41 / 255, 138 / 255),
         (102 / 255, 166 / 255, 30 / 255),
         (230 / 255, 171 / 255, 2 / 255),
         (166 / 255, 118 / 255, 29 / 255),
         (102 / 255, 102 / 255, 102 / 255)]

def plot_frf(data, plot_dir, x_range, y_range, title, axs_labels, figsize, fontsize):
    """
    Function for plotting of FRFs.

    Args:
      x_range: Range of the x-axis of the plot
      y_range: Range of the y-axis of the plot
      figsize: Size of the figure
      fontsize: Size of the fonts
    """
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=figsize)
    for idx, axis_data in enumerate(data):
        axs[idx].plot(axis_data[:, 0], axis_data[:, 1], color=DARK2[0])
        axs[idx].plot(axis_data[:, 0], axis_data[:, 1], color=DARK2[1])
    for idx, axis in enumerate(axs):
        axis.set_yticks(np.arange(y_range[0], y_range[1], y_range[2]))
        axis.set_ylim(y_range[0], y_range[1])
        axis.set_xlim(x_range[0], x_range[1])
        axis.set_xticks(np.arange(x_range[0], x_range[1], x_range[2]))
        axis.set_title(axs_labels[idx], fontsize=fontsize + 2)
    axs[-1].set_xlabel("Frequency")

    fig.canvas.draw()
    for idx, axis in enumerate(axs):
        axis.set_ylabel("Compliance")
        axs[idx] = modify_axis(axis, 'Hz', r'$\frac{µm}{N}$', -2, -2, fontsize)

    fig.suptitle(title, fontsize=fontsize + 4)

    plt.savefig(
        '{}/frf_{}.png'.format(plot_dir, title),
        format='png',
        dpi=600
    )
    plt.close()
    # plt.show()
def plot_frf_phase(data, plot_dir, x_range, y_range, title, axs_labels, figsize, fontsize):
    """
    Function for plotting of FRFs.

    Args:
      x_range: Range of the x-axis of the plot
      y_range: Range of the y-axis of the plot
      figsize: Size of the figure
      fontsize: Size of the fonts
    """
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=figsize)
    for idx, axis_data in enumerate(data):
        axs[idx].plot(axis_data[:, 0], axis_data[:, 2], color=DARK2[0])
        axs[idx].plot(axis_data[:, 0], axis_data[:, 2], color=DARK2[1])
    for idx, axis in enumerate(axs):
        axis.set_yticks(np.arange(y_range[0], y_range[1], y_range[2]))
        axis.set_ylim(y_range[0], y_range[1])
        axis.set_xlim(x_range[0], x_range[1])
        axis.set_xticks(np.arange(x_range[0], x_range[1], x_range[2]))
        axis.set_title(axs_labels[idx], fontsize=fontsize + 2)
    axs[-1].set_xlabel("Frequency")

    fig.canvas.draw()
    for idx, axis in enumerate(axs):
    #    axis.set_ylabel("Compliance")
        axs[idx] = modify_axis(axis, 'Hz', r'$\frac{µm}{N}$', -2, -2, fontsize)

    fig.suptitle(title, fontsize=fontsize + 4)

    plt.savefig(
        '{}/frf_phase_{}.png'.format(plot_dir, title),
        format='png',
        dpi=600
    )
    plt.close()