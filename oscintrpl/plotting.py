"""Script for FRF plotting"""

import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
from plot_utils import modify_axis
from properties import (
   plot_dir,
   x_range,
   y_range_amp,
   y_range_phase,
   figsize,
   fontsize
)
# plt.switch_backend('Agg') # LiDo
rc('font', family='Arial')
matplotlib.rcParams.update({'font.size': 14})


def plot_frf(data, labels, colors, linestyles, title):
    """
    Function for plotting of FRFs.

    Args:
        data: List of numpy arrays. Each list member is expected to be a data set to plot.
            Each data set is expected to contain members for frequency, amplitude and phase.
        labels: Labels of the data sets.
        title: Title of the plot. Also used as filename.
    """
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=figsize)
    for idx, data_set in enumerate(data):
        axs[0].plot(
            data_set[:, 0],
            data_set[:, 1],
            linestyle=linestyles[idx],
            color=colors[idx],
            label=labels[idx]
        )
        axs[1].plot(
            data_set[:, 0],
            data_set[:, 2],
            linestyle=linestyles[idx],
            color=colors[idx],
            label=labels[idx]
        )

    for __, axis in enumerate(axs):
        axis.set_xlim(x_range[0], x_range[1])
        axis.set_xticks(np.arange(x_range[0], x_range[1], x_range[2]))
    axs[0].set_yticks(np.arange(y_range_amp[0], y_range_amp[1], y_range_amp[2]))
    axs[0].set_ylim(y_range_amp[0], y_range_amp[1])
    axs[1].set_yticks(np.arange(y_range_phase[0], y_range_phase[1], y_range_phase[2]))
    axs[1].set_ylim(y_range_phase[0], y_range_phase[1])
    axs[-1].set_xlabel("Frequency")

    fig.canvas.draw()
    axs[0].set_ylabel("Compliance")
    axs[0] = modify_axis(axs[0], 'Hz', r'$\frac{µm}{N}$', -2, -2, fontsize)
    axs[1].set_ylabel("Phase angle")
    axs[1] = modify_axis(axs[1], 'Hz', '°', -2, -2, fontsize)

    fig.suptitle(title, fontsize=fontsize + 4, y=1.08)
    
    for axis in axs:
        axis.get_yaxis().set_label_coords(-0.1,0.5)

    fig.tight_layout()
    leg = axs[0].legend(
        prop={'size':fontsize},
        bbox_to_anchor=(0.95, 1.1),
        bbox_transform=plt.gcf().transFigure
    )
    leg.get_frame().set_linewidth(0.0)

    plt.setp(axs[0].get_xticklabels(), visible=False)

    plt.savefig(
        '{}/frf_{}.png'.format(plot_dir, title),
        format='png',
        dpi=600,
        bbox_extra_artists=(leg,),
        bbox_inches='tight'
    )
    plt.close()
    # plt.show()

