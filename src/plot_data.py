"""Plot pose dependent FRFs"""

import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
import misc
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

def main():
    """Main method"""
    data_dir = (
        '../data/!2_FRFs_defl'
    )
    delimiter = ' '
    doe_file = (
        '../data/Versuchsplan.xlsx'
    )
    figsize = (10, 10)
    fontsize = 14
    lower_cutoff = 200
    upper_cutoff = 3002

    filenames = [
        filename for filename in os.listdir(data_dir)
        if filename.endswith('.txt') and filename.split('_')[2] == 'YY'
    ]

    xls = pd.ExcelFile(doe_file)
    positions = pd.read_excel(xls, sheetname='positions')
    doe = pd.read_excel(xls, sheetname='DOE')


    for file_idx in range(len(filenames)):
        filename = filenames[file_idx]
        xx_file = filename.split('_')
        xx_file[2] = 'XX'
        xx_file[0] = str(int(xx_file[0]) + 48)
        xx_number = int(xx_file[0])
        xx_file = '_'.join(xx_file)
        yy_frf = np.loadtxt('{}/{}'.format(data_dir, filename), delimiter=delimiter)
        xx_frf = np.loadtxt('{}/{}'.format(data_dir, xx_file), delimiter=delimiter)
        xx_frf = xx_frf[(np.where((xx_frf[:, 0] > lower_cutoff) & (xx_frf[:, 0] < upper_cutoff)))]
        yy_frf = yy_frf[(np.where((yy_frf[:, 0] > lower_cutoff) & (yy_frf[:, 0] < upper_cutoff)))]

        pos_label = doe.loc[doe['XX_Nr'] == xx_number]['Position'].values[0]
        b_angle = doe.loc[doe['XX_Nr'] == xx_number]['B_angle'].values[0]
        x_pos, y_pos = positions.loc[positions['Label'] == pos_label][['X', 'Y']].values[0]
        title = 'X{}_Y{}_B{}'.format(x_pos, y_pos, b_angle)

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=figsize)
        axs[0][0].plot(xx_frf[:, 0], xx_frf[:, 1], color=DARK2[0])
        axs[0][1].plot(xx_frf[:, 0], xx_frf[:, 2], color=DARK2[0])
        axs[1][0].plot(yy_frf[:, 0], yy_frf[:, 1], color=DARK2[1])
        axs[1][1].plot(yy_frf[:, 0], yy_frf[:, 2], color=DARK2[1])
        for i in range(2):
            axs[i][0].set_yticks(np.arange(0, 0.9, 0.2))
            axs[i][0].set_ylim(0, 0.8)
            axs[i][0].set_xlim(lower_cutoff, upper_cutoff)
            axs[i][0].set_xticks(
                np.arange(lower_cutoff, upper_cutoff + 1, int((upper_cutoff - lower_cutoff) / 3))
            )
        axs[1][0].set_xlabel("Frequency")
        axs[0][0].set_title('XX', fontsize=fontsize + 2)
        axs[1][0].set_title('YY', fontsize=fontsize + 2)

        fig.canvas.draw()


        for i in range(2):
            axs[i][0].set_ylabel("Compliance")
            axs[i][0] = modify_axis(axs[i][0], 'Hz', r'$\frac{µm}{N}$', -2, -2, fontsize)

        for i in range(2):
            axs[i][1].set_yticks(np.arange(-150.0, 50.0, 50.0))
            axs[i][1].set_ylim(-180.0, 20.0)
            axs[i][1].set_xlim(lower_cutoff, upper_cutoff)
            axs[i][1].set_xticks(
                np.arange(lower_cutoff, upper_cutoff + 1, int((upper_cutoff - lower_cutoff) / 3))
            )
        axs[1][1].set_xlabel("Frequency")
        axs[0][1].set_title('XX', fontsize=fontsize + 2)
        axs[1][1].set_title('YY', fontsize=fontsize + 2)

        fig.suptitle(title, fontsize=fontsize + 4)

        plt.savefig(
            '{}/{}.png'.format(data_dir, title),
            format='png',
            dpi=600
        )
        plt.close()
        # plt.show()
        quit()

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()
