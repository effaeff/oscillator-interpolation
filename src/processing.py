"""
Process raw data by adding pose features to target filenames
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from plot_frf import plot_frf
import misc


def main():
    """Main method"""
    data_dir = '../data/01_raw'
    processed_dir = '../data/02_processed'
    plot_dir = '../figures'
    delimiter = ' '
    doe_file = '../data/01_raw/Versuchsplan.xlsx'
    figsize = (7, 7)
    fontsize = 14
    x_range = (200, 3201, 1000)
    y_range = (0, 0.9, 0.2)

    filenames = [
        filename for filename in os.listdir(data_dir)
        if filename.endswith('.txt') and filename.split('_')[2] == 'YY'
    ]

    positions = pd.read_excel(doe_file, sheet_name='positions')
    doe = pd.read_excel(doe_file, sheet_name='DOE')

    for file_idx in tqdm(range(len(filenames))):
        filename = filenames[file_idx]
        xx_file = filename.split('_')
        xx_file[2] = 'XX'
        xx_file[0] = str(int(xx_file[0]) + 48)
        xx_number = int(xx_file[0])
        xx_file = '_'.join(xx_file)
        yy_frf = np.loadtxt('{}/{}'.format(data_dir, filename), delimiter=delimiter)
        xx_frf = np.loadtxt('{}/{}'.format(data_dir, xx_file), delimiter=delimiter)
        xx_frf = xx_frf[(np.where((xx_frf[:, 0] > x_range[0]) & (xx_frf[:, 0] < x_range[1])))]
        yy_frf = yy_frf[(np.where((yy_frf[:, 0] > x_range[0]) & (yy_frf[:, 0] < x_range[1])))]

        pos_label = doe.loc[doe['XX_Nr'] == xx_number]['Position'].values[0]
        b_angle = doe.loc[doe['XX_Nr'] == xx_number]['B_angle'].values[0]
        x_pos, y_pos = positions.loc[positions['Label'] == pos_label][['X', 'Y']].values[0]
        title = 'X{}_Y{}_B{}'.format(x_pos, y_pos, b_angle)

        data = [xx_frf, yy_frf]

        if len(sys.argv) > 1 and sys.argv[1] == 'plot':
            plot_frf(data, plot_dir, x_range, y_range, title, ['XX', 'YY'], figsize, fontsize)

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()
