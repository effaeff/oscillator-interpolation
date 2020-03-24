"""
Process raw data by combining pose features with frequency features
for the prediction of each amplitude of each measurement direction.
"""

import os
import sys
import pandas as pd
import numpy as np
#from tqdm import tqdm
from plot_frf import plot_frf
import misc


def main():
    """Main method"""
    # Definitions of data and plotting properties
    data_dir = '../data/01_raw'
    processed_dir = '../data/02_processed'
    plot_dir = '../figures'
    delimiter = ' '
    doe_file = '../data/01_raw/Versuchsplan.xlsx'
    figsize = (7, 7)
    fontsize = 14
    freq_step = 0.25 # Precision of the measurement
    x_range = (200, 3200, 1000)
    y_range = (0, 0.9, 0.2)
    input_size = 4
    output_size = 2

    # Fetch all filenames for YY FRFs
    filenames = [
        filename for filename in os.listdir(data_dir)
        if filename.endswith('.txt') and filename.split('_')[2] == 'YY'
    ]

    # Read poses from design of experiments file
    xls = pd.ExcelFile(doe_file)
    positions = pd.read_excel(xls, sheetname='positions')
    doe = pd.read_excel(xls, sheetname='DOE')

    n_rows = int((x_range[1] - x_range[0]) / freq_step - 1)
    freq_steps_new = 10
    combine = int(freq_steps_new/freq_step)
    n_rows = int(n_rows/combine)

    processed = np.empty((len(filenames), n_rows, input_size + output_size))
    for file_idx in range(len(filenames)):
        filename = filenames[file_idx]
        # Get XX FRF based on filename of XX FRF
        xx_file = filename.split('_')
        xx_file[2] = 'XX'
        xx_file[0] = str(int(xx_file[0]) + 48)
        xx_number = int(xx_file[0])
        xx_file = '_'.join(xx_file)
        # Load data
        yy_frf = np.loadtxt('{}/{}'.format(data_dir, filename), delimiter=delimiter)
        xx_frf = np.loadtxt('{}/{}'.format(data_dir, xx_file), delimiter=delimiter)
        # Truncate data to relevant frequency range
        xx_frf = xx_frf[(np.where((xx_frf[:, 0] > x_range[0]) & (xx_frf[:, 0] < x_range[1])))]
        yy_frf = yy_frf[(np.where((yy_frf[:, 0] > x_range[0]) & (yy_frf[:, 0] < x_range[1])))]

        #get mean in blocks to get frf steps in 10Hz
        xx_frf_n = []
        yy_frf_n = []
        

        for i in range(int(len(xx_frf)/combine)):
            xx_frf_n.append((xx_frf[(i*combine) + int(combine/2)-1, 0], np.mean(xx_frf[i*combine : (i+1)*combine, 1])))
            yy_frf_n.append((yy_frf[(i*combine) + int(combine/2)-1, 0], np.mean(yy_frf[i*combine : (i+1)*combine, 1])))
        xx_frf_n = np.asarray(xx_frf_n)
        yy_frf_n = np.asarray(yy_frf_n)
        xx_frf = xx_frf_n
        yy_frf = yy_frf_n
        # Get pose features
        pos_label = doe.loc[doe['XX_Nr'] == xx_number]['Position'].values[0]
        b_angle = doe.loc[doe['XX_Nr'] == xx_number]['B_angle'].values[0]
        x_pos, y_pos = positions.loc[positions['Label'] == pos_label][['X', 'Y']].values[0]
        title = 'X{}_Y{}_B{}'.format(x_pos, y_pos, b_angle)

        # Plot FRF
        #if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        if True:
            plot_frf(
                [xx_frf, yy_frf],
                plot_dir,
                x_range,
                y_range,
                title,
                ['XX', 'YY'],
                figsize,
                fontsize
            )

        # Use pose and frequency as features and XX/YY amplitudes as targets,
        # but store them for each file separately for reasonable train/test split
        for freq_idx, __ in enumerate(xx_frf):
            processed[file_idx, freq_idx] = np.array(
                [
                    x_pos,
                    y_pos,
                    b_angle,
                    xx_frf[freq_idx, 0],
                    xx_frf[freq_idx, 1],
                    yy_frf[freq_idx, 1]
                ]
            )

    np.save('{}/processed_data.npy'.format(processed_dir), processed)

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()
