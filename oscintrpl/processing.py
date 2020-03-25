"""
Process raw data by combining pose features with frequency features
for the prediction of each amplitude of each measurement direction.
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from plotting import plot_frf 
import misc
from properties import (
    data_dir,
    processed_dir,
    plot_dir,
    dark2,
    delimiter,
    doe_file,
    freq_step,
    freq_steps_aggreg,
    pos_axes,
    x_range,
    input_size,
    output_size
)


def processing(store=True, plot=False):
    """Processing method"""
    # Fetch all filenames for YY FRFs
    filenames = [
        filename for filename in os.listdir(data_dir)
        if filename.endswith('.txt') and filename.startswith('YY')
    ]
    # Read poses from design of experiments file
    xls = pd.ExcelFile(doe_file)
    positions = pd.read_excel(xls, sheet_name='positions')

    n_rows = int((x_range[1] - x_range[0]) / freq_step - 1)
    aggreg = int(freq_steps_aggreg / freq_step)
    n_rows = int(n_rows / aggreg)

    processed = np.empty((len(filenames), n_rows, input_size + output_size))
    for file_idx in tqdm(range(len(filenames))):
        filename = filenames[file_idx]
        # Get XX FRF based on filename of XX FRF
        xx_file = filename.split('_')
        xx_file[0] = 'XX'
        b_angle = int(os.path.splitext(xx_file[-1][1:])[0])
        pos_label = xx_file[1]
        xx_file = '_'.join(xx_file)
        # Load data
        yy_frf = np.loadtxt('{}/{}'.format(data_dir, filename), delimiter=delimiter)
        xx_frf = np.loadtxt('{}/{}'.format(data_dir, xx_file), delimiter=delimiter)
        # Truncate data to relevant frequency range
        xx_frf = xx_frf[(np.where((xx_frf[:, 0] > x_range[0]) & (xx_frf[:, 0] < x_range[1])))]
        yy_frf = yy_frf[(np.where((yy_frf[:, 0] > x_range[0]) & (yy_frf[:, 0] < x_range[1])))]

        # Get mean in blocks to get frf steps in aggregation period
        xx_frf_n = []
        yy_frf_n = []

        for i in range(int(len(xx_frf) / aggreg)):
            xx_frf_n.append(
                    (
                        xx_frf[(i * aggreg) + int(aggreg / 2) - 1, 0],
                        np.mean(xx_frf[i * aggreg : (i + 1) * aggreg, 1]),
                        np.mean(xx_frf[i * aggreg : (i + 1) * aggreg, 2])
                    )
            )
            yy_frf_n.append(
                    (
                        yy_frf[(i * aggreg) + int(aggreg / 2) - 1, 0],
                        np.mean(yy_frf[i * aggreg : (i + 1) * aggreg, 1]),
                        np.mean(yy_frf[i * aggreg : (i + 1) * aggreg, 2])
                    )
            )
        xx_frf = np.asarray(xx_frf_n)
        yy_frf = np.asarray(yy_frf_n)
        # Get pose features
        x_pos, y_pos = positions.loc[positions['Label'] == pos_label][pos_axes].values[0]
        title = '{}{}_{}{}_B{}'.format(pos_axes[0], x_pos, pos_axes[1], y_pos, b_angle)

        # Plot FRF
        if plot:
            plot_frf(
                [xx_frf, yy_frf],
                ['XX', 'YY'],
                [dark2[0], dark2[1]],
                ['-', '-'],
                title
            )

        # Use pose and frequency as features and XX/YY amplitudes and phases as targets,
        # but store them for each file separately for reasonable train/test split
        for freq_idx, __ in enumerate(xx_frf):
            processed[file_idx, freq_idx] = np.array(
                [
                    x_pos,
                    y_pos,
                    b_angle,
                    xx_frf[freq_idx, 0],
                    xx_frf[freq_idx, 1],
                    yy_frf[freq_idx, 1],
                    xx_frf[freq_idx, 2],
                    yy_frf[freq_idx, 2]
                ]
            )
    if store:
        np.save('{}/processed_data.npy'.format(processed_dir), processed)

    return processed

