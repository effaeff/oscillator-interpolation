"""Aggregate FRFs"""

import os
import numpy as np

import misc
from oscintrpl.processing import aggregate
from oscintrpl.properties import (
    data_dir,
    x_range,
    freq_step,
    freq_steps_aggreg,
    delimiter
)


def main():
    """Main method"""
    filenames = [
        filename for filename in os.listdir(data_dir)
        if filename.endswith('.txt')
    ]

    n_rows = int((x_range[1] - x_range[0]) / freq_step - 1)
    aggreg = int(freq_steps_aggreg / freq_step)
    n_rows = int(n_rows / aggreg)

    for file_idx, __ in enumerate(filenames):
        filename = filenames[file_idx]
        # Get FRF based on filename of FRF
        frf = np.loadtxt('{}/{}'.format(data_dir, filename), delimiter=delimiter)
        # Truncate data to relevant frequency range
        frf = frf[(np.where((frf[:, 0] > x_range[0]) & (frf[:, 0] < x_range[1])))]

        frf = aggregate(frf, aggreg)
        np.savetxt('{}/{}_resampled.txt'.format(data_dir, os.path.splitext(filename)[0]), frf)


if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
