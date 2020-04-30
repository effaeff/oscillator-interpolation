"""Script for evaluation of trained models to predict FRFs"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import load

from oscintrpl.testing import eval_frf
from oscintrpl.processing import processing
from oscintrpl.properties import (
    model_dir,
    input_size,
    output_size,
    x_range,
    freq_steps_aggreg,
    test_configs
)
import misc


def main():
    """Main method"""
    eval_configs = [
        [-333.33, 300.0, -5.0]
    ]
    np.set_printoptions(suppress=True)
    data = processing(store=True, plot=False)

    # Train/test split
    train_data = np.empty(
        (np.shape(data)[0] - len(test_configs), np.shape(data)[1], np.shape(data)[2])
    )
    train_idx = 0

    for scenario in data:
        test_found = False
        for config in test_configs:
            if (
                    scenario[0, 0] == config[0] and
                    scenario[0, 1] == config[1] and
                    scenario[0, 2] == config[2]
                ):
                test_found = True
        if not test_found:
            train_data[train_idx] = scenario
            train_idx += 1

    # Flatten training data by one dimension but keep the shape of the testing data,
    # to being able to test different FRFs separately
    train_data = np.reshape(train_data, (-1, input_size + output_size))

    # Scale data
    x_scaler = MinMaxScaler()
    x_scaler.fit(train_data[:, :input_size])

    freqs = np.arange(x_range[0], x_range[1], freq_steps_aggreg)
    eval_data = np.empty((len(eval_configs), len(freqs), input_size))
    for eval_idx, eval_config in enumerate(eval_configs):
        eval_scenario = np.empty((len(freqs), input_size))
        for freq_idx, freq in enumerate(freqs):
            eval_scenario[freq_idx] = eval_config + [freq]
        eval_data[eval_idx] = x_scaler.transform(eval_scenario)

    model_names = [filename for filename in os.listdir(model_dir) if filename.endswith('.joblib')]
    for filename in model_names:
        hyperopt = load('{}/{}'.format(model_dir, filename))
        eval_frf(hyperopt, eval_data, eval_configs)

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
