"""Script for leaning FRFs"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump

from oscintrpl.processing import processing
from oscintrpl.training import training
from oscintrpl.testing import testing
from oscintrpl.properties import (
    data_dir,
    processed_dir,
    plot_dir,
    model_dir,
    results_dir,
    input_size,
    output_size,
    test_size,
    random_seed,
    test_configs
)
import misc


def write_results(hyperopts, errors, variances):
    """Write results to file"""
    with open('{}/results.txt'.format(results_dir), 'w') as res_file:
        res_file.write(
            "Regressor\t"
            "NRMSE XX Amplitude\t"
            "NRMSE YY Amplitude\t"
            "NRMSE XX Phase\t"
            "NRMSE YY Phase\n"
        )

        for hyper_idx, hyperopt in enumerate(hyperopts):
            res_file.write(
                "{0}\t"
                "{1:.2f} % +/- {2:.2f} %\t"
                "{3:.2f} % +/- {4:.2f} %\t"
                "{5:.2f} % +/- {6:.2f} %\t"
                "{7:.2f} % +/- {8:.2f} %\n".format(
                    hyperopt[0].best_estimator_.__class__.__name__,
                    errors[hyper_idx, 0], variances[hyper_idx, 0],
                    errors[hyper_idx, 1], variances[hyper_idx, 1],
                    errors[hyper_idx, 2], variances[hyper_idx, 2],
                    errors[hyper_idx, 3], variances[hyper_idx, 3]
                )
            )

def main():
    """Main method"""
    misc.to_local_dir(__file__)
    misc.gen_dirs([data_dir, processed_dir, plot_dir, model_dir, results_dir])
    # data = np.load(f"{processed_dir}/processed_data.npy")
    np.set_printoptions(suppress=True)
    data = processing(store=True, plot=False)
    print(np.shape(data))

    # Train/test split
    # train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)
    train_data = np.empty((np.shape(data)[0] - 2, np.shape(data)[1], np.shape(data)[2]))
    test_data = np.empty((len(test_configs), np.shape(data)[1], np.shape(data)[2]))
    test_idx = 0
    train_idx = 0
    for frf in data:
        test_found = False
        for config in test_configs:
            if frf[0, 0] == config[0] and frf[0, 1] == config[1] and frf[0, 2] == config[2]:
                test_data[test_idx] = frf
                test_idx += 1
                test_found = True
        if not test_found:
            train_data[train_idx] = frf
            train_idx += 1

    # Flatten training data by one dimension but keep the shape of the testing data,
    # to being able to test different FRFs separately
    train_data = np.reshape(train_data, (-1, input_size + output_size))

    # Scale data
    x_scaler = MinMaxScaler()
    train_data[:, :input_size] = x_scaler.fit_transform(train_data[:, :input_size])
    for test_idx, __ in enumerate(test_data):
        test_data[test_idx, :, :input_size] = x_scaler.transform(
            test_data[test_idx, :, :input_size]
        )

    hyperopts = training(train_data)
    total_errors = np.empty((len(hyperopts), output_size))
    total_variances = np.empty((len(hyperopts), output_size))
    for hyper_idx, hyperopt in enumerate(hyperopts):
        errors, variances = testing(hyperopt, test_data, x_scaler)
        total_errors[hyper_idx] = errors
        total_variances[hyper_idx] = variances
        dump(
            hyperopt,
            '{}/hyperopt_{}.joblib'.format(
                model_dir,
                hyperopt[0].best_estimator_.__class__.__name__
            )
        )
    write_results(hyperopts, total_errors, total_variances)

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()
