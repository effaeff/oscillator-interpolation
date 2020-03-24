"""Main script"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load

from processing import processing
from training import training
from testing import testing
from properties import (
    data_dir,
    processed_dir,
    plot_dir,
    model_dir,
    results_dir,
    input_size,
    output_size,
    test_size,
    random_seed
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

def gen_dirs():
    for directory in [data_dir, processed_dir, plot_dir, model_dir, results_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print(f"Error: Creation of directory {directory} failed.")

def main():
    """Main method"""
    misc.to_local_dir(__file__)
    gen_dirs()
    # Definition of data and learning properties
    data = np.load(f"{processed_dir}/processed_data.npy")
    # data = processing(store=True, plot=True) 

    # Train/test split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)

    # Flatten training data by one dimension but keep the shape of the testing data,
    # to being able to test different FRFs separately
    train_data = np.reshape(train_data, (-1, input_size + output_size))
    
    # Scale data
    x_scaler = MinMaxScaler()
    train_data[:, :input_size] = x_scaler.fit_transform(train_data[:, :input_size])
    for test_idx, __ in enumerate(test_data):
        test_data[test_idx, :, :input_size] = x_scaler.transform(test_data[test_idx, :, :input_size])

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

