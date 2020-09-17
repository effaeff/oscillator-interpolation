"""Script for leaning oscillators"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump

from oscintrpl.processing import read_osc
from oscintrpl.training import training
from oscintrpl.testing import test_osc
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
            "RMSE Mean freq\t"
            "RMSE Mean gamma\t"
            "RMSE Mean mass\n"
        )

        for hyper_idx, hyperopt in enumerate(hyperopts):
            local_errors = np.reshape(errors[hyper_idx], (int(len(errors[hyper_idx]) / 3), 3))
            local_variances = np.reshape(
                variances[hyper_idx], (int(len(variances[hyper_idx]) / 3), 3)
            )
            local_errors = np.mean(local_errors, axis=1)
            local_variances = np.mean(local_variances, axis=0)
            res_file.write(
                "{0}\t"
                "{1:.2f} +/- {2:.2f}\t"
                "{3:.2f} +/- {4:.2f}\t"
                "{5:.2f} +/- {6:.2f}\n".format(
                    hyperopt[0].best_estimator_.__class__.__name__,
                    local_errors[0], local_variances[0],
                    local_errors[1], local_variances[1],
                    local_errors[2], local_variances[2]
                )
            )

def main():
    """Main method"""
    misc.to_local_dir(__file__)
    misc.gen_dirs([data_dir, processed_dir, plot_dir, model_dir, results_dir])
    # Definition of data and learning properties
    np.set_printoptions(suppress=True)
    # data = np.load(f"{processed_dir}/processed_osc.npy")
    data = read_osc(store=True)

    # Train/test split
    # train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)
    train_data = np.empty((np.shape(data)[0] - len(test_configs), np.shape(data)[1]))
    test_data = np.empty((len(test_configs), np.shape(data)[1]))
    test_idx = 0
    train_idx = 0

    for config in test_configs:
        for row in data:
            if row[0] == config[0] and row[1] == config[1] and row[2] == config[2]:
                test_data[test_idx] = row
                test_idx += 1

    for row in data:
        test_found = False
        for config in test_configs:
            if row[0] == config[0] and row[1] == config[1] and row[2] == config[2]:
                test_found = True
        if not test_found:
            train_data[train_idx] = row
            train_idx += 1

    # import os
    # import math
    # from sklearn.metrics import mean_squared_error
    # regre = ['ElasticNet', 'RandomForestRegressor', 'XGBRegressor']
    # pred_dir = '{}/dmg_hsc75linear/osc_fitting_hypsrch300_predictions_2'.format(results_dir)

    # for reg in regre:
        # files = [
            # filename for filename in os.listdir(pred_dir)
            # if filename.startswith(reg)
        # ]
        # freq_errors = []
        # gamma_errors = []
        # mass_errors = []
        # for config_idx, config in enumerate(test_configs):
            # local_test = test_data[config_idx]
            # local_pred_file = files[0]
            # for idx in range(1, len(files)):
                # file_config = os.path.splitext(files[idx])[0].split('_')[2:]
                # if np.all([float(file_config[jdx]) == config[jdx] for jdx in range(len(config))]):
                    # local_pred_file = files[idx]
            # local_pred = np.load('{}/{}'.format(pred_dir, local_pred_file))
            # local_test = local_test[input_size:]
            # freq_error = math.sqrt(mean_squared_error(local_test[::3], local_pred[::3]))
            # gamma_error = math.sqrt(mean_squared_error(local_test[1::3], local_pred[1::3]))
            # mass_error = math.sqrt(mean_squared_error(local_test[2::3], local_pred[2::3]))
            # freq_errors.append(freq_error)
            # gamma_errors.append(gamma_error)
            # mass_errors.append(mass_error)


        # print(reg)
        # # print("Frequency error: {} +/- {}".format(np.mean(freq_errors), np.std(freq_errors)))
        # # print("Gamma error: {} +/- {}".format(np.mean(gamma_errors), np.std(gamma_errors)))
        # # print("Mass error: {} +/- {}".format(np.mean(mass_errors), np.std(mass_errors)))
        # print(r"{:.2f} \pm {:.2f} & {:.2f} \pm {:.2f} & {:.2f} \pm {:.2f}".format(
            # np.mean(freq_errors), np.std(freq_errors),
            # np.mean(gamma_errors), np.std(gamma_errors),
            # np.mean(mass_errors), np.std(mass_errors)
        # ))

    # quit()

    # Scale data
    x_scaler = MinMaxScaler()
    train_data[:, :input_size] = x_scaler.fit_transform(train_data[:, :input_size])
    test_data[:, :input_size] = x_scaler.transform(test_data[:, :input_size])

    hyperopts = training(train_data)

    total_errors = np.empty((len(hyperopts), output_size))
    total_variances = np.empty((len(hyperopts), output_size))
    for hyper_idx, hyperopt in enumerate(hyperopts):
        errors, variances = test_osc(hyperopt, test_data, test_configs)
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
