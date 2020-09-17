"""Script for a naive interpolation of frfs and oscillator parameter values"""

import math
import numpy as np
import sklearn.neighbors as neighbors
from sklearn.metrics import mean_squared_error

from oscintrpl.processing import read_osc, processing
from oscintrpl.plotting import plot_frf
from oscintrpl.properties import (
    data_dir,
    processed_dir,
    plot_dir,
    results_dir,
    input_size,
    output_size,
    test_configs
)
from oscintrpl.properties import OSC
import misc
from colors import dark2


def train_test_osc():
    """Preprocess oscillator data"""
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

    return train_data, test_data

def train_test_frf():
    # data = processing(store=True, plot=False)
    data = np.load(f"{processed_dir}/processed_data.npy")

    # Train/test split
    # train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)
    train_data = np.empty(
        (np.shape(data)[0] - len(test_configs), np.shape(data)[1], np.shape(data)[2])
    )
    test_data = np.empty((len(test_configs), np.shape(data)[1], np.shape(data)[2]))
    test_idx = 0
    train_idx = 0
    for config in test_configs:
        for scenario in data:
            if (
                    scenario[0, 0] == config[0] and
                    scenario[0, 1] == config[1] and
                    scenario[0, 2] == config[2]
                ):
                test_data[test_idx] = scenario
                test_idx += 1

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

    return train_data, test_data

def interpolate_osc(train_data, test_data, n_neighbors):
    """Method to interpolate oscillator parameter values"""
    nbrs = neighbors.NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="kd_tree",
        n_jobs=-1
    ).fit(train_data[:, :input_size])
    dist, ind = nbrs.kneighbors(test_data[:, :input_size])

    freq_errors = []
    gamma_errors = []
    mass_errors = []
    for test_idx, test_scenario in enumerate(test_data):
        knbrs = [train_data[ind[test_idx][nbr_idx], input_size:] for nbr_idx in range(n_neighbors)]


        knbrs = np.array(knbrs)

        print(knbrs[:, 3])

        print(
            np.mean(
                [
                    np.mean([abs(knbrs[jdx, idx] - test_scenario[3 + idx]) for jdx in range(n_neighbors)])
                    for idx in range(0, output_size, 3)
                ]
            )
        )
        quit()

        pred = [
            np.sum(
                [
                    dist[test_idx][nbr_idx] * knbrs[nbr_idx][param_idx]
                    for nbr_idx in range(n_neighbors)
                ]
            ) / np.sum(dist[test_idx])
            for param_idx in range(output_size)
        ]
        local_test = test_scenario[input_size:]
        freq_error = math.sqrt(mean_squared_error(local_test[::3], pred[::3]))
        gamma_error = math.sqrt(mean_squared_error(local_test[1::3], pred[1::3]))
        mass_error = math.sqrt(mean_squared_error(local_test[2::3], pred[2::3]))
        freq_errors.append(freq_error)
        gamma_errors.append(gamma_error)
        mass_errors.append(mass_error)

    return (
        freq_errors,
        gamma_errors,
        mass_errors,
        (
            np.mean(freq_errors) / max(freq_errors) +
            np.mean(gamma_errors) / np.max(gamma_errors) +
            np.mean(mass_errors) / np.max(mass_errors)
        )
    )

def interpolate_frf(train_data, test_data, n_neighbors):
    """Method to interpolate frfs"""
    nbrs = neighbors.NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm='kd_tree',
        n_jobs=-1
    ).fit(train_data[:, 0, :input_size-1])
    dist, ind = nbrs.kneighbors(test_data[:, 0, :input_size-1])

    errors = np.zeros(output_size)
    variances = np.zeros(output_size)
    for test_idx, test_scenario in enumerate(test_data):
        knbrs = [
            train_data[ind[test_idx][nbr_idx], :, input_size:] for nbr_idx in range(n_neighbors)
        ]

        pred = []
        for freq_idx in range(len(knbrs[0])):
            local_pred = [
                np.sum(
                    [
                        dist[test_idx][nbr_idx] * knbrs[nbr_idx][freq_idx][param_idx]
                        for nbr_idx in range(n_neighbors)
                    ]
                ) / np.sum(dist[test_idx])
                for param_idx in range(output_size)
            ]
            pred.append(local_pred)

        for out_idx in range(output_size):
            local_pred = np.transpose(pred)[out_idx]
            local_test = test_scenario[:, input_size + out_idx]
            errors[out_idx] += math.sqrt(
                mean_squared_error(local_test, local_pred)
            )
            variances[out_idx] += np.std(
                [
                    abs(local_test[idx] - local_pred[idx])
                    for idx in range(len(local_test))
                ]
            )
        pred = np.array(pred)
        np.save(
            'linear_pred_test-scenario{}.npy'.format(test_idx),
            pred
        )
        # test_plot = np.copy(test_scenario)
        # frequency = test_plot[:, input_size - 1]
        # test_output = test_plot[:, input_size:]
        # plot_frf(
            # [
                # np.c_[frequency, test_output[:, 0], test_output[:, 2]],
                # np.c_[frequency, test_output[:, 1], test_output[:, 3]],
                # np.c_[frequency, pred[:, 0], pred[:, 2]],
                # np.c_[frequency, pred[:, 1], pred[:, 3]]
            # ],
            # ['XX test', 'YY test', 'XX pred', 'YY pred'],
            # [dark2[0], dark2[1], dark2[0], dark2[1]],
            # ['-', '-', '--', '--'],
            # 'linear_pred_test-scenario{}'.format(
                # test_idx
            # )
        # )

    for out_idx in range(output_size):
        errors[out_idx] /= len(test_data)
        variances[out_idx] /= len(test_data)

    return errors, variances, np.sum(errors) + np.sum(variances)


def main():
    """Main method"""
    misc.to_local_dir(__file__)
    misc.gen_dirs([data_dir, processed_dir, plot_dir, results_dir])
    # Definition of data and learning properties
    np.set_printoptions(suppress=True)
    # data = np.load(f"{processed_dir}/processed_osc.npy")

    n_neighbors = range(10, 21)
    if OSC:
        train_data, test_data = train_test_osc()



        best_score = 0
        best_nbrs = 0
        for local_nbrs in n_neighbors:
            __, __, __, score = interpolate_osc(train_data, test_data, local_nbrs)
            if score > best_score:
                best_score = score
                best_nbrs = local_nbrs
        freq_errors, gamma_errors, mass_errors, __ = interpolate_osc(
            train_data, test_data, best_nbrs
        )
        # print("Frequency error: {} +/- {}".format(np.mean(freq_errors), np.std(freq_errors)))
        # print("Gamma error: {} +/- {}".format(np.mean(gamma_errors), np.std(gamma_errors)))
        # print("Mass error: {} +/- {}".format(np.mean(mass_errors), np.std(mass_errors)))
        print(r"{:.2f} \pm {:.2f} & {:.2f} \pm {:.2f} & {:.2f} \pm {:.2f}".format(
            np.mean(freq_errors), np.std(freq_errors),
            np.mean(gamma_errors), np.std(gamma_errors),
            np.mean(mass_errors), np.std(mass_errors)
        ))

    else:
        train_data, test_data = train_test_frf()
        best_score = 0
        best_nbrs = 0
        for local_nbrs in n_neighbors:
            __, __, score = interpolate_frf(train_data, test_data, local_nbrs)
            if score > best_score:
                best_score = score
                best_nbrs = local_nbrs
        errors, variances, __ = interpolate_frf(train_data, test_data, best_nbrs)
        print(r"{:.2f} \pm {:.2f} & {:.2f} \pm {:.2f} & {:.2f} \pm {:.2f} & {:.2f} \pm {:.2f}".format(
            errors[0], variances[0],
            errors[1], variances[1],
            errors[2], variances[2],
            errors[3], variances[3]
        ))

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()
