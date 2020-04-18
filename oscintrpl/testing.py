"""Testing routine using trained regressor"""

import math
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from oscintrpl.plotting import plot_frf
from oscintrpl.properties import (
    input_size,
    output_size,
    dark2
)


def testing(hyperopt, test_data, scaler):
    """
    Args:
        hyperopt:
            Numpy array of objects, which result from a hyperoptimization procedure.
            Each array member corresponds to one of the learning targets.
            Each array member contains several estimators.
            Best estimator of each array member can be accessed through object.best_estimator_.
            Provides predict().
        test_data:
            List of test scenarios.
        scaler:
            sklearn.MinMaxScaler instance, which was used prior to the taining procedure.
            Used to inverse scale the data for interpretable plots.
        plot:
            Boolean which indicates, if plots shoud be generated.
    Returns:
        errors:
            Numpy array of normalized root mean squared errors
            between measured and predicted target values.
            Each array member corresponds to one of the learning targets.
        variances:
            Numpy array of standard deviations between measured and predicted target values.
            Each array member corresponds to one of the learning targets.
    """
    errors = np.zeros(output_size)
    variances = np.zeros(output_size)
    for test_idx, test_scenario in enumerate(test_data):
        predictions = np.empty((output_size, len(test_scenario)))
        for out_idx in range(output_size):
            pred = hyperopt[out_idx].predict(test_scenario[:, :input_size])
            outputs = np.c_[
                np.copy(test_scenario[:, input_size + out_idx]),
                pred
            ]
            y_scaler = MinMaxScaler()
            outputs_scaled = y_scaler.fit_transform(outputs)
            errors[out_idx] += math.sqrt(
                    mean_squared_error(outputs_scaled[:, 0], outputs_scaled[:, 1])
            ) * 100.0
            variances[out_idx] += np.std(
                [
                    abs(outputs_scaled[idx, 0] - outputs_scaled[idx, 1])
                    for idx in range(len(outputs))
                ]
            ) * 100.0

            predictions[out_idx] = pred

        predictions = np.transpose(predictions)
        test_plot = np.copy(test_scenario)
        test_plot[:, :input_size] = scaler.inverse_transform(test_plot[:, :input_size])
        frequency = test_plot[:, input_size - 1]
        test_output = test_plot[:, input_size:]
        plot_frf(
            [
                np.c_[frequency, test_output[:, 0], test_output[:, 2]],
                np.c_[frequency, test_output[:, 1], test_output[:, 3]],
                np.c_[frequency, predictions[:, 0], predictions[:, 2]],
                np.c_[frequency, predictions[:, 1], predictions[:, 3]]
            ],
            ['XX test', 'YY test', 'XX pred', 'YY pred'],
            [dark2[0], dark2[1], dark2[0], dark2[1]],
            ['-', '-', '--', '--'],
            '{}_test-scenario{}'.format(
                hyperopt[0].best_estimator_.__class__.__name__,
                test_idx
            )
        )

    for out_idx in range(output_size):
        errors[out_idx] /= len(test_data)
        variances[out_idx] /= len(test_data)

    return errors, variances
