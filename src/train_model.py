"""Learning routine to model the similarity between simulation and measured data"""
import os
import math
from matplotlib import pyplot as plt
import numpy as np
from joblib import dump, load
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, randint

import misc


def learn(train, test, input_size, output_size, cv_folds, n_iter_search, random_seed):
    """Learning method"""
    param_dicts = [
        {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(2, 32),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(2, 11),
            'max_features': randint(1, input_size)
        }
    ]
    regressors = [
        [RandomForestRegressor(random_state=random_seed, n_jobs=-1) for __ in range(output_size)]
    ]

    errors = np.empty((len(regressors), output_size))
    variances = np.empty((len(regressors), output_size))
    hyperopts = np.empty((len(regressors), output_size), dtype=object)
    for reg_idx in tqdm(range(len(regressors))):
        for out_idx in range(output_size):
            target = train[:, -output_size + out_idx]
            rand_search = RandomizedSearchCV(
                regressors[reg_idx][out_idx],
                param_distributions=param_dicts[reg_idx],
                n_iter=n_iter_search,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                iid=False,
                n_jobs=-1
            )
            rand_search.fit(train[:, :input_size], target)
            pred = rand_search.predict(test[:, :input_size])
            hyperopts[reg_idx, out_idx] = rand_search

            errors[reg_idx, out_idx] = math.sqrt(
                mean_squared_error(test[:, -output_size + out_idx], pred)
            ) * 100.0
            variances[reg_idx, out_idx] = np.std(
                [
                    abs(test[idx, -output_size + out_idx] - pred[idx])
                    for idx in range(len(test))
                ]
            ) * 100.0
    return regressors, errors, variances, hyperopts

def evaluate(regressors, test, input_size, output_size):
    """Method to evaluate trained regressors"""
    errors = np.empty((len(regressors), output_size))
    variances = np.empty((len(regressors), output_size))
    for reg_idx in tqdm(range(len(regressors))):
        pred = regressors[reg_idx].predict(test[:, :input_size])

        for out_idx in range(output_size):
            errors[reg_idx, out_idx] = math.sqrt(
                mean_squared_error(test[:, -output_size + out_idx], pred[:, out_idx])
            ) * 100.0
            variances[reg_idx, out_idx] = np.std(
                [
                    abs(test[idx, -output_size + out_idx] - pred[idx, out_idx])
                    for idx in range(len(test))
                ]
            ) * 100.0
    return errors, variances

def write_results(regressors, errors, variances, filename):
    """Write results to file"""
    with open(filename, 'w') as res_file:
        res_file.write(
            "Regressor\t"
            "NRMSE Thickness_a\t"
            "NRMSE Thickness_b\t"
            "NRMSE Thickness_c\t"
            "NRMSE Thickness_d\t"
            "NRMSE Tickness_mean\t"
            "NRMSE Weight\n"
        )
        for reg_idx, regressor in enumerate(regressors):
            res_file.write(
                "{0}\t\t"
                "{1:.2f} % +/- {2:.2f} %\t"
                "{3:.2f} % +/- {4:.2f} %\t"
                "{5:.2f} % +/- {6:.2f} %\t"
                "{7:.2f} % +/- {8:.2f} %\t"
                "{9:.2f} % +/- {10:.2f} %\t"
                "{11:.2f} % +/- {12:.2f} %\n".format(
                    regressor[0].__class__.__name__,
                    errors[reg_idx, 0], variances[reg_idx, 0],
                    errors[reg_idx, 1], variances[reg_idx, 1],
                    errors[reg_idx, 2], variances[reg_idx, 2],
                    errors[reg_idx, 3], variances[reg_idx, 3],
                    np.mean([errors[reg_idx, idx] for idx in range(4)]),
                    np.mean([variances[reg_idx, idx] for idx in range(4)]),
                    errors[reg_idx, 4], variances[reg_idx, 4]
                )
            )

def replace_data(sim, train, similarities, input_size, percentage):
    """Replace measurements with simulations based on similarities and percentage"""
    similarities = similarities.tolist()
    similarities.sort(key=lambda sample: sample[-1], reverse=True)
    similarities = np.array(similarities)

    replace_samples = int(percentage * len(train) / 100)
    replaced = 0
    for similarity in similarities[:, :-1]:
        for idx, __ in enumerate(train):
            if np.all(similarity == train[idx, :input_size]):
                train[idx, input_size:] = sim[idx].copy()
                replaced += 1
                if replaced == replace_samples:
                    return train

def main():
    """Main method"""
    group = 'Recorder'
    keys_exp = [
        'thickness_a',
        'thickness_b',
        'thickness_c',
        'thickness_d',
        'weight'
    ]

    keys_sim = [
        'thickness_a',
        'thickness_b',
        'thickness_c',
        'thickness_d',
        'mass'
    ]
    random_seed = 1234
    parameter_keys = [
        'comp_thickness',
        'melt_temp',
        'tool_temp',
        'dyn_inj_flow',
        'pressure_level'
    ]
    data_dir = 'C:/Data/Projects/SFB_Lebensdauer/Vorversuche/Heim/processed'
    test_size = 0.1
    cv_folds = 10
    n_iter_search = 500

    similarities = np.load('similarities.npy')

    mean_similarities = np.empty((len(similarities), 6))
    for idx, __ in enumerate(similarities):
        mean_similarities[idx] = np.array(
            [
                similarities[idx, 0],
                similarities[idx, 1],
                similarities[idx, 2],
                similarities[idx, 3],
                similarities[idx, 4],
                np.mean(
                    [
                        similarities[idx, 8],
                        similarities[idx, 9],
                        similarities[idx, 10],
                        similarities[idx, 11],
                        similarities[idx, 12],
                        similarities[idx, 13],
                        similarities[idx, 14],
                        similarities[idx, 15]
                    ]
                )
            ]
        )

    # simulation_files = np.array([
    #     filename for filename in os.listdir('{}/simulation'.format(data_dir))
    #     if filename.endswith('.tdms')
    # ])

    sim_data, exp_data, parameters = load_data(
        data_dir,
        keys_sim,
        keys_exp,
        parameter_keys,
        group
    )

    # measured_data_mean, __, parameters = load_measurements(
    #     data_dir,
    #     simulation_files,
    #     keys_exp,
    #     parameter_keys,
    #     group
    # )
    input_size = len(parameter_keys)
    output_size = len(keys_exp)

    in_out = np.c_[parameters, exp_data]
    in_out_sim = list(zip(in_out, sim_data))
    train, test = train_test_split(in_out_sim, test_size=test_size, random_state=random_seed)
    train, sim_data = zip(*train)
    test, __ = zip(*test)
    train = np.array(train)
    test = np.array(test)
    sim_data = np.array(sim_data)

    for percentage in range(15, 91, 5):
        train_combined = replace_data(sim_data, train, mean_similarities, input_size, percentage)
        x_scaler = MinMaxScaler()
        # y_scaler = MinMaxScaler()
        train_combined[:, :input_size] = x_scaler.fit_transform(train_combined[:, :input_size])
        test[:, :input_size] = x_scaler.transform(test[:, :input_size])
        # train[:, -output_size:] = y_scaler.fit_transform(train[:, -output_size:])
        # test[:, -output_size:] = y_scaler.transform(test[:, -output_size:])
        regressors, errors, variances, hyperopts = learn(
            train_combined,
            test,
            input_size,
            output_size,
            cv_folds,
            n_iter_search,
            random_seed
        )
        write_results(regressors, errors, variances, "results_quality_{}.txt".format(percentage))
        for hyp_idx, hyperopt in enumerate(hyperopts):
            dump(
                hyperopt,
                'hyperopt_{}_{}.joblib'.format(
                    regressors[hyp_idx].__class__.__name__,
                    percentage
                )
            )
        # regressors = [
        #     load(filename).best_estimator_ for filename in os.listdir('.')
        #     if filename.endswith('.joblib')
        # ]
        # errors, variances = evaluate(regressors, test, input_size, output_size)
        # write_results(regressors, errors, variances)

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()
