"""Learning routine to model the amplitudes of FRFs based on pose and frequency information"""
import os
import math
import numpy as np
from joblib import dump, load
#from tqdm import tqdm
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
import time


def learn(train, test, input_size, output_size, cv_folds, n_iter_search, random_seed):
    """Learning method"""
    param_dicts = [
        # {
        #     'learning_rate': uniform(0.0001, 0.1),
        #     'max_depth': randint(2, 32),
        #     'subsample': uniform(0.5, 0.5),
        #     'n_estimators': randint(100, 1000),
        #     'colsample_bytree': uniform(0.4, 0.6),
        #     'lambda': randint(1, 100),
        #     'gamma': uniform()
        # },
        # {'alpha': uniform()},
        # {'alpha': uniform()},
        # {'alpha': uniform(), 'l1_ratio': uniform()},
        # {
        #     'learning_rate': uniform(0.0001, 0.1),
        #     'n_estimators': randint(100, 1000)
        # },
        # {
        #     'learning_rate': uniform(0.0001, 0.1),
        #     'n_estimators': randint(100, 1000),
        #     'max_depth': randint(2, 32),
        #     'min_samples_split': randint(2, 11),
        #     'min_samples_leaf': randint(2, 11),
        #     'max_features': randint(1, input_size)
        # },
        {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(2, 32),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(2, 11),
            'max_features': randint(1, input_size)
        }
    ]
    regressors = [
        #[xgb.XGBRegressor(objective='reg:squarederror') for __ in range(output_size)],
        # [Ridge(random_state=random_seed) for __ in range(output_size)],
        # [Lasso(random_state=random_seed) for __ in range(output_size)],
        # [ElasticNet(random_state=random_seed) for __ in range(output_size)],
        # [AdaBoostRegressor(random_state=random_seed) for __ in range(output_size)],

        #[GradientBoostingRegressor(random_state=random_seed) for __ in range(output_size)],
        [RandomForestRegressor(random_state=random_seed, n_jobs=-1) for __ in range(output_size)]
    ]

    errors = np.empty((len(regressors), output_size))
    variances = np.empty((len(regressors), output_size))
    hyperopts = np.empty((len(regressors), output_size), dtype=object)
    for reg_idx in range(len(regressors)):
        print('regressor {}'.format(reg_idx))
        for out_idx in range(output_size):
            print('output {}'.format(out_idx))
            target = train[:, -output_size + out_idx]
            start = time.time()
            rand_search = RandomizedSearchCV(
                regressors[reg_idx][out_idx],
                param_distributions=param_dicts[reg_idx],
                n_iter=n_iter_search,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            rand_search.fit(train[:, :input_size], target)
            end = time.time()
            print('time: {}'.format(end - start))
            hyperopts[reg_idx, out_idx] = rand_search

            # Test each FRF separately
            local_error = 0
            local_std = 0
            for test_scenario in test:
                pred = rand_search.predict(test_scenario[:, :input_size])
                local_error += math.sqrt(mean_squared_error(test_scenario[:, -output_size + out_idx], pred)) * 100.0
                local_std += np.std(
                    [
                        abs(test_scenario[idx, -output_size + out_idx] - pred[idx])
                        for idx in range(len(test_scenario))
                    ]
                ) * 100.0
                import matplotlib.pyplot as plt
                plt.clf()
                plt.plot(test_scenario[:, 3], pred, color='red')
                plt.plot(test_scenario[:, 3], test_scenario[:, -output_size + out_idx], color='green')
                plt.savefig('../output/fig_{}_{}_{}.png'.format(test_scenario[0,2], reg_idx, out_idx), dpi=600)
            errors[reg_idx, out_idx] = local_error / len(test)
            variances[reg_idx, out_idx] = local_std / len(test)
            end2 = time.time()
            print('time for test: {}'.format(end2 - end))
    return regressors, errors, variances, hyperopts

def evaluate(regressors, test, input_size, output_size):
    """Method to evaluate trained regressors"""
    errors = np.empty((len(regressors), output_size))
    variances = np.empty((len(regressors), output_size))
    for reg_idx in range(len(regressors)):
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
            "NRMSE XX\t"
            "NRMSE YY\n"
        )
        for reg_idx, regressor in enumerate(regressors):
            res_file.write(
                "{0}\t\t"
                "{1:.2f} % +/- {2:.2f} %\t"
                "{3:.2f} % +/- {4:.2f} %\n".format(
                    regressor[0].__class__.__name__,
                    errors[reg_idx, 0], variances[reg_idx, 0],
                    errors[reg_idx, 1], variances[reg_idx, 1]
                )
            )

def main():
    """Main method"""
    # Definition of data and learning properties
    data_dir = '../data/02_processed'
    test_size = 0.1
    cv_folds = 10
    n_iter_search = 500
    random_seed = 1234
    data = np.load('{}/processed_data.npy'.format(data_dir))
    input_size = 4
    output_size = 2

    # Train/test split
    train, test = train_test_split(data, test_size=test_size, random_state=random_seed)

    # Flatten training data by one dimension but keep the shape of the testing data,
    # to being able to test different FRFs separately
    train = np.reshape(train, (-1, input_size + output_size))
    # Scale data
    x_scaler = MinMaxScaler()
    train[:, :input_size] = x_scaler.fit_transform(train[:, :input_size])
    for test_idx, __ in enumerate(test):
        test[test_idx, :, :input_size] = x_scaler.transform(test[test_idx, :, :input_size])

    regressors, errors, variances, hyperopts = learn(
        train,
        test,
        input_size,
        output_size,
        cv_folds,
        n_iter_search,
        random_seed
    )
    # Save results in output folder
    write_results(regressors, errors, variances, "../output/results.txt")
    for hyp_idx, hyperopt in enumerate(hyperopts):
        dump(
            hyperopt,
            '../models/hyperopt_{}.joblib'.format(regressors[hyp_idx].__class__.__name__)
        )

if __name__ == '__main__':
    misc.to_local_dir('__file__')
    main()
