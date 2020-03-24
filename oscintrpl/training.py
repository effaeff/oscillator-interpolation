"""Learning routine to model the amplitudes of FRFs based on pose and frequency information"""

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV

from properties import (
    regressors,
    param_dicts,
    input_size,
    output_size,
    cv_folds,
    n_iter_search,
    random_seed
)


def training(train_data): 
    """Learning method"""

    hyperopts = np.empty((len(regressors), output_size), dtype=object)
    for reg_idx in tqdm(range(len(regressors))):
        for out_idx in range(output_size):
            target = train_data[:, input_size + out_idx]
            rand_search = RandomizedSearchCV(
                regressors[reg_idx][out_idx],
                param_distributions=param_dicts[reg_idx],
                n_iter=n_iter_search,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            rand_search.fit(train_data[:, :input_size], target)
            hyperopts[reg_idx, out_idx] = rand_search
    
    return hyperopts

