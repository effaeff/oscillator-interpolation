"""Definition of project-global variables"""

# Global colors
from colors import dark2

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
import xgboost as xgb
from scipy.stats import uniform, randint

HELLER = True
OSC = True
n_iter_search = 100
MERHOFE = True
freq_steps_aggreg = 2

test_configs = (
    [
        # [-50.0, 500.0, -60.0],
        # [-50.0, 500.0, 0.0],
        # [-200.0, 300.0, 0.0],
        # [-50.0, 100.0, 10.0]
        # [-50.0, 500.0, -45.0],
        # [-200.0, 300.0, -45.0],
        # [-350.0, 100.0, -45.0],
        # [-350.0, 500.0, -45.0]
        # [-50.0, 100.0, -30.0],
        # [-50.0, 500.0, -30.0],
        # [-200.0, 300.0, -30.0],
        # [-350.0, 100.0, -30.0],
        # [-350.0, 500.0, -30.0]
        [-50.0, 100.0, -20.0],
        [-50.0, 500.0, -20.0],
        [-200.0, 300.0, -20.0],
        [-350.0, 100.0, -20.0],
        [-350.0, 500.0, -20.0]
    ] if not HELLER else
    [
#        [-266.66, 233.33, 0.0],
#        [266.66, 233.33, -150.0],
#        [-266.66, 233.33, -30.0],
#        [0.0, 500.0, -90.0]
       [-266.66, 233.33, -30.0],
       [-266.66, 766.66, -30.0],
       [0.0, 500.0, -30.0],
       [266.66, 233.33, -30.0],
       [266.66, 766.66, -30.0]
#        [-266.66, 233.33, -60.0],
#        [-266.66, 766.66, -60.0],
#        [0.0, 500.0, -60.0],
#        [266.66, 233.33, -60.0],
#        [266.66, 766.66, -60.0]
        # [-266.66, 233.33, -120.0],
        # [-266.66, 766.66, -120.0],
        # [0.0, 500.0, -120.0],
        # [266.66, 233.33, -120.0],
        # [266.66, 766.66, -120.0]
    ]
)

# Data properties
data_dir = (
    '../data/01_raw/heller_ft4000/frfs' if HELLER and not OSC else
    '../data/01_raw/heller_ft4000/oscis' if HELLER and OSC else
    '../data/01_raw/dmg_hsc75linear/frfs' if not HELLER and not OSC else
    '../data/01_raw/dmg_hsc75linear/oscis'
)
processed_dir = '../data/02_processed'
plot_dir = '../figures'
model_dir = '../models'
results_dir = '../results'
delimiter = ' ' if HELLER else '\t'
doe_file = (
    '../data/01_raw/heller_ft4000/doe.xlsx' if HELLER
    else '../data/01_raw/dmg_hsc75linear/doe.xlsx'
)
pos_axes = ['X', 'Y'] if HELLER else ['Z', 'Y']
# Plot properties
figsize = (7, 7)
fontsize = 14
freq_step = 0.25 if HELLER else 0.5
x_range = (200, 3201, 1000) if HELLER else (1000, 3001, 500)
y_range_amp = (0, 0.9, 0.2)
y_range_phase = (-180.0, 81.0, 50.0)
# Model properties
input_size = 3 if OSC else 4
n_fitted_osc_x = (
    10 if HELLER and not MERHOFE else
    6 if not HELLER and not MERHOFE else
    5 if HELLER and MERHOFE else
    5
)
n_fitted_osc_y = (
    10 if HELLER and not MERHOFE else
    6 if not HELLER and not MERHOFE else
    4 if HELLER and MERHOFE else
    4
)
output_size = (n_fitted_osc_x + n_fitted_osc_y) * 3 if OSC else 4
test_size = 0.1
cv_folds = 10
random_seed = 1234
param_dicts = [
    {
        'learning_rate': uniform(0.0001, 0.1),
        'max_depth': randint(2, 32),
        'subsample': uniform(0.5, 0.5),
        'n_estimators': randint(100, 1000),
        'colsample_bytree': uniform(0.4, 0.6),
        'lambda': randint(1, 100),
        'gamma': uniform()
    },
    # {'alpha': uniform()},
    # {'alpha': uniform()},
    {'alpha': uniform(), 'l1_ratio': uniform()},
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'n_estimators': randint(100, 1000)
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
    [xgb.XGBRegressor(objective='reg:squarederror') for __ in range(output_size)],
    # [Ridge(random_state=random_seed) for __ in range(output_size)],
    # [Lasso(random_state=random_seed) for __ in range(output_size)],
    [ElasticNet(random_state=random_seed, tol=0.01, max_iter=100000) for __ in range(output_size)],
    # [AdaBoostRegressor(random_state=random_seed) for __ in range(output_size)],
    # [GradientBoostingRegressor(random_state=random_seed) for __ in range(output_size)],
    [RandomForestRegressor(random_state=random_seed, n_jobs=-1) for __ in range(output_size)]
]
