from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
# import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor
from numpy import sort
from preprocessor import convert_to_classification

estimator_params = {
    'tree_method': "hist",
    'single_precision_histogram': True,
    'n_jobs': -1,
    'n_estimators': 1500,
    'importance_type': 'weight',
    'use_label_encoder': False,
    'booster': 'gbtree',
    'scale_pos_weight': 1,
    'reg_alpha': 5,
    'colsample_bytree': .8,
    'learning_rate': .1,
    'min_child_weight': 7,
    'subsample': .5,
    'max_depth': 6,
    'gamma': 0.1,
    'reg_lambda': 1,  # 100
}

# is_classifier = True


descriptive_param = 'JobSeekingStatus_class'

regression_dependent_variables = ['JobSeekingStatus', 'JobSatisfaction']
class_dependent_variables = ['JobSeekingStatus_class', 'JobSatisfaction_class']


def get_data(year):
    return pd.read_csv(f'data/{year}_pandas.csv')


def fit():
    years = [2017, 2018, 2019]
    for year in years:
        for var in regression_dependent_variables:
            fit_model(get_data(year), var, False, f'{year}_{var}')
        for var in class_dependent_variables:
            fit_model(get_data(year), var, True, f'{year}_{var}')


def fit_model(data, dependent_variable, name):
    def fit_estimator(x_set, y_set):
        estimator = xgb.XGBRegressor(**estimator_params, eval_metric='rmsle')
        estimator.fit(x_set, y_set)
        return estimator

    x = data.drop(regression_dependent_variables + class_dependent_variables, axis=1)
    y = data[dependent_variable]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

    selection = SelectFromModel(fit_estimator(x_train, y_train), threshold=.015, prefit=True)

    selection_estimator = fit_estimator(selection.transform(x_train), y_train)
    preds = selection_estimator.predict(selection.transform(x_test))
    print("RMSE: %.2f" % math.sqrt(abs(mean_squared_error(y_test, preds))))
    print("R2: %.2f" % r2_score(y_test, preds))

    xgb.plot_importance(selection_estimator, max_num_features=10)
    plt.tight_layout()
    plt.savefig(f'data/feat_importance_{name}.png')


# def search_params(estimator, x_data, y_data):
#     searcher = GridSearchCV(
#         estimator,
#         {
#             # 'reg_alpha': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
#             # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 0.75, .85, 1],
#             # 'gamma': [0, .001, .6, 0.1, 0.2, 1, 2],
#             # 'learning_rate': [.01, 0.1, 0.005, .05, 0.3, 0.5, 0.7],
#             # 'min_child_weight': [1,2,3,4,5,7,8, 9],
#             # 'subsample': [0.6, 0.7, 0.8, 0.9, .55, 1],
#             # 'max_depth': [3, 4, 5,6, 7,8,9],
#             # 'reg_lambda': [ 1, 10, 100, 0],
#             'importance_type': ['weight', 'gain', 'cover', 'total_gain', 'total_cover'],
#             # 'eval_metric': ['rmsle', 'rmse']
#         },
#         cv=2,
#         scoring='r2',  # neg_mean_squared_error , roc_auc or r2
#         verbose=10,
#         n_jobs=-1
#     )
#
#     searcher.fit(x_data, y_data)
#     print(searcher.best_params_)
#     print(math.sqrt(abs(searcher.best_score_)))


if __name__ == '__main__':
    fit()