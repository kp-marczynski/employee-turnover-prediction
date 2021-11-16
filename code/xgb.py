from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_squared_log_error, make_scorer
from numpy import sort
from sklearn.feature_selection import SelectFromModel
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


def get_estimator(is_classifier):
    if is_classifier:
        return xgb.XGBClassifier(**estimator_params, eval_metric='auc')
    else:
        return xgb.XGBRegressor(**estimator_params, eval_metric='rmsle')


descriptive_param = 'JobSeekingStatus_class'

regression_dependent_variables = ['JobSeekingStatus', 'JobSatisfaction']
class_dependent_variables = ['JobSeekingStatus_class', 'JobSatisfaction_class']

environment_constants = []
dev_profile_features = []
company_features = []
person_features = []


def fit():
    years = [2017, 2018, 2019]
    for year in years:
        for var in regression_dependent_variables:
            fit_model(year, var, False)
        for var in class_dependent_variables:
            fit_model(year, var, True)


def fit_model(year, dependent_variable, is_classifier):
    print(f'Fitting for {dependent_variable} over year {year}')
    data = pd.read_csv(f'data/{year}_pandas.csv')

    # if is_classifier:
    #     data = convert_to_classification(data)

    estimator = get_estimator(is_classifier)

    X = data.drop(regression_dependent_variables + class_dependent_variables, axis=1)
    y = data[descriptive_param]
    # search_params(estimator, X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    # ens_model = AdaBoostRegressor(base_estimator=estimator)
    estimator.fit(X_train, y_train)

    selection = SelectFromModel(estimator, threshold=.015, prefit=True)
    select_X_train = selection.transform(X_train)
    print(f'Selected {select_X_train.shape[1]} features')
    # train model
    selection_model = get_estimator(is_classifier)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)

    preds = selection_model.predict(select_X_test)
    if is_classifier:
        print("Accuracy: %.2f" % accuracy_score(y_test, preds))
    else:
        print("RMSE: %.2f" % math.sqrt(abs(mean_squared_error(y_test, preds))))
        # print("RMSLE: %.2f" % math.sqrt(abs(mean_squared_log_error(y_test, preds))))
        print("R2: %.2f" % r2_score(y_test, preds))

    xgb.plot_importance(estimator, max_num_features=10)
    plt.tight_layout()
    plt.savefig(f'data/feat_importance_{year}_{dependent_variable}.png')


def search_params(estimator, x_data, y_data):
    searcher = GridSearchCV(
        estimator,
        {
            # 'reg_alpha': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
            # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 0.75, .85, 1],
            # 'gamma': [0, .001, .6, 0.1, 0.2, 1, 2],
            # 'learning_rate': [.01, 0.1, 0.005, .05, 0.3, 0.5, 0.7],
            # 'min_child_weight': [1,2,3,4,5,7,8, 9],
            # 'subsample': [0.6, 0.7, 0.8, 0.9, .55, 1],
            # 'max_depth': [3, 4, 5,6, 7,8,9],
            # 'reg_lambda': [ 1, 10, 100, 0],
            'importance_type': ['weight', 'gain', 'cover', 'total_gain', 'total_cover'],
            # 'eval_metric': ['rmsle', 'rmse']
        },
        cv=2,
        scoring='r2',  # neg_mean_squared_error , roc_auc or r2
        verbose=10,
        n_jobs=-1
    )

    searcher.fit(x_data, y_data)
    print(searcher.best_params_)
    print(math.sqrt(abs(searcher.best_score_)))


if __name__ == '__main__':
    fit()
