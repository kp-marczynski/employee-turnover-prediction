from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, mean_squared_log_error, make_scorer
from sklearn.feature_selection import SelectFromModel
import numpy as np
# import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor
from numpy import sort
from preprocessor import convert_to_classification
import shap

estimator_params = {
    'tree_method': "hist",
    'single_precision_histogram': True,
    'n_jobs': -1,
    'n_estimators': 1500,
    'importance_type': 'gain',
    'use_label_encoder': False,
    'booster': 'gbtree',
    # 'scale_pos_weight': 9,
    'reg_alpha': 5,
    'colsample_bytree': .8,
    'learning_rate': .05,
    'min_child_weight': 100,
    'subsample': 0.7,
    'max_depth': 20,
    'gamma': 0.01,
    # 'reg_lambda': 1,  # 100
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


def get_data(year):
    return pd.read_csv(f'data/{year}_pandas.csv')


def fit():
    years = [2017, 2018, 2019]
    for year in years:
        # for var in regression_dependent_variables:
        #     fit_model(get_data(year), var, False, f'{year}_{var}', year)
        for var in class_dependent_variables:
            fit_model(get_data(year), var, True, f'{year}_{var}', year)


def fit_model(data, dependent_variable, is_classifier, name, year):
    print(f'Fitting for {name}')
    estimator = get_estimator(is_classifier)

    X = data.drop(regression_dependent_variables + class_dependent_variables, axis=1)
    y = data[dependent_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    estimator.fit(X_train, y_train)

    selection = SelectFromModel(estimator, threshold=.01, prefit=True, max_features=10)
    feature_idx = selection.get_support()
    feature_names = X.columns[feature_idx]
    # print(feature_names)
    select_X_train = selection.transform(X_train)
    print(f'Selected {select_X_train.shape[1]} features')
    # train model
    selection_model = get_estimator(is_classifier)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)

    preds = selection_model.predict(select_X_test)
    title = ''
    if is_classifier:
        title = "Accuracy: %.2f" % accuracy_score(y_test, preds) + ", Precision: %.2f" % precision_score(y_test, preds)+ ", Recall: %.2f" % recall_score(y_test, preds)
        # print(confusion_matrix(y_test, preds))
        # print(classification_report(y_test, preds))
    else:
        title = "RMSLE: %.2f" % math.sqrt(abs(mean_squared_error(y_test, preds))) + ", R2: %.2f" % r2_score(y_test,
                                                                                                            preds)

    # xgb.plot_importance(estimator, max_num_features=10, importance_type='gain')
    # values = list(selection_model.get_booster().get_score(importance_type='gain').values())
    # keys= list(feature_names)
    # result = pd.DataFrame(data=values, index=keys, columns=["score"])
    # lst = list()
    # short_keys = list(map(lambda x: x[:30], keys))
    # for x in range(len(values)):
    #     lst.append("("+f"{values[x]:.2f}"+",{"+short_keys[x]+"})")
    # print("\\bargraph{"+(", ".join(list(map(lambda x: "{"+x+"}", short_keys)))) + "}{"+(" ".join(lst))+"}{feat_importance_"+name+"}{feat_importance_"+name+"}{10cm}")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
    # print("")
    # print(result)
    # result.plot(kind='barh')
    plt.tight_layout()
    plt.xlabel('mean(|SHAP values|)')
    plt.title('Year: ' + str(year) + ', Depended variable: ' + dependent_variable)
    plt.suptitle(title)
    plt.savefig(f'feat_importance/feat_importance_{name}.png', bbox_inches='tight')


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
