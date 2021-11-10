from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    years = [2017, 2018, 2019]
    for year in years:
        data = pd.read_csv(f'data/{year}_pandas.csv')
        searcher = GridSearchCV(xgb.XGBRegressor(tree_method="hist", single_precision_histogram=True), {
            'max_depth': [4],  # tested 2,3,4,5,7,9
            'min_child_weight': [7],  # tested 1,3,5,6,7,8
            'gamma': [0],  # tested 0,0.1,0.2,1,2
            'n_estimators': [1000],  # todo 1500
            'subsample': [0.8],  # tested 0.6,0.7,0.8,0.9
            'colsample_bytree': [0.6],  # tested 0.5,0.6,0.7,0.8,0.9
            'scale_pos_weight': [1],
            'reg_alpha': [1],  # tested 1e-5, 1e-2, 0.1, 1, 10, 100
            'reg_lambda': [100],  # tested 1e-5, 1e-2, 0.1, 1, 10, 100
            'learning_rate': [0.01],
        }, cv=3, scoring='neg_mean_squared_error', verbose=10, n_jobs=4)

        searcher.fit(data.drop(['JobSatisfaction', 'CareerSatisfaction'], axis=1), data['JobSatisfaction'])
        # print(searcher.best_params_)
        print(math.sqrt(abs(searcher.best_score_)))
        xgb.plot_importance(searcher.best_estimator_, max_num_features=10)
        plt.tight_layout()
        plt.savefig(f'data/feat_importance_{year}.png')
        # print(searcher.best_params_)
        # print(searcher.best_score_)

        # X = data.drop(['JobSatisfaction'], axis=1)
        # y = data['JobSatisfaction']
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        # ens_model = AdaBoostRegressor(base_estimator=searcher)
        # ens_model.fit(X_train, y_train)
        # preds = ens_model.predict(X_test)
        # mean_squared_error(preds, y_test)
        # print(f'RMSE: {math.sqrt(abs(mean_squared_error(preds, y_test)))}')
        # print(f'R2: {r2_score(preds, y_test)}')


if __name__ == '__main__':
    main()
