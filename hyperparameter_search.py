import os
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn



def main():
    # wine_path = os.read("/content/mlflow/examples/r_wine/wine-quality.csv")
    data = pd.read_csv("./winequality-red.csv")
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    # iterating the columns
        # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    param_grid = {
        'alpha': [0.05, .1, 0.5, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 1.0]
    }
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():
        # GridSearchCV with ElasticNet
        grid_search = GridSearchCV(ElasticNet(random_state=65), param_grid, scoring='neg_mean_squared_error')

        # Fit the GridSearchCV on training data
        grid_search.fit(train.drop("quality", axis=1), train["quality"])

        # Get the best hyperparameters
        best_alpha = grid_search.best_params_['alpha']
        best_l1_ratio = grid_search.best_params_['l1_ratio']

        print("Best hyperparameters:")
        print("  alpha: %s" % best_alpha)
        print("  l1_ratio: %s" % best_l1_ratio)

        mlflow.log_metric("best_alpha", best_alpha)
        mlflow.log_metric("best_l1_ratio", best_l1_ratio)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    main()
