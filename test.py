
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
    data = pd.read_csv("./winequality-red.csv")
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    model_name = sys.argv[3]
    model_version = sys.argv[4]
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run():
        model = load_and_use_model(model_name, model_version, test_x, test_y, alpha, l1_ratio)
        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


def load_and_use_model(model_name, model_version, test_x, test_y, alpha, l1_ratio):
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return model

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    main()
