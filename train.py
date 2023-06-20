# Run from the root of MLflow
# Read the wine-quality csv file
import os
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
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

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run():
        # (rmse, r2, mae, model) = load_and_use_model(test_x, test_y, alpha, l1_ratio)
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def tracking():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_id = "0"
    runs = mlflow.search_runs(experiment_ids=experiment_id)
    df = pd.DataFrame(runs)
    print(df.head())

def load_and_use_model(test_x, test_y, alpha, l1_ratio):
    model_name = "Model A"
    model_version = 1

    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
    predicted_qualities = model.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    return rmse, r2, mae, model

if __name__ == "__main__":
    main()