# mlflow

# Hyperparameter search
```mlflow run . --env-manager=local -P file_name='hyperparameter_search.py'```

From Github

```mlflow run https://github.com/rohan2810/mlflow.git --env-manager=local -P file_name='hyperparameter_search.py'```
# model training
### use alpha and l1_ratio from hyperparameter search

```mlflow run . --env-manager=local -P file_name='train.py' -P alpha=0.05 -P l1_ratio=0.1```

From Github

```mlflow run https://github.com/rohan2810/mlflow.git --env-manager=local -P file_name='train.py' -P alpha=0.05 -P l1_ratio=0.1```

# model testing
```mlflow run . --env-manager=local -P file_name='test.py' -P model_name='Model A' -P  model_version=1```

From Github

```mlflow run https://github.com/rohan2810/mlflow.git --env-manager=local -P file_name='test.py' -P model_name='Model A' -P  model_version=1```


# Notes:

## MLFlow:
>  experiment tracking, project management, model management/registry

## Installation:
> mlflow: pip3 install mlflow

> UI (localhost:5000): run```  mlflow ui``` on a seperate terminal (same project directory) 

## Experiment tracking:

> After data pre-processing step/before running the model(training, testing, hyperparamter), add ```mlflow.set_tracking_uri("http://127.0.0.1:5000")``` to log everything to localhost.

> Logging parameters: mlflow.log_param("var", var)

> Logging metrics:  mlflow.log_metric("var", var)

> Logging artifacts: mlflow.<library>.log_model
	
> localhost:5000 lists individual runs with params, metrics, and artifacts and models

## Project management:
> Can package MLflow projects for reusability

> Conda.yml file to list dependencies

> MlFlow file for parameters and command

#### Eg. To find best hyperparameters:

1. From local project: ```mlflow run . --env-manager=local -P file_name='hyperparameter_search.py'```


    ```'--env-manager=local'``` forces to use local env instead of creating a new conda env

2. From github: ```mlflow run https://github.com/rohan2810/mlflow.git --env-manager=local -P file_name='hyperparameter_search.py'```


## Model Registry:
> Can register models and reuse using APIs and CLI

> Versions, Stage (None, staging, production, archived)

> Models can be loaded using ```model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")```

### Eg. https://github.com/rohan2810/mlflow/blob/main/test.py#L44
