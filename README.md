# mlflow

# Hyperparameter search
mlflow run . --env-manager=local -P file_name='hyperparameter_search.py'

# model training
mlflow run . --env-manager=local -P file_name='train.py'   

# model testing
mlflow run . --env-manager=local -P file_name='test.py' -P model_name='Model A' -P  model_version=1