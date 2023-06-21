# mlflow

# Hyperparameter search
```mlflow run . --env-manager=local -P file_name='hyperparameter_search.py'```

# model training
### use alpha and l1_ratio from hyperparameter search

```mlflow run . --env-manager=local -P file_name='train.py' -P alpha=0.05 -P l1_ratio=0.1```

# model testing
```mlflow run . --env-manager=local -P file_name='test.py' -P model_name='Model A' -P  model_version=1```