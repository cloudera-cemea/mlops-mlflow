# Lab 1: Experiment Tracking with MLflow

## Introduction

Experiment tracking is a core component of MLOps, ensuring reproducibility, collaboration, and efficiency when developing machine learning models. Without proper tracking, experiments become difficult to manage, and results may not be reproducible.

MLflow provides an easy-to-use tracking API that allows logging of parameters, metrics, and models. This lab will guide you through the importance of experiment tracking and demonstrate best practices with MLflow on Cloudera Machine Learning.

## Overview

- [Lab 1: Experiment Tracking with MLflow](#lab-1-experiment-tracking-with-mlflow)
  - [Introduction](#introduction)
  - [Overview](#overview)
  - [Why Experiment Tracking?](#why-experiment-tracking)
  - [Bad Example: No Experiment Tracking in Basic Model Training](#bad-example-no-experiment-tracking-in-basic-model-training)
  - [Best Practice 1: Using MLflow with Scikit-Learn](#best-practice-1-using-mlflow-with-scikit-learn)
  - [Best Practice 2: Using MLflow with PyTorch](#best-practice-2-using-mlflow-with-pytorch)
  - [Experiment Tracking Additional Best Practices](#experiment-tracking-additional-best-practices)
    - [Best Practice: Using Experiment Names](#best-practice-using-experiment-names)
    - [Best Practice: Autologging vs. Manual Logging](#best-practice-autologging-vs-manual-logging)
    - [Best Practice: Working with Experiment Runs](#best-practice-working-with-experiment-runs)
    - [Best Practice: Signatures and Input Examples](#best-practice-signatures-and-input-examples)
  - [Experiment Tracking Additional Features](#experiment-tracking-additional-features)
    - [Feature: Artifacts](#feature-artifacts)
    - [Feature: Traditional Machine Learning vs. Deep Learning in MLflow](#feature-traditional-machine-learning-vs-deep-learning-in-mlflow)
  - [Summary](#summary)

## Why Experiment Tracking?

Experiment tracking transforms chaotic model development into a structured, reproducible workflow by systematically recording training runs, parameters, and results. Without it, answering critical questions about model performance, hyperparameters, and reproducibility becomes nearly impossible, making it essential for building reliable machine learning systems.

Experimemt Tracking with MLflow on Cloudera Machine Learning provides:

- ✅ **Logging APIs**: For tracking experiments, parameters, metrics, and artifacts.
- ✅ **MLflow UI**: For visualizing and comparing experiments.
- ✅ **Autologging**: Automatically logs parameters and metrics from libraries like scikit-learn and PyTorch.

![MLflow Tracking Overview](https://mlflow.org/docs/latest/assets/images/tracking-basics-dd24b77b7d7b32c5829e257316701801.png)

## Bad Example: No Experiment Tracking in Basic Model Training

Below is a minimalistic example with Scikit-Learn.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
```

> [!Caution]
> There are some issues with this:
>
> - No record of hyperparameters or metrics.
> - Cannot compare multiple runs.
> - No way to reproduce the result reliably.

## Best Practice 1: Using MLflow with Scikit-Learn

Based on the [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html), we can track experiments with Scikit-Learn by making few adjustments to our code.

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create named MLflow experiment
mlflow.set_experiment("sklearn_experiment")

with mlflow.start_run():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    # Log parameters and metrics
    mlflow.log_param("solver", model.solver)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "logistic_regression")
```

> [!Tip]
> Benefits:
>
> - Experiment results are logged and stored.
> - Allows comparison across multiple runs.
> - Ensures reproducibility.

## Best Practice 2: Using MLflow with PyTorch

Based on the [MLflow Deep Learning Guide](https://mlflow.org/docs/latest/deep-learning/pytorch/guide/index.html), we can track PyTorch models similarly.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc1(x)

# Create named MLflow experiment
mlflow.set_experiment("pytorch_experiment")

with mlflow.start_run():
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Log parameters
    mlflow.log_param("learning_rate", 0.01)

    # Log model
    mlflow.pytorch.log_model(model, "simple_pytorch_model")
```

## Experiment Tracking Additional Best Practices

### Best Practice: Using Experiment Names

It is a best practice to use experiment names to organize runs. Use `mlflow.set_experiment("experiment_name")` to specify experiments before running tracking code.

> [!Warning]
> If no experiment name is given, all data will be logged to the "Default" experiment.

### Best Practice: Autologging vs. Manual Logging

MLflow supports **autologging** for various libraries, including scikit-learn and PyTorch. With autologging:

```python
mlflow.sklearn.autolog()
```

MLflow will automatically log parameters, metrics, and models without explicit calls to `mlflow.log_*` functions.

![MLflow Autologging](https://mlflow.org/docs/latest/assets/images/autologging-intro-8e1315dec6527d392563f06b36abeb56.png)

**Autologging Pros**:

- Requires minimal code changes.
- Automatically tracks model training details.

**Manual Logging Pros**:

- More control over what gets logged.
- Custom tracking of additional artifacts.

### Best Practice: Working with Experiment Runs

- Consistent Naming: Start each new run with a clear name or use with `mlflow.start_run(run_name="...")` to provide context for the experiment.
- Context Manager: The `with mlflow.start_run():` block automatically ends the run, so there's less chance of accidentally leaving runs open.
- Log Metadata: Record run IDs or timestamps as tags, e.g. `mlflow.set_tag("run_timestamp", "<timestamp>")` to keep track of experiment details.
- Browsing Experiment Runs: Browse and retrieve data from specific experiments, for example:

```python
# Fetching the best model based on accuracy
run_results = mlflow.search_runs(search_all_experiments=True)
best_run_artifact_uri = run_results.loc[run_results["metrics.accuracy"].idxmax(), "artifact_uri"]
best_model = mlflow.pyfunc.load_model(f"{best_run_artifact_uri}/logistic_regression")
print(f"Loaded best model from artifacts location: {best_model}")
```

### Best Practice: Signatures and Input Examples

- Model Signatures: Use MLflow's model signature to capture expected input shapes and data types.
- Input Examples: Provide input examples to make it easier to use the model later on.

```python
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train[:1])
```

## Experiment Tracking Additional Features

### Feature: Artifacts

- Store Additional Outputs: Log relevant plots, checkpoints, or data samples for each run:

```python
mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("sample_predictions.csv")
```

### Feature: Traditional Machine Learning vs. Deep Learning in MLflow

| Feature         | Traditional ML (e.g., Scikit-Learn) | Deep Learning (e.g., PyTorch) |
|----------------|--------------------------------|-----------------------------|
| Logging Models | `mlflow.sklearn.log_model()`  | `mlflow.pytorch.log_model()` |
| Metrics        | Accuracy, Precision, Recall   | Loss, Epoch-based metrics  |
| Training Speed | Faster                        | Computationally Intensive  |

## Summary

- Experiment tracking is essential for reproducibility.
- MLflow enables easy tracking, logging, and model management.
- Use experiment names to structure and organize your results.
- Autologging simplifies tracking but manual logging offers more flexibility.

Proceed to the next lab: [Model Registry](../lab-2-model-registry/README.md).
