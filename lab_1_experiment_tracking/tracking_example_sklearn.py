# Based on the [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)

import flask

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Define MLflow Experiment
mlflow.set_experiment("sklearn_experiment_lab_1")

# Define Model and Hyperparameters
hyperparameter_sets = [
    {"solver": "liblinear", "C": 0.1},
    {"solver": "liblinear", "C": 1.0},
    {"solver": "liblinear", "C": 10.0}
]

# Run Experiments and define mlflow tracking
for params in hyperparameter_sets:
    with mlflow.start_run():
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(solver=params["solver"], C=params["C"])
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "logistic_regression", signature=signature, input_example=X_train[:1])

# Compare Experiment Results
print("Experiment runs completed. Check MLflow UI for comparison.")
