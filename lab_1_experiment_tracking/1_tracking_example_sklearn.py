"""
Example script demonstrating MLflow experiment tracking with scikit-learn models.
This script:
1. Sets up an MLflow experiment
2. Trains multiple Logistic Regression models with different hyperparameters
3. Logs parameters, metrics, and models to MLflow
4. Creates model signatures for better model serving capabilities

Based on the [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)
"""

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Define MLflow Experiment - this creates a named experiment in MLflow UI
mlflow.set_experiment("sklearn_experiment_lab_1")

# Define different sets of hyperparameters to test
hyperparameter_sets = [
    {"solver": "liblinear", "C": 0.1},  # Low regularization
    {"solver": "liblinear", "C": 1.0},  # Medium regularization
    {"solver": "liblinear", "C": 10.0}  # High regularization
]

# Run experiments with different hyperparameter sets
for params in hyperparameter_sets:
    # Start a new MLflow run for each hyperparameter set
    with mlflow.start_run():
        # Load and split the Iris dataset
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model with current hyperparameters
        model = LogisticRegression(solver=params["solver"], C=params["C"])
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # Log hyperparameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Create a model signature that describes the model's input and output schema
        # This is important for model serving and validation
        # The signature helps MLflow understand what inputs the model expects and what outputs it produces
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log the model to MLflow with its signature and an example input
        # This enables model serving and makes it easier to understand how to use the model
        mlflow.sklearn.log_model(
            model, 
            "logistic_regression", 
            signature=signature, 
            input_example=X_train[:1]  # Provide an example input for documentation
        )

# Notify user that experiments are complete
print("Experiment runs completed. Check MLflow UI for comparison.")
