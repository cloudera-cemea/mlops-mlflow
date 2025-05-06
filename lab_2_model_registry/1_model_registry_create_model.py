"""
Script to create and log multiple model versions with different hyperparameters to MLflow.
This script:
1. Sets up an MLflow experiment
2. Trains multiple Logistic Regression models with different regularization strengths
3. Logs parameters, metrics, and models to MLflow
4. Creates model signatures for better model serving capabilities
"""

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Experiment and model configuration
EXPERIMENT_NAME = "sklearn_experiment_lab_2"
MODEL_ARTIFACT_LOCATION = "logistic_regression"

# Set up MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Define different sets of hyperparameters to test
# C is the inverse of regularization strength - smaller values specify stronger regularization
hyperparameter_sets = [
    {"solver": "liblinear", "C": 0.1},  # Strong regularization
    {"solver": "liblinear", "C": 0.5},  # Medium-strong regularization
    {"solver": "liblinear", "C": 1.0},  # Medium regularization
    {"solver": "liblinear", "C": 5.0},  # Medium-weak regularization
    {"solver": "liblinear", "C": 10.0}  # Weak regularization
]

# Train and log models with different hyperparameters
for params in hyperparameter_sets:
    with mlflow.start_run():
        # Load and prepare the Iris dataset
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model with current hyperparameters
        model = LogisticRegression(solver=params["solver"], C=params["C"])
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # Log parameters and metrics to MLflow
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
            MODEL_ARTIFACT_LOCATION, 
            signature=signature, 
            input_example=X_train[:1]  # Provide an example input for documentation
        )
