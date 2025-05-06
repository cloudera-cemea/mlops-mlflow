"""
Script to register the best performing model from an MLflow experiment to the model registry.
This script:
1. Searches for the best model based on accuracy metric
2. Registers the model in MLflow's model registry
3. Makes the model available for deployment and versioning

The registered model can then be:
- Versioned
- Staged (Staging/Production)
- Deployed to serving environments
"""

import mlflow

# Configuration for experiment and model
EXPERIMENT_NAME = "sklearn_experiment_lab_2"
MODEL_ARTIFACT_LOCATION = "logistic_regression"
MODEL_NAME = "sklearn_model"

# Set up MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Search for all runs in the experiment and find the best one based on accuracy
run_results = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
best_run_id = run_results.loc[run_results["metrics.accuracy"].idxmax(), "run_id"]

# Register the best model in the model registry
# This creates a new version of the model in the registry
registered_model = mlflow.register_model(
    f"runs:/{best_run_id}/{MODEL_ARTIFACT_LOCATION}",  # Source model location
    MODEL_NAME  # Name of the model in the registry
)
