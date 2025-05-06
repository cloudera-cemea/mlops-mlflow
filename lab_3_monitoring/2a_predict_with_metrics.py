"""
Script demonstrating model monitoring and metrics tracking in production.
This script:
1. Loads the best performing model from MLflow
2. Sets up a prediction function with metrics tracking
3. Demonstrates how to track input and output metrics for monitoring

The metrics tracking system allows for:
- Monitoring model performance in production
- Tracking input/output distributions
- Detecting data drift
- Analyzing prediction patterns
"""

import mlflow
import pandas as pd

import cml.metrics_v1 as metrics
import cml.models_v1 as models

# Configuration for experiment and model
EXPERIMENT_NAME = "sklearn_experiment_lab_2"
MODEL_ARTIFACT_LOCATION = "logistic_regression"

# Load the best performing model from MLflow based on accuracy
run_results = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
best_run_artifact_uri = run_results.loc[run_results["metrics.accuracy"].idxmax(), "artifact_uri"]
model = mlflow.pyfunc.load_model(f"{best_run_artifact_uri}/{MODEL_ARTIFACT_LOCATION}")
print(f"Loaded best model from artifacts location: {best_run_artifact_uri}")

# The model_metrics decorator equips the predict function to
# call track_metrics. It also changes the return type. If the
# raw predict function returns a value "result", the wrapped
# function will return eg
# {
#   "uuid": "612a0f17-33ad-4c41-8944-df15183ac5bd",
#   "prediction": "result"
# }
# The UUID can be used to query the stored metrics for this
# prediction later.
@models.cml_model(metrics=True)
def predict(inputs: pd.DataFrame):
    """
    Make predictions with the model while tracking metrics.
    
    Args:
        inputs: DataFrame containing input features
        
    Returns:
        Dictionary containing:
        - prediction: The model's prediction
        - uuid: Unique identifier for tracking this prediction's metrics
    """
    # Track the input features for monitoring
    metrics.track_metric("input", inputs["inputs"][0])

    # Run model inference
    result = model.predict(inputs)

    # Track the prediction output
    metrics.track_metric("output", str(result[0]))

    return result
