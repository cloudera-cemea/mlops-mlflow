# fetches best model from experimemt runs and pushes it to the registry

import mlflow

EXPERIMENT_NAME = "sklearn_experiment_lab_2"
MODEL_ARTIFACT_LOCATION = "logistic_regression"
MODEL_NAME = "sklearn_model"

mlflow.set_experiment(EXPERIMENT_NAME)

run_results = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
best_run_id = run_results.loc[run_results["metrics.accuracy"].idxmax(), "run_id"]

registered_model = mlflow.register_model(f"runs:/{best_run_id}/{MODEL_ARTIFACT_LOCATION}", MODEL_NAME)
