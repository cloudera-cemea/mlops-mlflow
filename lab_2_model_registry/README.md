# Lab 2: Model Registry and Deployments

## Introduction

In Lab 1 you learned how to track experiments with MLflow on Cloudera Machine Learning. This lab will take you a step further: once you have multiple models and runs, how do you keep them organized, versioned, and easy to discover? And finally, how to you expose the model to production following standards and best practices? Cloudera's **Model Registry** and **Model Deployments** solve these problems.

    experiments → model registry → deployment

This lab will teach you how to register models, store them in the Model Registry, and manage different versions systematically — and deploy models as endpoints, directly from the registry.

## Overview

- [Lab 2: Model Registry and Deployments](#lab-2-model-registry-and-deployments)
  - [Introduction](#introduction)
  - [Overview](#overview)
  - [Model Registry](#model-registry)
    - [Why Model Registry?](#why-model-registry)
    - [Best Practice 1: Version and Manage your Models via Model Registry](#best-practice-1-version-and-manage-your-models-via-model-registry)
    - [Best Practice 2: Model Registry Workflow Automation](#best-practice-2-model-registry-workflow-automation)
    - [Model Registry Additional Features](#model-registry-additional-features)
      - [Registering a model via the MLflow SDK](#registering-a-model-via-the-mlflow-sdk)
      - [Interacting with the Model Registry via the Cloudera API](#interacting-with-the-model-registry-via-the-cloudera-api)
  - [Model Deployments](#model-deployments)
    - [Why Model Deployments?](#why-model-deployments)
    - [Best Practice 1: Model Deployments from Model Registry](#best-practice-1-model-deployments-from-model-registry)
    - [Best Practice 2: Model Deployment Workflow Automation](#best-practice-2-model-deployment-workflow-automation)
  - [Summary](#summary)

## Model Registry

### Why Model Registry?

When collaborating on machine learning projects, it's easy to lose track of which model is the "latest" or "best." You might end up with `model_v1.pkl`, `model_v2.pkl`, and `model_latest.pkl` files scattered in different folders, with little to no documentation on origin and differences between those files. Model Registry is the next logical step in the MLOps pipeline:

    experiments → model registry

> [!Tip]
> Benefits:
>
> - ✅ **Centralized Storage**: A single source of truth for all models in your project.  
> - ✅ **Automatic Versioning**: Each new model or model update is assigned a unique version number.  
> - ✅ **Traceability**: Every model can be linked back to the experiment run (metrics, hyperparameters, etc.) that produced it.

### Best Practice 1: Version and Manage your Models via Model Registry

In this scenario, a model is trained and pushed the model to the registry manually through the Experiments User Interface.

1. Create an experiment, e.g. via the Python script [`1_model_registry_create_model.py`](./1_model_registry_create_model.py)
2. Register the logged model to the Registry via the User Interface:

![experiments user interface](/images/experiments-user-interface.png)

3. Browse the model details in the Registry User Interface:

![registry user interface](/images/registry-user-interface.png)

> [!Tip]
> Benefits:
>
> - ✅ You can now see all its versions (including the one just registered), details like creation time, run ID, and metadata logged by MLflow.
> - ✅ Other team members can also view this registry entry, download artifacts, or further update the model version as needed.

### Best Practice 2: Model Registry Workflow Automation

These workflows can (and should, depending on the requirements and complexity of your use case) also be automated. Following example demonstrates how the simple Model Registry workflow from above can be automated using the MLflow SDK and Cloudera Machine Learning Jobs and Pipelines.

Example workflow:

1. An initial Job creates experiments/runs and logs model artifacts, e.g. [`1_model_registry_create_model.py`](./1_model_registry_create_model.py).

```python
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
    ...
    mlflow.sklearn.log_model(model, MODEL_ARTIFACT_LOCATION, signature=signature, input_example=X_train[:1])
```

2. A subsequent Job retrieves the best performing model and pushes it to the registry, e.g. [`2_model_registry_push_to_registry.py`](./2_model_registry_push_to_registry.py).

```python
# Search for all runs in the experiment and find the best one based on accuracy
run_results = mlflow.search_runs(search_all_experiments=True)
best_run_id = run_results.loc[run_results["metrics.accuracy"].idxmax(), "run_id"]

# Register the best model in the model registry
# This creates a new version of the model in the registry
registered_model = mlflow.register_model(f"runs:/{best_run_id}/{MODEL_ARTIFACT_LOCATION}", "sklearn_model")
```

![registry workflow](/images/registry-workflow.png)

- Cloudera Documentation for Jobs and Pipelines: <https://docs.cloudera.com/machine-learning/1.5.3/jobs-pipelines/topics/ml-creating-a-job-c.html>

### Model Registry Additional Features

Cloudera Machine Learning supports different ways to interact with the Model Registry. This section covers different examples via MLflow SDK and Cloudera APIs.

#### Registering a model via the MLflow SDK

- Example for registering a model via the `mlflow.log_model` API:

```python
mlflow.sklearn.log_model(model, "sklearn_model", registered_model_name="sklearn_model")
```

> [!Note]
> This will also create a new experiment (or new experiment run, if an experiment witht the same name already exists).

- Example for registering a model via the `mlflow.register_model` API:

```python
registered_model = mlflow.register_model(f"runs:/{best_run_id}/{model_artifact_location}", "sklearn_model")
```

> [!Note]
> This will **not** create a new experiment or experiment run.

#### Interacting with the Model Registry via the Cloudera API

The Cloudera API extends MLflow's capabilities, enabling sophisticated MLOps workflows through automation. Note that some Model Registry APIs require Data Services version 1.5.4 or higher. For detailed API documentation, visit the [Cloudera API Reference](https://docs.cloudera.com/machine-learning/1.5.3/api/topics/ml-api-v2.html).

- Example for registering a model:

```python
import cmlapi
import os

workspace_domain = os.getenv("CDSW_DOMAIN")
client = cmlapi.default_client(url=f"https://{workspace_domain}")

CreateRegisteredModelRequest = {
    "project_id": "<project-id>", 
    "experiment_id" : "<experiment-id>",
    "run_id": "<run-id>", 
    "model_name": "<model-name>", 
    "model_path": "<model-artifact-uri>", 
    "visibility": "PUBLIC"
}

api_response = client.create_registered_model(CreateRegisteredModelRequest)
```

- Example for retrieving metadata from a registered model

```python
import os
import json
import cmlapi

# Set up client
workspace_domain = os.getenv("CDSW_DOMAIN")
client = cmlapi.default_client(url=f"https://{workspace_domain}")

# Retrieve model ID by model name
search_filter = {"model_name": "sklearn_model"}
response_dict = client.list_registered_models(search_filter=json.dumps(search_filter)).to_dict()
model_id = response_dict["models"][0]["model_id"]
print(f"Model ID: {model_id}")

# Retrieve details for the latest model version
model_details = client.get_registered_model(model_id).to_dict()
latest_version = model_details["model_versions"][0]
print(f"Latest Model Version: {latest_version}")

# Extract experiment ID and run ID from the latest model version
mlflow_meta = latest_version["model_version_metadata"]["mlflow_metadata"]
experiment_id = mlflow_meta["experiment_id"]
history_tag = next(
    tag["value"]
    for tag in mlflow_meta["tags"]
    if tag["key"] == "mlflow.log-model.history"
)
run_id = json.loads(history_tag)[0]["run_id"]

print(f"Experiment ID: {experiment_id}")
print(f"Run ID: {run_id}")
```

## Model Deployments

### Why Model Deployments?

The Model Registry is naturally complemented by Model Deployments, which allow registered models to be served as APIs for real-time inference or batch processing. Once a model is stored in the registry, it can be deployed in Cloudera Machine Learning either via the User Interface (UI) or programmatically using the Cloudera API/SDK.

Cloudera Machine Learning automatically handles containerization, scaling, and monitoring, making it easy to integrate models into real-time applications and business workflows.

> [!Tip]
> Benefits:
>
> - ✅ Standardized deployments as REST APIs
> - ✅ Built-in security with secret key authentication and Cloudera governance
> - ✅ Built-in monitoring capabilities (see lab 3 for more details)

### Best Practice 1: Model Deployments from Model Registry

Models can be deployed directly from the registry, ensuring version control and traceability throughout the deployment process. This creates a natural flow from experimentation to production:

    experiments → model registry → deployment

Expanding on the pipeline example from [Lab #2](../lab_2_model_registry/README.md#good-example-2-automation-with-cloudera-machine-learning-jobs-and-pipelines), we can add a third step to the pipeline to deploy the registered model as an endpoint. The script [`deploy_from_registry.py`](./deployment/deploy_from_registry.py) retrieves the latest version of the registered model and deploys it:

1. First the model metadata is retrieved from the Model Registry:

```python
# Search through registered models to find the one matching our MODEL_NAME
response_dict = cml_client.list_registered_models().to_dict()
model_id = next((model["model_id"] for model in response_dict["models"] if model["name"] == MODEL_NAME), None)
print(f"Model ID: {model_id}")

...
```

2. Then the model deployment is triggered via the Cloudera API:

```python
import cmlapi

# Set up client
workspace_domain = os.getenv("CDSW_DOMAIN")
cml_client = cmlapi.default_client(url=f"https://{workspace_domain}")

# Create and configure the model deployment
# Abbreviated, full examples in 3_deploy_from_registry.py/ipynb
CreateModelRequest = {
    "project_id": os.getenv("CDSW_PROJECT_ID"), 
    "name" : MODEL_NAME,
    "description": f"Production model deployment for model name: {MODEL_NAME}",
    "registered_model_id": model_id,
    ...
}

model_api_response = cml_client.create_model(CreateModelRequest, os.getenv("CDSW_PROJECT_ID"))
...
```

### Best Practice 2: Model Deployment Workflow Automation

Similar to the Registry workflows described above, deployment workflows can (and should, depending on the requirements and complexity of your use case) be automated. Following example demonstrates how the simple Model Registry workflow from above can be extended by a final deployment step, using the Cloudera APIs and Cloudera Machine Learning Jobs and Pipelines.

Extended example workflow:

1. An initial Job creates experiments/runs and logs model artifacts, e.g. [`1_model_registry_create_model.py`](./1_model_registry_create_model.py).

2. A subsequent Job retrieves the best performing model and pushes it to the registry, e.g. [`2_model_registry_push_to_registry.py`](./2_model_registry_push_to_registry.py).

3. A final Job retrieves the latest version of the registered model and deploys it as a prediction service: [`3_deploy_from_registry.py`](./3_deploy_from_registry.py)

![registry workflow](/images/deployment-workflow.png)

## Summary

The Model Registry and Deployments provide a seamless pipeline from experimentation to production by enabling centralized model versioning and scalable API deployments.

Proceed to the next lab: [Monitoring](../lab_3_monitoring/README.md).
