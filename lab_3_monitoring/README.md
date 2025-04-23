# Lab 3: Deployment and Monitoring Workflows

## Introduction

A complete machine learning lifecycle follows a structured pipeline:

    experiments → model registry → deployment → monitoring

Labs 1 and 2 have already covered:
- Experiment Tracking: Using MLflow to log model parameters, metrics, and artifacts.
- Model Registry: Versioning and managing models for reproducibility.

This lab chains together the previous steps and adds **Deployment** and **Monitoring** to demonstrate components of a full MLOps pipeline.

Once a model is in production, continuous monitoring is essential to ensure:
- ✅ Infrastructure scales effectively (low latency, stable request handling).
- ✅ Performance remains high (accuracy, prediction quality).
- ✅ Model drift and data shifts are detected early to prevent degradation.

## Overview

- [Lab 3: Deployment and Monitoring Workflows](#lab-3-deployment-and-monitoring-workflows)
  - [Introduction](#introduction)
  - [Overview](#overview)
  - [Model Deployment as REST endpoint from Model Registry](#model-deployment-as-rest-endpoint-from-model-registry)
  - [Problem: Lack of Visibility into Model Behavior in Production](#problem-lack-of-visibility-into-model-behavior-in-production)
  - [Solution #1: Technical Monitoring](#solution-1-technical-monitoring)
  - [Solution #2: Prediction Monitoring with Model Metrics](#solution-2-prediction-monitoring-with-model-metrics)
  - [Further Reading: CI/CD with Cloudera Machine Learning APIs and GitLab](#further-reading-cicd-with-cloudera-machine-learning-apis-and-gitlab)

## Model Deployment as REST endpoint from Model Registry

In Cloudera Machine Learning, Model Deployments allow you to serve models as scalable REST API endpoints directly from your Projects. Models can also be deployed direclty from registered models, ensuring traceability and version control. Cloudera Machine Learning automatically handles containerization, scaling, and monitoring, making it easy to integrate models into real-time applications and business workflows.

    experiments → model registry → deployment

Expanding on the pipeline example from [Lab #2](../lab_2_model_registry/README.md#good-example-2-automation-with-cloudera-machine-learning-jobs-and-pipelines), we can add a third step to the pipeline to deploy the registered model as an endpoint. The script [`deploy_from_registry.py`](./deployment/deploy_from_registry.py) retrieves the latest version of the registered model and deploys it:

1. First the model metadata is retrieved from the Model Registry:

```python
# Retrieve model id by model name
search_filter = {"model_name": MODEL_NAME}
response_dict = cml_client.list_registered_models(search_filter=json.dumps(search_filter)).to_dict()
model_id = response_dict["models"][0]["model_id"]
print(f"Model ID: {model_id}")
```

2. Then the model deployment is triggered via the Cloudera API:

```python
import cmlapi

# Set up client
workspace_domain = os.getenv("CDSW_DOMAIN")
cml_client = cmlapi.default_client(url=f"https://{workspace_domain}")

CreateModelRequest = {
    "project_id": os.getenv("CDSW_PROJECT_ID"), 
    "name" : MODEL_NAME,
    "description": f"Production model deployment for model name: {MODEL_NAME}",
    "registered_model_id": model_id
}

model_api_response = cml_client.create_model(CreateModelRequest, os.getenv("CDSW_PROJECT_ID"))

...
```

## Problem: Lack of Visibility into Model Behavior in Production

Once a model is deployed, not knowing how it is performing in real-world conditions can lead to performance degradation, undetected failures, and poor decision-making. Without proper monitoring, issues may go unnoticed until they start impacting users or business processes.

Key Challenges
- Performance Monitoring: How do we track if the model is making accurate predictions over time?
- Data & Concept Drift: Is the input data distribution changing compared to what the model was trained on?
- Technical Scalability: Can the model handle increasing requests without latency or failures?

## Solution #1: Technical Monitoring

To simulate load, the Notebook [1_ops_simulation.ipynb](./monitoring/1_ops_simulation.ipynb) creates synthetic data and API calls to the model endpoint.

```python
for sample in synthetic_sample:
    input_sample = {"inputs": [sample.tolist()]}
    cdsw.call_model(model_access_key=MODEL_ACCESS_KEY, ipt=input_sample)
```

Technical Metrics on Cloudera Machine Learning provide insights into the operational aspects of deployed models, specifically those running as REST API endpoints. These metrics help determine if models are appropriately resourced and functioning correctly.

Key Features:
- **Scope: Applicable exclusively to models deployed as REST API endpoints.**
- Metrics Tracked:
- Requests per Second
- Total Number of Requests
- Number of Failed Requests
- Model Response Time
- CPU Usage across all Model Replicas
- Memory Usage across all Model Replicas
- Model Request and Response Sizes

Documentation: https://docs.cloudera.com/machine-learning/1.5.3/models/topics/ml-model-tech-metrics.html

## Solution #2: Prediction Monitoring with Model Metrics

To answer questions such as data & concept drift and to add custom business logic to monitoring workflows, Model Metrics offer a customizable approach to monitor both REST API endpoints and batch inference processes. By integrating custom code, users can track specific performance indicators tailored to their models.

The notebook [2_model_metrics.ipynb](./monitoring/2_model_metrics.ipynb) shows a full example how to make use of Model Metrics to track predictions along with inputs and a delayed ground truth. The notebook makes use of the decorated predict function defined in the [2_predict_with_metrics.py](./monitoring/2_predict_with_metrics.py) module.

```python
import cml.metrics_v1 as metrics
import cml.models_v1 as models

@models.cml_model(metrics=True)
def predict(args):
    # Track the input.
    metrics.track_metric("input", args)
    result = model.predict(args)
    # Track the output.
    metrics.track_metric("output",result)
    return result
```

The ground truth is often available after predictions are made and served to end users/applications. By tracking ground truth after and correlating them with predictions, model and prediction quality can be monitored over time to account for concepts like data & concept drift.

```python
# Track the true values alongside the corresponding predictions
# with track_delayed_metrics function
for i in range(len(ground_truth_array)):
    ground_truth = ground_truth_array[i]
    metrics.track_delayed_metrics({"actual_result": str(ground_truth)}, uuids[i], dev=True)
```

Model Metrics key features:
- **Scope: Applicable to both REST API endpoints and batch inference models.**
- Customization: Allows tracking of user-defined metrics relevant to model performance and business objectives.
- Storage: Metrics are stored in a scalable metrics store, either managed by CML or an external Postgres database.

Model Metrics implementing custom metrics tracking:
- For REST API Endpoints: Incorporate custom code within the model’s prediction function to log desired metrics during each API call.
- For Batch Inference: Embed metric tracking code within the batch processing scripts to log performance indicators after each batch run.

To summarize, by leveraging Model Metrics, data scientists and engineers can gain deeper insights into model behavior, facilitating proactive maintenance and optimization.

Documentation: https://docs.cloudera.com/machine-learning/1.5.3/model-metrics/topics/ml-enabling-model-metrics.html

## Further Reading: CI/CD with Cloudera Machine Learning APIs and GitLab

By leveraging GitLab’s CI/CD pipelines in conjunction with Cloudera APIs, teams can automate the processes of model training, testing, deployment, and monitoring. This integration ensures that models are consistently updated and deployed without manual intervention, promoting a robust MLOps culture.

Example GitLab pipeline utilizing Cloudera APIs: https://gitlab.com/vish3004/end-to-end-devops-demo
