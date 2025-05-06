# Lab 3: Deployment and Monitoring Workflows

## Introduction

Labs 1 and 2 have already covered Experiment Tracking, Model Registry and Model Deployments. This lab adds the critical last step in the MLOps pipeline: **Monitoring**.

    experiments → model registry → deployment → monitoring

> [!Tip] Once a model is in production, continuous monitoring is crucial to ensure:
>
> - ✅ Infrastructure scales effectively (low latency, stable request handling).
> - ✅ Performance remains high (accuracy, prediction quality).
> - ✅ Model drift and data shifts are detected early to prevent degradation.

## Overview

- [Lab 3: Deployment and Monitoring Workflows](#lab-3-deployment-and-monitoring-workflows)
  - [Introduction](#introduction)
  - [Overview](#overview)
  - [Why Monitoring?](#why-monitoring)
  - [Best Practice #1: Technical Monitoring](#best-practice-1-technical-monitoring)
  - [Best Practice #2: Prediction Monitoring with Model Metrics](#best-practice-2-prediction-monitoring-with-model-metrics)
    - [Setup Metrics Store and Prediction Function](#setup-metrics-store-and-prediction-function)
    - [Monitor Drift and Prediction Quality with Delayed Metrics](#monitor-drift-and-prediction-quality-with-delayed-metrics)
  - [Summary](#summary)

## Why Monitoring?

Monitoring is essential for maintaining model performance and reliability in production. Without it, models can degrade silently, leading to poor predictions. Key challenges include tracking accuracy, detecting drift, and ensuring scalability. Effective monitoring provides visibility into model behavior and enables proactive maintenance.

## Best Practice #1: Technical Monitoring

Technical monitoring focuses on the operational health of deployed models, especially those running as REST API endpoints. By tracking metrics like request rates, failures, response times, CPU, and memory usage, you can quickly identify and resolve infrastructure issues before they impact users. The notebook [1_technical_metrics_create_traffic.ipynb](./1_technical_metrics_create_traffic.ipynb) demonstrates how to simulate load and monitor these metrics in practice.

```python
from cml.models_v1 import call_model

for sample in synthetic_sample:
    input_sample = {"inputs": [sample.tolist()]}
    call_model(model_access_key=MODEL_ACCESS_KEY, ipt=input_sample)
```

After a few minutes, the Model Deployment Monitoring UI should show the load.

![technical monitoring](../images/tech-monitoring.png)

Documentation: <https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-model-tech-metrics.html>

## Best Practice #2: Prediction Monitoring with Model Metrics

To answer questions such as data & concept drift and to add custom business logic to monitoring workflows, Model Metrics offer a customizable approach to monitor both REST API endpoints and batch inference processes. By integrating custom code, users can track specific performance indicators tailored to their models.

### Setup Metrics Store and Prediction Function

As long as Model Metrics are enabled for the Machine Learning Workbench, Cloudera deploys an embedded Postgres Database to store Model Metrics related data. See also <https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-enabling-model-metrics.html>. To make use of the metrics store, decorate your prediction function with the Cloudera Model Metrics decorator. The notebook [2a_predict_with_metrics.ipynb](2a_predict_with_metrics.ipynb) shows a full example how to set up a decorated prediction function, based on the example model from the previous labs.

```python
import cml.metrics_v1 as metrics
import cml.models_v1 as models

...
# Load the model from mlflow
model = mlflow.pyfunc.load_model(...)

# Set up model metrics with decorator
@models.cml_model(metrics=True)
def predict(args):
    metrics.track_metric("input", args) # Track the input with every inference
    result = model.predict(args)
    metrics.track_metric("output",result) # Track the output with every inference
    return result
```

The returned predictions are enhanced by Model Metrics details, i.e. `model_deployment_crn` and `uuid` that can later be used to retrieve and analyse the tracked data:

```python
...
# invoke the decorated predict function
prediction = predict(example_input)
print(prediction)

>>> {'prediction': array([0]),
 'model_deployment_crn': 'crn:cdp:ml:::workspace:dev/model-deployment',
 'uuid': '5f94cb41-eb5b-4b8b-a750-67874f7d4aad'}
```

>[!Note] Model Metrics with Deployments from Registry
>
> With the current Workbench version `2.0.50-b52`, model deployments from registry are always done with an (auto-generated) decorated predict_with_metrics function. This allows any direct deployments from the registry to be used with the Model Metrics feature, unless the Workbench has Model Metrics feature disabled.

### Monitor Drift and Prediction Quality with Delayed Metrics

The notebook [2b_use_model_metrics.ipynb](2b_use_model_metrics.ipynb) shows a full example how to make use of Model Metrics to monitor prediction quality and data drift. This is done by tracking inputs, predictions and a delayed ground truth. The ground truth is often available after predictions are made and served to end users/applications.

Note that the notebook follows a simplified example where the uuid are assumed to be known at the time when ground truth becomes available. In a real life scenario, this may not be the case. Ground truth may become available but the uuid of the associated prediction may have to be retrieved, e.g. by using a timestamp:

```
t0 -- model is invoked with query
    --> prediction generated with uuid "5f94cb41-eb5b-4b8b-a750-67874f7d4aad"
    --> uuid and timestamp added to metrics store (timestamp=t0, uuid=...)

t1 -- actual answer for query at t0 becomes available
    --> uuid retrieved with timestamp read_metrics(timestamp=t0, ...) 
    --> returned uuid "5f94cb41-eb5b-4b8b-a750-67874f7d4aad"
    --> ground truth written to metric store with uuid
```

Example workflow for retrieving prediction uuid via timestamp:

```python
...
# Ground truth with associated timestamp becomes available
ground_truth_start_ts = 1739926103995
ground_truth_end_ts = 1739926103996
ground_truth = "1"

# metrics store data array example:
# [{'modelDeploymentCrn': 'crn:cdp:ml:::workspace:dev/model-deployment',
#   'modelBuildCrn': 'crn:cdp:ml:::workspace:dev/model-build',
#   'modelCrn': 'crn:cdp:ml:::workspace:dev/model-deployment',
#   'startTimeStampMs': 1739926103995,
#   'endTimeStampMs': 1739926103996,
#   'predictionUuid': '3f5bcd30-e017-45ac-b20e-02f01ca5c8e3',
#   'metrics': {'input': [4.6, 3.6, 1.0, 0.2], 'output': '0'}}, ...]
data = metrics.read_metrics(
    model_deployment_crn=model_deployment_crn,
    start_timestamp_ms=ground_truth_start_ts,
    end_timestamp_ms=ground_truth_end_ts
)["metrics"][0]["predictionUuid"]

# Track the true value along with corresponding prediction using the uuid
metrics.track_delayed_metrics(
    {"actual_result": str(ground_truth)},
    uuid, dev=True
)
```

The metrics store will now contain the ground truth as well:

```
[{'modelDeploymentCrn': 'crn:cdp:ml:::workspace:dev/model-deployment',
  'modelBuildCrn': 'crn:cdp:ml:::workspace:dev/model-build',
  'modelCrn': 'crn:cdp:ml:::workspace:dev/model-deployment',
  'startTimeStampMs': 1739926103995,
  'endTimeStampMs': 1739926103996,
  'predictionUuid': '3f5bcd30-e017-45ac-b20e-02f01ca5c8e3',
  'metrics': {'input': [4.6, 3.6, 1.0, 0.2], 'output': '0', 'actual_result': '1'}}, ...]
```

See also <https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-model-metrics.html>.

## Summary

Monitoring is a critical component of the MLOps pipeline that ensures models remain reliable and performant in production. This lab covered two key aspects of monitoring:

- Technical monitoring for infrastructure health and performance
- Prediction monitoring with model metrics for tracking model quality and drift
