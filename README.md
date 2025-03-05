# MLflow Workshop for Cloudera Machine Learning

## Introduction

This repository provides a hands-on MLflow workshop tailored for Cloudera Machine Learning (CML). The workshop is designed for data scientists looking to integrate MLflow into their MLOps workflows, covering key aspects such as experiment tracking, model versioning, deployment, and monitoring.

By the end of this workshop, participants will be able to:
✅ Track machine learning experiments with MLflow.
✅ Manage model versions and automate workflows using the Model Registry.
✅ Deploy models as endpoints and implement monitoring best practices.

## Workshop Structure

The workshop is divided into three labs:

1. **Experiment Tracking**: Learn how to use MLflow for tracking experiments, logging metrics, and comparing models.
2. **Model Registry and Automation**: Understand how MLflow's Model Registry helps in versioning and managing models.
3. **Deployment and Monitoring**: Implement deployment and monitoring workflows using both technical and custom model metrics.

## Prerequisites

| Component | Specification |
|-----------|---------------|
| Python Dependencies | See `requirements.txt` |
| MLflow | `2.19.0` (Provided by Cloudera Machine Learning Runtime) |
| Cloudera Data Services | >= `1.5.3` |
| Cloudera Machine Learning Runtime | `docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-jupyterlab-python3.11-standard:2024.10.1-b12` |
| Public Internet Access | Not required* |

*Some examples may make use of public data sets that may require public internet access. Please contact your platform team in case of issues.

## Repository Structure

```
.
├── README.md
├── lab_1_experiment_tracking
│   ├── README.md
│   ├── lab_1_exersice_experiment_tracking.ipynb
│   ├── tracking_example_sklearn.py
│   └── tracking_example_torch.py
├── lab_2_model_registry
│   ├── README.md
│   ├── lab_2_exercise_model_registry.ipynb
│   ├── model_registry_create_model.py
│   ├── model_registry_push_to_registry.py
│   ├── registry-user-interface.png
│   └── registry-workflow.png
├── lab_3_monitoring
│   ├── README.md
│   ├── deployment
│   │   ├── deploy_from_registry.ipynb
│   │   └── deploy_from_registry.py
│   └── monitoring
│       ├── model_metrics.ipynb
│       ├── ops_simulation.ipynb
│       ├── predict_with_metrics.ipynb
│       └── predict_with_metrics.py
├── requirements_dev.txt
└── requirements_prod.txt
```

## Getting Started

Each lab is self-contained with step-by-step instructions.
Follow the labs in order to get started with MLOps using MLflow and Cloudera Machine Learning. 🚀

1. Creata a new Cloudera Machine Learning Project
2. Launch a session and install the dependencies

```bash
pip install -r requirements_dev.txt
```

## Contact

- Cloudera Solutions Engineering : Maximilian Engelhardt mengelhardt@cloudera.com
