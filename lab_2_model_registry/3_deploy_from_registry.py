"""
Script to automatically deploy a registered model to production using Cloudera Machine Learning.

This script demonstrates a streamlined deployment pipeline that:
1. Retrieves the latest version of a registered model
2. Creates a model instance
3. Automatically builds the model with specified runtime
4. Automatically deploys the model as a production service

The automated deployment process includes:
- Model version selection
- Runtime environment setup
- Automatic build process
- Automatic deployment with resource allocation
- Single-step deployment configuration

Note: This script uses environment variables for configuration:
- CDSW_PROJECT_ID: Target project ID for deployment
- CDSW_DOMAIN: Machine Learning Workspace domain
"""

import os

import cmlapi

# Configuration for model deployment
# Runtime environment specification for model serving
TARGET_RUNTIME = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.11-standard:2025.01.3-b8"
TARGET_PROJECT_ID = os.getenv("CDSW_PROJECT_ID")
MODEL_NAME = "sklearn_model"

# Initialize CML API client
workspace_domain = os.getenv("CDSW_DOMAIN")
cml_client = cmlapi.default_client(url=f"https://{workspace_domain}")

# Step 1: Retrieve the model ID from the registry
# Search through all registered models to find the one matching our MODEL_NAME
response_dict = cml_client.list_registered_models().to_dict()
model_id = next((model["model_id"] for model in response_dict["models"] if model["name"] == MODEL_NAME), None)
print(f"Model ID: {model_id}")

# Step 2: Get the latest version of the model
# Sort by version number in descending order to get the most recent version
registered_model = cml_client.get_registered_model(model_id, sort="-version_number")
model_version_id = registered_model.model_versions[0].model_version_id
print(f"Version ID: {model_version_id}")

# Step 3: Create and configure the model deployment
# This single request handles model creation, building, and deployment automatically
CreateModelRequest = {
    "project_id": os.getenv("CDSW_PROJECT_ID"), 
    "name" : MODEL_NAME,
    "description": f"Production model deployment for model name: {MODEL_NAME}",
    "disable_authentication": True,  # Note: Disabling authentication is not recommended for production
    "registered_model_id": model_id,
    # Enable automatic build process
    "auto_build_model": True,
    "auto_build_config": {
        "registered_model_version_id": model_version_id,
        "runtime_identifier": TARGET_RUNTIME
    },
    # Enable automatic deployment
    "auto_deploy_model": True,
    "auto_deploy_config": {
        "cpu": 1,      # Allocate 1 CPU core
        "memory": 2    # Allocate 2 GB memory
    }
}

# Execute the deployment request
model_api_response = cml_client.create_model(CreateModelRequest, os.getenv("CDSW_PROJECT_ID"))
print(model_api_response)
