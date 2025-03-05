from datetime import datetime
import time
import sys
import os
import json

import cmlapi

MODEL_NAME = "sklearn_model"
TARGET_PROJECT_ID = os.getenv("CDSW_PROJECT_ID")
TARGET_RUNTIME = "ares-ecs-docker-repo.cloudera-field.org/cloudera/cdsw/ml-runtime-jupyterlab-python3.11-standard:2024.02.1-b4"

# Set up client
workspace_domain = os.getenv("CDSW_DOMAIN")
cml_client = cmlapi.default_client(url=f"https://{workspace_domain}")

# Retrieve model id by model name
search_filter = {"model_name": MODEL_NAME}
response_dict = cml_client.list_registered_models(search_filter=json.dumps(search_filter)).to_dict()
model_id = response_dict["models"][0]["model_id"]
print(f"Model ID: {model_id}")

# Retrieve version id for deployment
registered_model = cml_client.get_registered_model(model_id, sort="-version_number")
model_version_id = registered_model.model_versions[0].model_version_id
print(f"Version ID: {model_version_id}")

# 1. Create Cloudera Machine Learning Model
CreateModelRequest = {
    "project_id": os.getenv("CDSW_PROJECT_ID"), 
    "name" : MODEL_NAME,
    "description": f"Production model deployment for model name: {MODEL_NAME}",
    "disable_authentication": True, # not recommended for production
    "registered_model_id": model_id
}

model_api_response = cml_client.create_model(CreateModelRequest, os.getenv("CDSW_PROJECT_ID"))

# 2. Create Cloudera Machine Learning Model Build
CreateModelBuildRequest = {
    "registered_model_version_id": model_version_id,
    "comment": "Invoking model build for model ",
    "runtime_identifier": TARGET_RUNTIME,
    "model_id": model_api_response.id
}

model_build_api_response = cml_client.create_model_build(CreateModelBuildRequest, TARGET_PROJECT_ID, model_api_response.id)

start_time = datetime.now()
print(start_time.strftime("%H:%M:%S"))

# keep track of the build process
while model_build_api_response.status not in ["built", "build failed"]:
    print("Waiting for model to build...")
    time.sleep(10)
    model_build_api_response = cml_client.get_model_build(
        TARGET_PROJECT_ID,
        model_api_response.id,
        model_build_api_response.id
    )
    if model_build_api_response.status == "build failed" :
        print("Model build failed, see UI for more information")
        sys.exit(1)

build_time = datetime.now()   
print(f"Time required for building model (sec): {(build_time - start_time).seconds}.")
print("Model build finished successfully.")

# 3. Create Cloudera Machine Learning Model Deployment
CreateModelDeploymentRequest = {
    "project_id": TARGET_PROJECT_ID,
    "model_id": model_api_response.id,
    "model_build_id": model_build_api_response.id,
    "cpu": 1,
    "memory": 2
}

model_deployment_api_response = cml_client.create_model_deployment(
    CreateModelDeploymentRequest,
    TARGET_PROJECT_ID,
    model_api_response.id,
    model_build_api_response.id
)

start_time = datetime.now()
print(start_time.strftime("%H:%M:%S"))

while model_deployment_api_response.status not in ["stopped", "failed", "deployed"]:
    print("Waiting for model to deploy...")
    time.sleep(10)
    model_deployment_api_response = cml_client.get_model_deployment(
        TARGET_PROJECT_ID,
        model_api_response.id,
        model_build_api_response.id,
        model_deployment_api_response.id
)

if model_deployment_api_response.status != "deployed":
    print("Model deployment failed, see UI for more information.")
    sys.exit(1)

if model_deployment_api_response.status == "deployed" :
    print(f"Time required for deploying model (sec): {(datetime.now() - start_time).seconds}.")
    print("Model deployed successfully!")
