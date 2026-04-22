# Databricks notebook source
# MAGIC %md
# MAGIC ## 04 — Model Serving Endpoint
# MAGIC
# MAGIC Creates a Databricks Model Serving endpoint for the Unity Catalog model and
# MAGIC demonstrates how to send real-time inference requests.
# MAGIC
# MAGIC **Run this notebook interactively** — it is not part of the DAB job pipeline.

# COMMAND ----------

# Parameters
dbutils.widgets.text("catalog", "cmegdemos_catalog")
dbutils.widgets.text("schema",  "geospatial_analytics")
dbutils.widgets.text("endpoint_name", "signal-strength-predictor")

CATALOG       = dbutils.widgets.get("catalog")
SCHEMA        = dbutils.widgets.get("schema")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
MODEL_NAME    = f"{CATALOG}.{SCHEMA}.signal_strength_predictor"

print(f"Model    : {MODEL_NAME}")
print(f"Endpoint : {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md ### 1. Get latest Champion model version

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

champion_version = client.get_model_version_by_alias(MODEL_NAME, "Champion")
model_version = champion_version.version

print(f"Champion version: {model_version}")

# COMMAND ----------

# MAGIC %md ### 2. Create model serving endpoint
# MAGIC
# MAGIC Uses the Databricks SDK to create a serverless model serving endpoint.
# MAGIC Scale-to-zero is enabled to minimize cost for demo purposes.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

# Check if endpoint already exists
existing = [ep for ep in w.serving_endpoints.list() if ep.name == ENDPOINT_NAME]

if existing:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists — updating config...")
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=MODEL_NAME,
                entity_version=model_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
    )
else:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=MODEL_NAME,
                    entity_version=model_version,
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        ),
    )

print(f"Endpoint '{ENDPOINT_NAME}' configured with model version {model_version}")

# COMMAND ----------

# MAGIC %md ### 3. Wait for endpoint to be ready
# MAGIC
# MAGIC Model serving endpoints take a few minutes to provision. Poll until ready.

# COMMAND ----------

import time

def wait_for_endpoint(w, endpoint_name, timeout_minutes=20):
    """Poll endpoint status until READY or timeout."""
    start = time.time()
    while time.time() - start < timeout_minutes * 60:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state
        print(f"  Status: {state.ready} | Config: {state.config_update}")
        if state.ready == "READY":
            print(f"Endpoint '{endpoint_name}' is READY!")
            return endpoint
        time.sleep(30)
    raise TimeoutError(f"Endpoint not ready after {timeout_minutes} minutes")

endpoint = wait_for_endpoint(w, ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md ### 4. Query the serving endpoint
# MAGIC
# MAGIC Send a sample request with feature values matching the training schema.

# COMMAND ----------

# Sample input: a signal reading near Chandler Fashion Center
sample_request = {
    "dataframe_split": {
        "columns": [
            "latitude",
            "longitude",
            "h3_index",
            "distance_to_nearest_tower",
            "nearest_tower_type_enc",
            "nearest_tower_freq_band_enc",
            "tower_count_within_500m",
            "network_type_enc",
            "hour_of_day",
            "day_of_week",
            "wifi_rssi_filled",
        ],
        "data": [
            [33.3062, -111.8713, 617733122422996991, 50.0, 0, 0, 3, 0, 14, 3, 0.0],
            [33.3075, -111.8700, 617733122423062527, 120.0, 1, 2, 2, 1, 10, 5, -65.0],
        ],
    }
}

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_split=sample_request["dataframe_split"],
)

print("Predictions:")
print(response.predictions)

# COMMAND ----------

# MAGIC %md ### 5. Query using REST API (alternative)
# MAGIC
# MAGIC Shows how to call the endpoint from any HTTP client outside of Databricks.

# COMMAND ----------

import json

# Get workspace URL and token for the REST example
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print("Example curl command:\n")
print(f"""curl -X POST \\
  https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(sample_request, indent=2)}'""")

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC
# MAGIC | Item | Value |
# MAGIC |------|-------|
# MAGIC | Endpoint Name | `signal-strength-predictor` |
# MAGIC | Model | `cmegdemos_catalog.geospatial_analytics.signal_strength_predictor` |
# MAGIC | Workload Size | Small |
# MAGIC | Scale to Zero | Enabled |
# MAGIC | Input Format | `dataframe_split` with 11 feature columns |
