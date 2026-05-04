# Databricks notebook source
# MAGIC %md
# MAGIC ## 04 — Online Feature Store & Model Serving
# MAGIC
# MAGIC Publishes the `signal_features` table to a **Lakebase online store** and deploys
# MAGIC the **Champion model** (logged with `fe.log_model()`) to a serving endpoint.
# MAGIC
# MAGIC Because the model carries feature lookup metadata, the endpoint automatically
# MAGIC retrieves features from the online store — the request only needs the lookup key
# MAGIC (`h3_index`).
# MAGIC
# MAGIC **Run this notebook interactively** — it is not part of the DAB job pipeline.

# COMMAND ----------

# Parameters
dbutils.widgets.text("catalog", "cmegdemos_catalog")
dbutils.widgets.text("schema",  "geospatial_analytics")
dbutils.widgets.text("endpoint_name", "signal-strength-predictor")
dbutils.widgets.text("online_store_name", "telco-call-center")

CATALOG           = dbutils.widgets.get("catalog")
SCHEMA            = dbutils.widgets.get("schema")
ENDPOINT_NAME     = dbutils.widgets.get("endpoint_name")
ONLINE_STORE_NAME = dbutils.widgets.get("online_store_name")
MODEL_NAME        = f"{CATALOG}.{SCHEMA}.signal_strength_predictor"
FEATURE_TABLE     = f"{CATALOG}.{SCHEMA}.signal_features"

print(f"Model         : {MODEL_NAME}")
print(f"Feature Table : {FEATURE_TABLE}")
print(f"Endpoint      : {ENDPOINT_NAME}")
print(f"Online Store  : {ONLINE_STORE_NAME}")

# COMMAND ----------

# MAGIC %md ### 1. Connect to the Lakebase online store
# MAGIC
# MAGIC Uses an existing Lakebase Autoscaling instance that serves features
# MAGIC at low latency for real-time inference.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
import time

fe = FeatureEngineeringClient()

online_store = fe.get_online_store(name=ONLINE_STORE_NAME)
print(f"Online store: {online_store.name} (state: {online_store.state})")

# COMMAND ----------

# MAGIC %md ### 2. Publish feature table to the online store
# MAGIC
# MAGIC `publish_table()` syncs the `signal_features` Delta table to the online store.

# COMMAND ----------

ONLINE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.signal_hex_features_online"

fe.publish_table(
    online_store=online_store,
    source_table_name=FEATURE_TABLE,
    online_table_name=ONLINE_TABLE_NAME,
    publish_mode="SNAPSHOT",
)

print(f"Published '{FEATURE_TABLE}' -> '{ONLINE_TABLE_NAME}' (SNAPSHOT mode)")

# COMMAND ----------

# MAGIC %md ### 3. Create model serving endpoint
# MAGIC
# MAGIC Deploys the **Champion** model version — the one logged with `fe.log_model()`.
# MAGIC Because feature lookup metadata is embedded in the model, the serving endpoint
# MAGIC automatically retrieves features from the online store using `h3_index`.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
w = WorkspaceClient()

# Get the Champion model version
champion_version = client.get_model_version_by_alias(MODEL_NAME, "Champion").version
print(f"Champion version: {champion_version}")

# Check if endpoint already exists and is up-to-date
needs_update = True
existing = [ep for ep in w.serving_endpoints.list() if ep.name == ENDPOINT_NAME]
if existing:
    ep = existing[0]
    state = ep.state
    # Delete if stuck in failed state
    if state and "UPDATE_FAILED" in str(state.config_update):
        print(f"Deleting endpoint with failed config...")
        w.serving_endpoints.delete(ENDPOINT_NAME)
        time.sleep(10)
        existing = []
    elif "READY" in str(state.ready) and "NOT_UPDATING" in str(state.config_update):
        # Check if already serving the correct version
        served = ep.config.served_entities or []
        current_versions = [e.entity_version for e in served]
        if champion_version in current_versions:
            print(f"Endpoint '{ENDPOINT_NAME}' already serving Champion v{champion_version} -- skipping update")
            needs_update = False

if needs_update:
    if existing:
        print(f"Endpoint '{ENDPOINT_NAME}' exists -- updating to Champion v{champion_version}...")
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_entities=[
                ServedEntityInput(
                    entity_name=MODEL_NAME,
                    entity_version=champion_version,
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ],
        )
    else:
        print(f"Creating endpoint '{ENDPOINT_NAME}' with Champion v{champion_version}...")
        w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(
                name=ENDPOINT_NAME,
                served_entities=[
                    ServedEntityInput(
                        entity_name=MODEL_NAME,
                        entity_version=champion_version,
                        workload_size="Small",
                        scale_to_zero_enabled=True,
                    )
                ]
            ),
        )

# COMMAND ----------

# MAGIC %md ### 4. Wait for endpoint to be ready

# COMMAND ----------

def wait_for_endpoint(w, endpoint_name, timeout_minutes=30):
    """Poll endpoint status until READY or timeout."""
    start = time.time()
    while time.time() - start < timeout_minutes * 60:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state
        ready = str(state.ready)
        config = str(state.config_update) if state.config_update else ""
        print(f"  Status: {ready} | Config: {config}")
        if "READY" in ready and ("NOT_UPDATING" in config or not config):
            print(f"Endpoint '{endpoint_name}' is READY!")
            return endpoint
        if "UPDATE_FAILED" in config:
            raise RuntimeError(f"Endpoint update failed: check event logs")
        time.sleep(30)
    raise TimeoutError(f"Endpoint not ready after {timeout_minutes} minutes")

if needs_update:
    endpoint = wait_for_endpoint(w, ENDPOINT_NAME)
else:
    print(f"Endpoint already ready -- skipping wait")

# COMMAND ----------

# MAGIC %md ### 5. Query the serving endpoint
# MAGIC
# MAGIC The model was logged with `fe.log_model()` and features are published to the
# MAGIC online store. The request only needs the **lookup key** (`h3_index`) — the
# MAGIC endpoint automatically retrieves all features.

# COMMAND ----------

from databricks.sdk.service.serving import DataframeSplitInput

# Get a few real h3_index values from the holdout set for testing
holdout_hexes = (
    spark.table(f"`{CATALOG}`.`{SCHEMA}`.signal_observations")
    .filter("split = 'holdout'")
    .select("h3_index", "avg_rsrp")
    .limit(5)
    .toPandas()
)

h3_list = holdout_hexes["h3_index"].values.tolist()
print(f"Querying {len(h3_list)} holdout hexes")

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_split=DataframeSplitInput(
        columns=["h3_index"],
        data=[[h3] for h3 in h3_list],
    ),
)

print("\nPredicted vs Actual RSRP:")
for i, h3 in enumerate(h3_list):
    actual = holdout_hexes["avg_rsrp"].iloc[i]
    predicted = response.predictions[i]
    print(f"  Hex {h3}: predicted={predicted:.2f} dBm, actual={actual:.2f} dBm")

# COMMAND ----------

# MAGIC %md ### 6. Query using REST API (alternative)
# MAGIC
# MAGIC Shows how to call the endpoint from any HTTP client outside of Databricks.
# MAGIC Only the lookup key (`h3_index`) is required — features are auto-looked up.

# COMMAND ----------

import json, requests

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

curl_payload = {
    "dataframe_split": {
        "columns": ["h3_index"],
        "data": [[h3] for h3 in h3_list],
    }
}

# Execute the REST API call and show the response
url = f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

resp = requests.post(url, headers=headers, json=curl_payload)
print(f"Status: {resp.status_code}")
print(f"Response:\n{json.dumps(resp.json(), indent=2)}")

print(f"\nEquivalent curl command:\n")
print(f"""curl -X POST \\
  {url} \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(curl_payload, indent=2)}'""")

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC
# MAGIC | Item | Value |
# MAGIC |------|-------|
# MAGIC | Endpoint Name | `signal-strength-predictor` |
# MAGIC | Model | `signal_strength_predictor` @ Champion |
# MAGIC | Logged with | `fe.log_model()` — auto feature lookup enabled |
# MAGIC | Online Store | Lakebase Autoscaling instance |
# MAGIC | Online Table | `signal_features_online` (synced from `signal_features`) |
# MAGIC | Request Payload | `h3_index` only — features retrieved automatically |
# MAGIC | Workload Size | Small |
# MAGIC | Scale to Zero | Enabled |
