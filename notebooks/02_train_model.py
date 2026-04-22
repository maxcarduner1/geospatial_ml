# Databricks notebook source
# MAGIC %md
# MAGIC ## 02 — Train LightGBM Model & Register to Unity Catalog
# MAGIC
# MAGIC Trains a `LGBMRegressor` to predict RSRP signal strength from spatial, temporal,
# MAGIC and network features. Logs the experiment to MLflow and registers the model to
# MAGIC Unity Catalog for governed downstream consumption.

# COMMAND ----------

# Parameters (injected by DAB; fall back to defaults for interactive runs)
dbutils.widgets.text("catalog", "cmegdemos_catalog")
dbutils.widgets.text("schema",  "geospatial_analytics")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")

MODEL_NAME = f"{CATALOG}.{SCHEMA}.signal_strength_predictor"

print(f"Catalog    : {CATALOG}")
print(f"Schema     : {SCHEMA}")
print(f"Model name : {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ### 1. Configure MLflow for Unity Catalog

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

# Resolve current user for experiment path
username = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{username}/ml_geospatial_signal_strength"
mlflow.set_experiment(experiment_path)

print(f"MLflow registry : {mlflow.get_registry_uri()}")
print(f"Experiment      : {experiment_path}")

# COMMAND ----------

# MAGIC %md ### 2. Load training data

# COMMAND ----------

import pandas as pd

train_sdf = spark.table(f"`{CATALOG}`.`{SCHEMA}`.signal_training_data")
train_pdf = train_sdf.toPandas()

print(f"Training rows: {len(train_pdf):,}")
display(train_sdf.limit(5))

# COMMAND ----------

# MAGIC %md ### 3. Prepare features and target

# COMMAND ----------

feature_cols = [
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
]

target_col = "rsrp"

X = train_pdf[feature_cols]
y = train_pdf[target_col]

print(f"Feature matrix: {X.shape}")
print(f"Target stats:\n{y.describe()}")

# COMMAND ----------

# MAGIC %md ### 4. Train/validation split for metric evaluation

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]:,} rows")
print(f"Val:   {X_val.shape[0]:,} rows")

# COMMAND ----------

# MAGIC %md ### 5. Train LightGBM regressor with MLflow tracking

# COMMAND ----------

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
import numpy as np

with mlflow.start_run(run_name="lightgbm_signal_strength") as run:

    # --- Model definition ---
    model = LGBMRegressor(
        objective="regression",
        metric="rmse",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    # --- Train ---
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    # --- Evaluate ---
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Validation RMSE : {rmse:.4f}")
    print(f"Validation MAE  : {mae:.4f}")
    print(f"Validation R²   : {r2:.4f}")

    # --- Log metrics ---
    mlflow.log_metric("val_rmse", rmse)
    mlflow.log_metric("val_mae", mae)
    mlflow.log_metric("val_r2", r2)

    # --- Log model with signature ---
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:3]

    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id
    print(f"\nMLflow run ID: {run_id}")
    print(f"Model registered: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ### 6. Set "Champion" alias on the latest model version

# COMMAND ----------

client = MlflowClient()

# Get the latest version number
latest_version = max(
    client.search_model_versions(f"name='{MODEL_NAME}'"),
    key=lambda v: int(v.version),
)

client.set_registered_model_alias(MODEL_NAME, "Champion", latest_version.version)

print(f"Alias 'Champion' set on {MODEL_NAME} version {latest_version.version}")

# COMMAND ----------

# MAGIC %md ### 7. Feature importance

# COMMAND ----------

import pandas as pd

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

display(spark.createDataFrame(importance_df))

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC
# MAGIC | Item | Value |
# MAGIC |------|-------|
# MAGIC | Model | LightGBM Regressor |
# MAGIC | Target | RSRP signal strength (dBm) |
# MAGIC | Registry | Unity Catalog |
# MAGIC | Model Name | `cmegdemos_catalog.geospatial_analytics.signal_strength_predictor` |
# MAGIC | Alias | Champion |
