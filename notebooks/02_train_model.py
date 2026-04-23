# Databricks notebook source
# MAGIC %md
# MAGIC ## 02 — Train LightGBM Model with Feature Store
# MAGIC
# MAGIC Uses the **Feature Engineering client** to build a training set from the `signal_features`
# MAGIC feature table, trains a `LGBMRegressor`, and logs the model with `fe.log_model()` so that
# MAGIC feature lookup metadata is packaged with the model for downstream scoring and serving.

# COMMAND ----------

# Parameters (injected by DAB; fall back to defaults for interactive runs)
dbutils.widgets.text("catalog", "cmegdemos_catalog")
dbutils.widgets.text("schema",  "geospatial_analytics")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")

MODEL_NAME    = f"{CATALOG}.{SCHEMA}.signal_strength_predictor"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.signal_features"

print(f"Catalog       : {CATALOG}")
print(f"Schema        : {SCHEMA}")
print(f"Model name    : {MODEL_NAME}")
print(f"Feature table : {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md ### 1. Configure MLflow for Unity Catalog

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

username = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{username}/ml_geospatial_signal_strength"
mlflow.set_experiment(experiment_path)

print(f"MLflow registry : {mlflow.get_registry_uri()}")
print(f"Experiment      : {experiment_path}")

# COMMAND ----------

# MAGIC %md ### 2. Build training set with Feature Engineering client
# MAGIC
# MAGIC `FeatureLookup` defines which features to pull from the feature table.
# MAGIC `fe.create_training_set()` joins them onto the observations DataFrame automatically.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

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

feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        feature_names=feature_cols,
        lookup_key="signal_id",
    )
]

# COMMAND ----------

# Load training observations (signal_id + rsrp, filtered to train split)
train_obs_df = (
    spark.table(f"`{CATALOG}`.`{SCHEMA}`.signal_observations")
    .filter("split = 'train'")
    .select("signal_id", "rsrp")
)

print(f"Training observations: {train_obs_df.count():,}")

# Create training set — auto-joins features from feature table
training_set = fe.create_training_set(
    df=train_obs_df,
    feature_lookups=feature_lookups,
    label="rsrp",
    exclude_columns=["signal_id"],
)

training_pdf = training_set.load_df().toPandas()
print(f"Training set loaded: {training_pdf.shape}")
display(training_set.load_df().limit(5))

# COMMAND ----------

# MAGIC %md ### 3. Prepare features and target

# COMMAND ----------

X = training_pdf[feature_cols]
y = training_pdf["rsrp"]

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

# MAGIC %md ### 5. Train LightGBM regressor and log with Feature Store

# COMMAND ----------

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

with mlflow.start_run(run_name="lightgbm_signal_strength_fe") as run:

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

    # --- Log model with Feature Engineering client ---
    # This packages feature lookup metadata with the model so that
    # fe.score_batch() and serving endpoints can auto-retrieve features.
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.lightgbm,
        training_set=training_set,
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id
    print(f"\nMLflow run ID: {run_id}")
    print(f"Model registered: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ### 6. Set "Champion" alias on the latest model version

# COMMAND ----------

client = MlflowClient()

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
# MAGIC | Logged with | `fe.log_model()` (Feature Engineering client) |
# MAGIC | Feature Table | `signal_features` (auto-lookup at scoring/serving) |
# MAGIC | Registry | Unity Catalog |
# MAGIC | Model Name | `cmegdemos_catalog.geospatial_analytics.signal_strength_predictor` |
# MAGIC | Alias | Champion |
