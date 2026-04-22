# Databricks notebook source
# MAGIC %md
# MAGIC ## 03 — Batch Scoring: Signal Strength Predictions
# MAGIC
# MAGIC Loads the Champion model from Unity Catalog and scores the holdout set in batch.
# MAGIC Evaluates prediction quality and persists results for downstream analysis.

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

# MAGIC %md ### 1. Load Champion model from Unity Catalog

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

model_uri = f"models:/{MODEL_NAME}@Champion"
model = mlflow.pyfunc.load_model(model_uri)

print(f"Loaded model: {model_uri}")
print(f"Model metadata: {model.metadata}")

# COMMAND ----------

# MAGIC %md ### 2. Load holdout data

# COMMAND ----------

holdout_sdf = spark.table(f"`{CATALOG}`.`{SCHEMA}`.signal_holdout_data")
holdout_pdf = holdout_sdf.toPandas()

print(f"Holdout rows: {len(holdout_pdf):,}")
display(holdout_sdf.limit(5))

# COMMAND ----------

# MAGIC %md ### 3. Generate predictions

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

X_holdout = holdout_pdf[feature_cols]
holdout_pdf["predicted_rsrp"] = model.predict(X_holdout)

print(f"Predictions generated for {len(holdout_pdf):,} rows")

# COMMAND ----------

# MAGIC %md ### 4. Evaluate prediction quality

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_actual = holdout_pdf["rsrp"]
y_pred = holdout_pdf["predicted_rsrp"]

rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"Holdout RMSE : {rmse:.4f}")
print(f"Holdout MAE  : {mae:.4f}")
print(f"Holdout R²   : {r2:.4f}")

# COMMAND ----------

# MAGIC %md ### 5. Actual vs Predicted comparison

# COMMAND ----------

from pyspark.sql import functions as F

results_sdf = spark.createDataFrame(holdout_pdf)

# Summary statistics
display(
    results_sdf.select(
        F.avg("rsrp").alias("avg_actual_rsrp"),
        F.avg("predicted_rsrp").alias("avg_predicted_rsrp"),
        F.expr("corr(rsrp, predicted_rsrp)").alias("correlation"),
    )
)

# COMMAND ----------

# Scatter-friendly view: actual vs predicted
display(
    results_sdf.select("rsrp", "predicted_rsrp", "distance_to_nearest_tower", "network_type")
)

# COMMAND ----------

# MAGIC %md ### 6. Persist predictions

# COMMAND ----------

(
    results_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.signal_predictions")
)

print(f"Predictions saved to: {CATALOG}.{SCHEMA}.signal_predictions")

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | RMSE | `{rmse:.4f}` |
# MAGIC | MAE | `{mae:.4f}` |
# MAGIC | R² | `{r2:.4f}` |
# MAGIC | Output Table | `signal_predictions` |
