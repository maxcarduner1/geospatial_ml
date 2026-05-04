# Databricks notebook source
# MAGIC %md
# MAGIC ## 03 — Batch Scoring with Feature Store
# MAGIC
# MAGIC Uses `fe.score_batch()` to score the holdout hexes. The Feature Engineering client
# MAGIC automatically looks up features from the `signal_features` table — the scoring
# MAGIC DataFrame only needs to contain the lookup key (`h3_index`).

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

# MAGIC %md ### 1. Load holdout observations
# MAGIC
# MAGIC Only `h3_index` is needed for feature lookup — `avg_rsrp` is kept for evaluation.

# COMMAND ----------

holdout_df = (
    spark.table(f"`{CATALOG}`.`{SCHEMA}`.signal_observations")
    .filter("split = 'holdout'")
    .select("h3_index", "avg_rsrp")
)

print(f"Holdout hexes: {holdout_df.count():,}")
display(holdout_df.limit(5))

# COMMAND ----------

# MAGIC %md ### 2. Score with `fe.score_batch()`
# MAGIC
# MAGIC The model was logged with `fe.log_model()`, so it contains feature lookup metadata.
# MAGIC `score_batch()` automatically joins features from the `signal_features` table
# MAGIC using `h3_index`, then runs the model to produce predictions.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
import mlflow

mlflow.set_registry_uri("databricks-uc")

fe = FeatureEngineeringClient()

model_uri = f"models:/{MODEL_NAME}@Champion"

predictions_df = fe.score_batch(
    model_uri=model_uri,
    df=holdout_df,
)

print(f"Predictions generated via fe.score_batch()")
display(predictions_df.limit(5))

# COMMAND ----------

# MAGIC %md ### 3. Evaluate prediction quality

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

results_pdf = predictions_df.toPandas()

y_actual = results_pdf["avg_rsrp"]
y_pred = results_pdf["prediction"]

rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"Holdout RMSE : {rmse:.4f}")
print(f"Holdout MAE  : {mae:.4f}")
print(f"Holdout R²   : {r2:.4f}")

# COMMAND ----------

# MAGIC %md ### 4. Actual vs Predicted comparison

# COMMAND ----------

from pyspark.sql import functions as F

# Summary statistics
display(
    predictions_df.select(
        F.count("*").alias("n_hexes"),
        F.avg("avg_rsrp").alias("avg_actual_rsrp"),
        F.avg("prediction").alias("avg_predicted_rsrp"),
        F.avg(F.abs(F.col("avg_rsrp") - F.col("prediction"))).alias("avg_abs_error"),
    )
)

# COMMAND ----------

# Scatter-friendly view: actual vs predicted with key features
display(
    predictions_df.select("h3_index", "avg_rsrp", "prediction", "avg_distance_to_nearest_tower", "dominant_network_type_enc")
)

# COMMAND ----------

# MAGIC %md ### 5. Persist predictions

# COMMAND ----------

(
    predictions_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.signal_predictions")
)

print(f"Predictions saved to: {CATALOG}.{SCHEMA}.signal_predictions")

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC
# MAGIC | Item | Value |
# MAGIC |------|-------|
# MAGIC | Scoring method | `fe.score_batch()` — automatic feature lookup |
# MAGIC | Input | `h3_index` (lookup key) + `avg_rsrp` (for evaluation) |
# MAGIC | Feature source | `signal_features` table (auto-joined) |
# MAGIC | Output table | `signal_predictions` |
