# Databricks notebook source
# MAGIC %md
# MAGIC ## 01 — Feature Engineering: Signal Strength Prediction
# MAGIC
# MAGIC Builds ML-ready training and holdout tables from the raw `signal_points` table.
# MAGIC
# MAGIC **Features engineered:**
# MAGIC - Spatial: distance to nearest cell tower, tower density, nearest tower type/frequency
# MAGIC - H3: hexagonal spatial index at resolution 9
# MAGIC - Temporal: hour of day, day of week
# MAGIC - Network: encoded network type, cell type, connectivity type
# MAGIC
# MAGIC **Target:** RSRP signal strength (dBm) — continuous regression

# COMMAND ----------

# Parameters (injected by DAB; fall back to defaults for interactive runs)
dbutils.widgets.text("catalog", "cmegdemos_catalog")
dbutils.widgets.text("schema",  "geospatial_analytics")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")

print(f"Catalog : {CATALOG}")
print(f"Schema  : {SCHEMA}")

# COMMAND ----------

# MAGIC %md ### 1. Load raw signal points

# COMMAND ----------

raw_df = spark.table(f"`{CATALOG}`.`{SCHEMA}`.signal_points")
print(f"Raw rows: {raw_df.count():,}")
display(raw_df.limit(5))

# COMMAND ----------

# MAGIC %md ### 2. Clean and rename columns

# COMMAND ----------

from pyspark.sql import functions as F

# Filter out null RSRP values and select/rename relevant columns
clean_df = (
    raw_df
    .filter(F.col("ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLSIGNALSTRENGTH_RSRP").isNotNull())
    .select(
        F.col("LATITUDE").cast("double").alias("latitude"),
        F.col("LONGITUDE").cast("double").alias("longitude"),
        F.col("ED_TIMESTAMP").cast("timestamp").alias("timestamp"),
        F.col("ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLSIGNALSTRENGTH_RSRP").cast("double").alias("rsrp"),
        F.col("ED_ENVIRONMENT_NET_CONNECTEDWIFISTATUS_RSSILEVEL").cast("double").alias("wifi_rssi"),
        F.col("ED_ENVIRONMENT_TELEPHONY_NETWORKTYPE").alias("network_type"),
        F.col("ED_ENVIRONMENT_NET_CONNECTIVITYTYPE").alias("connectivity_type"),
        F.col("ED_ENVIRONMENT_TELEPHONY_PRIMARYCELL_CELLTYPE").alias("cell_type"),
        F.col("ED_ENVIRONMENT_TELEPHONY_SERVICESTATE").alias("service_state"),
    )
    .filter(F.col("latitude").isNotNull() & F.col("longitude").isNotNull())
)

print(f"Clean rows (non-null RSRP): {clean_df.count():,}")
display(clean_df.limit(5))

# COMMAND ----------

# MAGIC %md ### 3. H3 spatial indexing

# COMMAND ----------

h3_df = clean_df.withColumn(
    "h3_index",
    F.expr("h3_longlatash3(latitude, longitude, 9)")
)

display(h3_df.select("latitude", "longitude", "h3_index", "rsrp").limit(5))

# COMMAND ----------

# MAGIC %md ### 4. Synthesize cell tower reference data
# MAGIC
# MAGIC No real tower data exists in this dataset, so we create a realistic set of cell towers
# MAGIC near Chandler Fashion Center (33.3062° N, 111.8713° W) for spatial feature engineering.

# COMMAND ----------

import pandas as pd

# Realistic cell tower positions around Chandler Fashion Center area
towers_data = [
    {"tower_id": 1, "tower_lat": 33.3062, "tower_lon": -111.8713, "tower_type": "macro",      "freq_band": "n71"},
    {"tower_id": 2, "tower_lat": 33.3085, "tower_lon": -111.8680, "tower_type": "macro",      "freq_band": "B66"},
    {"tower_id": 3, "tower_lat": 33.3045, "tower_lon": -111.8740, "tower_type": "small_cell", "freq_band": "n41"},
    {"tower_id": 4, "tower_lat": 33.3078, "tower_lon": -111.8750, "tower_type": "macro",      "freq_band": "B2"},
    {"tower_id": 5, "tower_lat": 33.3030, "tower_lon": -111.8695, "tower_type": "small_cell", "freq_band": "n41"},
    {"tower_id": 6, "tower_lat": 33.3095, "tower_lon": -111.8720, "tower_type": "macro",      "freq_band": "n71"},
    {"tower_id": 7, "tower_lat": 33.3055, "tower_lon": -111.8660, "tower_type": "small_cell", "freq_band": "B66"},
    {"tower_id": 8, "tower_lat": 33.3040, "tower_lon": -111.8770, "tower_type": "macro",      "freq_band": "B2"},
]

towers_pdf = pd.DataFrame(towers_data)
towers_sdf = spark.createDataFrame(towers_pdf)

# Persist for reference
(
    towers_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.cell_towers")
)

display(towers_sdf)

# COMMAND ----------

# MAGIC %md ### 5. Compute spatial features (distance to towers)
# MAGIC
# MAGIC Uses the Haversine formula to compute distance from each signal reading to every tower,
# MAGIC then extracts nearest-tower features and tower density.

# COMMAND ----------

import math

# Cross join signal points with towers and compute haversine distance
cross_df = h3_df.crossJoin(towers_sdf.alias("t"))

# Haversine distance in meters using SQL expression
cross_df = cross_df.withColumn(
    "distance_m",
    F.expr("""
        6371000 * 2 * asin(sqrt(
            pow(sin(radians(t.tower_lat - latitude) / 2), 2) +
            cos(radians(latitude)) * cos(radians(t.tower_lat)) *
            pow(sin(radians(t.tower_lon - longitude) / 2), 2)
        ))
    """)
)

# COMMAND ----------

# MAGIC %md #### 5a. Nearest tower features

# COMMAND ----------

from pyspark.sql.window import Window

# Rank towers by distance for each signal point
w = Window.partitionBy("latitude", "longitude", "timestamp").orderBy("distance_m")

nearest_df = (
    cross_df
    .withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") == 1)
    .select(
        h3_df.columns + [
            F.col("distance_m").alias("distance_to_nearest_tower"),
            F.col("t.tower_type").alias("nearest_tower_type"),
            F.col("t.freq_band").alias("nearest_tower_freq_band"),
        ]
    )
)

# COMMAND ----------

# MAGIC %md #### 5b. Tower density (count within 500m)

# COMMAND ----------

tower_density_df = (
    cross_df
    .filter(F.col("distance_m") <= 500)
    .groupBy("latitude", "longitude", "timestamp")
    .agg(F.count("*").alias("tower_count_within_500m"))
)

features_df = nearest_df.join(
    tower_density_df,
    on=["latitude", "longitude", "timestamp"],
    how="left"
).fillna(0, subset=["tower_count_within_500m"])

# COMMAND ----------

# MAGIC %md ### 6. Encode categorical features & extract temporal features

# COMMAND ----------

# Encode network_type
network_type_map = {
    "NETWORK_TYPE_LTE": 0,
    "NETWORK_TYPE_NR": 1,
    "NETWORK_TYPE_UMTS": 2,
    "NETWORK_TYPE_HSPAP": 3,
    "NETWORK_TYPE_HSPA": 4,
}

# Encode tower_type
tower_type_map = {"macro": 0, "small_cell": 1}

# Encode freq_band
freq_band_map = {"n71": 0, "B66": 1, "n41": 2, "B2": 3}

# Apply mappings + temporal features
encoded_df = (
    features_df
    .withColumn("network_type_enc", F.coalesce(
        *[F.when(F.col("network_type") == k, F.lit(v)) for k, v in network_type_map.items()],
        F.lit(-1)
    ))
    .withColumn("nearest_tower_type_enc", F.coalesce(
        *[F.when(F.col("nearest_tower_type") == k, F.lit(v)) for k, v in tower_type_map.items()],
        F.lit(-1)
    ))
    .withColumn("nearest_tower_freq_band_enc", F.coalesce(
        *[F.when(F.col("nearest_tower_freq_band") == k, F.lit(v)) for k, v in freq_band_map.items()],
        F.lit(-1)
    ))
    .withColumn("hour_of_day", F.hour("timestamp"))
    .withColumn("day_of_week", F.dayofweek("timestamp"))
    .withColumn("wifi_rssi_filled", F.coalesce(F.col("wifi_rssi"), F.lit(0.0)))
)

# COMMAND ----------

# MAGIC %md ### 7. Select final feature columns and split train/holdout

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

# Keep original columns for context + feature columns + target
final_df = encoded_df.select(
    *feature_cols,
    F.col(target_col),
    # Keep raw columns for interpretability
    F.col("network_type"),
    F.col("nearest_tower_type"),
    F.col("nearest_tower_freq_band"),
    F.col("timestamp"),
)

print(f"Final feature set: {final_df.count():,} rows, {len(feature_cols)} features")
display(final_df.limit(10))

# COMMAND ----------

# MAGIC %md ### 8. 80/20 train/holdout split and persist

# COMMAND ----------

train_df, holdout_df = final_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training rows : {train_df.count():,}")
print(f"Holdout rows  : {holdout_df.count():,}")

# COMMAND ----------

(
    train_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.signal_training_data")
)

(
    holdout_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.signal_holdout_data")
)

print("Tables saved:")
print(f"  {CATALOG}.{SCHEMA}.signal_training_data")
print(f"  {CATALOG}.{SCHEMA}.signal_holdout_data")

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `cell_towers` | Synthesized cell tower reference data (8 towers) |
# MAGIC | `signal_training_data` | 80% split — ML-ready features + RSRP target |
# MAGIC | `signal_holdout_data` | 20% split — for batch scoring evaluation |
