# Geospatial ML — Signal Strength Prediction on Databricks

**Repository:** [github.com/maxcarduner1/geospatial_ml](https://github.com/maxcarduner1/geospatial_ml)

End-to-end Databricks demo that predicts **RSRP cellular signal strength** (dBm) from handset readings relative to **synthesized cell towers** (Haversine distance, tower density). Features use **H3** indexing (resolution 9), temporal attributes, and encoded network fields. Models are trained with **LightGBM**, packaged with the **Databricks Feature Engineering** client (`FeatureLookup` + `fe.log_model()`), tracked in **MLflow**, and registered in **Unity Catalog**.

## Architecture

1. **Notebook 01** reads raw `signal_points`, engineers features, writes reference **`cell_towers`** (8 towers near Chandler Fashion Center), registers a governed **`signal_features`** feature table (primary key `signal_id`), and writes **`signal_observations`** (`signal_id`, `rsrp`, `split` = train/holdout).
2. **Notebook 02** builds a training set via **`create_training_set()`**, trains LightGBM, and logs/registers the model with feature metadata for UC.
3. **Notebook 03** scores holdout rows with **`fe.score_batch()`** so features are joined automatically from `signal_features`.
4. **Notebook 04** (manual) provisions **Model Serving** and shows sample requests.

## What’s in this repo

| Piece | Purpose |
|--------|---------|
| [`databricks.yml`](databricks.yml) | Databricks Asset Bundle (DAB): serverless job pipeline, `fevm-cmegdemos` workspace target |
| [`notebooks/01_feature_engineering.py`](notebooks/01_feature_engineering.py) | Raw data → `cell_towers`, `signal_features`, `signal_observations` |
| [`notebooks/02_train_model.py`](notebooks/02_train_model.py) | Feature Engineering training set → LightGBM → `fe.log_model()` → UC registry |
| [`notebooks/03_batch_scoring.py`](notebooks/03_batch_scoring.py) | Holdout via `signal_observations` → `fe.score_batch()` → metrics + persisted predictions |
| [`notebooks/04_model_serving.py`](notebooks/04_model_serving.py) | Serving endpoint + inference example (**interactive only**, not part of the job) |

## Data & Unity Catalog

| Asset | Description |
|-------|-------------|
| **Workspace** | [fevm-cmegdemos](https://fevm-cmegdemos.cloud.databricks.com) — CLI profile `fevm-cmegdemos` |
| **Catalog.schema** | `cmegdemos_catalog.geospatial_analytics` (override via bundle variables) |
| **`signal_points`** | Source telemetry |
| **`cell_towers`** | Synthetic tower reference for spatial features |
| **`signal_features`** | UC feature table (`signal_id` PK); CDF enabled for online patterns |
| **`signal_observations`** | Labels + 80/20 train/holdout split |
| **Registered model** | `{catalog}.{schema}.signal_strength_predictor` — MLflow registry URI `databricks-uc` |

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) authenticated as profile `fevm-cmegdemos`.
- UC access to the catalog/schema, Feature Engineering / feature table permissions, and MLflow model registry rights.

## Deploy and run

```bash
cd ml_geospatial   # or your clone path

databricks bundle deploy --profile fevm-cmegdemos
databricks bundle run ml_geospatial_pipeline --profile fevm-cmegdemos
```

Optional: set `catalog` and `schema` in `databricks.yml` variables or pass overrides supported by your CLI version for multi-environment demos.

### Job tasks

1. **feature_engineering** — Delta + Feature Table registration  
2. **train_model** — LightGBM, MLflow, UC model with packaged feature specs  
3. **batch_scoring** — Champion model, `score_batch`, evaluation output  

Serverless environment uses client `"5"` with dependencies including `lightgbm`, `mlflow`, `scikit-learn`, `databricks-sdk`, and **`databricks-feature-engineering`**.

## Model serving (interactive)

After the **Champion** alias exists on the UC model, run **`04_model_serving.py`** in the workspace to create or update a serverless endpoint (demo-friendly scale-to-zero). See [Model Serving — create and manage endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints).

## References

- [MLflow LightGBM API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.lightgbm.html)
- [Databricks Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)

## License

Use and modify for demos and internal Field Engineering purposes unless you add a separate license.
