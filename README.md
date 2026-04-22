# Geospatial ML — Signal Strength Prediction on Databricks

End-to-end Databricks demo that predicts **RSRP cellular signal strength** (dBm) for geographic locations relative to synthetic cell towers. Features include **H3 spatial indexing**, distances and tower density, temporal signals, and encoded network metadata. Models are trained with **LightGBM**, tracked in **MLflow**, and registered in **Unity Catalog**.

## What’s in this repo

| Piece | Purpose |
|--------|---------|
| [`databricks.yml`](databricks.yml) | Databricks Asset Bundle (DAB): serverless job pipeline and workspace target |
| [`notebooks/01_feature_engineering.py`](notebooks/01_feature_engineering.py) | Builds ML-ready training/holdout tables from `signal_points` |
| [`notebooks/02_train_model.py`](notebooks/02_train_model.py) | Trains LightGBM, logs runs, registers `signal_strength_predictor` to UC |
| [`notebooks/03_batch_scoring.py`](notebooks/03_batch_scoring.py) | Loads Champion model, batch-scores holdout data |
| [`notebooks/04_model_serving.py`](notebooks/04_model_serving.py) | Creates/updates a Model Serving endpoint and sample inference (run interactively; not in the job) |

## Data & Unity Catalog

- **Workspace:** Field Engineering CMEG demos — `https://fevm-cmegdemos.cloud.databricks.com` (CLI profile `fevm-cmegdemos`).
- **Catalog / schema:** `cmegdemos_catalog.geospatial_analytics`
- **Source table:** `signal_points`
- **Derived tables:** training and holdout tables produced by notebook 01 (e.g. `signal_training_data`, `signal_holdout_data`).
- **Registered model:** `{catalog}.{schema}.signal_strength_predictor` with MLflow registry URI `databricks-uc`.

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) with auth for profile `fevm-cmegdemos`.
- Unity Catalog access to `cmegdemos_catalog.geospatial_analytics` and MLflow UC model registry permissions.

## Deploy and run the pipeline

From this directory:

```bash
databricks bundle deploy --profile fevm-cmegdemos
databricks bundle run ml_geospatial_pipeline --profile fevm-cmegdemos
```

Override catalog/schema if needed via bundle variables (defined in `databricks.yml`).

The job runs three tasks in order:

1. **feature_engineering** — features + labels for regression on RSRP  
2. **train_model** — LightGBM + MLflow + UC registration  
3. **batch_scoring** — Champion model, holdout evaluation, persisted predictions  

Job dependencies (`lightgbm`, `mlflow`, `pandas`, etc.) are declared in the bundle environment spec (serverless client `"5"`).

## Model serving (interactive)

After the Champion alias exists on the UC model, open **`04_model_serving.py`** in the workspace. It provisions or updates a **serverless Model Serving** endpoint (optional scale-to-zero) and shows how to issue inference requests against the endpoint.

See [Create and manage model serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints) for operations and monitoring.

## References

- [MLflow LightGBM API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.lightgbm.html)
- [Databricks Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)

## License

Use and modify for demos and internal Field Engineering purposes unless you add a separate license.
