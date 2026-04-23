I need you to help me prep and end to end ML demo on databricks related to a geospatial workload where we want to predict the expected signal strength of a h3 bin given it's relative positioning to a cell tower (and any other relevant data fields you can think of) 
use cmeg workspace
use cmegdemos_catalog.geospatial_analytics schema
use signal points dataset as a source table to develop target from potentially (or just make it up) and create another table that's ready for training a ML model
train a litegbm model and see resource link below for how to use with mlflow but make sure we are using the Unity Catalog version of MLflow where we are registering models to UC
show an example scoring pipeline of calling said model on a holdout set of the training data in a batch manor
also show an example request to a served version of the model to a model serving endpoint 
use databricks asset bundles


resources:
https://mlflow.org/docs/latest/api_reference/python_api/mlflow.lightgbm.html
https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints
