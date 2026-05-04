"""
Configuration and authentication helpers.

Detects whether we are running inside a Databricks App (service principal
credentials auto-injected) or locally (uses CLI profile).
"""

import os
from databricks.sdk import WorkspaceClient

IS_DATABRICKS_APP = bool(os.environ.get("DATABRICKS_APP_NAME"))

CATALOG = "cmegdemos_catalog"
SCHEMA = "geospatial_analytics"
SERVING_ENDPOINT = os.environ.get("SERVING_ENDPOINT", "signal-strength-predictor")

# Databricks profile for local development
LOCAL_PROFILE = "fevm-cmegdemos"


def get_workspace_client() -> WorkspaceClient:
    """Return an authenticated WorkspaceClient."""
    if IS_DATABRICKS_APP:
        return WorkspaceClient()
    return WorkspaceClient(profile=LOCAL_PROFILE)


def get_workspace_host() -> str:
    """Return the workspace URL with https:// prefix."""
    if IS_DATABRICKS_APP:
        host = os.environ.get("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            host = f"https://{host}"
        return host
    w = get_workspace_client()
    return w.config.host


def get_oauth_token() -> str:
    """Get an OAuth token for Databricks API calls."""
    w = get_workspace_client()
    if w.config.token:
        return w.config.token
    auth_headers = w.config.authenticate()
    if auth_headers and "Authorization" in auth_headers:
        return auth_headers["Authorization"].replace("Bearer ", "")
    raise RuntimeError("Unable to obtain authentication token")
