"""
Database access layer using databricks-sql-connector.

Queries the SQL warehouse for hex data, features, and cell tower locations.

Hex boundaries are computed server-side using Databricks SQL's built-in
h3_boundaryasgeojson() function, so the frontend does not need h3-js
to render polygons — it uses deck.gl's PolygonLayer directly.
"""

import json
import os
from databricks.sql import connect
from databricks.sql.client import Connection

from server.config import (
    CATALOG,
    SCHEMA,
    IS_DATABRICKS_APP,
    get_workspace_client,
    get_oauth_token,
)

# ---------------------------------------------------------------------------
# Decode maps for categorical feature encodings
# ---------------------------------------------------------------------------
TOWER_TYPE_MAP = {0: "Macro", 1: "Small Cell", -1: "Unknown"}
FREQ_BAND_MAP = {0: "n71", 1: "B66", 2: "n41", 3: "B2", -1: "Unknown"}
NETWORK_TYPE_MAP = {0: "LTE", 1: "NR (5G)", 2: "UMTS", 3: "HSPAP", 4: "HSPA", -1: "Unknown"}


def _h3_int_to_hex(h3_int: int) -> str:
    """Convert an H3 BIGINT to the hex string that h3-js expects."""
    return format(int(h3_int), "x")


def _get_connection() -> Connection:
    """Create a Databricks SQL connection."""
    warehouse_id = os.environ.get("SQL_WAREHOUSE", os.environ.get("DATABRICKS_WAREHOUSE_ID", ""))

    if IS_DATABRICKS_APP:
        host = os.environ.get("DATABRICKS_HOST", "")
        token = get_oauth_token()
    else:
        w = get_workspace_client()
        host = w.config.host.replace("https://", "")
        token = get_oauth_token()

    return connect(
        server_hostname=host if IS_DATABRICKS_APP else host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}" if warehouse_id else "",
        access_token=token,
    )


def fetch_all_hex_data() -> list[dict]:
    """Fetch ALL hexes with features, observations, and batch predictions.

    Joins:
      - signal_features: feature columns + lat/lon
      - signal_observations: actual RSRP + split (train/holdout)
      - signal_predictions: batch-scored predicted RSRP (holdout only)

    Returns decoded categorical features for human-readable tooltips.
    """
    # NOTE: The stored h3_index values were computed with swapped lat/lon args
    # in h3_longlatash3(). We recompute the correct H3 index and boundary here
    # from the stored (correct) lat/lon columns. Joins still use the stored
    # h3_index since all tables share the same (wrong) values consistently.
    sql = f"""
        SELECT
            f.h3_index,
            h3_longlatash3(f.longitude, f.latitude, 10) AS correct_h3,
            h3_boundaryasgeojson(h3_longlatash3(f.longitude, f.latitude, 10)) AS hex_boundary,
            f.latitude,
            f.longitude,
            f.avg_distance_to_nearest_tower,
            f.dominant_tower_type_enc,
            f.dominant_freq_band_enc,
            f.avg_tower_count_within_500m,
            f.dominant_network_type_enc,
            f.avg_wifi_rssi,
            f.measurement_count,
            o.avg_rsrp   AS actual_rsrp,
            o.split,
            p.prediction AS predicted_rsrp
        FROM `{CATALOG}`.`{SCHEMA}`.signal_features f
        LEFT JOIN `{CATALOG}`.`{SCHEMA}`.signal_observations o
            ON f.h3_index = o.h3_index
        LEFT JOIN `{CATALOG}`.`{SCHEMA}`.signal_predictions p
            ON f.h3_index = p.h3_index
        ORDER BY f.h3_index
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(zip(columns, row))
                # Keep the stored h3_index as hex string — the model/online
                # store uses these (wrong) values consistently, so predictions
                # must send the stored values to get a feature lookup match.
                d["h3_hex"] = _h3_int_to_hex(d.pop("h3_index"))
                d.pop("correct_h3")  # used only for boundary computation
                # Parse GeoJSON boundary → [[lng, lat], ...] coordinate ring
                geojson = json.loads(d.pop("hex_boundary"))
                d["boundary"] = geojson["coordinates"][0]  # outer ring
                # Decode categoricals to human-readable strings
                d["tower_type"] = TOWER_TYPE_MAP.get(d.pop("dominant_tower_type_enc", -1), "Unknown")
                d["freq_band"] = FREQ_BAND_MAP.get(d.pop("dominant_freq_band_enc", -1), "Unknown")
                d["network_type"] = NETWORK_TYPE_MAP.get(d.pop("dominant_network_type_enc", -1), "Unknown")
                # Round floats for cleaner JSON
                d["avg_distance_to_nearest_tower"] = round(float(d["avg_distance_to_nearest_tower"]), 1) if d["avg_distance_to_nearest_tower"] is not None else None
                d["avg_tower_count_within_500m"] = round(float(d["avg_tower_count_within_500m"]), 1) if d["avg_tower_count_within_500m"] is not None else None
                d["avg_wifi_rssi"] = round(float(d["avg_wifi_rssi"]), 1) if d["avg_wifi_rssi"] is not None else None
                d["measurement_count"] = int(d["measurement_count"]) if d["measurement_count"] is not None else None
                d["actual_rsrp"] = round(float(d["actual_rsrp"]), 2) if d["actual_rsrp"] is not None else None
                d["predicted_rsrp"] = round(float(d["predicted_rsrp"]), 2) if d["predicted_rsrp"] is not None else None
                d["latitude"] = float(d["latitude"])
                d["longitude"] = float(d["longitude"])
                results.append(d)
            return results
    finally:
        conn.close()


def fetch_cell_towers() -> list[dict]:
    """Fetch cell tower reference data."""
    sql = f"""
        SELECT tower_id, tower_lat, tower_lon, tower_type, freq_band
        FROM `{CATALOG}`.`{SCHEMA}`.cell_towers
        ORDER BY tower_id
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()
