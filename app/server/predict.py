"""
Model serving client.

Calls the `signal-strength-predictor` endpoint with h3_index lookup keys.
The model was logged with `fe.log_model()` so the endpoint automatically
retrieves features from the online store -- only h3_index is needed.

The frontend sends H3 hex strings (e.g. "8a2ab1072c97fff") to avoid
JavaScript BigInt precision issues.  This module converts them back to
integers for the serving endpoint payload.
"""

from typing import Optional

import httpx

from server.config import SERVING_ENDPOINT, get_workspace_host, get_oauth_token


def _h3_hex_to_int(h3_hex: str) -> int:
    """Convert an H3 hex string back to the BIGINT the model expects."""
    return int(h3_hex, 16)


async def predict_rsrp(h3_hex_values: list[str]) -> list[Optional[float]]:
    """Call the serving endpoint for a batch of H3 hex string values.

    Args:
        h3_hex_values: List of H3 hex strings (e.g. ["8a2ab1072c97fff"]).

    Returns:
        List of predicted RSRP values (dBm), one per input hex.
    """
    host = get_workspace_host()
    token = get_oauth_token()
    url = f"{host}/serving-endpoints/{SERVING_ENDPOINT}/invocations"

    # Convert hex strings to ints for the serving endpoint
    h3_ints = [_h3_hex_to_int(h) for h in h3_hex_values]

    payload = {
        "dataframe_split": {
            "columns": ["h3_index"],
            "data": [[h3] for h3 in h3_ints],
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Serving endpoint returned {response.status_code}: {response.text}"
            )
        result = response.json()
        return result.get("predictions", [])
