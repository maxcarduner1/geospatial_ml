"""
Databricks App entry point -- FastAPI backend serving the static frontend
and API routes for hex data, cell towers, and model predictions.
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.db import fetch_all_hex_data, fetch_cell_towers
from server.predict import predict_rsrp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache so we only query SQL warehouse once on first load
# ---------------------------------------------------------------------------
_cache: dict = {"hexes": None, "towers": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Signal Strength Prediction App starting up")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Signal Strength Prediction App", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    h3_indices: list[str]  # hex strings like "8a2ab1072c97fff"


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/hexes")
async def get_hexes():
    """Return ALL hexes with features, observations, and batch predictions."""
    if _cache["hexes"] is None:
        try:
            _cache["hexes"] = fetch_all_hex_data()
            predicted = sum(1 for h in _cache["hexes"] if h["predicted_rsrp"] is not None)
            logger.info(f"Loaded {len(_cache['hexes'])} hexes ({predicted} with predictions)")
        except Exception as e:
            logger.error(f"Failed to fetch hex data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"hexes": _cache["hexes"]}


@app.get("/api/towers")
async def get_towers():
    """Return cell tower reference data."""
    if _cache["towers"] is None:
        try:
            _cache["towers"] = fetch_cell_towers()
            logger.info(f"Loaded {len(_cache['towers'])} cell towers")
        except Exception as e:
            logger.error(f"Failed to fetch tower data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"towers": _cache["towers"]}


@app.post("/api/predict")
async def predict(request: PredictRequest):
    """Call the serving endpoint for selected h3 hex values.

    Returns the raw prediction values from the model. The frontend
    matches these back to the hex data for display.
    """
    if not request.h3_indices:
        raise HTTPException(status_code=400, detail="h3_indices cannot be empty")
    try:
        predictions = await predict_rsrp(request.h3_indices)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Serve static frontend (single HTML file — no build step needed)
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{full_path:path}")
    async def serve_static(full_path: str):
        """Serve static files or fall back to index.html."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Run with uvicorn when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
