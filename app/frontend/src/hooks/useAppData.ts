import { useState, useEffect, useCallback } from "react";
import type { HexData, PredictedHex, CellTower } from "../types";

interface AppState {
  hexes: HexData[];
  towers: CellTower[];
  predictions: PredictedHex[];
  loading: boolean;
  predicting: boolean;
  error: string | null;
  selectedHex: PredictedHex | null;
  hasPredictions: boolean;
}

export function useAppData() {
  const [state, setState] = useState<AppState>({
    hexes: [],
    towers: [],
    predictions: [],
    loading: true,
    predicting: false,
    error: null,
    selectedHex: null,
    hasPredictions: false,
  });

  // ------------------------------------------------------------------
  // Initial data load
  // ------------------------------------------------------------------
  useEffect(() => {
    async function loadData() {
      try {
        const [hexRes, towerRes] = await Promise.all([
          fetch("/api/hexes"),
          fetch("/api/towers"),
        ]);

        if (!hexRes.ok) throw new Error(`Hex fetch failed: ${hexRes.status}`);
        if (!towerRes.ok)
          throw new Error(`Tower fetch failed: ${towerRes.status}`);

        const hexData = await hexRes.json();
        const towerData = await towerRes.json();

        setState((prev) => ({
          ...prev,
          hexes: hexData.hexes,
          towers: towerData.towers,
          loading: false,
        }));
      } catch (err: any) {
        setState((prev) => ({
          ...prev,
          loading: false,
          error: err.message || "Failed to load data",
        }));
      }
    }

    loadData();
  }, []);

  // ------------------------------------------------------------------
  // Predict all holdout hexes
  // ------------------------------------------------------------------
  const predictAll = useCallback(async () => {
    setState((prev) => ({ ...prev, predicting: true, error: null }));

    try {
      const res = await fetch("/api/predict-all", { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Prediction failed: ${res.status}`);
      }

      const data = await res.json();
      setState((prev) => ({
        ...prev,
        predictions: data.predictions,
        predicting: false,
        hasPredictions: true,
      }));
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        predicting: false,
        error: err.message || "Prediction failed",
      }));
    }
  }, []);

  // ------------------------------------------------------------------
  // Predict a single hex (on click)
  // ------------------------------------------------------------------
  const predictSingle = useCallback(
    async (h3Index: number) => {
      // If we already have predictions, just select it
      const existing = state.predictions.find((p) => p.h3_index === h3Index);
      if (existing) {
        setState((prev) => ({ ...prev, selectedHex: existing }));
        return;
      }

      // Otherwise, get prediction from the endpoint
      setState((prev) => ({ ...prev, predicting: true, error: null }));

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ h3_indices: [h3Index] }),
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `Prediction failed: ${res.status}`);
        }

        const data = await res.json();
        const hexInfo = state.hexes.find((h) => h.h3_index === h3Index);

        if (hexInfo && data.predictions && data.predictions.length > 0) {
          const predicted = data.predictions[0];
          const actual = hexInfo.avg_rsrp;
          const newPrediction: PredictedHex = {
            h3_index: h3Index,
            latitude: hexInfo.latitude,
            longitude: hexInfo.longitude,
            actual_rsrp: Math.round(actual * 100) / 100,
            predicted_rsrp:
              predicted != null ? Math.round(predicted * 100) / 100 : null,
            error:
              predicted != null
                ? Math.round((predicted - actual) * 100) / 100
                : null,
          };

          setState((prev) => ({
            ...prev,
            predictions: [...prev.predictions, newPrediction],
            selectedHex: newPrediction,
            predicting: false,
          }));
        }
      } catch (err: any) {
        setState((prev) => ({
          ...prev,
          predicting: false,
          error: err.message || "Prediction failed",
        }));
      }
    },
    [state.hexes, state.predictions]
  );

  // ------------------------------------------------------------------
  // Select a hex
  // ------------------------------------------------------------------
  const selectHex = useCallback(
    (h3Index: number | null) => {
      if (h3Index === null) {
        setState((prev) => ({ ...prev, selectedHex: null }));
        return;
      }
      const hex = state.predictions.find((p) => p.h3_index === h3Index);
      setState((prev) => ({ ...prev, selectedHex: hex || null }));
    },
    [state.predictions]
  );

  return {
    ...state,
    predictAll,
    predictSingle,
    selectHex,
  };
}
