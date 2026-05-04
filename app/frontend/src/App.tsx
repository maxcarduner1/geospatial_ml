import React from "react";
import HexMap from "./components/HexMap";
import Sidebar from "./components/Sidebar";
import { useAppData } from "./hooks/useAppData";

export default function App() {
  const {
    hexes,
    towers,
    predictions,
    loading,
    predicting,
    error,
    selectedHex,
    hasPredictions,
    predictAll,
    predictSingle,
    selectHex,
  } = useAppData();

  return (
    <div className="app-layout">
      <Sidebar
        hexCount={hexes.length}
        towers={towers}
        predictions={predictions}
        selectedHex={selectedHex}
        hasPredictions={hasPredictions}
        predicting={predicting}
        loading={loading}
        error={error}
        onPredictAll={predictAll}
      />

      <div className="map-container">
        {loading && (
          <div className="loading-overlay">
            <div className="spinner" />
          </div>
        )}

        {predicting && (
          <div className="loading-overlay">
            <div style={{ textAlign: "center" }}>
              <div className="spinner" style={{ margin: "0 auto 12px" }} />
              <div style={{ fontSize: 14, color: "#94a3b8" }}>
                Running predictions on serving endpoint...
              </div>
            </div>
          </div>
        )}

        <HexMap
          hexes={hexes}
          predictions={predictions}
          towers={towers}
          hasPredictions={hasPredictions}
          onHexClick={predictSingle}
        />
      </div>
    </div>
  );
}
