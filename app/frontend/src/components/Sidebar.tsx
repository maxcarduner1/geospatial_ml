import React from "react";
import {
  Radio,
  Zap,
  MapPin,
  BarChart3,
  Activity,
  Hexagon,
} from "lucide-react";
import type { PredictedHex, CellTower } from "../types";
import { formatRsrp, computeMetrics } from "../utils";

interface SidebarProps {
  hexCount: number;
  towers: CellTower[];
  predictions: PredictedHex[];
  selectedHex: PredictedHex | null;
  hasPredictions: boolean;
  predicting: boolean;
  loading: boolean;
  error: string | null;
  onPredictAll: () => void;
}

export default function Sidebar({
  hexCount,
  towers,
  predictions,
  selectedHex,
  hasPredictions,
  predicting,
  loading,
  error,
  onPredictAll,
}: SidebarProps) {
  const metrics = computeMetrics(predictions);

  return (
    <div className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <h1>
          <Radio size={18} style={{ verticalAlign: "middle", marginRight: 6 }} />
          Signal Strength Prediction
        </h1>
        <p>
          Real-time RSRP predictions for H3 hex cells near Chandler Fashion
          Center, AZ. Powered by LightGBM on Databricks.
        </p>
      </div>

      <div className="sidebar-body">
        {/* Error banner */}
        {error && <div className="error-banner">{error}</div>}

        {/* Status */}
        <div style={{ marginBottom: 16 }}>
          {loading ? (
            <span className="status-badge loading">
              <span className="status-dot" /> Loading data
            </span>
          ) : hasPredictions ? (
            <span className="status-badge ready">
              <span className="status-dot" /> Predictions loaded
            </span>
          ) : (
            <span className="status-badge loading">
              <span className="status-dot" /> Data loaded
            </span>
          )}
        </div>

        {/* Stats */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="label">Holdout Hexes</div>
            <div className="value">
              {hexCount}
              <span className="unit">cells</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Cell Towers</div>
            <div className="value">
              {towers.length}
              <span className="unit">sites</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">Predicted</div>
            <div className="value">
              {predictions.filter((p) => p.predicted_rsrp !== null).length}
              <span className="unit">hexes</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">MAE</div>
            <div className="value">
              {hasPredictions ? metrics.mae.toFixed(2) : "--"}
              <span className="unit">dBm</span>
            </div>
          </div>
        </div>

        {/* Predict All button */}
        <button
          className="btn btn-primary btn-full"
          onClick={onPredictAll}
          disabled={predicting || loading || hexCount === 0}
          style={{ marginBottom: 20 }}
        >
          <Zap size={16} />
          {predicting
            ? "Running predictions..."
            : hasPredictions
              ? "Re-run All Predictions"
              : "Predict All Hexes"}
        </button>

        {/* Legend */}
        <div className="legend">
          <h3>RSRP Signal Strength</h3>
          <div className="legend-gradient" />
          <div className="legend-labels">
            <span>-120 dBm (weak)</span>
            <span>-90 dBm</span>
            <span>-60 dBm (strong)</span>
          </div>
        </div>

        {/* Prediction metrics */}
        {hasPredictions && (
          <div className="metrics-summary">
            <h3>
              <BarChart3 size={14} style={{ verticalAlign: "middle" }} />{" "}
              Prediction Metrics
            </h3>
            <div className="metrics-row">
              <span className="key">RMSE</span>
              <span className="val">{metrics.rmse.toFixed(3)} dBm</span>
            </div>
            <div className="metrics-row">
              <span className="key">MAE</span>
              <span className="val">{metrics.mae.toFixed(3)} dBm</span>
            </div>
            <div className="metrics-row">
              <span className="key">Avg Actual</span>
              <span className="val">{metrics.avgActual.toFixed(1)} dBm</span>
            </div>
            <div className="metrics-row">
              <span className="key">Avg Predicted</span>
              <span className="val">{metrics.avgPredicted.toFixed(1)} dBm</span>
            </div>
          </div>
        )}

        {/* Selected hex detail */}
        {selectedHex && (
          <div className="hex-detail">
            <h3>
              <Hexagon size={14} /> Selected Hex
            </h3>
            <div className="hex-detail-row">
              <span className="key">H3 Index</span>
              <span className="val" style={{ fontSize: 11, letterSpacing: "-0.02em" }}>
                {BigInt(selectedHex.h3_index).toString(16)}
              </span>
            </div>
            <div className="hex-detail-row">
              <span className="key">Location</span>
              <span className="val" style={{ fontSize: 12 }}>
                {selectedHex.latitude.toFixed(4)}, {selectedHex.longitude.toFixed(4)}
              </span>
            </div>
            <div className="hex-detail-row">
              <span className="key">Actual RSRP</span>
              <span className="val">{formatRsrp(selectedHex.actual_rsrp)}</span>
            </div>
            <div className="hex-detail-row">
              <span className="key">Predicted RSRP</span>
              <span className="val good">
                {formatRsrp(selectedHex.predicted_rsrp)}
              </span>
            </div>
            <div className="hex-detail-row">
              <span className="key">Error</span>
              <span
                className={`val ${
                  selectedHex.error !== null
                    ? Math.abs(selectedHex.error) < 3
                      ? "good"
                      : Math.abs(selectedHex.error) < 6
                        ? "neutral"
                        : "bad"
                    : ""
                }`}
              >
                {selectedHex.error !== null
                  ? `${selectedHex.error > 0 ? "+" : ""}${selectedHex.error.toFixed(2)} dBm`
                  : "--"}
              </span>
            </div>

            {/* Comparison bars */}
            {selectedHex.predicted_rsrp !== null && (
              <div style={{ marginTop: 12 }}>
                <ComparisonBar
                  label="Actual"
                  value={selectedHex.actual_rsrp}
                  className="actual"
                />
                <ComparisonBar
                  label="Predicted"
                  value={selectedHex.predicted_rsrp}
                  className="predicted"
                />
              </div>
            )}
          </div>
        )}

        {/* Tower list */}
        <div className="section-title">
          <MapPin size={12} style={{ verticalAlign: "middle", marginRight: 4 }} />
          Cell Towers
        </div>
        <div className="tower-list">
          {towers.map((t) => (
            <div className="tower-item" key={t.tower_id}>
              <div className={`tower-dot ${t.tower_type}`} />
              <div className="tower-info">
                <div className="tower-name">Tower {t.tower_id}</div>
                <div className="tower-meta">
                  {t.tower_type} | {t.freq_band} | {t.tower_lat.toFixed(4)},{" "}
                  {t.tower_lon.toFixed(4)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** A simple horizontal comparison bar. */
function ComparisonBar({
  label,
  value,
  className,
}: {
  label: string;
  value: number;
  className: string;
}) {
  // Normalize: RSRP typically ranges from -120 to -60
  const pct = Math.max(0, Math.min(100, ((value + 120) / 60) * 100));

  return (
    <div className="comparison-row">
      <span className="comparison-label">{label}</span>
      <div className="comparison-bar-wrap">
        <div
          className={`comparison-bar ${className}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="comparison-value">{value.toFixed(1)}</span>
    </div>
  );
}
