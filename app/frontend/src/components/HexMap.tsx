import React, { useMemo, useCallback } from "react";
import { Map as MapGL } from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { ScatterplotLayer } from "@deck.gl/layers";
import type { HexData, PredictedHex, CellTower } from "../types";
import { h3IndexToString, rsrpToColor } from "../utils";

const CHANDLER_CENTER = {
  longitude: -111.8713,
  latitude: 33.3062,
  zoom: 15,
  pitch: 45,
  bearing: -15,
};

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

interface HexMapProps {
  hexes: HexData[];
  predictions: PredictedHex[];
  towers: CellTower[];
  hasPredictions: boolean;
  onHexClick: (h3Index: number) => void;
}

export default function HexMap({
  hexes,
  predictions,
  towers,
  hasPredictions,
  onHexClick,
}: HexMapProps) {
  // Build a lookup for predicted hexes
  const predictionMap = useMemo(() => {
    const map = new Map<number, PredictedHex>();
    for (const p of predictions) {
      map.set(p.h3_index, p);
    }
    return map;
  }, [predictions]);

  // Compute RSRP range from actual data for color scaling
  const rsrpRange = useMemo(() => {
    const values = hexes.map((h) => h.avg_rsrp);
    if (values.length === 0) return { min: -120, max: -60 };
    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [hexes]);

  const getTooltip = useCallback(
    ({ object }: any) => {
      if (!object) return null;

      const h3Index = object.h3_index;
      const pred = predictionMap.get(h3Index);
      const hex = hexes.find((h) => h.h3_index === h3Index);
      const h3Str = h3IndexToString(h3Index);

      let html = `<div style="min-width:200px">`;
      html += `<div style="font-weight:700;margin-bottom:8px;font-size:14px">H3 Hex</div>`;
      html += `<div style="font-size:11px;color:#94a3b8;margin-bottom:8px;font-family:monospace;word-break:break-all">${h3Str}</div>`;

      if (hex) {
        html += `<div style="display:flex;justify-content:space-between;padding:3px 0"><span style="color:#94a3b8">Actual RSRP</span><span style="font-family:monospace;font-weight:600">${hex.avg_rsrp.toFixed(1)} dBm</span></div>`;
      }

      if (pred && pred.predicted_rsrp !== null) {
        html += `<div style="display:flex;justify-content:space-between;padding:3px 0"><span style="color:#94a3b8">Predicted RSRP</span><span style="font-family:monospace;font-weight:600;color:#22c55e">${pred.predicted_rsrp.toFixed(1)} dBm</span></div>`;
        html += `<div style="display:flex;justify-content:space-between;padding:3px 0"><span style="color:#94a3b8">Error</span><span style="font-family:monospace;font-weight:600;color:${Math.abs(pred.error!) < 3 ? "#22c55e" : Math.abs(pred.error!) < 6 ? "#eab308" : "#ef4444"}">${pred.error! > 0 ? "+" : ""}${pred.error!.toFixed(2)} dBm</span></div>`;
      } else {
        html += `<div style="padding:3px 0;color:#64748b;font-style:italic;margin-top:4px">Click to predict</div>`;
      }

      html += `</div>`;
      return { html, className: "deck-tooltip" };
    },
    [hexes, predictionMap]
  );

  const layers = useMemo(() => {
    // H3 hexagon layer -- shows predicted color if available, else blue tint
    const hexLayer = new H3HexagonLayer({
      id: "h3-hexes",
      data: hexes,
      pickable: true,
      filled: true,
      extruded: hasPredictions,
      elevationScale: hasPredictions ? 4 : 0,
      getHexagon: (d: HexData) => h3IndexToString(d.h3_index),
      getFillColor: (d: HexData) => {
        const pred = predictionMap.get(d.h3_index);
        if (pred && pred.predicted_rsrp !== null) {
          return rsrpToColor(pred.predicted_rsrp, rsrpRange.min, rsrpRange.max);
        }
        // Before predictions, color by actual RSRP with lower opacity
        return rsrpToColor(d.avg_rsrp, rsrpRange.min, rsrpRange.max).map(
          (v, i) => (i === 3 ? 100 : v)
        ) as [number, number, number, number];
      },
      getElevation: (d: HexData) => {
        if (!hasPredictions) return 0;
        const pred = predictionMap.get(d.h3_index);
        if (pred && pred.predicted_rsrp !== null) {
          // Elevation based on signal strength -- stronger signal = taller
          const normalized =
            (pred.predicted_rsrp - rsrpRange.min) /
            (rsrpRange.max - rsrpRange.min);
          return Math.max(0, normalized) * 50;
        }
        return 0;
      },
      onClick: ({ object }: any) => {
        if (object) onHexClick(object.h3_index);
      },
      updateTriggers: {
        getFillColor: [predictions.length, hasPredictions],
        getElevation: [predictions.length, hasPredictions],
      },
      transitions: {
        getFillColor: 300,
        getElevation: 500,
      },
    });

    // Cell tower markers
    const towerLayer = new ScatterplotLayer({
      id: "cell-towers",
      data: towers,
      pickable: true,
      stroked: true,
      filled: true,
      radiusMinPixels: 6,
      radiusMaxPixels: 12,
      lineWidthMinPixels: 2,
      getPosition: (d: CellTower) => [d.tower_lon, d.tower_lat],
      getFillColor: (d: CellTower) =>
        d.tower_type === "macro" ? [245, 158, 11, 230] : [139, 92, 246, 230],
      getLineColor: [255, 255, 255, 200],
      getRadius: 30,
    });

    return [hexLayer, towerLayer];
  }, [hexes, predictions, towers, hasPredictions, predictionMap, rsrpRange, onHexClick]);

  return (
    <DeckGL
      initialViewState={CHANDLER_CENTER}
      controller={true}
      layers={layers}
      getTooltip={getTooltip}
      style={{ position: "relative", width: "100%", height: "100%" }}
    >
      <MapGL reuseMaps mapStyle={MAP_STYLE} />
    </DeckGL>
  );
}
