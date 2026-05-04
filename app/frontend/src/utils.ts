/**
 * Convert a BigInt H3 index to the hex string that h3-js expects.
 *
 * H3 indices in the database are BIGINT (e.g. 622236750609498111).
 * h3-js expects a 15-character hex string (e.g. "8a2ab1072c97fff").
 *
 * We convert via BigInt -> hex string, then zero-pad to 15 chars.
 */
export function h3IndexToString(h3Index: number | bigint): string {
  // Use BigInt to avoid floating-point precision issues with large integers
  const bi = BigInt(h3Index);
  return bi.toString(16).padStart(15, "0");
}

/**
 * RSRP color scale: red (weak) -> yellow (medium) -> green (strong).
 *
 * Typical RSRP range: -120 dBm (very weak) to -60 dBm (excellent)
 */
export function rsrpToColor(
  rsrp: number,
  minRsrp: number = -120,
  maxRsrp: number = -60
): [number, number, number, number] {
  // Normalize to 0..1
  const t = Math.max(0, Math.min(1, (rsrp - minRsrp) / (maxRsrp - minRsrp)));

  let r: number, g: number, b: number;

  if (t < 0.5) {
    // Red -> Yellow (0.0 -> 0.5)
    const s = t * 2;
    r = 239;
    g = Math.round(68 + (158 - 68) * s);
    b = Math.round(68 * (1 - s) + 11 * s);
  } else {
    // Yellow -> Green (0.5 -> 1.0)
    const s = (t - 0.5) * 2;
    r = Math.round(245 * (1 - s) + 34 * s);
    g = Math.round(158 + (197 - 158) * s);
    b = Math.round(11 * (1 - s) + 94 * s);
  }

  return [r, g, b, 180];
}

/**
 * Color for hexes that have no prediction yet (just showing actual data).
 * Semi-transparent blue.
 */
export function unpredictedColor(): [number, number, number, number] {
  return [59, 130, 246, 100];
}

/**
 * Format RSRP value for display.
 */
export function formatRsrp(value: number | null): string {
  if (value === null || value === undefined) return "--";
  return `${value.toFixed(1)} dBm`;
}

/**
 * Compute summary metrics from predictions.
 */
export function computeMetrics(
  predictions: Array<{ actual_rsrp: number; predicted_rsrp: number | null }>
): {
  mae: number;
  rmse: number;
  count: number;
  avgActual: number;
  avgPredicted: number;
} {
  const valid = predictions.filter((p) => p.predicted_rsrp !== null);
  if (valid.length === 0) {
    return { mae: 0, rmse: 0, count: 0, avgActual: 0, avgPredicted: 0 };
  }

  let sumAbsErr = 0;
  let sumSqErr = 0;
  let sumActual = 0;
  let sumPredicted = 0;

  for (const p of valid) {
    const err = p.predicted_rsrp! - p.actual_rsrp;
    sumAbsErr += Math.abs(err);
    sumSqErr += err * err;
    sumActual += p.actual_rsrp;
    sumPredicted += p.predicted_rsrp!;
  }

  return {
    mae: sumAbsErr / valid.length,
    rmse: Math.sqrt(sumSqErr / valid.length),
    count: valid.length,
    avgActual: sumActual / valid.length,
    avgPredicted: sumPredicted / valid.length,
  };
}
