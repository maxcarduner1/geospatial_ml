/** A holdout hex with coordinates and actual RSRP from the SQL query. */
export interface HexData {
  h3_index: number;
  latitude: number;
  longitude: number;
  avg_rsrp: number;
}

/** A hex after prediction, combining actual + predicted values. */
export interface PredictedHex {
  h3_index: number;
  latitude: number;
  longitude: number;
  actual_rsrp: number;
  predicted_rsrp: number | null;
  error: number | null;
}

/** A cell tower from the reference table. */
export interface CellTower {
  tower_id: number;
  tower_lat: number;
  tower_lon: number;
  tower_type: string;
  freq_band: string;
}
