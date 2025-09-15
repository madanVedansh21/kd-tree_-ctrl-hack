# kd_tree_upgrade.py
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import time
from bisect import bisect_left, bisect_right
from scipy.spatial import cKDTree
import warnings
import math
from typing import Dict, List


warnings.filterwarnings("ignore")
def format_dt_hours(dt_hours):
    seconds = dt_hours * 3600
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    return f"{days}d {hours}h {minutes}m"

def compute_confidence_score(temporal_score=0.0, spatial_score=0.0, significance_score=0.0,
                             reliability=1.0, weights=None):
    """
    Compute confidence score from weighted combination of temporal, spatial, and significance scores.
    Rules:
      - Any component that is zero or None is treated as missing and ignored.
      - Provided `weights` should be a mapping for available components (or a superset).
      - We renormalize weights across the **available** components only.
      - Returns tuple (adaptive_score, confidence_score) where adaptive_score is
        the renormalized weighted sum BEFORE applying reliability.
    """
    # default full set weights (used only to pick defaults if weights missing keys)
    default_weights = {"temporal": 0.4, "spatial": 0.35, "significance": 0.25}
    weights = weights or default_weights

    components = {
        "temporal": float(temporal_score) if temporal_score is not None else 0.0,
        "spatial": float(spatial_score) if spatial_score is not None else 0.0,
        "significance": float(significance_score) if significance_score is not None else 0.0,
    }

    # Keep only positive (non-zero) components — treat zero as 'missing' per your rule.
    available = {c: components[c] for c in components if components[c] > 0.0}

    if not available:
        return 0.0, 0.0

    # Build weight set for available components (fallback to default_weights if missing)
    w_avail = {c: float(weights.get(c, default_weights.get(c, 0.0))) for c in available.keys()}
    total_w = sum(w_avail.values())
    if total_w <= 0:
        # equally distribute if weights are degenerate
        normalized_w = {c: 1.0 / len(w_avail) for c in w_avail}
    else:
        normalized_w = {c: w_avail[c] / total_w for c in w_avail}

    # adaptive score (before reliability)
    adaptive_score = sum(normalized_w[c] * available[c] for c in available)

    # final confidence applying reliability
    confidence = adaptive_score * float(reliability)

    return adaptive_score, confidence


class RobustMultimessengerCorrelator:
    """
    Scalable multi-messenger correlator using KD-tree spatial matching and temporal binary search.
    """

    def __init__(self, weights=None, csv_directory="./data"):
        # default weights for temporal, spatial, significance
        self.weights = weights or {"temporal": 0.4, "spatial": 0.35, "significance": 0.25}
        self.csv_directory = csv_directory
        self.datasets = {}  # filename -> dataframe (cleaned)
        self.dataset_stats = {}
        self.combined_data = None
        # spatial / temporal indices per dataset (dataset name without .csv)
        self.spatial_kdtrees = {}  # name -> {'tree': cKDTree, 'positions': np.array, 'indices': np.array}
        self.temporal_indices = {}  # name -> {'times_sorted': np.array (seconds), 'indices_by_time': np.array}
        self.column_mappings = {
            "event_id": ["event_id", "id", "name", "event_name"],
            "source": ["source", "instrument", "detector", "origin"],
            "event_type": ["event_type", "type", "classification", "category"],
            "utc_time": ["utc_time", "time", "timestamp", "datetime", "obs_time"],
            "ra_deg": ["ra_deg", "ra", "right_ascension", "ra_j2000"],
            "dec_deg": ["dec_deg", "dec", "declination", "dec_j2000"],
            "pos_error_deg": ["pos_error_deg", "pos_error", "position_error", "error_radius"],
            "signal_strength": ["signal_strength", "flux", "snr", "magnitude", "amplitude"],
        }

    # -------------------------
    # Loading and cleaning
    # -------------------------
    def detect_and_map_columns(self, df: pd.DataFrame, filename: str) -> Dict[str, str]:
        detected_mapping = {}
        cols_lower = [c.lower().strip() for c in df.columns]
        for std_col, candidates in self.column_mappings.items():
            for cand in candidates:
                if cand.lower() in cols_lower:
                    detected_mapping[std_col] = df.columns[cols_lower.index(cand.lower())]
                    break
        return detected_mapping

    def clean_and_standardize_data(self, df: pd.DataFrame, column_mapping: Dict[str, str], filename: str):
        clean_df = pd.DataFrame()
        for std_col, actual in column_mapping.items():
            if actual in df.columns:
                clean_df[std_col] = df[actual].copy()

        # add dataset identifier
        dataset_name = os.path.basename(filename).replace(".csv", "")
        if "source" not in clean_df.columns:
            clean_df["source"] = dataset_name

        if "event_id" not in clean_df.columns:
            clean_df["event_id"] = [f"{dataset_name}_{i}" for i in range(len(clean_df))]

        # drop fully empty rows
        clean_df = clean_df.dropna(how="all").reset_index(drop=True)

        # convert times
        if "utc_time" in clean_df.columns:
            # handle a variety of time formats, ensure tz-aware (UTC)
            clean_df["utc_time"] = pd.to_datetime(clean_df["utc_time"], errors="coerce", utc=True)

        # numeric conversions
        for col in ["ra_deg", "dec_deg", "pos_error_deg", "signal_strength"]:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        # flags
        clean_df["has_temporal"] = clean_df["utc_time"].notna() if "utc_time" in clean_df.columns else False
        clean_df["has_spatial"] = (
            clean_df[["ra_deg", "dec_deg"]].notna().all(axis=1) if all(c in clean_df.columns for c in ["ra_deg", "dec_deg"]) else False
        )
        clean_df["has_signal"] = clean_df["signal_strength"].notna() if "signal_strength" in clean_df.columns else False

        # attach dataset column (name without extension)
        clean_df["dataset"] = dataset_name

        return clean_df

    def load_csv_files(self):
        if not os.path.exists(self.csv_directory):
            os.makedirs(self.csv_directory, exist_ok=True)
            self._create_sample_data()

        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        if not csv_files:
            self._create_sample_data()
            csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))

        all_data = []
        for csv_file in csv_files:
            fname = os.path.basename(csv_file)
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue
            if df.shape[0] == 0:
                continue
            mapping = self.detect_and_map_columns(df, fname)
            if not mapping:
                # try a permissive fallback: if file has at least ra/dec/time columns heuristically
                print(f"Skipping {fname}: no column mapping detected.")
                continue
            clean_df = self.clean_and_standardize_data(df, mapping, fname)
            if clean_df.shape[0] == 0:
                continue
            self.dataset_stats[fname] = {
                "total_events": len(clean_df),
                "temporal_events": int(clean_df["has_temporal"].sum()),
                "spatial_events": int(clean_df["has_spatial"].sum()),
                "signal_events": int(clean_df["has_signal"].sum()),
                "complete_events": int((clean_df["has_temporal"] & clean_df["has_spatial"] & clean_df["has_signal"]).sum()),
            }
            self.datasets[fname] = clean_df
            all_data.append(clean_df)

        if all_data:
            self.combined_data = pd.concat(all_data, ignore_index=True)
            # normalize signal strengths (z-score) for significance calculations
            if "signal_strength" in self.combined_data.columns:
                sig = self.combined_data["signal_strength"].dropna()
                if len(sig) >= 2:
                    mu, sigma = sig.mean(), sig.std(ddof=0)
                    if sigma == 0:
                        sigma = 1.0
                    self.combined_data["signal_z"] = (self.combined_data["signal_strength"] - mu) / sigma
                else:
                    self.combined_data["signal_z"] = 0.0
            else:
                self.combined_data["signal_z"] = 0.0

            # rebuild per-dataset cleaned dataframes from combined to ensure indices align
            for fname in list(self.datasets.keys()):
                name = fname
                df_sub = self.combined_data[self.combined_data["dataset"] == name.replace(".csv", "")].reset_index(drop=True)
                self.datasets[fname] = df_sub

            self._build_indices()
        else:
            self.combined_data = pd.DataFrame()

        return self

    # -------------------------
    # Index building
    # -------------------------
    def _spherical_to_cartesian(self, ra_deg_arr, dec_deg_arr):
        ra = np.radians(ra_deg_arr)
        dec = np.radians(dec_deg_arr)
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.column_stack([x, y, z])

    def _build_indices(self):
        self.spatial_kdtrees = {}
        self.temporal_indices = {}
        for fname, df in self.datasets.items():
            dataset_name = fname.replace(".csv", "")
            # spatial
            if df["has_spatial"].any():
                sp = df[df["has_spatial"]].copy()
                positions = self._spherical_to_cartesian(sp["ra_deg"].values, sp["dec_deg"].values)
                if len(positions) > 0:
                    tree = cKDTree(positions)
                    # indices are dataset-local (after reset_index in load_csv_files)
                    self.spatial_kdtrees[dataset_name] = {
                        "tree": tree,
                        "positions": positions,
                        "indices": sp.index.values,  # indices into dataset df
                    }
            # temporal
            if df["has_temporal"].any():
                tp = df[df["has_temporal"]].copy() 
                tp["utc_time"] = tp["utc_time"].dt.tz_convert("UTC")
                times = tp["utc_time"].view("int64") // 10**9

                sorted_idx = np.argsort(times)
                self.temporal_indices[dataset_name] = {
                    "times_sorted": times.values[sorted_idx],
                    "indices_by_time": tp.index.values[sorted_idx],
                }

    # -------------------------
    # Matching helpers
    # -------------------------
    @staticmethod
    def angular_deg_to_chord_dist(theta_deg):
        # For unit sphere points, chord distance = 2*sin(theta/2)
        theta_rad = np.deg2rad(theta_deg)
        return 2.0 * np.sin(theta_rad / 2.0)

    def _query_spatial_candidates(self, dataset_name, ra_deg, dec_deg, max_sep_deg):
        # returns dataset-local indices (indices into dataset df) of candidates within max_sep_deg
        if dataset_name not in self.spatial_kdtrees:
            return np.array([], dtype=int)
        tree_info = self.spatial_kdtrees[dataset_name]
        query_point = self._spherical_to_cartesian([ra_deg], [dec_deg])[0]
        chord_thresh = self.angular_deg_to_chord_dist(max_sep_deg)
        idxs = tree_info["tree"].query_ball_point(query_point, r=chord_thresh)
        if not idxs:
            return np.array([], dtype=int)
        return tree_info["indices"][np.array(idxs, dtype=int)]

    def _query_temporal_candidates(self, dataset_name, timestamp_sec, max_time_window_sec):
        # returns dataset-local indices (indices into dataset df) within +/- window
        if dataset_name not in self.temporal_indices:
            return np.array([], dtype=int)
        times_sorted = self.temporal_indices[dataset_name]["times_sorted"]
        idxs_by_time = self.temporal_indices[dataset_name]["indices_by_time"]
        lo = bisect_left(times_sorted, timestamp_sec - max_time_window_sec)
        hi = bisect_right(times_sorted, timestamp_sec + max_time_window_sec)
        if lo >= hi:
            return np.array([], dtype=int)
        return idxs_by_time[lo:hi]

    # -------------------------
    # Scoring
    # -------------------------
    def _component_scores(self, e1, e2):
        temporal_score = 0.0
        spatial_score = 0.0
        significance_score = 0.0
        components = []

        # temporal
        if e1.get("has_temporal", False) and e2.get("has_temporal", False):
            # both are pandas.Timestamp (tz-aware)
            dt = abs((e1["utc_time"] - e2["utc_time"]).total_seconds())
            # exponential decay with 1 hour scale (parameterizable)
            temporal_score = math.exp(-dt / 3600.0)
            components.append("temporal")

        # spatial
        if e1.get("has_spatial", False) and e2.get("has_spatial", False):
            p1 = self._spherical_to_cartesian([e1["ra_deg"]], [e1["dec_deg"]])[0]
            p2 = self._spherical_to_cartesian([e2["ra_deg"]], [e2["dec_deg"]])[0]
            cosang = np.clip(np.dot(p1, p2), -1.0, 1.0)
            ang_deg = math.degrees(math.acos(cosang))
            err1 = e1.get("pos_error_deg", 1.0) if not pd.isna(e1.get("pos_error_deg", np.nan)) else 1.0
            err2 = e2.get("pos_error_deg", 1.0) if not pd.isna(e2.get("pos_error_deg", np.nan)) else 1.0
            combined_err = err1 + err2
            # spatial score scaled by ratio of separation to combined error
            if ang_deg < combined_err:
                spatial_score = math.exp(-ang_deg / (combined_err + 1e-9))
            else:
                spatial_score = math.exp(-ang_deg / (combined_err + 1e-9)) * 0.1
            components.append("spatial")

        # significance (use z-score normalized signal)
        if (not pd.isna(e1.get("signal_z", np.nan))) and (not pd.isna(e2.get("signal_z", np.nan))):
            z1 = float(e1.get("signal_z", 0.0))
            z2 = float(e2.get("signal_z", 0.0))
            s1 = (math.tanh(z1 / 3.0) + 1.0) / 2.0
            s2 = (math.tanh(z2 / 3.0) + 1.0) / 2.0
            # significance positive mapping
            significance_score = math.sqrt(max(s1 * s2, 0.0))
            # treat extremely small significance as zero (and therefore missing)
            if significance_score > 0.0:
                components.append("significance")
            else:
                significance_score = 0.0

        return temporal_score, spatial_score, significance_score, components

    def calculate_adaptive_correlation_score(self, event1_row, event2_row):
        # expects rows (pandas Series) from dataset dataframes
        # skip identical events across same dataset (defensive)
        if event1_row["dataset"] == event2_row["dataset"] and event1_row["event_id"] == event2_row["event_id"]:
            return None

        temporal_score, spatial_score, significance_score, comps = self._component_scores(event1_row, event2_row)
        if not comps:
            return None

        # reliability heuristic based on number of components used
        reliability = {
            3: 0.95,
            2: 0.8,
            1: 0.6,
        }.get(len(comps), 0.5)

        # build a reduced weight map for available components
        avail_weight_map = {k: self.weights.get(k, 0.0) for k in comps}

        # compute adaptive and confidence using compute_confidence_score
        adaptive_score, confidence = compute_confidence_score(
            temporal_score=temporal_score,
            spatial_score=spatial_score,
            significance_score=significance_score,
            reliability=reliability,
            weights=avail_weight_map,
        )

        # assemble result, include astrophysical diagnostics requested for front-end
        result = {
            "event1_id": event1_row.get("event_id", None),
            "event2_id": event2_row.get("event_id", None),
            "dataset1": event1_row.get("dataset", None),
            "dataset2": event2_row.get("dataset", None),
            "temporal_score": temporal_score,
            "spatial_score": spatial_score,
            "significance_score": significance_score,
            "adaptive_score": adaptive_score,
            "confidence_score": confidence,
            "reliability": reliability,
            "available_components": comps,
        }

        # add extra diagnostics that the website needs (may be NaN if missing)
        if "temporal" in comps:
            dt = abs((event1_row["utc_time"] - event2_row["utc_time"]).total_seconds())
            result.update({"time_diff_sec": dt, "time_diff_hours": dt / 3600.0})
        else:
            result.update({"time_diff_sec": np.nan, "time_diff_hours": np.nan})

        if "spatial" in comps:
            p1 = self._spherical_to_cartesian([event1_row["ra_deg"]], [event1_row["dec_deg"]])[0]
            p2 = self._spherical_to_cartesian([event2_row["ra_deg"]], [event2_row["dec_deg"]])[0]
            cosang = np.clip(np.dot(p1, p2), -1.0, 1.0)
            ang_deg = math.degrees(math.acos(cosang))
            err1 = event1_row.get("pos_error_deg", 1.0) if not pd.isna(event1_row.get("pos_error_deg", np.nan)) else 1.0
            err2 = event2_row.get("pos_error_deg", 1.0) if not pd.isna(event2_row.get("pos_error_deg", np.nan)) else 1.0
            result.update({"angular_sep_deg": ang_deg, "combined_error_deg": err1 + err2, "within_error_circle": ang_deg < (err1 + err2)})
        else:
            result.update({"angular_sep_deg": np.nan, "combined_error_deg": np.nan, "within_error_circle": False})

        return result

    # -------------------------
    # Main correlation function (efficient)
    # -------------------------
    def find_correlations(self, max_time_window=86400, max_spatial_search_deg=90.0, min_confidence=0.01, target_top_n=50, output_file=None):
        """
        Find cross-dataset correlations using spatial & temporal pruning.
        - max_time_window: seconds
        - max_spatial_search_deg: degrees
        - min_confidence: minimal confidence to keep
        - target_top_n: number of top correlations to return/display
        """
        if self.combined_data is None or self.combined_data.shape[0] == 0:
            print("No data loaded.")
            return pd.DataFrame()

        correlations = []
        total_comp = 0
        dataset_files = list(self.datasets.keys())

        # pairwise dataset matching
        for i in range(len(dataset_files)):
            for j in range(i + 1, len(dataset_files)):
                f1 = dataset_files[i]
                f2 = dataset_files[j]
                df1 = self.datasets[f1]
                df2 = self.datasets[f2]
                name1 = f1.replace(".csv", "")
                name2 = f2.replace(".csv", "")

                # choose smaller-larger to iterate over smaller set
                if len(df1) <= len(df2):
                    small_df, large_df, small_name, large_name = df1, df2, name1, name2
                else:
                    small_df, large_df, small_name, large_name = df2, df1, name2, name1

                # precompute times arrays if needed
                for idx_small, row_small in small_df.iterrows():
                    # candidate indices in large_df (dataset-local indices)
                    candidate_idx = set()

                    # spatial candidates if both datasets have spatial index
                    if row_small.get("has_spatial", False) and large_name in self.spatial_kdtrees:
                        spatial_candidates = self._query_spatial_candidates(large_name, row_small["ra_deg"], row_small["dec_deg"], max_spatial_search_deg)
                        for ci in spatial_candidates:
                            candidate_idx.add(int(ci))

                    # temporal candidates
                    if row_small.get("has_temporal", False) and large_name in self.temporal_indices:
                        tsec = int(row_small["utc_time"].timestamp())
                        temporal_candidates = self._query_temporal_candidates(large_name, tsec, max_time_window)
                        for ci in temporal_candidates:
                            candidate_idx.add(int(ci))

                    # fallback: if no pruning possible, compare against all in large_df (but skip if too big)
                    if len(candidate_idx) == 0:
                        # limit brute force to a reasonable cap (avoid O(N^2) blowup)
                        if len(large_df) <= 5000:
                            candidate_idx.update(list(large_df.index.values))
                        else:
                            # skip heavy brute force when no pruning available
                            continue

                    # compute scores
                    for ci in candidate_idx:
                        total_comp += 1
                        row_large = large_df.loc[ci]
                        score_data = self.calculate_adaptive_correlation_score(row_small, row_large)
                        if score_data and score_data["confidence_score"] >= min_confidence:
                            correlations.append(score_data)

        # make DataFrame
        if not correlations:
            print("No correlations found.")
            return pd.DataFrame()

        results_df = pd.DataFrame(correlations)

        # filter obvious bad entries
        results_df = results_df[results_df["event1_id"] != results_df["event2_id"]]
        results_df = results_df[results_df["dataset1"] != results_df["dataset2"]]

        # Some sanity: drop pathological confidence==1.0 results if they are exact duplicates (but keep them otherwise)
        results_df = results_df[results_df["confidence_score"] < 0.999999]

        # sort and rank (prefer higher reliability then higher confidence then adaptive_score)
        results_df = results_df.sort_values(["reliability", "confidence_score", "adaptive_score"], ascending=[False, False, False]).reset_index(drop=True)
        results_df["rank"] = range(1, len(results_df) + 1)

        # ensure CSV contains the requested columns for frontend
        output_columns = [
            "rank",
            "event1_id", "dataset1",
            "event2_id", "dataset2",
            "temporal_score", "spatial_score", "significance_score",
            "adaptive_score", "confidence_score", "reliability",
            "available_components",
            "angular_sep_deg", "combined_error_deg", "within_error_circle",
            "time_diff_sec", "time_diff_hours","time_diff_hrs_readable" , 
        ]
        # add any missing columns as NaN to preserve consistent schema
        for c in output_columns:
            if c not in results_df.columns:
                results_df[c] = np.nan

        results_df = results_df.dropna(subset=["time_diff_hours"])

        # Convert hours to timedelta correctly
        results_df["time_diff_hrs_readable"] = results_df["time_diff_hours"].apply(
            lambda x: "N/A" if pd.isna(x) else str(timedelta(hours=x))
        )



        results_df_out = results_df[output_columns].copy()

        # save full results optionally
        if output_file:
            # ensure UTF-8 with BOM for Excel compatibility
            results_df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"Saved results to {output_file}")

        # display top-N
        topn = results_df.head(target_top_n)
        self._display_results(topn)

        return results_df_out

    # -------------------------
    # Utility display and saving
    # -------------------------
    def _display_results(self, top_results: pd.DataFrame):
        if top_results is None or top_results.shape[0] == 0:
            print("No top results to display.")
            return
        print(f"\nTOP {len(top_results)} CORRELATIONS")
        print("-" * 80)
        for _, row in top_results.iterrows():
            comps = ", ".join(row.get("available_components", [])) if isinstance(row.get("available_components", []), (list, tuple)) else str(row.get("available_components"))
            s = f"#{int(row['rank'])}. {row['event1_id']} ({row['dataset1']}) <-> {row['event2_id']} ({row['dataset2']}) | conf={row['confidence_score']:.4f} rel={row['reliability']:.2f} comps=[{comps}]"
            if not pd.isna(row.get("time_diff_hours", np.nan)):
                s += f" | dt={row['time_diff_hours']:.2f}h"
            if not pd.isna(row.get("angular_sep_deg", np.nan)):
                s += f" | sep={row['angular_sep_deg']:.3f}°"
            print(s)

    def _create_sample_data(self):
        # creates minimal sample CSVs for testing
        gw_data = [
            {"event_id": "GW150914", "utc_time": "2015-09-14 09:50:44.400", "ra_deg": 112.5, "dec_deg": -70.2, "pos_error_deg": 0.164, "signal_strength": 24},
            {"event_id": "GW170817", "utc_time": "2017-08-17 12:41:04.400", "ra_deg": 197.45, "dec_deg": -23.38, "pos_error_deg": 0.0044, "signal_strength": 32},
            {"event_id": "GW190814", "utc_time": "2019-08-14 21:10:39.000", "ra_deg": 134.56, "dec_deg": 2.69, "pos_error_deg": 0.005, "signal_strength": 25},
        ]
        grb_data = [
            {"event_id": "GRB150914A", "utc_time": "2015-09-14 10:15:30.000", "ra_deg": 114.2, "dec_deg": -68.9, "pos_error_deg": 12.5, "signal_strength": 8.4},
            {"event_id": "bn170817529", "utc_time": "2017-08-17 12:41:06.470", "ra_deg": 197.42, "dec_deg": -23.42, "pos_error_deg": 3.2, "signal_strength": 6.2},
            {"event_id": "GRB_INCOMPLETE", "utc_time": "2019-08-14 22:30:45.000", "ra_deg": None, "dec_deg": None, "pos_error_deg": None, "signal_strength": 12.4},
        ]
        os.makedirs(self.csv_directory, exist_ok=True)
        pd.DataFrame(gw_data).to_csv(os.path.join(self.csv_directory, "gravitational_waves.csv"), index=False, encoding="utf-8")
        pd.DataFrame(grb_data).to_csv(os.path.join(self.csv_directory, "gamma_ray_bursts.csv"), index=False, encoding="utf-8")
        print(f"Sample CSV files created in {self.csv_directory}")


# -------------------------
# Reporting & statistics (fixed)
# -------------------------
def generate_advanced_statistics(correlator: RobustMultimessengerCorrelator, results: pd.DataFrame):
    if correlator.combined_data is None or correlator.combined_data.shape[0] == 0:
        print("No combined data available for statistics.")
        return

    total_events = len(correlator.combined_data)
    temporal_events = int(correlator.combined_data["has_temporal"].sum())
    spatial_events = int(correlator.combined_data["has_spatial"].sum())
    signal_events = int(correlator.combined_data["has_signal"].sum())
    complete_events = int((correlator.combined_data["has_temporal"] & correlator.combined_data["has_spatial"] & correlator.combined_data["has_signal"]).sum())

    print("\nDATA COMPLETENESS METRICS")
    print(f"  Total events: {total_events}")
    print(f"  Temporal coverage: {temporal_events}/{total_events} ({temporal_events/total_events*100:.1f}%)")
    print(f"  Spatial coverage: {spatial_events}/{total_events} ({spatial_events/total_events*100:.1f}%)")
    print(f"  Signal coverage: {signal_events}/{total_events} ({signal_events/total_events*100:.1f}%)")
    print(f"  Complete triplets: {complete_events}/{total_events} ({complete_events/total_events*100:.1f}%)")

    if results is None or results.shape[0] == 0:
        print("No correlations to analyze.")
        return

    high = results[results["reliability"] >= 0.8]
    med = results[(results["reliability"] >= 0.6) & (results["reliability"] < 0.8)]
    low = results[results["reliability"] < 0.6]
    print("\nCORRELATION QUALITY")
    print(f"  High (>=0.8): {len(high)}")
    print(f"  Medium (0.6-0.8): {len(med)}")
    print(f"  Low (<0.6): {len(low)}")

    # component stats
    comp_stats = {}
    for comps in results["available_components"]:
        key = ", ".join(sorted(comps)) if isinstance(comps, (list, tuple)) else str(comps)
        comp_stats[key] = comp_stats.get(key, 0) + 1
    print("\nCOMPONENT BREAKDOWN")
    for k, v in sorted(comp_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")

    # cross-dataset matrix
    print("\nCROSS-DATASET CORRELATION MATRIX")
    datasets = [fname for fname in correlator.datasets.keys()]
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            ds1 = datasets[i].replace(".csv", "")
            ds2 = datasets[j].replace(".csv", "")
            cross_corrs = results[(results["dataset1"] == ds1) & (results["dataset2"] == ds2)]
            cross_corrs_rev = results[(results["dataset1"] == ds2) & (results["dataset2"] == ds1)]
            total = len(cross_corrs) + len(cross_corrs_rev)
            print(f"  {ds1} <-> {ds2}: {total}")


def generate_hackathon_report(correlator: RobustMultimessengerCorrelator, results: pd.DataFrame, filename="hackathon_technical_report.txt"):
    report = []
    report.append("HACKATHON SUBMISSION REPORT")
    report.append("=" * 60)
    report.append("TECHNICAL INNOVATION HIGHLIGHTS:")
    report.append("- Adaptive missing data handling with dynamic scoring.")
    report.append("- Intelligent column detection across variable CSV formats.")
    report.append("- KD-tree spatial indexing and temporal binary search for scalable matching.")
    report.append("- Normalized significance using z-scores.")
    report.append("")
    report.append("PERFORMANCE METRICS:")
    report.append(f"Total Events Processed: {len(correlator.combined_data) if correlator.combined_data is not None else 0}")
    report.append(f"Datasets Successfully Loaded: {len(correlator.datasets)}")
    report.append(f"Valid Correlations Found: {len(results) if results is not None else 0}")
    report_text = "\n".join(report)

    print("\n" + report_text + "\n")
    # save with utf-8 to avoid encoding errors
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved hackathon report to {filename}")


# -------------------------
# Main execution
# -------------------------
def main():
    correlator = RobustMultimessengerCorrelator(csv_directory="./data")
    correlator.load_csv_files()

    print("\nDATASET STATISTICS:")
    for fname, stats in correlator.dataset_stats.items():
        print(f"{fname}: total={stats['total_events']} temporal={stats['temporal_events']} spatial={stats['spatial_events']} signal={stats['signal_events']} complete={stats['complete_events']}")

    # adjust parameters as needed here
    results = correlator.find_correlations(
        max_time_window=86400 * 7,  # 7 days
        max_spatial_search_deg=180.0,  # full sky
        min_confidence=0.01,
        target_top_n=50,
        output_file="multimessenger_correlations.csv",
    )

    # generate stats and report
    if results is not None and not results.empty:
        # results is the exported schema containing astrophysical fields
        generate_advanced_statistics(correlator, results)
        generate_hackathon_report(correlator, results, filename="hackathon_technical_report.txt")
        print(f"\nFound {len(results)} correlations. Top {min(50, len(results))} saved/displayed.")
    else:
        print("\nNo correlations found.")

    return correlator, results


if __name__ == "__main__":
    correlator, results = main()
