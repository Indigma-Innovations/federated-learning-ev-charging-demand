#!/usr/bin/env python3
"""
Build an early-session regression dataset.

Goal
----
One row per charging session.

Inputs:
- plug-in/session context
- first N minutes of charging behavior

Output:
- total session energy (kWhDelivered)

Reads:
- raw/sessions/<site>.jsonl
- raw/timeseries/<site>.jsonl

Writes:
- processed/dataset_early_session_<site>.parquet
- processed/dataset_early_session_<site>.csv

Example
-------
uv run python src/federated_acn/data_dl/build_early_session_dataset.py \
    --data-dir ./acn_fl_data \
    --early-window-min 10 \
    --site caltech
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SITES = ["caltech", "jpl", "office001"]
DEFAULT_SITE = "caltech"


# I/O helpers
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_dt(x: Any) -> Optional[pd.Timestamp]:
    if x is None or x == "":
        return None
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


# Session features
def extract_user_input_fields(user_inputs: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if isinstance(user_inputs, list) and len(user_inputs) > 0:
        ui = user_inputs[0]
    elif isinstance(user_inputs, dict):
        ui = user_inputs
    else:
        ui = None

    if not isinstance(ui, dict):
        return out

    # Keep a few common fields if present.
    for k in [
        "kWhRequested",
        "requestedDeparture",
        "minutesAvailable",
        "requested_energy",
    ]:
        if k in ui:
            out[k] = ui[k]

    return out


def build_sessions_df(data_dir: Path, sites: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for site in sites:
        path = data_dir / "raw" / "sessions" / f"{site}.jsonl"
        for i, rec in enumerate(read_jsonl(path), 1):
            if i % 5000 == 0:
                print(f"[sessions:{site}] processed {i:,}", flush=True)
            connection_time = parse_dt(rec.get("connectionTime"))
            disconnect_time = parse_dt(rec.get("disconnectTime"))
            done_charging_time = parse_dt(rec.get("doneChargingTime"))

            row: Dict[str, Any] = {
                "session_id": str(rec.get("sessionID") or rec.get("_id")),
                "site_name": site,
                "site_id": rec.get("siteID"),
                "station_id": rec.get("stationID"),
                "space_id": rec.get("spaceID"),
                "cluster_id": rec.get("clusterID"),
                "timezone": rec.get("timezone"),
                "user_id": rec.get("userID"),
                "kwh_delivered": rec.get("kWhDelivered"),
                "connection_time": connection_time,
                "disconnect_time": disconnect_time,
                "done_charging_time": done_charging_time,
            }

            if connection_time is not None:
                row["connect_hour"] = connection_time.hour
                row["connect_weekday"] = connection_time.weekday()
                row["connect_month"] = connection_time.month
                row["connect_dayofyear"] = connection_time.dayofyear
                row["connect_is_weekend"] = int(connection_time.weekday() >= 5)
            else:
                row["connect_hour"] = np.nan
                row["connect_weekday"] = np.nan
                row["connect_month"] = np.nan
                row["connect_dayofyear"] = np.nan
                row["connect_is_weekend"] = np.nan

            # Optional user input fields, if present at plug-in time.
            row.update(extract_user_input_fields(rec.get("userInputs")))

            rows.append(row)

    df = pd.DataFrame(rows)

    # Optional derived fields from user inputs.
    if "requestedDeparture" in df.columns:
        req_dep = pd.to_datetime(df["requestedDeparture"], utc=True, errors="coerce")
        df["requested_departure_minutes_from_connect"] = (
            req_dep - df["connection_time"]
        ).dt.total_seconds() / 60.0

    if "minutesAvailable" in df.columns:
        df["minutes_available"] = pd.to_numeric(df["minutesAvailable"], errors="coerce")

    if "kWhRequested" in df.columns:
        df["kwh_requested"] = pd.to_numeric(df["kWhRequested"], errors="coerce")

    return df


# Time-series parsing
def extract_signal_arrays(
    rec: Dict[str, Any], field: str
) -> Tuple[List[Any], List[Any]]:
    """
    ACN raw /ts records store:
    - pilotSignal: {"timestamps": [...], "pilot": [...]}
    - chargingCurrent: {"timestamps": [...], "current": [...]}
    """
    obj = rec.get(field, {})
    if not isinstance(obj, dict):
        return [], []

    timestamps = obj.get("timestamps", []) or []

    if field == "pilotSignal":
        values = obj.get("pilot", obj.get("values", [])) or []
    elif field == "chargingCurrent":
        values = obj.get("current", obj.get("values", [])) or []
    else:
        values = obj.get("values", []) or []

    return list(timestamps), list(values)


def compute_slope(x_seconds: np.ndarray, y: np.ndarray) -> float:
    if len(x_seconds) < 2 or len(y) < 2:
        return np.nan
    if np.allclose(x_seconds, x_seconds[0]):
        return np.nan
    try:
        slope = np.polyfit(x_seconds, y, 1)[0]
        return float(slope)
    except Exception:
        return np.nan


def summarize_early_window(
    session_id: str,
    site_name: str,
    connection_time: pd.Timestamp,
    current_ts: List[Any],
    current_vals: List[Any],
    pilot_ts: List[Any],
    pilot_vals: List[Any],
    early_window_min: int,
) -> Dict[str, Any]:
    """
    Build early-window features for one session.
    """
    early_end = connection_time + pd.Timedelta(minutes=early_window_min)

    current_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(current_ts, utc=True, errors="coerce"),
            "charging_current": pd.to_numeric(current_vals, errors="coerce"),
        }
    ).dropna(subset=["timestamp"])

    pilot_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pilot_ts, utc=True, errors="coerce"),
            "pilot_signal": pd.to_numeric(pilot_vals, errors="coerce"),
        }
    ).dropna(subset=["timestamp"])

    if not current_df.empty:
        current_df = current_df[
            (current_df["timestamp"] >= connection_time)
            & (current_df["timestamp"] <= early_end)
        ].sort_values("timestamp")

    if not pilot_df.empty:
        pilot_df = pilot_df[
            (pilot_df["timestamp"] >= connection_time)
            & (pilot_df["timestamp"] <= early_end)
        ].sort_values("timestamp")

    # Outer merge on exact timestamps. Good enough for ACN because both signals are sampled similarly.
    if current_df.empty and pilot_df.empty:
        merged = pd.DataFrame(columns=["timestamp", "charging_current", "pilot_signal"])
    elif current_df.empty:
        merged = pilot_df.copy()
        merged["charging_current"] = np.nan
    elif pilot_df.empty:
        merged = current_df.copy()
        merged["pilot_signal"] = np.nan
    else:
        merged = pd.merge(
            current_df, pilot_df, on="timestamp", how="outer"
        ).sort_values("timestamp")

    feat: Dict[str, Any] = {
        "session_id": session_id,
        "site_name": site_name,
        "early_window_min": early_window_min,
        "n_points_window": int(len(merged)),
        "n_current_points": int(len(current_df)),
        "n_pilot_points": int(len(pilot_df)),
        "window_observed_minutes": np.nan,
        "mean_current": np.nan,
        "max_current": np.nan,
        "std_current": np.nan,
        "min_current": np.nan,
        "last_current": np.nan,
        "mean_pilot": np.nan,
        "max_pilot": np.nan,
        "std_pilot": np.nan,
        "min_pilot": np.nan,
        "last_pilot": np.nan,
        "mean_utilization": np.nan,
        "max_utilization": np.nan,
        "current_slope_per_sec": np.nan,
        "pilot_slope_per_sec": np.nan,
        "approx_energy_first_window_kwh": np.nan,
        "has_enough_early_data": 0,
    }

    if not merged.empty:
        feat["window_observed_minutes"] = (
            merged["timestamp"].max() - merged["timestamp"].min()
        ).total_seconds() / 60.0

    if not current_df.empty:
        cur = current_df["charging_current"].astype(float).to_numpy()
        feat["mean_current"] = float(np.nanmean(cur))
        feat["max_current"] = float(np.nanmax(cur))
        feat["std_current"] = float(np.nanstd(cur))
        feat["min_current"] = float(np.nanmin(cur))
        feat["last_current"] = float(cur[-1])

        x_sec = (
            (current_df["timestamp"] - current_df["timestamp"].min())
            .dt.total_seconds()
            .to_numpy()
        )
        feat["current_slope_per_sec"] = compute_slope(x_sec, cur)

        # Approximate energy in the first window using:
        # power ~= current * voltage / 1000
        # This is only an approximate early-window feature, not the target.
        if len(current_df) >= 2:
            t = current_df["timestamp"].astype("int64").to_numpy() / 1e9
            power_kw = cur * 208.0 / 1000.0
            energy_kwh = np.trapezoid(power_kw, t) / 3600.0
            feat["approx_energy_first_window_kwh"] = float(energy_kwh)

    if not pilot_df.empty:
        pil = pilot_df["pilot_signal"].astype(float).to_numpy()
        feat["mean_pilot"] = float(np.nanmean(pil))
        feat["max_pilot"] = float(np.nanmax(pil))
        feat["std_pilot"] = float(np.nanstd(pil))
        feat["min_pilot"] = float(np.nanmin(pil))
        feat["last_pilot"] = float(pil[-1])

        x_sec = (
            (pilot_df["timestamp"] - pilot_df["timestamp"].min())
            .dt.total_seconds()
            .to_numpy()
        )
        feat["pilot_slope_per_sec"] = compute_slope(x_sec, pil)

    if not current_df.empty and not pilot_df.empty:
        util_df = pd.merge(current_df, pilot_df, on="timestamp", how="inner")
        if not util_df.empty:
            valid = util_df["pilot_signal"] > 0
            util = (
                util_df.loc[valid, "charging_current"].astype(float).to_numpy()
                / util_df.loc[valid, "pilot_signal"].astype(float).to_numpy()
            )
            if len(util) > 0:
                feat["mean_utilization"] = float(np.nanmean(util))
                feat["max_utilization"] = float(np.nanmax(util))

    feat["has_enough_early_data"] = int(
        feat["n_current_points"] >= 5
        and feat["window_observed_minutes"] >= max(1.0, early_window_min / 2.0)
    )

    return feat


def build_early_ts_features_df(
    data_dir: Path, early_window_min: int, sites: Iterable[str]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for site in sites:
        ts_path = data_dir / "raw" / "timeseries" / f"{site}.jsonl"
        for i, rec in enumerate(read_jsonl(ts_path), 1):
            if i % 5000 == 0:
                print(f"[timeseries:{site}] processed {i:,}", flush=True)
            session_id = str(rec.get("sessionID") or rec.get("_id"))
            connection_time = parse_dt(rec.get("connectionTime"))
            if connection_time is None:
                continue

            current_ts, current_vals = extract_signal_arrays(rec, "chargingCurrent")
            pilot_ts, pilot_vals = extract_signal_arrays(rec, "pilotSignal")

            feat = summarize_early_window(
                session_id=session_id,
                site_name=site,
                connection_time=connection_time,
                current_ts=current_ts,
                current_vals=current_vals,
                pilot_ts=pilot_ts,
                pilot_vals=pilot_vals,
                early_window_min=early_window_min,
            )
            rows.append(feat)

    return pd.DataFrame(rows)


# Final merge
def add_client_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["client_site"] = out["site_name"].astype(str)
    out["client_station"] = (
        out["site_name"].astype(str) + "::" + out["station_id"].astype(str)
    )
    out["client_cluster"] = (
        out["site_name"].astype(str)
        + "::"
        + out["cluster_id"].fillna("unassigned").astype(str)
    )
    return out


def resolve_sites(site_arg: str) -> List[str]:
    if site_arg == "all":
        return list(SITES)
    if site_arg in SITES:
        return [site_arg]
    raise ValueError(
        f"Unsupported site '{site_arg}'. Choose one of: {', '.join(SITES)} or 'all'."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", required=True, help="Root ACN data directory, e.g. ./acn_fl_data"
    )
    parser.add_argument(
        "--early-window-min", type=int, default=10, help="Early window size in minutes"
    )
    parser.add_argument(
        "--site",
        default=DEFAULT_SITE,
        help=f"Site to process ({', '.join(SITES)}) or 'all'. Default: {DEFAULT_SITE}",
    )
    parser.add_argument(
        "--min-current-points",
        type=int,
        default=5,
        help="Minimum current points required",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    selected_sites = resolve_sites(args.site)
    processed_dir = data_dir / "processed"
    safe_mkdir(processed_dir)

    print("Loading sessions...", flush=True)
    sessions_df = build_sessions_df(data_dir, sites=selected_sites)
    print(f"Loaded sessions: {len(sessions_df):,}", flush=True)

    print("Building early-session time-series features...", flush=True)
    early_df = build_early_ts_features_df(
        data_dir, early_window_min=args.early_window_min, sites=selected_sites
    )
    print(f"Built early-window feature rows: {len(early_df):,}", flush=True)

    print("Merging datasets...", flush=True)
    df = sessions_df.merge(early_df, on=["session_id", "site_name"], how="inner")

    # Keep only rows with target and enough early data.
    df["kwh_delivered"] = pd.to_numeric(df["kwh_delivered"], errors="coerce")
    df = df[df["kwh_delivered"].notna()].copy()
    df = df[df["n_current_points"] >= args.min_current_points].copy()

    df = add_client_ids(df)

    # Select a clean set of columns.
    keep_cols = [
        # IDs / grouping
        "session_id",
        "site_name",
        "site_id",
        "station_id",
        "cluster_id",
        "space_id",
        "timezone",
        "client_site",
        "client_station",
        "client_cluster",
        # Plug-in context
        "connection_time",
        "connect_hour",
        "connect_weekday",
        "connect_month",
        "connect_dayofyear",
        "connect_is_weekend",
        # Optional user-side fields
        "kwh_requested",
        "minutes_available",
        "requested_departure_minutes_from_connect",
        # Early-window features
        "early_window_min",
        "n_points_window",
        "n_current_points",
        "n_pilot_points",
        "window_observed_minutes",
        "mean_current",
        "max_current",
        "std_current",
        "min_current",
        "last_current",
        "mean_pilot",
        "max_pilot",
        "std_pilot",
        "min_pilot",
        "last_pilot",
        "mean_utilization",
        "max_utilization",
        "current_slope_per_sec",
        "pilot_slope_per_sec",
        "approx_energy_first_window_kwh",
        "has_enough_early_data",
        # Target
        "kwh_delivered",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # drop duplicate sessions if any
    df = df.drop_duplicates(subset=["session_id"]).reset_index(drop=True)

    for col in ["site_id", "station_id", "cluster_id", "space_id", "timezone"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    site_suffix = args.site
    out_parquet = processed_dir / f"dataset_early_session_{site_suffix}.parquet"
    out_csv = processed_dir / f"dataset_early_session_{site_suffix}.csv"

    print("Saving dataset...", flush=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_parquet}", flush=True)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Final rows: {len(df):,}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
