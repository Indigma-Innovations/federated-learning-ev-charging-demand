#!/usr/bin/env python3
"""
Resumable ACN downloader and preprocessor for the ACN dataset.

Features
------------
- Resumes after timeout/crash
- Saves every page immediately to disk
- Retries with exponential backoff
- Stores per-endpoint state
- Can rebuild processed outputs from partial/full raw downloads

Data downloaded
---------------
1. Non-time-series sessions for:
   - caltech
   - jpl
   - office001

2. Time-series sessions for:
   - caltech
   - jpl
   - office001
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError, Timeout

BASE_URL = "https://ev.caltech.edu/api/v1/"
SITES = ["caltech", "jpl", "office001"]

# Helpers
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def flatten_dict(prefix: str, obj: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flatten_dict(key, v, out)
        else:
            out[key] = v


def json_dump(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_acn_token() -> str:
    load_env_file()
    token = os.getenv("ACN_TOKEN")
    if not token:
        raise RuntimeError(
            "ACN_TOKEN not found. Please create a .env file with:\n"
            "ACN_TOKEN=your_token_here"
        )
    return token


# API client
class ACNClient:
    def __init__(
        self,
        token: str,
        base_url: str = BASE_URL,
        sleep_s: float = 0.25,
        connect_timeout_s: int = 15,
        read_timeout_s: int = 180,
        max_retries: int = 8,
        backoff_base_s: float = 2.0,
    ) -> None:
        self.base_url = base_url
        self.sleep_s = sleep_s
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

        self.session = requests.Session()
        self.session.auth = (token, "")

    def get_json(self, url: str) -> Dict[str, Any]:
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                print(
                    f"    [GET] {url} (attempt {attempt}/{self.max_retries})",
                    flush=True,
                )
                r = self.session.get(
                    url,
                    timeout=(self.connect_timeout_s, self.read_timeout_s),
                )
                r.raise_for_status()

                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)

                return r.json()

            except (ReadTimeout, Timeout, ConnectionError) as e:
                last_err = e
                wait_s = self.backoff_base_s ** min(attempt, 6)
                print(
                    f"    [WARN] network timeout/error: {type(e).__name__}: {e}. "
                    f"Retrying in {wait_s:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_s)

            except HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (429, 500, 502, 503, 504):
                    last_err = e
                    wait_s = self.backoff_base_s ** min(attempt, 6)
                    print(
                        f"    [WARN] HTTP {status}: retrying in {wait_s:.1f}s...",
                        flush=True,
                    )
                    time.sleep(wait_s)
                else:
                    raise

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts for URL: {url}"
        ) from last_err


# Resumable download state
def get_endpoint_paths(outdir: Path, kind: str, site: str) -> Dict[str, Path]:
    """
    kind: 'sessions' or 'timeseries'
    """
    raw_root = outdir / "raw" / kind
    state_root = outdir / "state" / kind
    merged_root = outdir / "raw_merged" / kind

    safe_mkdir(raw_root)
    safe_mkdir(state_root)
    safe_mkdir(merged_root)

    return {
        "jsonl": raw_root / f"{site}.jsonl",
        "state": state_root / f"{site}.json",
        "merged_json": merged_root / f"{site}.json",
    }


def default_state(initial_url: str) -> Dict[str, Any]:
    return {
        "next_url": initial_url,
        "pages_done": 0,
        "items_done": 0,
        "completed": False,
        "last_success_at": None,
    }


def load_state(path: Path, initial_url: str) -> Dict[str, Any]:
    if path.exists():
        return json_load(path, default_state(initial_url))
    return default_state(initial_url)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    json_dump(path, state)


def merge_jsonl_to_json(jsonl_path: Path, json_path: Path) -> int:
    items = read_jsonl(jsonl_path)
    json_dump(json_path, items)
    return len(items)


def fetch_endpoint_resumable(
    client: ACNClient,
    outdir: Path,
    endpoint: str,
    kind: str,
    site: str,
) -> List[Dict[str, Any]]:
    """
    Resumable page-by-page download.
    Saves each page immediately to JSONL and updates state after every success.
    """
    paths = get_endpoint_paths(outdir, kind, site)
    initial_url = urljoin(client.base_url, endpoint)
    state = load_state(paths["state"], initial_url)

    if state.get("completed"):
        print(
            f"[resume] {kind}/{site}: already completed, loading local JSONL",
            flush=True,
        )
        items = read_jsonl(paths["jsonl"])
        merge_jsonl_to_json(paths["jsonl"], paths["merged_json"])
        print(
            f"[resume] {kind}/{site}: loaded {len(items)} items from local cache",
            flush=True,
        )
        return items

    url = state.get("next_url") or initial_url
    print(
        f"[resume] {kind}/{site}: starting from pages_done={state['pages_done']} "
        f"items_done={state['items_done']}",
        flush=True,
    )

    while url:
        next_page_num = state["pages_done"] + 1
        print(f"  [fetch] {kind}/{site} page={next_page_num}", flush=True)

        payload = client.get_json(url)
        page_items = payload.get("_items", [])
        append_jsonl(paths["jsonl"], page_items)

        state["pages_done"] += 1
        state["items_done"] += len(page_items)
        state["last_success_at"] = pd.Timestamp.utcnow().isoformat()

        next_href = payload.get("_links", {}).get("next", {}).get("href")
        if next_href:
            url = urljoin(client.base_url, next_href)
            state["next_url"] = url
        else:
            url = None
            state["next_url"] = None
            state["completed"] = True

        save_state(paths["state"], state)

        print(
            f"  [fetch] {kind}/{site} page={state['pages_done']} "
            f"items_this_page={len(page_items)} total_so_far={state['items_done']} "
            f"completed={state['completed']}",
            flush=True,
        )

    total = merge_jsonl_to_json(paths["jsonl"], paths["merged_json"])
    print(
        f"[done] {kind}/{site}: merged {total} items to {paths['merged_json']}",
        flush=True,
    )
    return read_jsonl(paths["jsonl"])


# Download
def download_non_ts(client: ACNClient, outdir: Path) -> Dict[str, List[Dict[str, Any]]]:
    all_sessions: Dict[str, List[Dict[str, Any]]] = {}
    for site in SITES:
        print(f"[non-ts] downloading site={site}", flush=True)
        all_sessions[site] = fetch_endpoint_resumable(
            client=client,
            outdir=outdir,
            endpoint=f"sessions/{site}",
            kind="sessions",
            site=site,
        )
        print(f"[non-ts] {site}: {len(all_sessions[site])} sessions", flush=True)
    return all_sessions


def download_ts(client: ACNClient, outdir: Path) -> Dict[str, List[Dict[str, Any]]]:
    all_ts: Dict[str, List[Dict[str, Any]]] = {}
    for site in SITES:
        print(f"[ts] downloading site={site}", flush=True)
        all_ts[site] = fetch_endpoint_resumable(
            client=client,
            outdir=outdir,
            endpoint=f"sessions/{site}/ts",
            kind="timeseries",
            site=site,
        )
        print(f"[ts] {site}: {len(all_ts[site])} sessions", flush=True)
    return all_ts


def extract_signal_arrays(
    rec: Dict[str, Any], field: str
) -> Tuple[List[Any], List[Any]]:
    obj = rec.get(field, {})
    if not isinstance(obj, dict):
        return [], []
    timestamps = ensure_list(obj.get("timestamps"))
    values = ensure_list(obj.get("values"))
    return timestamps, values


def normalize_ts_record(site: str, rec: Dict[str, Any]) -> pd.DataFrame:
    session_id = rec.get("sessionID") or rec.get("session_id") or rec.get("_id")
    station_id = rec.get("stationID")
    cluster_id = rec.get("clusterID")

    cur_ts, cur_vals = extract_signal_arrays(rec, "chargingCurrent")
    pil_ts, pil_vals = extract_signal_arrays(rec, "pilotSignal")

    current_df = pd.DataFrame(
        {
            "timestamp": cur_ts,
            "charging_current": cur_vals,
        }
    )
    pilot_df = pd.DataFrame(
        {
            "timestamp": pil_ts,
            "pilot_signal": pil_vals,
        }
    )

    if current_df.empty and pilot_df.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "session_id",
                "station_id",
                "cluster_id",
                "timestamp",
                "charging_current",
                "pilot_signal",
            ]
        )

    if current_df.empty:
        merged = pilot_df.copy()
        merged["charging_current"] = pd.NA
    elif pilot_df.empty:
        merged = current_df.copy()
        merged["pilot_signal"] = pd.NA
    else:
        merged = pd.merge(current_df, pilot_df, on="timestamp", how="outer")

    merged["site_id"] = site
    merged["session_id"] = session_id
    merged["station_id"] = station_id
    merged["cluster_id"] = cluster_id
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    merged["connection_time"] = rec.get("connectionTime")
    merged["disconnect_time"] = rec.get("disconnectTime")
    merged["done_charging_time"] = rec.get("doneChargingTime")
    merged["kwh_delivered"] = rec.get("kWhDelivered")

    return merged


# Rebuild from disk
def load_downloaded_raw(outdir: Path, kind: str) -> Dict[str, List[Dict[str, Any]]]:
    data: Dict[str, List[Dict[str, Any]]] = {}
    for site in SITES:
        jsonl_path = outdir / "raw" / kind / f"{site}.jsonl"
        data[site] = read_jsonl(jsonl_path)
    return data


# Main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--skip-ts", action="store_true", help="Skip time-series download"
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Do not download; rebuild processed outputs from local raw files",
    )
    parser.add_argument(
        "--sleep-s", type=float, default=0.25, help="Sleep between API calls"
    )
    parser.add_argument("--connect-timeout-s", type=int, default=20)
    parser.add_argument("--read-timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=32)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    if args.rebuild_only:
        print("Rebuilding processed files from local raw data...", flush=True)
        non_ts = load_downloaded_raw(outdir, "sessions")
        ts_data = {} if args.skip_ts else load_downloaded_raw(outdir, "timeseries")
    else:
        token = get_acn_token()
        client = ACNClient(
            token=token,
            sleep_s=args.sleep_s,
            connect_timeout_s=args.connect_timeout_s,
            read_timeout_s=args.read_timeout_s,
            max_retries=args.max_retries,
        )

        print("Downloading non-time-series sessions...", flush=True)
        non_ts = download_non_ts(client, outdir)

        ts_data: Dict[str, List[Dict[str, Any]]] = {}
        if args.skip_ts:
            print("Skipping time-series download.", flush=True)
        else:
            print("Downloading time-series sessions...", flush=True)
            ts_data = download_ts(client, outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
