"""
build_running_bests.py
──────────────────────
Finds best running times and Cooper Test results from .fit files.
Produces running_bests.json with results per time window (year + recent).

Best efforts: 1k, 3k, 5k, 10k, HM (21.097k), 25k, 30k, Marathon (42.195k)
Cooper Test: best distance covered in any 12-minute window → VO2max estimate

Incremental — only processes new .fit files.

Run after garmin_download.py:
    python3 build_running_bests.py

To rebuild:
    rm running_bests.json && python3 build_running_bests.py
"""

from __future__ import annotations
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import fitparse
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
FIT_DIR     = Path("fit_files")
BESTS_FILE  = Path("running_bests.json")
RECENT_DAYS = 42

# Target distances in metres
DISTANCES = {
    "1k"  : 1000,
    "3k"  : 3000,
    "5k"  : 5000,
    "10k" : 10000,
    "HM"  : 21097,
    "25k" : 25000,
    "30k" : 30000,
    "Mar" : 42195,
}

COOPER_DURATION = 720   # 12 minutes in seconds


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_fit(path: Path) -> fitparse.FitFile:
    raw = path.read_bytes()
    if raw[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            fit_name = next(n for n in zf.namelist() if n.endswith('.fit'))
            fit_bytes = zf.read(fit_name)
        return fitparse.FitFile(io.BytesIO(fit_bytes))
    return fitparse.FitFile(str(path))


def extract_distance_time_series(fit_path: Path) -> list[tuple]:
    """
    Extract (timestamp, cumulative_distance_m, speed_mps) from a .fit file.
    """
    fitfile = open_fit(fit_path)
    records = []
    for rec in fitfile.get_messages("record"):
        data = {f.name: f.value for f in rec}
        ts   = data.get("timestamp")
        dist = data.get("distance")
        spd  = data.get("enhanced_speed") or data.get("speed")
        if ts and dist is not None:
            records.append((ts, float(dist), float(spd) if spd else 0.0))
    return records


def secs_to_time(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def find_best_times(records: list[tuple]) -> dict[str, dict]:
    """
    Find best time for each target distance using cumulative distance data.
    Uses a two-pointer approach on the distance series.
    """
    if not records:
        return {}

    bests = {}
    n     = len(records)
    times = np.array([(r[0] - records[0][0]).total_seconds() for r in records])
    dists = np.array([r[1] for r in records])

    for label, target_m in DISTANCES.items():
        if dists[-1] < target_m:
            continue   # ride not long enough

        best_time = float('inf')
        j = 0
        for i in range(n):
            # Advance j until distance covered >= target
            while j < n and (dists[j] - dists[i]) < target_m:
                j += 1
            if j >= n:
                break
            elapsed = times[j] - times[i]
            if elapsed < best_time:
                best_time = elapsed

        if best_time < float('inf'):
            pace_sec = best_time / (target_m / 1000)
            bests[label] = {
                "time_s"     : round(best_time, 1),
                "time_str"   : secs_to_time(best_time),
                "pace_sec_km": round(pace_sec, 1),
                "pace_label" : secs_to_time(pace_sec),
            }

    return bests


def find_cooper_distance(records: list[tuple]) -> dict | None:
    """
    Find the maximum distance covered in any 12-minute (720s) window.
    Returns distance and estimated VO2max.
    Cooper formula: VO2max = (distance_m - 504.9) / 44.73
    """
    if not records:
        return None

    times = np.array([(r[0] - records[0][0]).total_seconds() for r in records])
    dists = np.array([r[1] for r in records])

    if times[-1] < COOPER_DURATION:
        return None

    best_dist = 0.0
    j = 0
    n = len(records)

    for i in range(n):
        while j < n and (times[j] - times[i]) < COOPER_DURATION:
            j += 1
        if j >= n:
            break
        covered = dists[j] - dists[i]
        if covered > best_dist:
            best_dist = covered

    if best_dist < 1000:
        return None

    vo2max = (best_dist - 504.9) / 44.73

    return {
        "distance_m" : round(best_dist),
        "vo2max"     : round(vo2max, 1),
    }


def merge_bests(existing: dict, new: dict) -> dict:
    """Keep the best (fastest) time for each distance."""
    merged = dict(existing)
    for label, val in new.items():
        if label not in merged or val["time_s"] < merged[label]["time_s"]:
            merged[label] = val
    return merged


def merge_cooper(existing: dict | None, new: dict | None) -> dict | None:
    """Keep the best (highest distance) Cooper result."""
    if not existing:
        return new
    if not new:
        return existing
    return new if new["distance_m"] > existing["distance_m"] else existing


def load_bests() -> dict:
    if BESTS_FILE.exists():
        return json.loads(BESTS_FILE.read_text())
    return {"series": {}, "processed_files": [], "last_updated": ""}


# ── Main ──────────────────────────────────────────────────────────────────────

def build_running_bests() -> None:
    bests     = load_bests()
    processed = set(bests["processed_files"])
    cutoff    = (datetime.now() - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")

    new_files = [
        f for f in sorted(FIT_DIR.glob("*.fit"))
        if "_running_" in f.name and f.name not in processed
    ]
    print(f"Found {len(new_files)} new running .fit files to process.")

    series = bests.get("series", {})

    for fit_path in new_files:
        name     = fit_path.name
        date_str = name[:10]
        year     = name[:4]
        sk_year  = f"year_{year}"
        is_recent = date_str >= cutoff

        try:
            records = extract_distance_time_series(fit_path)
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            processed.add(name)
            continue

        if not records or records[-1][1] < 500:
            print(f"  – {name}: too short")
            processed.add(name)
            continue

        best_times = find_best_times(records)
        cooper     = find_cooper_distance(records)
        top_dist   = max(best_times.keys(),
                        key=lambda k: DISTANCES[k]) if best_times else "—"

        print(f"  ✓ {name}  [{sk_year}]  "
              f"best={top_dist}  "
              f"cooper={cooper['distance_m']}m" if cooper else f"  ✓ {name}")

        for sk in [sk_year] + (["recent"] if is_recent else []):
            if sk not in series:
                series[sk] = {"best_times": {}, "cooper": None}
            series[sk]["best_times"] = merge_bests(
                series[sk].get("best_times", {}), best_times)
            series[sk]["cooper"] = merge_cooper(
                series[sk].get("cooper"), cooper)

        processed.add(name)

    bests["series"]          = series
    bests["processed_files"] = sorted(processed)
    bests["last_updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")

    BESTS_FILE.write_text(json.dumps(bests, indent=2))
    size_kb = BESTS_FILE.stat().st_size // 1024
    print(f"\nWritten → {BESTS_FILE}  ({size_kb} KB)")
    for sk, s in series.items():
        cooper = s.get("cooper")
        vo2    = f"  Cooper VO2={cooper['vo2max']}" if cooper else ""
        print(f"  {sk:<12} bests={list(s['best_times'].keys())}{vo2}")


if __name__ == "__main__":
    build_running_bests()
