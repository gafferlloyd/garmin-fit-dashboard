"""
build_cycling_cloud.py
──────────────────────
Extracts 1-minute windowed power + HR data from cycling .fit files and
builds an incremental binned dataset (cycling_cloud.json) for the dashboard.

Series produced:
  indoor_YYYY, outdoor_YYYY  — one per year present in data
  recent                     — last 42 days (all cycling)

Each series contains:
  - Binned avg HR per power bucket (10W wide, 50–450W)
  - Linear fit with HRR marker extrapolations
  - Raw scatter points for recent window

Run after garmin_download.py:
    python build_cycling_cloud.py

Requirements:
    pip install fitparse numpy
"""

from __future__ import annotations
import io
import json
import math
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from athlete_config import (
    HR_REST, HR_MAX, HRR, THRESHOLD_HR, VT1_HR,
    HRR_MARKERS_PCT, hrr_to_bpm,
    FTP_INDOOR, FTP_OUTDOOR,
)
from fit_window_extractor import extract_clean_windows, open_fit

# ── Config ────────────────────────────────────────────────────────────────────
FIT_DIR       = Path("fit_files")
CLOUD_FILE    = Path("cycling_cloud.json")
WINDOW_SECS   = 60       # 1-minute averaging window (configurable)
POWER_MIN     = 50       # watts
POWER_MAX     = 450      # watts
BUCKET_W      = 10       # watts per bucket
HR_MIN        = 60
RECENT_DAYS   = 42
MIN_POINTS    = 5        # minimum points per bucket to include


# ── Helpers ───────────────────────────────────────────────────────────────────

from fit_window_extractor import extract_clean_windows


def bucket_index(power_w: float) -> int | None:
    if not (POWER_MIN <= power_w <= POWER_MAX):
        return None
    return int((power_w - POWER_MIN) // BUCKET_W)


def bucket_centre(idx: int) -> float:
    return POWER_MIN + idx * BUCKET_W + BUCKET_W / 2


def series_key(filename: str, is_indoor: bool) -> str:
    """Return series key e.g. 'indoor_2025', 'outdoor_2024'."""
    year = filename[:4]
    return f"{'indoor' if is_indoor else 'outdoor'}_{year}"


def compute_bucket_stats(raw_buckets: dict) -> dict:
    """Convert raw sum/sum_sq/count buckets to derived stats."""
    stats = {}
    for key, b in raw_buckets.items():
        count = b["count"]
        if count < MIN_POINTS:
            continue
        idx     = int(key)
        centre  = bucket_centre(idx)
        avg_hr  = b["sum_hr"] / count
        variance = max(0, b["sum_sq_hr"] / count - avg_hr ** 2)
        std_hr  = math.sqrt(variance)
        stats[key] = {
            "power_w"  : round(centre, 1),
            "avg_hr"   : round(avg_hr, 2),
            "std_hr"   : round(std_hr, 2),
            "count"    : count,
        }
    return stats


def fit_linear(bucket_stats: dict, ftp: float) -> dict | None:
    """
    Fit linear regression to power/HR data and extrapolate HRR markers.
    Expects roughly linear relationship across sub-threshold efforts.
    Uses all buckets up to ~110% FTP.
    """
    items = sorted(bucket_stats.values(), key=lambda b: b["power_w"])
    # Filter to sub-maximal range where relationship is linear
    items = [b for b in items if b["power_w"] <= ftp * 1.05 and b["count"] >= MIN_POINTS]

    if len(items) < 5:
        return None

    powers = np.array([b["power_w"] for b in items])
    hrs    = np.array([b["avg_hr"]  for b in items])

    coeffs = np.polyfit(powers, hrs, 1)
    slope, intercept = coeffs
    y_pred = np.polyval(coeffs, powers)
    ss_res = np.sum((hrs - y_pred) ** 2)
    ss_tot = np.sum((hrs - np.mean(hrs)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    def hr_to_power(hr_target: float) -> float | None:
        if slope == 0:
            return None
        return round((hr_target - intercept) / slope, 1)

    # HRR marker extrapolations
    markers = {}
    for pct in HRR_MARKERS_PCT:
        bpm = hrr_to_bpm(pct)
        pwr = hr_to_power(bpm)
        markers[str(pct)] = {"hr": round(bpm, 1), "power_w": pwr}

    markers["vt1"]       = {"hr": VT1_HR,       "power_w": hr_to_power(VT1_HR)}
    markers["threshold"] = {"hr": THRESHOLD_HR,  "power_w": hr_to_power(THRESHOLD_HR)}

    # Regression line for plotting
    x_line = np.linspace(powers[0], powers[-1], 60)
    y_line = np.polyval(coeffs, x_line)

    return {
        "slope"           : round(float(slope), 4),
        "intercept"       : round(float(intercept), 2),
        "r2"              : round(float(r2), 4),
        "markers"         : markers,
        "regression_line" : [
            {"power_w": round(float(x), 1), "hr": round(float(y), 2)}
            for x, y in zip(x_line, y_line)
        ],
    }


def load_cloud() -> dict:
    if CLOUD_FILE.exists():
        return json.loads(CLOUD_FILE.read_text())
    return {
        "series"          : {},   # key → {raw_buckets, recent_points}
        "processed_files" : [],
        "last_updated"    : "",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def build_cycling_cloud() -> None:
    cloud    = load_cloud()
    processed = set(cloud["processed_files"])
    cutoff_recent = datetime.now() - timedelta(days=RECENT_DAYS)
    cutoff_str    = cutoff_recent.strftime("%Y-%m-%d")

    # Find unprocessed cycling .fit files with power
    new_files = []
    for fit_path in sorted(FIT_DIR.glob("*.fit")):
        name = fit_path.name
        if "_cycling_" not in name:
            continue
        if name in processed:
            continue
        new_files.append(fit_path)

    print(f"Found {len(new_files)} new cycling .fit files to process.")

    series = cloud.get("series", {})

    # Ensure recent series exists
    if "recent" not in series:
        series["recent"] = {"raw_buckets": {}, "recent_points": []}

    for fit_path in new_files:
        name      = fit_path.name
        date_str  = name[:10]
        is_indoor = "indoor_cycling" in name or (
            # fall back: check companion json
            (FIT_DIR / name.replace(".fit", ".json")).exists() and
            json.loads((FIT_DIR / name.replace(".fit", ".json")).read_text())
                .get("is_indoor", False)
        )

        # Determine year series key
        sk = series_key(name, is_indoor)

        try:
            windows = extract_clean_windows(
                fit_path,
                is_indoor   = is_indoor,
                effort_field= "power",
                effort_min  = POWER_MIN,
                effort_max  = POWER_MAX,
            )
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            processed.add(name)
            continue

        if not windows:
            print(f"  – {name}: no valid windows")
            processed.add(name)
            continue

        print(f"  ✓ {name}: {len(windows)} windows  [{sk}]")

        # Ensure series bucket store exists
        for s in [sk, "recent"]:
            if s not in series:
                series[s] = {"raw_buckets": {}, "recent_points": []}

        is_recent = date_str >= cutoff_str

        for pwr, hr in windows:
            idx = bucket_index(pwr)
            if idx is None:
                continue
            key = str(idx)
            # Add to year series
            b = series[sk]["raw_buckets"].setdefault(
                key, {"sum_hr": 0.0, "sum_sq_hr": 0.0, "count": 0})
            b["sum_hr"]    += hr
            b["sum_sq_hr"] += hr * hr
            b["count"]     += 1

            # Add to recent series
            if is_recent:
                b2 = series["recent"]["raw_buckets"].setdefault(
                    key, {"sum_hr": 0.0, "sum_sq_hr": 0.0, "count": 0})
                b2["sum_hr"]    += hr
                b2["sum_sq_hr"] += hr * hr
                b2["count"]     += 1

        # Raw scatter for recent (capped at 2000 points per series for size)
        if is_recent:
            pts = series["recent"].setdefault("recent_points", [])
            for pwr, hr in windows[:50]:   # max 50 points per file
                pts.append({"date": date_str, "power_w": pwr, "hr": hr,
                            "indoor": is_indoor})

        processed.add(name)

    # Prune recent points older than cutoff
    for sk in series:
        pts = series[sk].get("recent_points", [])
        series[sk]["recent_points"] = [p for p in pts if p.get("date","") >= cutoff_str]

    # Compute derived stats and linear fits for each series
    output_series = {}
    for sk, s in series.items():
        is_indoor_series = sk.startswith("indoor") or sk == "recent"
        ftp = FTP_INDOOR if "indoor" in sk else FTP_OUTDOOR

        stats = compute_bucket_stats(s["raw_buckets"])
        fit   = fit_linear(stats, ftp)

        output_series[sk] = {
            "raw_buckets"   : s["raw_buckets"],
            "bucket_stats"  : stats,
            "linear_fit"    : fit,
            "recent_points" : s.get("recent_points", []),
            "n_windows"     : sum(b["count"] for b in stats.values()),
        }

    # HRR reference lines for dashboard
    hrr_markers = {str(pct): round(hrr_to_bpm(pct), 1) for pct in HRR_MARKERS_PCT}
    hrr_markers["vt1"]       = round(VT1_HR, 1)
    hrr_markers["threshold"] = round(THRESHOLD_HR, 1)

    cloud["series"]          = {sk: {k: v for k, v in s.items() if k != "raw_buckets"}
                                 for sk, s in output_series.items()}
    # Keep raw buckets for incremental updates
    cloud["_raw_series"]     = {sk: {"raw_buckets": s["raw_buckets"],
                                     "recent_points": s.get("recent_points",[])}
                                 for sk, s in output_series.items()}
    cloud["processed_files"] = sorted(processed)
    cloud["hrr_markers"]     = hrr_markers
    cloud["last_updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")

    CLOUD_FILE.write_text(json.dumps(cloud, indent=2))
    size_kb = CLOUD_FILE.stat().st_size // 1024
    print(f"\nWritten → {CLOUD_FILE}  ({size_kb} KB)")
    print(f"  Series: {sorted(output_series.keys())}")
    for sk, s in output_series.items():
        fit = s.get("linear_fit")
        r2  = f"R²={fit['r2']:.3f}" if fit else "no fit"
        print(f"  {sk:<20} {s['n_windows']:>6} windows  {r2}")


if __name__ == "__main__":
    build_cycling_cloud()
