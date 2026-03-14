"""
build_data.py
─────────────
Merges all session .json files from fit_files/ into a single
dashboard_data.json for the GitHub Pages dashboard.

Run this after garmin_download.py:
    python build_data.py

Output: dashboard_data.json  (committed to repo, read by index.html)
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from analysis import (
    load_all_json,
    cardiac_efficiency_index,
    training_load_summary,
    indoor_outdoor_delta,
    running_tss,
    detect_linear_region,
)
from pathlib import Path as _Path
import json as _json

FIT_DIR    = Path("fit_files")
OUTPUT     = Path("dashboard_data.json")
MIN_POWER_CEI = 150   # watts — exclude warmups/cooldowns from CEI


def date_from_filename(filename: str) -> str:
    """Extract YYYY-MM-DD from filename like 2026-03-12_17-04-42_cycling_xxx.json"""
    return filename[:10]


def build_dashboard_data() -> dict:
    sessions = load_all_json(FIT_DIR)
    if not sessions:
        raise RuntimeError("No session .json files found in fit_files/")

    print(f"Loaded {len(sessions)} sessions.")

    # ── PMC (all sessions, chronological) ────────────────────────────────────
    pmc_rows = training_load_summary(sessions)

    # Attach date to each PMC row
    pmc_chart = []
    for s, pmc in zip(sessions, pmc_rows):
        date = date_from_filename(s.get("file", ""))
        pmc_chart.append({
            "date"  : date,
            "tss"   : pmc["tss"],
            "rtss"  : pmc["rtss"],
            "stss"  : pmc["stss"],
            "ctl"   : pmc["ctl"],
            "atl"   : pmc["atl"],
            "tsb"   : pmc["tsb"],
            "type"  : s.get("activity_type", ""),
        })

    # ── CEI trend (cycling with power, min power filter) ─────────────────────
    cei_series = []
    for s in sessions:
        cei = cardiac_efficiency_index(s)
        if not cei:
            continue
        if cei["avg_power_w"] < MIN_POWER_CEI:
            continue
        date = date_from_filename(s.get("file", ""))
        cei_series.append({
            "date"       : date,
            "cei_inv"    : cei["cei_inv"],      # W/bpm — higher = better
            "cei"        : cei["cei"],           # bpm/W — lower = better
            "avg_power_w": cei["avg_power_w"],
            "avg_hr_bpm" : cei["avg_hr_bpm"],
            "is_indoor"  : cei["is_indoor"],
        })

    # ── Recent sessions table (last 10, all types) ────────────────────────────
    recent = []
    for s in reversed(sessions[-10:]):
        hr   = s.get("heart_rate", {})
        cad  = s.get("cadence", {})
        row  = {
            "date"         : date_from_filename(s.get("file", "")),
            "type"         : s.get("activity_type", ""),
            "is_indoor"    : s.get("is_indoor", False),
            "duration"     : s.get("duration", ""),
            "distance_km"  : s.get("distance_km"),
            "elevation_m"  : s.get("elevation_m"),
            "avg_hr"       : hr.get("avg_bpm"),
            "max_hr"       : hr.get("max_bpm"),
            "avg_hrr_pct"  : hr.get("avg_hrr_pct"),
            "avg_cadence"  : cad.get("avg_rpm"),
            "avg_power_w"  : s.get("avg_power_w"),
            "norm_power_w" : s.get("norm_power_w"),
            "tss"          : s.get("tss"),
            "if"           : s.get("intensity_factor"),
            "avg_speed_kph": s.get("avg_speed_kph"),
            "avg_pace"     : s.get("avg_pace"),
            "hr_zones"     : s.get("hr_zones_pct", {}),
        }
        recent.append(row)

    # ── Indoor/outdoor delta ──────────────────────────────────────────────────
    delta = indoor_outdoor_delta(sessions)

    # ── Summary stats ─────────────────────────────────────────────────────────
    latest_pmc   = pmc_chart[-1] if pmc_chart else {}
    cycling_sess = [s for s in sessions if "cycl" in s.get("activity_type","") or "bik" in s.get("activity_type","")]
    running_sess = [s for s in sessions if "run"  in s.get("activity_type","")]

    summary = {
        "total_sessions"  : len(sessions),
        "cycling_sessions": len(cycling_sess),
        "running_sessions": len(running_sess),
        "current_ctl"     : latest_pmc.get("ctl"),
        "current_atl"     : latest_pmc.get("atl"),
        "current_tsb"     : latest_pmc.get("tsb"),
        "date_range"      : {
            "from": date_from_filename(sessions[0].get("file", "")),
            "to"  : date_from_filename(sessions[-1].get("file", "")),
        },
    }

    # ── Running pace/HR cloud ─────────────────────────────────────────────────
    running_cloud = {}
    cloud_file = _Path("running_cloud.json")
    if cloud_file.exists():
        cloud_raw    = _json.loads(cloud_file.read_text())
        series_raw   = cloud_raw.get("series", {})
        # Compute linear fit for each series
        output_series = {}
        for sk, s in series_raw.items():
            stats  = s.get("bucket_stats", {})
            fit    = detect_linear_region(stats) if stats else None
            output_series[sk] = {
                "bucket_stats"  : stats,
                "recent_points" : s.get("recent_points", []),
                "linear_fit"    : fit,
                "n_windows"     : s.get("n_windows", 0),
            }
        running_cloud = {
            "series"      : output_series,
            "hrr_markers" : cloud_raw.get("hrr_markers", {}),
            "last_updated": cloud_raw.get("last_updated", ""),
        }

    # ── Cycling power curve ─────────────────────────────────────────────────────
    cycling_curve = {}
    curve_file = _Path("cycling_curve.json")
    if curve_file.exists():
        cycling_curve = _json.loads(curve_file.read_text())

    # ── Running bests ─────────────────────────────────────────────────────────
    running_bests = {}
    bests_file = _Path("running_bests.json")
    if bests_file.exists():
        running_bests = _json.loads(bests_file.read_text())

    # ── Cycling power/HR cloud ───────────────────────────────────────────────
    cycling_cloud = {}
    cycling_file = _Path("cycling_cloud.json")
    if cycling_file.exists():
        cycling_raw = _json.loads(cycling_file.read_text())
        cycling_cloud = {
            "series"      : cycling_raw.get("series", {}),
            "hrr_markers" : cycling_raw.get("hrr_markers", {}),
            "last_updated": cycling_raw.get("last_updated", ""),
        }

    return {
        "generated"     : __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        "summary"       : summary,
        "pmc_chart"     : pmc_chart,
        "cei_series"    : cei_series,
        "recent_sessions": recent,
        "indoor_outdoor_delta": delta,
        "running_cloud"       : running_cloud,
        "cycling_cloud"       : cycling_cloud,
        "cycling_curve"       : cycling_curve,
        "running_bests"       : running_bests,
    }


if __name__ == "__main__":
    data = build_dashboard_data()
    OUTPUT.write_text(json.dumps(data, indent=2))
    print(f"Written → {OUTPUT}  ({OUTPUT.stat().st_size // 1024} KB)")
    print(f"  PMC rows    : {len(data['pmc_chart'])}")
    print(f"  CEI points  : {len(data['cei_series'])}")
    print(f"  Recent sess : {len(data['recent_sessions'])}")
