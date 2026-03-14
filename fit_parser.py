"""
fit_parser.py
─────────────
Parses a Garmin .fit file and returns a structured metrics dict.
Handles cycling (indoor + outdoor) and running separately.

Metrics extracted:
  - Duration, distance, elevation
  - Avg / max speed or pace
  - Avg / max / min heart rate + time-in-zone (5-zone %HRR / Karvonen)
  - Avg / max cadence
  - Power: avg, max, normalised (NP), Intensity Factor (IF), TSS
    (cycling only; skipped gracefully if no power data)

Constants (update as fitness changes):
  - FTP_INDOOR / FTP_OUTDOOR
  - HR_REST / HR_MAX

Requirements:
    pip install fitparse
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitparse

# ── Config — loaded from athlete_config.json ─────────────────────────────────
from athlete_config import (
    HR_REST, HR_MAX, HRR, HRR_ZONE_UPPER_PCT, ZONE_LABELS,
    FTP_INDOOR, FTP_OUTDOOR, NP_WINDOW, WEIGHT_KG,
    hrr_zone as _hrr_zone,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

hrr_zone = _hrr_zone   # imported from athlete_config


def normalised_power(power_series: list[float], window: int = NP_WINDOW) -> float:
    """
    Standard NP calculation:
      1. 30 s rolling average of power
      2. Raise each value to the 4th power
      3. Take the mean
      4. Take the 4th root
    Returns 0.0 if insufficient data.
    """
    if len(power_series) < window:
        return 0.0
    rolling = [
        sum(power_series[i:i + window]) / window
        for i in range(len(power_series) - window + 1)
    ]
    mean_fourth = sum(x ** 4 for x in rolling) / len(rolling)
    return mean_fourth ** 0.25


def mps_to_pace(mps: float) -> str:
    """Convert metres/sec → 'MM:SS /km' string. Returns '--' for zero."""
    if mps <= 0:
        return "--"
    secs_per_km = 1000 / mps
    mins, secs = divmod(int(secs_per_km), 60)
    return f"{mins}:{secs:02d} /km"


def seconds_to_hms(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_fit(filepath: str | Path, activity_type: str) -> dict:
    """
    Parse a .fit file and return a metrics dict.

    Parameters
    ----------
    filepath      : path to the .fit file
    activity_type : 'cycling', 'indoor_cycling', 'road_biking', or 'running'
    """
    filepath = Path(filepath)
    is_cycling = activity_type in ("cycling", "indoor_cycling", "road_biking")
    is_indoor  = activity_type == "indoor_cycling"
    ftp        = FTP_INDOOR if is_indoor else FTP_OUTDOOR

    # Garmin's ORIGINAL download wraps the .fit in a zip — handle both cases
    import zipfile, io
    raw = Path(filepath).read_bytes()
    if raw[:2] == b'PK':   # zip magic bytes
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            fit_name = next(n for n in zf.namelist() if n.endswith('.fit'))
            fit_bytes = zf.read(fit_name)
        fitfile = fitparse.FitFile(io.BytesIO(fit_bytes))
    else:
        fitfile = fitparse.FitFile(str(filepath))

    # ── Accumulators ─────────────────────────────────────────────────────────
    hr_values      : list[int]   = []
    power_values   : list[float] = []
    cadence_values : list[int]   = []
    speed_values   : list[float] = []   # m/s
    zone_seconds   : list[float] = [0.0] * 5

    total_distance = 0.0   # metres
    total_ascent   = 0.0   # metres
    duration_s     = 0.0   # seconds (moving/elapsed from session record)
    start_time     = None
    end_time       = None

    prev_timestamp = None
    prev_alt       = None

    # Running biomechanics accumulators
    vertical_osc_values : list[float] = []
    stance_time_values  : list[float] = []
    step_length_values  : list[float] = []

    # ── Iterate records ───────────────────────────────────────────────────────
    for record in fitfile.get_messages("record"):
        data = {f.name: f.value for f in record}

        ts = data.get("timestamp")
        if ts is None:
            continue

        dt = (ts - prev_timestamp).total_seconds() if prev_timestamp else 1.0
        dt = max(0.0, min(dt, 10.0))   # clamp: ignore gaps > 10 s
        prev_timestamp = ts

        if start_time is None:
            start_time = ts
        end_time = ts

        # Heart rate — filter implausible values
        hr = data.get("heart_rate")
        if hr is not None and 40 <= hr <= (HR_MAX + 5):
            hr_values.append(hr)
            zone_idx = hrr_zone(hr) - 1
            zone_seconds[zone_idx] += dt

        # Power — present in both cycling and running records
        pwr = data.get("power")
        if pwr is not None and 0 < pwr < 2500:
            power_values.append(float(pwr))

        # Cadence
        cad = data.get("cadence")
        if cad is not None and cad > 0:
            cadence_values.append(cad)

        # Speed — Garmin uses enhanced_speed (m/s); fall back to speed
        spd = data.get("enhanced_speed") or data.get("speed")
        if spd is not None and spd >= 0:
            speed_values.append(float(spd))

        # Distance (cumulative field — take the last value via session record)
        dist = data.get("distance")
        if dist is not None:
            total_distance = dist         # Garmin stores cumulative metres

        # Elevation gain via enhanced_altitude
        alt = data.get("enhanced_altitude") or data.get("altitude")
        if alt is not None and prev_alt is not None:
            delta = alt - prev_alt
            if 0 < delta < 50:   # ignore GPS spikes
                total_ascent += delta
        if alt is not None:
            prev_alt = alt

        # Running biomechanics
        vo = data.get("vertical_oscillation")
        if vo is not None and vo > 0:
            vertical_osc_values.append(vo)
        st = data.get("stance_time")
        if st is not None and st > 0:
            stance_time_values.append(st)
        sl = data.get("step_length")
        if sl is not None and sl > 0:
            step_length_values.append(sl)

    # Session summary — more reliable than accumulating from records
    for session in fitfile.get_messages("session"):
        data = {f.name: f.value for f in session}
        if data.get("total_elapsed_time"):
            duration_s     = data["total_elapsed_time"]
        if data.get("total_distance"):
            total_distance = data["total_distance"]
        if data.get("total_ascent"):
            total_ascent   = data["total_ascent"]
        # Running uses avg_running_cadence; cycling uses avg_cadence
        sess_cad = data.get("avg_running_cadence") or data.get("avg_cadence")
        sess_max_cad = data.get("max_running_cadence") or data.get("max_cadence")
        # Session-level enhanced speed is more reliable than record mean
        sess_avg_spd = data.get("enhanced_avg_speed") or data.get("avg_speed")
        sess_max_spd = data.get("enhanced_max_speed") or data.get("max_speed")
        if sess_avg_spd:
            speed_values_session = (float(sess_avg_spd), float(sess_max_spd or 0))
        else:
            speed_values_session = None

    if duration_s == 0 and start_time and end_time:
        duration_s = (end_time - start_time).total_seconds()

    # ── Derived metrics ───────────────────────────────────────────────────────
    avg_hr  = round(sum(hr_values) / len(hr_values)) if hr_values else None
    max_hr  = max(hr_values)                          if hr_values else None
    min_hr  = min(hr_values)                          if hr_values else None

    avg_cad = round(sum(cadence_values) / len(cadence_values)) if cadence_values else None
    max_cad = max(cadence_values)                               if cadence_values else None
    # Override with session-level running cadence if available (includes fractional)
    if 'sess_max_cad' in dir() and sess_max_cad:
        max_cad = sess_max_cad
    if 'sess_cad' in dir() and sess_cad:
        avg_cad = sess_cad

    # Use session-level speed (more accurate than averaging record-level)
    if 'speed_values_session' in dir() and speed_values_session:
        avg_spd, max_spd = speed_values_session
    else:
        avg_spd = sum(speed_values) / len(speed_values) if speed_values else 0.0
        max_spd = max(speed_values)                      if speed_values else 0.0

    # HR zone percentages
    total_hr_time = sum(zone_seconds)
    zone_pct = (
        {ZONE_LABELS[i]: round(zone_seconds[i] / total_hr_time * 100, 1)
         for i in range(5)}
        if total_hr_time > 0 else {}
    )

    # Power metrics (cycling only)
    power_metrics = {}
    if is_cycling and power_values:
        avg_pwr = sum(power_values) / len(power_values)
        max_pwr = max(power_values)
        np      = normalised_power(power_values)
        IF      = np / ftp if ftp > 0 else 0.0
        tss     = (duration_s * np * IF) / (ftp * 3600) * 100 if ftp > 0 else 0.0
        power_metrics = {
            "avg_power_w"  : round(avg_pwr, 1),
            "max_power_w"  : round(max_pwr),
            "norm_power_w" : round(np, 1),
            "intensity_factor": round(IF, 3),
            "tss"          : round(tss, 1),
            "ftp_used_w"   : ftp,
            "w_per_kg"     : round(avg_pwr / WEIGHT_KG, 2),
            "np_per_kg"    : round(np / WEIGHT_KG, 2),
        }

    # Speed / pace formatting
    speed_metrics: dict = {}
    if is_cycling:
        speed_metrics = {
            "avg_speed_kph": round(avg_spd * 3.6, 1),
            "max_speed_kph": round(max_spd * 3.6, 1),
        }
    else:
        speed_metrics = {
            "avg_pace"     : mps_to_pace(avg_spd),
            "max_pace"     : mps_to_pace(max_spd),
            "avg_speed_kph": round(avg_spd * 3.6, 1),
        }

    # ── Assemble result ───────────────────────────────────────────────────────
    result = {
        "file"          : filepath.name,
        "activity_type" : activity_type,
        "is_indoor"     : is_indoor,
        "duration"      : seconds_to_hms(duration_s),
        "duration_s"    : round(duration_s),
        "distance_km"   : round(total_distance / 1000, 2),
        "elevation_m"   : round(total_ascent),
        "heart_rate"    : {
            "avg_bpm"   : avg_hr,
            "max_bpm"   : max_hr,
            "min_bpm"   : min_hr,
            "avg_hrr_pct": round((avg_hr - HR_REST) / HRR * 100, 1) if avg_hr else None,
        },
        "hr_zones_pct"  : zone_pct,
        "cadence"       : {
            "avg_rpm"   : avg_cad,
            "max_rpm"   : max_cad,
        },
        **speed_metrics,
        **power_metrics,
    }

    # Running biomechanics (only populated for running activities)
    if not is_cycling and vertical_osc_values:
        result["running_biomechanics"] = {
            "avg_vertical_osc_mm" : round(sum(vertical_osc_values) / len(vertical_osc_values), 1),
            "avg_stance_time_ms"  : round(sum(stance_time_values)  / len(stance_time_values),  1) if stance_time_values else None,
            "avg_step_length_mm"  : round(sum(step_length_values)  / len(step_length_values))     if step_length_values  else None,
        }

    return result


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python fit_parser.py <path/to/file.fit> <activity_type>")
        print("  activity_type: cycling | indoor_cycling | road_biking | running")
        sys.exit(1)

    metrics = parse_fit(sys.argv[1], sys.argv[2])
    print(json.dumps(metrics, indent=2))
