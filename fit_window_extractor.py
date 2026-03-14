"""
fit_window_extractor.py
───────────────────────
Shared utility for extracting clean 1-minute windows from .fit files.

Applies:
  1. Warm-up exclusion (configurable per indoor/outdoor)
  2. Primary window limit (default 70 min)
  3. Cardiac drift detection — excludes data where residual HR drift
     between first and second half of a block exceeds threshold
  4. Optional extension blocks beyond 70 min if drift-free

Used by both build_running_cloud.py and build_cycling_cloud.py.
"""

from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Callable

import fitparse

from athlete_config import (
    OUTDOOR_WARMUP_MIN, INDOOR_WARMUP_MIN,
    PRIMARY_WINDOW_MIN, EXTENSION_BLOCK_MIN,
    DRIFT_THRESHOLD_BPM,
)

WINDOW_SECS = 60
STEP_SECS   = 30   # 50% overlap


def open_fit(path: Path) -> fitparse.FitFile:
    raw = path.read_bytes()
    if raw[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            fit_name = next(n for n in zf.namelist() if n.endswith('.fit'))
            fit_bytes = zf.read(fit_name)
        return fitparse.FitFile(io.BytesIO(fit_bytes))
    return fitparse.FitFile(str(path))


def _residual_drift(windows: list[tuple], predict_fn: Callable | None) -> float:
    """
    Compute mean HR residual drift between first and second half of a window list.
    If predict_fn is None (no characteristic yet), uses raw HR drift.
    Returns drift in bpm (positive = second half higher).
    """
    if len(windows) < 4:
        return 0.0
    mid   = len(windows) // 2
    first = windows[:mid]
    second= windows[mid:]

    if predict_fn is not None:
        # Residual = actual HR - predicted HR for that effort
        def residual(w):
            effort, hr = w[0], w[1]
            pred = predict_fn(effort)
            return hr - pred if pred is not None else 0.0
        drift = (sum(residual(w) for w in second) / len(second) -
                 sum(residual(w) for w in first)  / len(first))
    else:
        # Cold start — use raw HR drift
        drift = (sum(w[1] for w in second) / len(second) -
                 sum(w[1] for w in first)  / len(first))
    return drift


def extract_clean_windows(
    fit_path     : Path,
    is_indoor    : bool,
    effort_field : str,          # 'pace' or 'power'
    effort_min   : float,
    effort_max   : float,
    hr_min       : float = 60,
    predict_fn   : Callable | None = None,
) -> list[tuple[float, float]]:
    """
    Extract clean (effort, hr) windows from a .fit file.

    Parameters
    ----------
    fit_path     : path to .fit file
    is_indoor    : True for indoor sessions (shorter warm-up)
    effort_field : 'pace' (sec/km) or 'power' (watts)
    effort_min   : minimum effort value to include
    effort_max   : maximum effort value to include
    hr_min       : minimum plausible HR
    predict_fn   : optional function(effort) → predicted_hr for drift residuals

    Returns list of (effort, hr) tuples — clean windows only.
    """
    fitfile = open_fit(fit_path)

    # ── Collect raw records ───────────────────────────────────────────────────
    records = []
    for rec in fitfile.get_messages("record"):
        data = {f.name: f.value for f in rec}
        ts = data.get("timestamp")
        hr = data.get("heart_rate")
        if not ts or not hr or hr < hr_min:
            continue

        if effort_field == "pace":
            spd = data.get("enhanced_speed") or data.get("speed")
            if not spd or spd <= 0:
                continue
            effort = 1000 / spd   # sec/km
        else:  # power
            effort = data.get("power")
            if effort is None:
                continue
            effort = float(effort)

        if effort_min <= effort <= effort_max:
            records.append((ts, effort, float(hr)))

    if not records:
        return []

    # ── Apply warm-up exclusion ───────────────────────────────────────────────
    # Use the FIRST timestamp in the entire file (not first in-range record)
    # to avoid double-penalising slow warm-up paces
    fitfile2  = open_fit(fit_path)
    all_ts    = []
    for rec in fitfile2.get_messages("record"):
        data = {f.name: f.value for f in rec}
        ts   = data.get("timestamp")
        if ts:
            all_ts.append(ts)
    t_start    = all_ts[0] if all_ts else records[0][0]

    warmup_min = INDOOR_WARMUP_MIN if is_indoor else OUTDOOR_WARMUP_MIN
    warmup_end = t_start.timestamp() + warmup_min * 60
    primary_end= t_start.timestamp() + PRIMARY_WINDOW_MIN * 60

    records_post_warmup = [r for r in records
                           if r[0].timestamp() >= warmup_end]
    if not records_post_warmup:
        return []

    # ── Build 1-minute sliding windows ───────────────────────────────────────
    def make_windows(recs: list) -> list[tuple[float, float]]:
        windows = []
        for i in range(0, len(recs) - WINDOW_SECS, STEP_SECS):
            window = recs[i:i + WINDOW_SECS]
            dt = (window[-1][0] - window[0][0]).total_seconds()
            if not (45 <= dt <= 90):
                continue
            avg_effort = sum(r[1] for r in window) / len(window)
            avg_hr     = sum(r[2] for r in window) / len(window)
            windows.append((round(avg_effort, 1), round(avg_hr, 1)))
        return windows

    # ── Primary window (warmup → 70 min) ─────────────────────────────────────
    primary_records = [r for r in records_post_warmup
                       if r[0].timestamp() <= primary_end]
    primary_windows = make_windows(primary_records)

    # Drift check on primary window
    drift = _residual_drift(primary_windows, predict_fn)
    if drift > DRIFT_THRESHOLD_BPM:
        # Keep only first half of primary window
        primary_windows = primary_windows[:len(primary_windows) // 2]

    clean_windows = primary_windows

    # ── Extension blocks beyond 70 min ───────────────────────────────────────
    block_start = primary_end
    while True:
        block_end = block_start + EXTENSION_BLOCK_MIN * 60
        block_records = [r for r in records
                         if block_start <= r[0].timestamp() < block_end]
        if len(block_records) < WINDOW_SECS:
            break

        block_windows = make_windows(block_records)
        drift = _residual_drift(block_windows, predict_fn)

        if drift <= DRIFT_THRESHOLD_BPM:
            clean_windows.extend(block_windows)
            block_start = block_end
        else:
            break   # drift detected — stop extending

    return clean_windows
