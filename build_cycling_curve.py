"""
build_cycling_curve.py
──────────────────────
Builds mean maximal power (MMP) curves and NP curves from cycling .fit files.
Produces cycling_curve.json with one curve per series (year + recent).

Series: indoor_YYYY, outdoor_YYYY, recent (42 days)

For each series:
  - MMP curve: best average power for durations 5s → longest ride
  - NP curve: best normalised power for 5min+ durations
  - HR annotations at: 1, 5, 10, 20, 30, 60, 90 min
  - Incremental — only processes new .fit files

Run after garmin_download.py:
    python3 build_cycling_curve.py

To rebuild from scratch:
    rm cycling_curve.json && python3 build_cycling_curve.py
"""

from __future__ import annotations
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import fitparse
import numpy as np

from athlete_config import (
    FTP_INDOOR, FTP_OUTDOOR, NP_WINDOW,
    HR_REST, HR_MAX,
)

# ── Config ────────────────────────────────────────────────────────────────────
FIT_DIR      = Path("fit_files")
CURVE_FILE   = Path("cycling_curve.json")
RECENT_DAYS  = 42

# Durations to sample on the MMP curve (seconds)
# Dense at short end, sparse at long end — log-spaced
MMP_DURATIONS = (
    [5, 10, 15, 20, 30, 45] +
    [60, 90, 120, 180, 240, 300] +           # 1–5 min
    [360, 420, 480, 540, 600] +              # 6–10 min
    [720, 900, 1200, 1500, 1800] +           # 12–30 min
    [2400, 3000, 3600, 4500, 5400, 7200]     # 40–120 min
)

# HR annotation durations (seconds)
HR_ANNOTATE = [60, 300, 600, 1200, 1800, 3600, 5400]
HR_LABELS   = ["1min", "5min", "10min", "20min", "30min", "60min", "90min"]

# NP curve starts at 5 min
NP_MIN_DURATION = 300


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_fit(path: Path) -> fitparse.FitFile:
    raw = path.read_bytes()
    if raw[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            fit_name = next(n for n in zf.namelist() if n.endswith('.fit'))
            fit_bytes = zf.read(fit_name)
        return fitparse.FitFile(io.BytesIO(fit_bytes))
    return fitparse.FitFile(str(path))


def extract_power_hr_series(fit_path: Path) -> tuple[list[float], list[float]]:
    """
    Extract aligned power and HR series (1 sample/sec) from a .fit file.
    Returns (power_list, hr_list) — None values filled with 0/previous.
    """
    fitfile = open_fit(fit_path)
    records = []
    for rec in fitfile.get_messages("record"):
        data = {f.name: f.value for f in rec}
        ts  = data.get("timestamp")
        pwr = data.get("power")
        hr  = data.get("heart_rate")
        if ts:
            records.append((ts, pwr, hr))

    if not records:
        return [], []

    # Resample to 1 Hz by interpolating gaps
    powers, hrs = [], []
    prev_pwr, prev_hr = 0.0, 0.0
    prev_ts = records[0][0]

    for ts, pwr, hr in records:
        dt = int((ts - prev_ts).total_seconds())
        fill_pwr = float(pwr) if pwr is not None else prev_pwr
        fill_hr  = float(hr)  if hr  is not None else prev_hr
        for _ in range(max(1, dt)):
            powers.append(fill_pwr)
            hrs.append(fill_hr)
        prev_ts  = ts
        prev_pwr = fill_pwr
        prev_hr  = fill_hr

    return powers, hrs


def best_average_power(powers: list[float], duration_s: int) -> float | None:
    """Find the best (highest) average power over any window of duration_s seconds."""
    n = len(powers)
    if n < duration_s:
        return None
    # Use numpy sliding window sum for efficiency
    arr = np.array(powers, dtype=np.float32)
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0)
    windows = (cumsum[duration_s:] - cumsum[:-duration_s]) / duration_s
    return float(np.max(windows))


def best_np_power(powers: list[float], duration_s: int,
                  window: int = NP_WINDOW) -> float | None:
    """Find the best NP over any window of duration_s seconds."""
    n = len(powers)
    if n < duration_s:
        return None
    arr = np.array(powers, dtype=np.float32)
    # Rolling NP across all windows of duration_s
    best = 0.0
    step = max(1, duration_s // 20)   # sample positions
    for start in range(0, n - duration_s, step):
        seg = arr[start:start + duration_s]
        # 30s rolling average → 4th power → mean → 4th root
        if len(seg) < window:
            continue
        roll = np.convolve(seg, np.ones(window)/window, mode='valid')
        np_val = float(np.mean(roll ** 4) ** 0.25)
        if np_val > best:
            best = np_val
    return best if best > 0 else None


def avg_hr_at_best_power(powers: list[float], hrs: list[float],
                          duration_s: int) -> float | None:
    """Return the average HR during the best power window of duration_s."""
    n = len(powers)
    if n < duration_s:
        return None
    arr = np.array(powers, dtype=np.float32)
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0)
    windows = (cumsum[duration_s:] - cumsum[:-duration_s]) / duration_s
    best_idx = int(np.argmax(windows))
    hr_seg = hrs[best_idx:best_idx + duration_s]
    valid_hrs = [h for h in hr_seg if 40 <= h <= HR_MAX + 5]
    return round(float(np.mean(valid_hrs)), 1) if valid_hrs else None


# ── MMP computation ───────────────────────────────────────────────────────────

def compute_mmp(powers: list[float], hrs: list[float]) -> dict:
    """Compute full MMP and NP curve for a power/HR series."""
    max_dur = len(powers)
    mmp, nmp, hr_ann = {}, {}, {}

    for dur in MMP_DURATIONS:
        if dur > max_dur:
            break
        p = best_average_power(powers, dur)
        if p and p > 0:
            mmp[str(dur)] = round(p, 1)

        if dur >= NP_MIN_DURATION:
            np_p = best_np_power(powers, dur)
            if np_p and np_p > 0:
                nmp[str(dur)] = round(np_p, 1)

    for dur in HR_ANNOTATE:
        if dur > max_dur:
            continue
        p   = best_average_power(powers, dur)
        hr  = avg_hr_at_best_power(powers, hrs, dur)
        if p and hr:
            hr_ann[str(dur)] = {"power_w": round(p, 1), "hr": hr}

    return {"mmp": mmp, "np_curve": nmp, "hr_annotations": hr_ann}


def merge_curves(existing: dict, new: dict) -> dict:
    """
    Merge two MMP curve dicts by taking the maximum power at each duration.
    Also merges HR annotations by keeping the one with higher power.
    """
    merged_mmp = dict(existing.get("mmp", {}))
    for dur, pwr in new.get("mmp", {}).items():
        if dur not in merged_mmp or pwr > merged_mmp[dur]:
            merged_mmp[dur] = pwr

    merged_np = dict(existing.get("np_curve", {}))
    for dur, pwr in new.get("np_curve", {}).items():
        if dur not in merged_np or pwr > merged_np[dur]:
            merged_np[dur] = pwr

    merged_hr = dict(existing.get("hr_annotations", {}))
    for dur, ann in new.get("hr_annotations", {}).items():
        if dur not in merged_hr or ann["power_w"] > merged_hr[dur]["power_w"]:
            merged_hr[dur] = ann

    return {"mmp": merged_mmp, "np_curve": merged_np,
            "hr_annotations": merged_hr}


def load_curve() -> dict:
    if CURVE_FILE.exists():
        return json.loads(CURVE_FILE.read_text())
    return {"series": {}, "processed_files": [], "last_updated": ""}


# ── Main ──────────────────────────────────────────────────────────────────────

def build_cycling_curve() -> None:
    curve     = load_curve()
    processed = set(curve["processed_files"])
    cutoff    = (datetime.now() - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")

    new_files = [
        f for f in sorted(FIT_DIR.glob("*.fit"))
        if "_cycling_" in f.name and f.name not in processed
    ]
    print(f"Found {len(new_files)} new cycling .fit files to process.")

    series = curve.get("series", {})

    for fit_path in new_files:
        name      = fit_path.name
        date_str  = name[:10]
        year      = name[:4]

        # Determine indoor/outdoor from companion json
        json_path = FIT_DIR / name.replace(".fit", ".json")
        is_indoor = False
        if json_path.exists():
            meta = json.loads(json_path.read_text())
            is_indoor = meta.get("is_indoor", False)

        sk_year   = f"{'indoor' if is_indoor else 'outdoor'}_{year}"
        is_recent = date_str >= cutoff

        try:
            powers, hrs = extract_power_hr_series(fit_path)
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            processed.add(name)
            continue

        if len(powers) < 30:
            print(f"  – {name}: insufficient power data")
            processed.add(name)
            continue

        has_power = sum(1 for p in powers if p > 0) > len(powers) * 0.3
        if not has_power:
            print(f"  – {name}: no power data")
            processed.add(name)
            continue

        print(f"  ✓ {name}  [{sk_year}]  {len(powers)//60}min")

        new_curve = compute_mmp(powers, hrs)

        # Merge into year series
        for sk in [sk_year] + (["recent"] if is_recent else []):
            if sk not in series:
                series[sk] = {"mmp": {}, "np_curve": {}, "hr_annotations": {}}
            series[sk] = merge_curves(series[sk], new_curve)

        processed.add(name)

    # Annotate with w/kg and %FTP
    for sk, s in series.items():
        ftp = FTP_INDOOR if "indoor" in sk else FTP_OUTDOOR
        annotated = {}
        for dur, pwr in s.get("mmp", {}).items():
            from athlete_config import WEIGHT_KG
            annotated[dur] = {
                "power_w" : pwr,
                "w_per_kg": round(pwr / WEIGHT_KG, 2),
                "pct_ftp" : round(pwr / ftp * 100, 1),
            }
        series[sk]["mmp_annotated"] = annotated

    curve["series"]          = series
    curve["processed_files"] = sorted(processed)
    curve["last_updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")

    CURVE_FILE.write_text(json.dumps(curve, indent=2))
    size_kb = CURVE_FILE.stat().st_size // 1024
    print(f"\nWritten → {CURVE_FILE}  ({size_kb} KB)")
    for sk, s in series.items():
        n = len(s.get("mmp", {}))
        print(f"  {sk:<20} {n} MMP points")


if __name__ == "__main__":
    build_cycling_curve()
