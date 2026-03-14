"""
athlete_config.py
─────────────────
Loads athlete_config.json and exposes all constants used across
fit_parser.py, analysis.py, build_running_cloud.py, and build_data.py.

Import this module instead of hardcoding constants anywhere.

Usage:
    from athlete_config import cfg, HR_REST, HR_MAX, HRR, FTP_INDOOR, ...
"""

from __future__ import annotations
import json
from pathlib import Path

_CONFIG_FILE = Path(__file__).parent / "athlete_config.json"

def _load() -> dict:
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"athlete_config.json not found at {_CONFIG_FILE}. "
            "Please create it from the template."
        )
    return json.loads(_CONFIG_FILE.read_text())

cfg = _load()

# ── Heart rate ────────────────────────────────────────────────────────────────
HR_REST       = cfg["heart_rate"]["rest_bpm"]
HR_MAX        = cfg["heart_rate"]["max_bpm"]
HRR           = HR_MAX - HR_REST
THRESHOLD_HR  = cfg["heart_rate"]["threshold_bpm"]
VT1_HR        = cfg["heart_rate"]["vt1_bpm"]

HRR_ZONE_UPPER_PCT = [
    cfg["heart_rate"]["zones"]["Z1_upper_pct"],
    cfg["heart_rate"]["zones"]["Z2_upper_pct"],
    cfg["heart_rate"]["zones"]["Z3_upper_pct"],
    cfg["heart_rate"]["zones"]["Z4_upper_pct"],
    cfg["heart_rate"]["zones"]["Z5_upper_pct"],
]
ZONE_LABELS = ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold", "Z5 VO2max"]

HRR_MARKERS_PCT = cfg["heart_rate"]["hrr_markers_pct"]   # [50, 60, 70, 80, 90]

def hrr_to_bpm(pct: float) -> float:
    """Convert %HRR to absolute bpm. e.g. hrr_to_bpm(70) → 134.0"""
    return HR_REST + (pct / 100) * HRR

def hrr_zone(bpm: float) -> int:
    """Return 1-based HR zone (1–5) for a given heart rate (bpm)."""
    pct = (bpm - HR_REST) / HRR
    for i, upper in enumerate(HRR_ZONE_UPPER_PCT):
        if pct <= upper:
            return i + 1
    return 5

# ── Cycling ───────────────────────────────────────────────────────────────────
FTP_INDOOR    = cfg["cycling"]["ftp_indoor_w"]
FTP_OUTDOOR   = cfg["cycling"]["ftp_outdoor_w"]
NP_WINDOW     = cfg["cycling"]["np_window_s"]

# ── Running ───────────────────────────────────────────────────────────────────
def _pace_str_to_mps(pace_str: str | None) -> float | None:
    """Convert 'M:SS' string to m/s. Returns None if not set."""
    if not pace_str:
        return None
    try:
        parts = pace_str.split(":")
        sec_per_km = int(parts[0]) * 60 + int(parts[1])
        return 1000 / sec_per_km
    except Exception:
        return None

THRESHOLD_PACE_MPS = _pace_str_to_mps(
    cfg["running"]["threshold_pace_per_km"]
)
PLATEAU_PACE_SEC = (
    lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])
)(cfg["running"]["plateau_pace_per_km"])

# ── Data quality ─────────────────────────────────────────────────────────────
DQ = cfg.get("data_quality", {})
OUTDOOR_WARMUP_MIN   = DQ.get("outdoor_warmup_min",   10)
INDOOR_WARMUP_MIN    = DQ.get("indoor_warmup_min",     5)
PRIMARY_WINDOW_MIN   = DQ.get("primary_window_min",   70)
EXTENSION_BLOCK_MIN  = DQ.get("extension_block_min",  30)
DRIFT_THRESHOLD_BPM  = DQ.get("drift_threshold_bpm",   5)

# ── Athlete ───────────────────────────────────────────────────────────────────
WEIGHT_KG     = cfg["athlete"]["weight_kg"]
