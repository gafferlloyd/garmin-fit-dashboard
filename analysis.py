"""
analysis.py
───────────
Proprietary analyses built on top of fit_parser.py output.
Import parse_fit from fit_parser, then pass the result dict here.

Current analyses:
  - cardiac_efficiency_index : steady-state HR per watt at sub-threshold
                               (leading indicator of adaptation)
  - indoor_outdoor_delta     : compares power/HR metrics across contexts
  - training_load_summary    : rolling TSS and ATL/CTL/TSB (PMC)

Add new analyses as standalone functions. Each takes a metrics dict
(or list of dicts for multi-session analyses) and returns a dict.

Usage:
    from fit_parser import parse_fit
    from analysis import cardiac_efficiency_index, training_load_summary

    metrics = parse_fit("my_ride.fit", "cycling")
    cei = cardiac_efficiency_index(metrics)
"""

from __future__ import annotations
from pathlib import Path
import json
import math

# ── Running threshold config ──────────────────────────────────────────────────
THRESHOLD_PACE_MPS = 1000 / (4 * 60)   # 4:00/km = 4.1667 m/s
THRESHOLD_HR       = 156                # bpm — from cross-country race data


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(json_path: str | Path) -> dict:
    """Load a parsed metrics .json file."""
    return json.loads(Path(json_path).read_text())


def load_all_json(fit_dir: str | Path = "fit_files") -> list[dict]:
    """Load all .json metrics files from fit_dir, sorted by filename (chronological)."""
    fit_dir = Path(fit_dir)
    files   = sorted(fit_dir.glob("*.json"))
    return [json.loads(f.read_text()) for f in files]


# ── Analysis 1: Cardiac Efficiency Index ─────────────────────────────────────

def cardiac_efficiency_index(metrics: dict) -> dict | None:
    """
    Cardiac Efficiency Index (CEI): steady-state HR per watt at sub-threshold.

    Intended for controlled sub-threshold efforts (~220–240W, 10–20 min,
    consistent conditions). A declining CEI over time = improved cardiac
    efficiency = leading indicator of adaptation ahead of FTP changes.

    Returns None if the activity has no power data or is not cycling.

    Output keys:
        avg_power_w       : average power for the session
        avg_hr_bpm        : average HR for the session
        cei               : bpm / watt  (lower = more efficient)
        cei_inv           : watts / bpm (higher = more efficient, easier to read)
        is_indoor         : bool
        context_note      : reminder of confounders to control for
    """
    if not metrics.get("avg_power_w") or not metrics.get("heart_rate", {}).get("avg_bpm"):
        return None

    avg_pwr = metrics["avg_power_w"]
    avg_hr  = metrics["heart_rate"]["avg_bpm"]

    cei     = avg_hr / avg_pwr          # bpm/W — lower is better
    cei_inv = avg_pwr / avg_hr          # W/bpm — higher is better

    return {
        "avg_power_w" : avg_pwr,
        "avg_hr_bpm"  : avg_hr,
        "cei"         : round(cei, 4),
        "cei_inv"     : round(cei_inv, 3),
        "is_indoor"   : metrics.get("is_indoor", False),
        "context_note": (
            "CEI is only comparable across sessions with similar power target, "
            "time of day, caffeine, hydration, fatigue, and temperature."
        ),
    }


# ── Analysis 2: Indoor / Outdoor Power Delta ──────────────────────────────────

def indoor_outdoor_delta(sessions: list[dict]) -> dict:
    """
    Compares average power, NP, and HR between indoor and outdoor cycling sessions.

    Takes a list of parsed metrics dicts (cycling only).
    Returns mean values per context and the delta.

    This is the quantification of your known indoor/outdoor cardiac efficiency gap.
    """
    indoor  = [s for s in sessions if s.get("is_indoor") and s.get("avg_power_w")]
    outdoor = [s for s in sessions if not s.get("is_indoor")
               and s.get("activity_type") in ("cycling", "road_biking")
               and s.get("avg_power_w")]

    def mean(lst, key):
        vals = [s[key] for s in lst if s.get(key)]
        return round(sum(vals) / len(vals), 1) if vals else None

    def mean_nested(lst, outer, inner):
        vals = [s[outer][inner] for s in lst if s.get(outer, {}).get(inner)]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        "indoor": {
            "n"            : len(indoor),
            "avg_power_w"  : mean(indoor, "avg_power_w"),
            "avg_np_w"     : mean(indoor, "norm_power_w"),
            "avg_hr_bpm"   : mean_nested(indoor, "heart_rate", "avg_bpm"),
            "avg_cei"      : round(
                mean_nested(indoor, "heart_rate", "avg_bpm") /
                mean(indoor, "avg_power_w"), 4
            ) if mean(indoor, "avg_power_w") else None,
        },
        "outdoor": {
            "n"            : len(outdoor),
            "avg_power_w"  : mean(outdoor, "avg_power_w"),
            "avg_np_w"     : mean(outdoor, "norm_power_w"),
            "avg_hr_bpm"   : mean_nested(outdoor, "heart_rate", "avg_bpm"),
            "avg_cei"      : round(
                mean_nested(outdoor, "heart_rate", "avg_bpm") /
                mean(outdoor, "avg_power_w"), 4
            ) if mean(outdoor, "avg_power_w") else None,
        },
        "note": (
            "Interpret delta carefully — indoor/outdoor sessions differ in "
            "intensity, duration, and conditions. Normalise by IF or power band "
            "for a cleaner comparison."
        ),
    }


# ── Analysis 3: Training Load (PMC — ATL / CTL / TSB) ────────────────────────

def training_load_summary(sessions: list[dict]) -> list[dict]:
    """
    Performance Management Chart metrics: ATL, CTL, TSB per session.

    ATL (Acute Training Load)   : 7-day exponential weighted TSS  — fatigue
    CTL (Chronic Training Load) : 42-day exponential weighted TSS — fitness
    TSB (Training Stress Balance): CTL - ATL                      — form

    Input: list of metrics dicts sorted chronologically.
    Only sessions with a 'tss' field contribute (i.e. cycling with power).
    Running TSS (rTSS) stub is included but requires threshold pace — returns
    None until running threshold is configured.

    Returns a list of dicts, one per session, with pmc fields added.
    """
    ATL_DECAY = math.exp(-1 / 7)
    CTL_DECAY = math.exp(-1 / 42)

    # ── Aggregate TSS by calendar day ────────────────────────────────────────
    # Multiple sessions on the same day should be summed before PMC decay,
    # otherwise each session triggers a full day's decay independently.
    from collections import defaultdict, OrderedDict

    # Build ordered dict: date → list of sessions
    days: dict[str, list] = OrderedDict()
    for s in sessions:
        date = s.get("file", "")[:10]   # YYYY-MM-DD from filename
        if date not in days:
            days[date] = []
        days[date].append(s)

    atl = 0.0
    ctl = 0.0
    result = []

    for date, day_sessions in days.items():
        day_tss  = 0.0
        day_has_tss = False

        for s in day_sessions:
            tss  = s.get("tss")
            rtss = running_tss(s)
            stss = tss if tss is not None else rtss
            if stss is not None:
                day_tss += stss
                day_has_tss = True

        if day_has_tss:
            atl = atl * ATL_DECAY + day_tss * (1 - ATL_DECAY)
            ctl = ctl * CTL_DECAY + day_tss * (1 - CTL_DECAY)

        # Emit one PMC row per session, all sharing the day's ATL/CTL
        for s in day_sessions:
            tss  = s.get("tss")
            rtss = running_tss(s)
            stss = tss if tss is not None else rtss
            result.append({
                "file"  : s.get("file"),
                "tss"   : tss,
                "rtss"  : rtss,
                "stss"  : stss,
                "atl"   : round(atl, 1),
                "ctl"   : round(ctl, 1),
                "tsb"   : round(ctl - atl, 1),
            })

    return result


# ── Analysis 4: Running TSS ──────────────────────────────────────────────────

def running_tss(session: dict) -> float | None:
    """
    Calculate rTSS for a running session.

    rTSS = (duration_s × IF²) / 3600 × 100
    IF   = avg_speed_mps / threshold_pace_mps

    Only calculated for outdoor running sessions with valid speed data.
    Returns None if data is insufficient.
    """
    if "run" not in session.get("activity_type", ""):
        return None

    duration_s = session.get("duration_s", 0)
    if duration_s < 300:   # ignore < 5 min
        return None

    # Derive avg speed from pace string or avg_speed_kph
    avg_spd_kph = session.get("avg_speed_kph")
    if not avg_spd_kph or avg_spd_kph <= 0:
        return None

    avg_spd_mps = avg_spd_kph / 3.6
    IF          = avg_spd_mps / THRESHOLD_PACE_MPS
    rtss        = (duration_s * IF * IF) / 3600 * 100

    # Sanity cap — rTSS > 400 in a single session is implausible
    return round(min(rtss, 400), 1)


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json

    sessions = load_all_json()
    if not sessions:
        print("No .json files found in fit_files/. Run garmin_download.py first.")
    else:
        print(f"\nLoaded {len(sessions)} session(s).\n")

        # CEI for each cycling session with power
        print("── Cardiac Efficiency Index ──────────────────────")
        for s in sessions:
            cei = cardiac_efficiency_index(s)
            if cei:
                context = "indoor" if cei["is_indoor"] else "outdoor"
                print(f"  {s['file'][:30]}  {context:8s}  "
                      f"{cei['avg_power_w']:5.0f}W  "
                      f"{cei['avg_hr_bpm']:3.0f}bpm  "
                      f"CEI={cei['cei']:.4f}  ({cei['cei_inv']:.2f} W/bpm)")

        # Indoor/outdoor delta
        print("\n── Indoor / Outdoor Delta ────────────────────────")
        delta = indoor_outdoor_delta(sessions)
        print(_json.dumps(delta, indent=2))

        # PMC
        print("\n── Training Load (PMC) ───────────────────────────")
        pmc = training_load_summary(sessions)
        for row in pmc:
            if row["tss"] is not None:
                print(f"  {row['file'][:35]}  TSS={row['tss']:5.1f}  "
                      f"CTL={row['ctl']:5.1f}  ATL={row['atl']:5.1f}  "
                      f"TSB={row['tsb']:+.1f}")
