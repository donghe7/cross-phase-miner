"""
Central configuration for the CrossPhase signal server.

All tunables live here so a demo can be reshaped from the command line or
environment without touching the code.

Threshold semantics (the "70%" requirement)
-------------------------------------------
Two different gates are both set to 0.70 by default, and they do different
jobs.  Keeping them separate means a deployment can loosen one without
touching the other:

1. ``OBS_CONFIDENCE_THRESHOLD`` - per-observation vision gate.  A robot's
   perception model reports a confidence with every frame; the server (and
   the robot's own transition extractor) ignore anything below this value.
   The simulator draws 0.88-0.99 for correct classifications and 0.55-0.75
   for misclassifications, so 0.70 removes roughly 80% of the vision
   errors before they can reach the learners.

2. ``CONSENSUS_THRESHOLD`` - fleet agreement gate.  When several robots
   watch the same crossing at the same time, the server commits a colour
   only if at least 70% of the confidence-weighted votes agree.  Below
   that the crossing is reported as DISPUTED and no robot is allowed to
   use the fleet state as evidence for crossing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# --- thresholds -----------------------------------------------------------

OBS_CONFIDENCE_THRESHOLD: float = _env_float("CP_OBS_THRESHOLD", 0.70)
CONSENSUS_THRESHOLD: float = _env_float("CP_CONSENSUS_THRESHOLD", 0.70)

# A vote older than this many (simulated) seconds no longer counts towards
# the live state of a crossing.
VOTE_WINDOW_SECONDS: float = _env_float("CP_VOTE_WINDOW", 3.0)

# After this long without a single fresh observation, the crossing falls
# back to model prediction (or UNKNOWN if there is no reliable model).
LIVE_STALE_SECONDS: float = _env_float("CP_LIVE_STALE", 4.0)

# De-bouncing for transition extraction: a colour change is only confirmed
# after the new colour has persisted this long.
TRANSITION_MIN_DURATION: float = _env_float("CP_MIN_DURATION", 2.0)

# Baseline crossing rule: consecutive seconds of confirmed green.
CONFIRM_SECONDS: float = _env_float("CP_CONFIRM_SECONDS", 5.0)


# --- scale ----------------------------------------------------------------

# Number of crossings the server manages.
NUM_INTERSECTIONS: int = _env_int("CP_INTERSECTIONS", 1200)

# Crossings inside the fleet's delivery zone (the ones robots actually
# visit).  The rest stay unobserved on purpose: the gap between "managed"
# and "learned" is the point of the coverage panel in the UI.
ZONE_SIZE: int = _env_int("CP_ZONE_SIZE", 48)

# Crossings that every robot route passes through, so that concurrent
# observation - and therefore the consensus gate - actually happens.
NUM_HUBS: int = _env_int("CP_HUBS", 8)

# How many (intersection, TOD period) learners are kept in memory.  Colder
# ones are serialised to SQLite and restored on demand.
LEARNER_CACHE_SIZE: int = _env_int("CP_LEARNER_CACHE", 4096)

# Run the UKF learner alongside the Bayesian one.  Off by default: it costs
# ~3x the CPU per update and the Bayesian learner wins on this data.
ENABLE_UKF: bool = os.environ.get("CP_ENABLE_UKF", "0") == "1"


# --- fleet ----------------------------------------------------------------

NUM_ROBOTS: int = _env_int("CP_ROBOTS", 20)

# Vision detections per simulated second, matching the physical robot's
# 15 FPS detector. Clock speed changes wall time, not this sampling interval.
OBS_RATE_HZ: float = _env_float("CP_OBS_RATE", 15.0)

# Keep live uploads comfortably inside the 3-second voting window.
# This interval is simulated time; HTTP request rate grows with clock speed.
UPLOAD_BATCH_SECONDS: float = _env_float("CP_BATCH_SECONDS", 1.0)

MISCLASS_PROB: float = _env_float("CP_MISCLASS", 0.03)


# --- ingest limits ----------------------------------------------------------
#
# The server trusts the fleet (no auth yet - see docs/architecture.md), but
# a single misbehaving or buggy client still should not be able to grow
# server memory or stall the event loop without bound.  These caps are sized
# generously above any real fleet traffic:
#
# * a real robot at 15 FPS falling behind for the entire scout timeout
#   (``SCOUT_TIMEOUT_SECONDS`` in server/fleet.py, 450s) would still only
#   catch up ~6750 frames in one batch;
# * a real arrival reports 2-4 phase transitions, never dozens.

# Longest a robot_id / intersection_id / destination_id string may be.
MAX_ID_LENGTH: int = _env_int("CP_MAX_ID_LENGTH", 128)

# Largest number of vision frames accepted in one observation batch.
MAX_OBSERVATIONS_PER_BATCH: int = _env_int("CP_MAX_OBS_BATCH", 10_000)

# Largest number of phase transitions accepted in one arrival report.
MAX_TRANSITIONS_PER_ARRIVAL: int = _env_int("CP_MAX_TRANSITIONS", 128)

# Simulated travel time between two crossings.
TRAVEL_SECONDS_RANGE = (
    _env_float("CP_TRAVEL_MIN", 90.0),
    _env_float("CP_TRAVEL_MAX", 300.0),
)


# --- clock ----------------------------------------------------------------

# Wall-clock seconds are multiplied by this factor to get simulated
# seconds, so a 150s signal cycle completes in 150/SPEED real seconds.
CLOCK_SPEED: float = _env_float("CP_SPEED", 5.0)

# Simulated start of the demo day: KST 06:00.
START_HOUR_KST: float = _env_float("CP_START_HOUR", 6.0)


def default_sim_start() -> float:
    """Unix timestamp of ``START_HOUR_KST`` on the demo day (KST)."""
    midnight = datetime(2024, 7, 19, 15, 0, 0, tzinfo=timezone.utc).timestamp()
    return midnight + START_HOUR_KST * 3600.0


# --- persistence ----------------------------------------------------------

DB_PATH: str = os.environ.get("DATABASE_URL") or os.environ.get("CP_DB", "server_state.sqlite3")

# Keep the demo reproducible.
SEED: int = _env_int("CP_SEED", 2024)

MQTT_ENABLED: bool = os.environ.get("CP_MQTT", "0") == "1"
MQTT_HOST: str = os.environ.get("CP_MQTT_HOST", "127.0.0.1")
MQTT_PORT: int = _env_int("CP_MQTT_PORT", 1883)
MQTT_PREFIX: str = os.environ.get("CP_MQTT_PREFIX", "crossphase/v1").strip("/")


@dataclass
class ServerConfig:
    """Snapshot of the active configuration, exposed via ``GET /v1/config``."""

    obs_confidence_threshold: float = OBS_CONFIDENCE_THRESHOLD
    consensus_threshold: float = CONSENSUS_THRESHOLD
    vote_window_seconds: float = VOTE_WINDOW_SECONDS
    live_stale_seconds: float = LIVE_STALE_SECONDS
    transition_min_duration: float = TRANSITION_MIN_DURATION
    confirm_seconds: float = CONFIRM_SECONDS
    num_intersections: int = NUM_INTERSECTIONS
    zone_size: int = ZONE_SIZE
    num_hubs: int = NUM_HUBS
    learner_cache_size: int = LEARNER_CACHE_SIZE
    enable_ukf: bool = ENABLE_UKF
    num_robots: int = NUM_ROBOTS
    obs_rate_hz: float = OBS_RATE_HZ
    upload_batch_seconds: float = UPLOAD_BATCH_SECONDS
    clock_speed: float = CLOCK_SPEED
    seed: int = SEED
    tod_periods: list = field(default_factory=lambda: ["day", "night"])
