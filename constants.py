"""
Central defaults and recommendations for the N-Queens GA.

This module provides:
 - BOARD_SIZE: default board size
 - Recommended default parameters derived from BOARD_SIZE
 - recommend_params(n): good starting values for any n
 - Optional persistent store integration (params/param_store.json) so
   recommendations can improve over time (across runs).
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict

# Problem constant: default board size
BOARD_SIZE = 8  # Try 8, 12, 16, 20, 36, ...

# Location for persisted parameter improvements
PARAM_STORE_PATH = Path("params/param_store.json")


def _load_param_store() -> Dict[str, Any]:
    try:
        if PARAM_STORE_PATH.exists():
            with PARAM_STORE_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def recommend_params(n: int) -> dict:
    """Return recommended starting GA parameters for a given board size n.

    Heuristics:
      - population_size ~ 22*n (min 200)
      - generations ~ 80*n (min 1000)
      - elitism in [2..6], scaled by n
      - tournament_k in [3..6], scaled by n
      - crossover_rate ~ 0.8 (<20) else ~0.75
      - mutation_rate ~ 0.15

    Then overlay with any persisted best settings from PARAM_STORE_PATH.
    """
    n = max(4, int(n))
    population_size = max(200, int(22 * n))
    generations = max(1000, int(80 * n))
    elitism = min(6, max(2, n // 6))
    tournament_k = min(6, max(3, n // 5))
    crossover_rate = 0.8 if n < 20 else 0.75
    mutation_rate = 0.15

    rec = {
        "population_size": population_size,
        "generations": generations,
        "elitism": elitism,
        "tournament_k": tournament_k,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
    }

    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    store = _load_param_store()
    boards = store.get("boards", {})
    best = boards.get(str(n), {}).get("best_settings")
    if isinstance(best, dict):
        rec.update({k: best.get(k, v) for k, v in rec.items()})
        return rec

    # Cross-n seeding: use nearest neighbour best settings and scale
    nearest = None
    try:
        candidates = [
            (abs(int(k) - n), int(k), v)
            for k, v in boards.items()
            if isinstance(v, dict) and isinstance(v.get("best_settings"), dict)
        ]
        if candidates:
            _, n0, v0 = min(candidates, key=lambda t: t[0])
            s0 = v0["best_settings"]

            # Scale population and generations roughly linearly with n
            scale = n / max(1, n0)
            pop0 = int(s0.get("population_size", rec["population_size"]))
            gen0 = int(s0.get("generations", rec["generations"]))
            pop = max(50, int(pop0 * scale))
            gens = max(200, int(gen0 * scale))

            # Elitism: scale with n but cap to fraction of population
            elit0 = int(s0.get("elitism", rec["elitism"]))
            elit = max(1, int(elit0 * scale))
            # Cap by MAX_ELITISM_FRAC if available in module scope
            try:
                cap = int(MAX_ELITISM_FRAC * pop)  # type: ignore[name-defined]
            except Exception:
                cap = max(1, pop // 10)
            elit = int(_clamp(elit, 1, max(1, cap)))

            # Tournament k: keep reasonable range and <= population
            tk0 = int(s0.get("tournament_k", rec["tournament_k"]))
            tk = int(_clamp(tk0, 3, 6))
            tk = int(_clamp(tk, 2, pop))

            # Rates: carry over and clamp
            cr = float(_clamp(float(s0.get("crossover_rate", rec["crossover_rate"])), 0.6, 0.95))
            mr = float(_clamp(float(s0.get("mutation_rate", rec["mutation_rate"])), 0.02, 0.5))

            seed = {
                "population_size": pop,
                "generations": gens,
                "elitism": elit,
                "tournament_k": tk,
                "crossover_rate": cr,
                "mutation_rate": mr,
            }
            rec.update(seed)
    except Exception:
        # If anything goes wrong, fall back to baseline rec
        pass

    return rec



# Evolutionary constants (derived from BOARD_SIZE by default)
_defaults = recommend_params(BOARD_SIZE)
POPULATION_SIZE = _defaults["population_size"]
GENERATIONS = _defaults["generations"]
ELITISM = _defaults["elitism"]
TOURNAMENT_K = _defaults["tournament_k"]
CROSSOVER_RATE = _defaults["crossover_rate"]
MUTATION_RATE = _defaults["mutation_rate"]

# Optional wall-clock time limit (in seconds). None = no time limit.
TIME_LIMIT_S = None

# Training summary (if store exists), useful for UI/printing
_store = _load_param_store()
TRAINING_SUMMARY = {
    "total_runs": int(_store.get("total_runs", 0)),
    "boards": {k: v.get("trained_runs", 0) for k, v in _store.get("boards", {}).items()},
    "last_updated": _store.get("last_updated"),
}

# Defaults for repeated running/training
SOLVE_REPEATS = 1000          # Default number of times to solve when repeating
TRAIN_BATCH_RUNS = 34         # Default runs per evaluation batch (statistical check)
TRAIN_KILO_DEFAULT = 0        # Default thousands of runs to train (0 = off)

# Per-board-size overrides for repeats and training defaults.
# Fill in as needed, e.g.:
# RUN_DEFAULTS_PER_BOARD = {
#     "8":  {"repeats": 1000, "train_batch": 34, "train_kilo": 10},
#     "16": {"repeats": 1000, "train_batch": 34, "train_kilo": 20},
#     "18": {"repeats": 2000, "train_batch": 34, "train_kilo": 30},
# }
RUN_DEFAULTS_PER_BOARD: Dict[str, Dict[str, int]] = {}

RUN_DEFAULTS = {
    "repeats": SOLVE_REPEATS,
    "train_batch": TRAIN_BATCH_RUNS,
    "train_kilo": TRAIN_KILO_DEFAULT,
}


def get_run_defaults(n: int) -> Dict[str, int]:
    """Return defaults for repeats/train for given n.

    Priority:
      1) Exact per-board override in RUN_DEFAULTS_PER_BOARD if present
      2) Nearest-neighbour per-board override (closest defined n)
      3) Global RUN_DEFAULTS fallback
    """
    n_int = int(n)
    nkey = str(n_int)
    defaults = RUN_DEFAULTS.copy()

    overrides = RUN_DEFAULTS_PER_BOARD.get(nkey)
    if overrides is None and RUN_DEFAULTS_PER_BOARD:
        # Find closest defined board size and use its policy as a heuristic default
        try:
            candidates = [(abs(int(k) - n_int), int(k)) for k in RUN_DEFAULTS_PER_BOARD.keys()]
            _, nearest = min(candidates, key=lambda t: t[0])
            overrides = RUN_DEFAULTS_PER_BOARD.get(str(nearest))
        except Exception:
            overrides = None

    if overrides:
        defaults.update({k: int(v) for k, v in overrides.items() if k in defaults})
    return defaults

# Optional default for multi-n training input if you don't want to pass CLI.
# IMPORTANT:
# - Leave TRAIN_MULTI_DEFAULT empty ("") to run the app in the normal mode
#   where BOARD_SIZE applies (single n), unless you explicitly pass
#   --train-multi or training flags.
# - If TRAIN_MULTI_DEFAULT is non-empty, the app behaves as if you passed
#   --train-multi with the same value: it will train for those board sizes
#   and then exit; BOARD_SIZE is ignored for that run.
# Accepted formats: "4,6,8,10" or ranges like "4-10,16,18" (trailing commas OK).
# Examples:
#   TRAIN_MULTI_DEFAULT = "4-8,18"   # trains for n=4,5,6,7,8,18
#   TRAIN_MULTI_DEFAULT = ""          # no auto multi-n training; BOARD_SIZE is used
TRAIN_MULTI_DEFAULT: str = "8, 16, 32, 64, 128, 256"

# Pre-filled per-n policies (you can edit these to your needs)
# Small n (4–10)
RUN_DEFAULTS_PER_BOARD.update({
    "4":  {"repeats": 500,  "train_batch": 50, "train_kilo": 5},
    "5":  {"repeats": 600,  "train_batch": 50, "train_kilo": 6},
    "6":  {"repeats": 700,  "train_batch": 50, "train_kilo": 7},
    "7":  {"repeats": 800,  "train_batch": 50, "train_kilo": 8},
    "8":  {"repeats": 1000, "train_batch": 38, "train_kilo": 10},
    "9":  {"repeats": 1000, "train_batch": 50, "train_kilo": 10},
    "10": {"repeats": 1000, "train_batch": 50, "train_kilo": 10},
})

# Medium n (12–18)
RUN_DEFAULTS_PER_BOARD.update({
    "12": {"repeats": 1500, "train_batch": 50, "train_kilo": 4},
    "14": {"repeats": 1700, "train_batch": 50, "train_kilo": 5},
    "15": {"repeats": 1800, "train_batch": 50, "train_kilo": 3},
    "16": {"repeats": 1800, "train_batch": 60, "train_kilo": 4},
    "18": {"repeats": 2000, "train_batch": 60, "train_kilo": 4},
})

# Larger n (20–36)
RUN_DEFAULTS_PER_BOARD.update({
    "20": {"repeats": 2200, "train_batch": 60, "train_kilo": 4},
    "24": {"repeats": 2500, "train_batch": 60, "train_kilo": 4},
    "28": {"repeats": 2800, "train_batch": 65, "train_kilo": 4},
    "32": {"repeats": 3000, "train_batch": 80, "train_kilo": 4},
    "36": {"repeats": 3000, "train_batch": 80, "train_kilo": 4},
    "64": {"repeats": 5000, "train_batch": 80, "train_kilo": 5},
    "128": {"repeats": 5000, "train_batch": 90, "train_kilo": 5},
    "256": {"repeats": 5000, "train_batch": 100, "train_kilo": 5},
})

# Exposed knobs for exploration and adaptation
# Exploration schedule during training
EXPLORE_PROB_START = 0.35
EXPLORE_PROB_MIN = 0.10
EXPLORE_PROB_MAX = 0.50
EXPLORE_PROB_REWARD = 0.05   # increase when an improvement is accepted
EXPLORE_PROB_COOL = 0.01     # decrease when no improvement

# Acceptance thresholds for improvements (relative)
ACCEPT_MEDIAN_IMPROVEMENT = 0.005  # 1% faster median time
ACCEPT_ATTACKS_IMPROVEMENT = 0.015 # 3% fewer average attacks

# In-run adaptation cadence and magnitudes
ADAPT_EVERY_GEN = 40
ADAPT_NEAR_THRESHOLD = 0.02  # remaining attacks fraction considered "near solution"
ADAPT_FAR_THRESHOLD = 0.20   # remaining attacks fraction considered "far from solution"
ADAPT_MUT_STEP_UP = 0.03
ADAPT_MUT_STEP_DOWN = 0.02

# Selection pressure and diversity controls
MAX_ELITISM_FRAC = 0.10     # cap elitism to at most 10% of population for robustness
IMMIGRANT_FRAC = 0.12       # fraction of population replaced by random individuals on stagnation

# Local repair (greedy improvement) controls
REPAIR_BASE_STEPS = 1       # baseline repair steps per child
REPAIR_MAX_FRAC = 1/6       # max repair steps as a fraction of n (e.g., n//6)

# Patience-based early stopping / restart
PATIENCE_MULTIPLIER = 2     # early-stop if no improvement for PATIENCE_MULTIPLIER * ADAPT_EVERY_GEN
SOFT_RESTART_FRAC = 0.4     # on severe stagnation, replace this fraction of worst individuals

# Persistence/checkpointing during training
STORE_CHECKPOINT_BATCHES = 10  # write param_store.json at least every N batches

# Dynamic time limit (optional, per run)
DYN_TIME_ENABLED_DEFAULT = True   # set True to enable without CLI
# Increase default dynamic time window slightly; promising runs can extend
# within this range when dynamic time is enabled.
DYN_TIME_BASE_S = 3.0             # initial time budget (seconds) if no --time-limit provided
DYN_TIME_MIN_S = 0.5              # never drop below this
DYN_TIME_MAX_S = 7.0              # never exceed this
DYN_TIME_EXTEND_S = 0.5          # extend when improvement happens
DYN_TIME_SHRINK_S = 0.0           # shrink on stagnation (0 = disabled)

# Optional per-board overrides for dynamic time windows
from typing import Dict as _Dict
DYN_TIME_MAX_PER_BOARD: _Dict[str, float] = {
    # Per-board max caps (override DYN_TIME_MAX_S). Keep within 3–5s window
    # unless an explicit larger window is desired.
    # Example override (disabled by default):
    # "20": 5.0,
}
DYN_TIME_BASE_PER_BOARD: _Dict[str, float] = {
    # Example: "20": 2.0,
}

def dyn_time_window_for(n: int) -> tuple[float, float]:
    """Return a recommended (base_s, max_s) dynamic time window for board size n.

    Priority order:
      1) Explicit per-board overrides in DYN_TIME_BASE_PER_BOARD / DYN_TIME_MAX_PER_BOARD
      2) Interpolated heuristic based on anchor sizes
      3) Global defaults (DYN_TIME_BASE_S, DYN_TIME_MAX_S)
    """
    try:
        key = str(int(n))
    except Exception:
        key = str(n)

    # 1) Explicit overrides
    if key in DYN_TIME_BASE_PER_BOARD or key in DYN_TIME_MAX_PER_BOARD:
        base = float(DYN_TIME_BASE_PER_BOARD.get(key, DYN_TIME_BASE_S))
        mx = float(DYN_TIME_MAX_PER_BOARD.get(key, DYN_TIME_MAX_S))
        return base, mx

    # 2) Interpolated heuristic between anchors
    # Anchors are chosen so that time increases with n
    anchors: list[tuple[int, tuple[float, float]]] = [
        (8,  (1.0, 3.0)),
        (16, (2.0, 6.0)),
        (24, (3.0, 9.0)),
        (32, (4.0, 12.0)),
        (48, (5.0, 15.0)),
        (64, (6.0, 18.0)),
        (80, (7.0, 21.0)),
        (100, (25.0, 40.0)),
        (260, (355.0, 700.0)),
    ]
    try:
        ni = int(n)
    except Exception:
        ni = 0

    # Below the smallest anchor → clamp to first
    if ni <= anchors[0][0]:
        return anchors[0][1]
    # Above the largest anchor → clamp to last
    if ni >= anchors[-1][0]:
        return anchors[-1][1]
    # Find surrounding anchors and linearly interpolate
    lo_idx = 0
    for i in range(1, len(anchors)):
        if ni <= anchors[i][0]:
            lo_idx = i - 1
            hi_idx = i
            break
    n_lo, (base_lo, max_lo) = anchors[lo_idx]
    n_hi, (base_hi, max_hi) = anchors[hi_idx]
    span = max(1, n_hi - n_lo)
    t = (ni - n_lo) / span
    base = base_lo + t * (base_hi - base_lo)
    mx = max_lo + t * (max_hi - max_lo)
    return float(base), float(mx)

# Ensure-solution behavior for single solves
SOLVE_GUARANTEE_DEFAULT = True  # try extra steps/restarts to reach 0 attacks in normal mode
SOLVE_MAX_RESTARTS = 3          # additional GA restarts if not solved
SOLVE_FINAL_STEPS = 2000        # greedy finish steps (min-conflicts style)

# Systematic probe settings (when plateau persists)
PROBE_PLATEAU_TRIGGER = 10   # batches without acceptance before probing
PROBE_BATCH_RUNS = 20        # quick probe runs per candidate
PROBE_COARSE_SCALE = 0.30    # ±30% for pop/gens probes
PROBE_RATE_STEP = 0.01       # ±0.01 for rates (fine sweep)
CONFIRM_BATCH_MIN = 50       # confirm with at least this many runs (enables ±0.005 fine rate sweeps)


def default_schedule() -> dict:
    """Return default schedule knobs used inside GA runs.

    These are the parameters the trainer can meta-learn in addition to the
    baseline GA settings.
    """
    return {
        "adapt_every": ADAPT_EVERY_GEN,
        "near_thr": ADAPT_NEAR_THRESHOLD,
        "far_thr": ADAPT_FAR_THRESHOLD,
        "mut_step_up": ADAPT_MUT_STEP_UP,
        "mut_step_down": ADAPT_MUT_STEP_DOWN,
        "immigrant_frac": IMMIGRANT_FRAC,
        "repair_base_steps": REPAIR_BASE_STEPS,
        "repair_max_frac": REPAIR_MAX_FRAC,
        "patience_multiplier": PATIENCE_MULTIPLIER,
        "max_elitism_frac": MAX_ELITISM_FRAC,
    }


def recommend_schedule(n: int) -> dict:
    """Return a recommended dynamic schedule for a given board size n.

    If a best_schedule exists for this n in the param store, return it.
    Otherwise, use the nearest neighbour's best_schedule and lightly scale
    adapt_every in proportion to n, with sensible clamps. Fall back to
    default_schedule() if no data exists.
    """
    n = int(n)
    sched = default_schedule()
    try:
        # Load store
        store = _load_param_store()
        boards = store.get("boards", {})
        here = boards.get(str(n), {})
        if isinstance(here.get("best_schedule"), dict):
            s = dict(here["best_schedule"])  # copy
            # Ensure required keys exist
            merged = default_schedule()
            merged.update(s)
            return merged

        # Find nearest neighbour with best_schedule
        candidates = [
            (abs(int(k) - n), int(k), v)
            for k, v in boards.items()
            if isinstance(v, dict) and isinstance(v.get("best_schedule"), dict)
        ]
        if not candidates:
            return sched
        _, n0, v0 = min(candidates, key=lambda t: t[0])
        seed = dict(v0["best_schedule"])  # type: ignore[index]
        merged = default_schedule()
        merged.update(seed)
        # Lightly scale adapt_every with n
        try:
            scale = max(0.5, min(2.0, n / max(1, n0)))
            merged["adapt_every"] = int(max(10, min(200, int(merged["adapt_every"]) * scale)))
        except Exception:
            pass
        return merged
    except Exception:
        return sched

# Aggressive training settings (adaptive exploration on long plateaus)
AGGR_ENABLE = True
AGGR_PLATEAU_L1 = 20        # batches without accept → escalate probes
AGGR_PLATEAU_L2 = 50        # larger escalation
AGGR_PROBE_SCALE = 0.50     # coarse scale for pop/gens when escalated
AGGR_RATE_STEP = 0.01       # larger rate steps when escalated
AGGR_TIE_MARGIN = 0.01      # allow 2% tie-accept margin when escalated (baseline move only)
AGGR_MACRO_RESEED = 80      # try macro baseline reseed after this many plateau batches
AGGR_MACRO_CAND = 3         # number of reseed candidates to try

# Auto-aggression based on total GA runs without improvement (not batches)
AGGR_AUTO_ENABLE = True
AGGR_AUTO_RUNS = 800       # if no accepted improvement in this many runs, escalate aggression
