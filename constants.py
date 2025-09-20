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

    store = _load_param_store()
    best = store.get("boards", {}).get(str(n), {}).get("best_settings")
    if isinstance(best, dict):
        rec.update({k: best.get(k, v) for k, v in rec.items()})
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
