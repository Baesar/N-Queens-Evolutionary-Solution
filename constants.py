"""
Central defaults and recommendations for the N-Queens GA.

This module provides:
 - BOARD_SIZE: default board size
 - Recommended default parameters derived from BOARD_SIZE
 - A helper function recommend_params(n) to get good starting values for any n

The app uses dynamic parameters at runtime (via CLI and the GAP tuner), but
these values are still useful as sensible starting points.
"""

# Problem constant: default board size
BOARD_SIZE = 18  # Try 8, 12, 16, 20, 36, ...


def recommend_params(n: int) -> dict:
    """Return recommended starting GA parameters for a given board size n.

    Heuristics:
      - population_size ~ 22*n (min 200)
      - generations ~ 80*n (min 1000)
      - elitism in [2..6], scaled by n
      - tournament_k in [3..6], scaled by n
      - crossover_rate ~ 0.8 (<20) else ~0.75
      - mutation_rate ~ 0.15
    """
    n = max(4, int(n))
    population_size = max(200, int(22 * n))
    generations = max(1000, int(80 * n))
    elitism = min(6, max(2, n // 6))
    tournament_k = min(6, max(3, n // 5))
    crossover_rate = 0.8 if n < 20 else 0.75
    mutation_rate = 0.15
    return {
        "population_size": population_size,
        "generations": generations,
        "elitism": elitism,
        "tournament_k": tournament_k,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
    }


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
