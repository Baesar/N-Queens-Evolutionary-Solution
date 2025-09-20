import argparse
import json
import os
import time
import random
from statistics import median
from typing import List, Tuple, Optional

from board import Board
from constants import (
    BOARD_SIZE,
    POPULATION_SIZE,
    GENERATIONS,
    ELITISM,
    TOURNAMENT_K,
    CROSSOVER_RATE,
    MUTATION_RATE,
    TIME_LIMIT_S,
    recommend_params,
    PARAM_STORE_PATH,
)
import view


# ---------------------------
# GA configuration structures
# ---------------------------

class GASettings:
    def __init__(
        self,
        population_size: int = POPULATION_SIZE,
        generations: int = GENERATIONS,
        elitism: int = ELITISM,
        tournament_k: int = TOURNAMENT_K,
        crossover_rate: float = CROSSOVER_RATE,
        mutation_rate: float = MUTATION_RATE,
    ) -> None:
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.elitism = int(elitism)
        self.tournament_k = int(tournament_k)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)

    def to_dict(self) -> dict:
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "elitism": self.elitism,
            "tournament_k": self.tournament_k,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GASettings":
        return cls(
            population_size=d.get("population_size", POPULATION_SIZE),
            generations=d.get("generations", GENERATIONS),
            elitism=d.get("elitism", ELITISM),
            tournament_k=d.get("tournament_k", TOURNAMENT_K),
            crossover_rate=d.get("crossover_rate", CROSSOVER_RATE),
            mutation_rate=d.get("mutation_rate", MUTATION_RATE),
        )


# ---------------------------
# GA operators
# ---------------------------

def tournament_select(pop: List[Board], k: int) -> Board:
    cand = random.sample(pop, k)
    best = max(cand, key=lambda b: b.fitness())
    return best.clone()


def one_point_crossover(p1: Board, p2: Board) -> Tuple[Board, Board]:
    n = p1.size
    cut = random.randint(1, n - 1)
    c1_state = p1.state[:cut] + p2.state[cut:]
    c2_state = p2.state[:cut] + p1.state[cut:]
    return Board(n, c1_state), Board(n, c2_state)


def uniform_crossover(p1: Board, p2: Board) -> Tuple[Board, Board]:
    n = p1.size
    c1_state, c2_state = [], []
    for i in range(n):
        if random.random() < 0.5:
            c1_state.append(p1.state[i])
            c2_state.append(p2.state[i])
        else:
            c1_state.append(p2.state[i])
            c2_state.append(p1.state[i])
    return Board(n, c1_state), Board(n, c2_state)


def crossover(p1: Board, p2: Board, crossover_rate: float) -> Tuple[Board, Board]:
    if random.random() > crossover_rate:
        return p1.clone(), p2.clone()
    if random.random() < 0.5:
        return one_point_crossover(p1, p2)
    else:
        return uniform_crossover(p1, p2)


# ---------------------------
# GA main loop
# ---------------------------

def init_population(board_size: int, population_size: int) -> List[Board]:
    return [Board(board_size) for _ in range(population_size)]


class GAResult:
    def __init__(self, best: Board, duration: float, generations: int) -> None:
        self.best = best
        self.duration = duration
        self.generations = generations


def run_ga(
    settings: GASettings,
    board_size: int,
    time_limit_s: Optional[float] = TIME_LIMIT_S,
    verbose: bool = True,
) -> GAResult:
    start = time.perf_counter()
    population = init_population(board_size, settings.population_size)
    best = max(population, key=lambda b: b.fitness()).clone()
    max_fit = best.max_non_attacking_pairs()

    if verbose:
        print(
            f"Initial best fitness: {best.fitness()} / {max_fit} (attacks={best.number_of_attacks})"
        )

    last_generation = 0
    for gen in range(1, settings.generations + 1):
        last_generation = gen
        if time_limit_s is not None and (time.perf_counter() - start) >= time_limit_s:
            if verbose:
                print(f"\nTime limit reached at generation {gen}.")
            break

        population.sort(key=lambda b: b.fitness(), reverse=True)
        if population[0].fitness() > best.fitness():
            best = population[0].clone()

        if best.number_of_attacks == 0:
            if verbose:
                print(
                    f"\nSolved at generation {gen} in {time.perf_counter() - start:.3f}s."
                )
            break

        elites = [population[i].clone() for i in range(min(settings.elitism, len(population)))]

        new_pop: List[Board] = elites[:]
        while len(new_pop) < settings.population_size:
            p1 = tournament_select(population, settings.tournament_k)
            p2 = tournament_select(population, settings.tournament_k)
            c1, c2 = crossover(p1, p2, settings.crossover_rate)
            c1.mutate(settings.mutation_rate)
            if len(new_pop) < settings.population_size:
                new_pop.append(c1)
            c2.mutate(settings.mutation_rate)
            if len(new_pop) < settings.population_size:
                new_pop.append(c2)

        population = new_pop

        if verbose and (gen % 100 == 0 or gen == 1):
            print(
                f"Gen {gen:5d} | best fit = {best.fitness():3d}/{max_fit} | attacks={best.number_of_attacks}"
            )

    duration = time.perf_counter() - start
    if verbose:
        print(
            f"\nFinished in {duration:.3f}s | Best fitness: {best.fitness()}/{max_fit} | attacks={best.number_of_attacks}"
        )
    return GAResult(best=best, duration=duration, generations=last_generation)


# ---------------------------
# Persistent parameter store utilities
# ---------------------------

def _load_store() -> dict:
    try:
        if PARAM_STORE_PATH.exists():
            with PARAM_STORE_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {"total_runs": 0, "boards": {}}


def _save_store(store: dict) -> None:
    PARAM_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with PARAM_STORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)


def evaluate_settings_over_runs(
    settings: GASettings,
    board_size: int,
    runs: int,
    time_limit_s: Optional[float],
) -> dict:
    durations = []
    successes = 0
    total_generations = 0
    total_fitness = 0
    for _ in range(runs):
        res = run_ga(settings, board_size, time_limit_s=time_limit_s, verbose=False)
        durations.append(res.duration)
        total_generations += res.generations
        total_fitness += res.best.fitness()
        if res.best.number_of_attacks == 0:
            successes += 1
    med = float(median(durations)) if durations else float("inf")
    return {
        "median_duration": med,
        "success_rate": successes / runs if runs else 0.0,
        "avg_generations": total_generations / runs if runs else 0.0,
        "avg_fitness": total_fitness / runs if runs else 0.0,
    }


def _perturb(settings: GASettings) -> GASettings:
    s = GASettings.from_dict(settings.to_dict())
    s.population_size = max(50, int(s.population_size + random.randint(-int(0.1 * s.population_size), int(0.1 * s.population_size))))
    s.generations = max(200, int(s.generations + random.randint(-int(0.1 * s.generations), int(0.1 * s.generations))))
    s.elitism = max(1, min(s.elitism + random.randint(-1, 2), s.population_size - 1))
    s.tournament_k = max(2, min(s.tournament_k + random.randint(-1, 2), s.population_size))
    s.crossover_rate = max(0.6, min(0.95, s.crossover_rate + random.uniform(-0.05, 0.05)))
    s.mutation_rate = max(0.02, min(0.5, s.mutation_rate + random.uniform(-0.05, 0.05)))
    return s


def train_parameters(
    board_size: int,
    kilo: int = 1,
    batch_runs: int = 34,
    explore_prob: float = 0.2,
    time_limit_s: Optional[float] = None,
) -> None:
    """Run many batches and keep parameters that improve median duration without
    hurting success rate, persisting improvements between runs.
    """
    total_budget = max(1, kilo) * 1000
    store = _load_store()
    bkey = str(board_size)
    entry = store["boards"].setdefault(bkey, {"trained_runs": 0})
    # Start from persisted best if exists, else from recommended
    if "best_settings" in entry:
        current = GASettings.from_dict(entry["best_settings"])
        current_metrics = entry.get("best_metrics", {})
    else:
        current = GASettings.from_dict(recommend_params(board_size))
        current_metrics = {}

    if not current_metrics:
        current_metrics = evaluate_settings_over_runs(current, board_size, batch_runs, time_limit_s)

    runs_done = 0
    accepted = 0
    while runs_done < total_budget:
        # Occasionally explore a new candidate
        if random.random() < explore_prob:
            candidate = _perturb(current)
        else:
            candidate = current  # re-measure stability

        cand_metrics = evaluate_settings_over_runs(candidate, board_size, batch_runs, time_limit_s)
        runs_done += batch_runs
        store["total_runs"] = store.get("total_runs", 0) + batch_runs
        entry["trained_runs"] = entry.get("trained_runs", 0) + batch_runs

        # Accept if success not worse and median duration improved by >=2%
        improved = cand_metrics["median_duration"] < current_metrics["median_duration"] * 0.98
        not_worse_success = cand_metrics["success_rate"] >= current_metrics.get("success_rate", 0.0)
        accept = improved and not_worse_success
        # With small probability, accept equal performance to keep exploration
        if not accept and random.random() < 0.05 and not_worse_success:
            accept = True

        if accept:
            current = candidate
            current_metrics = cand_metrics
            accepted += 1
            # small reward: briefly increase exploration for next step
            explore_prob = min(0.5, explore_prob + 0.05)
            # persist improvement
            entry["best_settings"] = current.to_dict()
            entry["best_metrics"] = current_metrics
            _save_store(store)
        else:
            # cool exploration a bit
            explore_prob = max(0.1, explore_prob - 0.01)

        print(
            f"Trained {runs_done}/{total_budget} runs | accept={accept} | med={current_metrics['median_duration']:.4f}s "
            f"success={current_metrics['success_rate']:.2f} | accepted={accepted}"
        )

    # Final persist
    entry["best_settings"] = current.to_dict()
    entry["best_metrics"] = current_metrics
    _save_store(store)


# ---------------------------
# Entry point and CLI
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="N-Queens GA with persistent parameter training")
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S if TIME_LIMIT_S is not None else None)
    parser.add_argument("--population", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--elitism", type=int)
    parser.add_argument("--tournament-k", type=int)
    parser.add_argument("--crossover-rate", type=float)
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--train-params", action="store_true", help="Run persistent training batches and save best params")
    parser.add_argument("--train-kilo", type=int, default=0, help="How many thousands of runs to train (0 = skip)")
    parser.add_argument("--train-batch", type=int, default=34, help="Runs per evaluation batch (statistical check)")
    parser.add_argument("--no-run", action="store_true", help="Parse and exit")

    args = parser.parse_args()
    if args.no_run:
        print("Execution disabled (--no-run). Exiting.")
        return

    n = max(4, int(args.board_size))
    base = recommend_params(n)
    settings = GASettings.from_dict(base)

    # Overrides
    for name in ("population", "generations", "elitism", "tournament_k", "crossover_rate", "mutation_rate"):
        val = getattr(args, name if name != "population" else "population")
        if val is not None:
            setattr(settings, name if name != "population" else "population_size", val)

    if args.train_params or args.train_kilo > 0:
        train_parameters(
            board_size=n,
            kilo=max(1, args.train_kilo or 1),
            batch_runs=max(10, args.train_batch),
            time_limit_s=args.time_limit if args.time_limit is not None else None,
        )
        return

    # Solve once with current settings
    result = run_ga(settings=settings, board_size=n, time_limit_s=args.time_limit, verbose=True)
    view.print_board_box(result.best, title="\nBest board:")


if __name__ == "__main__":
    main()
