import argparse
import json
import csv
import os
import concurrent.futures as cf
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
    SOLVE_REPEATS,
    TRAIN_BATCH_RUNS,
    TRAIN_KILO_DEFAULT,
    get_run_defaults,
    TRAIN_MULTI_DEFAULT,
    EXPLORE_PROB_START,
    EXPLORE_PROB_MIN,
    EXPLORE_PROB_MAX,
    EXPLORE_PROB_REWARD,
    EXPLORE_PROB_COOL,
    ACCEPT_MEDIAN_IMPROVEMENT,
    ACCEPT_ATTACKS_IMPROVEMENT,
    ADAPT_EVERY_GEN,
    ADAPT_NEAR_THRESHOLD,
    ADAPT_FAR_THRESHOLD,
    ADAPT_MUT_STEP_UP,
    ADAPT_MUT_STEP_DOWN,
    MAX_ELITISM_FRAC,
    IMMIGRANT_FRAC,
    REPAIR_BASE_STEPS,
    REPAIR_MAX_FRAC,
    PATIENCE_MULTIPLIER,
    SOFT_RESTART_FRAC,
    STORE_CHECKPOINT_BATCHES,
    DYN_TIME_ENABLED_DEFAULT,
    DYN_TIME_BASE_S,
    DYN_TIME_MIN_S,
    DYN_TIME_MAX_S,
    DYN_TIME_EXTEND_S,
    DYN_TIME_SHRINK_S,
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


def local_improve(board: Board, steps: int) -> None:
    """Greedy repair: for a few random columns, move queen to row that
    minimizes conflicts. Modifies board in-place."""
    if steps <= 0:
        return
    n = board.size
    for _ in range(steps):
        col = random.randint(0, n - 1)
        current_row = board.state[col]
        best_row = current_row
        best_attacks = None
        for r in range(n):
            if r == current_row:
                continue
            old = board.state[col]
            board.state[col] = r
            board.calculate_number_of_attacks()
            att = board.number_of_attacks
            if best_attacks is None or att < best_attacks:
                best_attacks = att
                best_row = r
            # revert for next candidate
            board.state[col] = old
            board.calculate_number_of_attacks()
        board.state[col] = best_row
        board.calculate_number_of_attacks()


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
    dynamic_time: bool = False,
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
    no_improve = 0
    adapt_every = ADAPT_EVERY_GEN
    # Operator weights for simple bandit-style crossover selection
    op_w_one, op_w_uni = 0.5, 0.5
    bandit_alpha = 0.1

    patience_limit = max(ADAPT_EVERY_GEN, ADAPT_EVERY_GEN * PATIENCE_MULTIPLIER)

    # Dynamic time window handling
    if dynamic_time:
        base_cap = time_limit_s if time_limit_s is not None else DYN_TIME_BASE_S
        deadline = start + base_cap
        min_deadline = start + DYN_TIME_MIN_S
        max_deadline = start + DYN_TIME_MAX_S

    for gen in range(1, settings.generations + 1):
        last_generation = gen
        now = time.perf_counter()
        if dynamic_time and now >= deadline:
            if verbose:
                print(f"\nDynamic time window reached at generation {gen}.")
            break
        if not dynamic_time and time_limit_s is not None and (now - start) >= time_limit_s:
            if verbose:
                print(f"\nTime limit reached at generation {gen}.")
            break

        population.sort(key=lambda b: b.fitness(), reverse=True)
        if population[0].fitness() > best.fitness():
            best = population[0].clone()
            no_improve = 0
            # Extend dynamic deadline a bit on improvement
            if dynamic_time:
                deadline = min(deadline + DYN_TIME_EXTEND_S, max_deadline)
        else:
            no_improve += 1

        if best.number_of_attacks == 0:
            if verbose:
                print(
                    f"\nSolved at generation {gen} in {time.perf_counter() - start:.3f}s."
                )
            break

        # Cap elitism to a fraction of population for robustness
        elit_cap = max(0, int(MAX_ELITISM_FRAC * settings.population_size))
        elit_count = min(settings.elitism, elit_cap, len(population))
        elites = [population[i].clone() for i in range(elit_count)]

        new_pop: List[Board] = elites[:]
        while len(new_pop) < settings.population_size:
            p1 = tournament_select(population, settings.tournament_k)
            p2 = tournament_select(population, settings.tournament_k)
            # Choose crossover operator by current weights
            if random.random() > settings.crossover_rate:
                c1, c2 = p1.clone(), p2.clone()
                used_op = None
            else:
                r = random.random() * (op_w_one + op_w_uni)
                if r < op_w_one:
                    c1, c2 = one_point_crossover(p1, p2)
                    used_op = 'one'
                else:
                    c1, c2 = uniform_crossover(p1, p2)
                    used_op = 'uni'
            c1.mutate(settings.mutation_rate)
            # Adaptive repair based on how far we are from solution
            max_pairs = best.max_non_attacking_pairs()
            remaining_ratio = best.number_of_attacks / max(1, max_pairs)
            max_repair = max(0, int(REPAIR_MAX_FRAC * board_size))
            repair_steps = min(max_repair, REPAIR_BASE_STEPS + int(remaining_ratio * max_repair))
            local_improve(c1, steps=repair_steps)
            if len(new_pop) < settings.population_size:
                new_pop.append(c1)
            c2.mutate(settings.mutation_rate)
            local_improve(c2, steps=repair_steps)
            if len(new_pop) < settings.population_size:
                new_pop.append(c2)

            # Bandit reward: if children beat parents, slightly increase the used operator's weight
            if used_op is not None:
                parent_best = max(p1.fitness(), p2.fitness())
                child_best = max(c1.fitness(), c2.fitness())
                if child_best >= parent_best:
                    if used_op == 'one':
                        op_w_one = (1 - bandit_alpha) * op_w_one + bandit_alpha * (op_w_one + op_w_uni)
                    else:
                        op_w_uni = (1 - bandit_alpha) * op_w_uni + bandit_alpha * (op_w_one + op_w_uni)
                else:
                    # small decay to avoid stagnation
                    if used_op == 'one':
                        op_w_one *= (1 - bandit_alpha * 0.5)
                    else:
                        op_w_uni *= (1 - bandit_alpha * 0.5)

        # Inject random immigrants into the worst fraction on stagnation
        if IMMIGRANT_FRAC > 0 and no_improve >= max(1, ADAPT_EVERY_GEN // 2):
            k = max(1, int(IMMIGRANT_FRAC * settings.population_size))
            new_pop.sort(key=lambda b: b.fitness())  # ascending: worst first
            for i in range(min(k, len(new_pop))):
                new_pop[i] = Board(board_size)

        population = new_pop

        # Simple dynamic hyperparameter adaptation based on progress
        if (gen % adapt_every == 0) or (no_improve >= adapt_every):
            # Estimate difficulty by remaining attacks ratio
            max_pairs = best.max_non_attacking_pairs()
            remaining_ratio = best.number_of_attacks / max(1, max_pairs)
            # Adjust mutation: more if far from solution, less if close
            if remaining_ratio > ADAPT_FAR_THRESHOLD:
                settings.mutation_rate = min(0.5, settings.mutation_rate + ADAPT_MUT_STEP_UP)
                settings.tournament_k = max(2, settings.tournament_k - 1)
            elif remaining_ratio < ADAPT_NEAR_THRESHOLD:
                settings.mutation_rate = max(0.02, settings.mutation_rate - ADAPT_MUT_STEP_DOWN)
                settings.tournament_k = min(max(3, settings.tournament_k + 1), settings.population_size)
            # Optionally shrink dynamic deadline a bit on stagnation
            if dynamic_time and DYN_TIME_SHRINK_S > 0 and no_improve >= adapt_every:
                deadline = max(deadline - DYN_TIME_SHRINK_S, min_deadline)
            # Reset stagnation counter after adapting
            no_improve = 0

        # Early stop on severe stagnation
        if no_improve >= patience_limit:
            if verbose:
                print(f"\nEarly stopping due to stagnation at generation {gen}.")
            # Optionally do a soft restart rather than full stop
            worst_replace = int(SOFT_RESTART_FRAC * settings.population_size)
            if worst_replace > 0:
                population.sort(key=lambda b: b.fitness(), reverse=True)
                keep = population[: max(0, settings.population_size - worst_replace)]
                add = [Board(board_size) for _ in range(worst_replace)]
                population = keep + add
            break

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


def _worker_run_once(payload: dict) -> dict:
    s = GASettings.from_dict(payload["settings"])  # reconstruct settings in worker
    res = run_ga(
        s,
        payload["board_size"],
        time_limit_s=payload["time_limit"],
        verbose=False,
        dynamic_time=payload.get("dynamic_time", False),
    )
    best = res.best
    return {
        "duration": res.duration,
        "generations": res.generations,
        "fitness": best.fitness(),
        "success": best.number_of_attacks == 0,
}


def _worker_run_many(payload: dict) -> dict:
    """Run GA multiple times in one worker to reduce process startup overhead."""
    s = GASettings.from_dict(payload["settings"])  # reconstruct settings in worker
    board_size = payload["board_size"]
    time_limit = payload["time_limit"]
    dynamic_time = payload.get("dynamic_time", False)
    repeat = int(payload.get("repeat", 1))
    durations = []
    successes = 0
    total_generations = 0
    total_fitness = 0
    for _ in range(max(1, repeat)):
        res = run_ga(
            s,
            board_size,
            time_limit_s=time_limit,
            verbose=False,
            dynamic_time=dynamic_time,
        )
        durations.append(res.duration)
        total_generations += res.generations
        best = res.best
        total_fitness += best.fitness()
        if best.number_of_attacks == 0:
            successes += 1
    return {
        "durations": durations,
        "successes": successes,
        "total_generations": total_generations,
        "total_fitness": total_fitness,
        "count": max(1, repeat),
    }


def evaluate_settings_over_runs(
    settings: GASettings,
    board_size: int,
    runs: int,
    time_limit_s: Optional[float],
    workers: Optional[int] = None,
    dynamic_time: bool = False,
) -> dict:
    durations = []
    successes = 0
    total_generations = 0
    total_fitness = 0
    runs = max(1, int(runs))
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(int(workers), runs))

    if workers == 1:
        for _ in range(runs):
            res = run_ga(settings, board_size, time_limit_s=time_limit_s, verbose=False, dynamic_time=dynamic_time)
            durations.append(res.duration)
            total_generations += res.generations
            total_fitness += res.best.fitness()
            if res.best.number_of_attacks == 0:
                successes += 1
    else:
        base = runs // workers
        extra = runs % workers
        payload = {
            "settings": settings.to_dict(),
            "board_size": board_size,
            "time_limit": time_limit_s,
            "dynamic_time": dynamic_time,
        }
        try:
            with cf.ProcessPoolExecutor(max_workers=workers) as ex:
                futures = []
                for i in range(workers):
                    repeat_i = base + (1 if i < extra else 0)
                    if repeat_i <= 0:
                        continue
                    p = dict(payload)
                    p["repeat"] = repeat_i
                    futures.append(ex.submit(_worker_run_many, p))
                for fut in cf.as_completed(futures):
                    try:
                        out = fut.result()
                        durations.extend(out.get("durations", []))
                        total_generations += out.get("total_generations", 0)
                        total_fitness += out.get("total_fitness", 0)
                        successes += out.get("successes", 0)
                    except Exception:
                        durations.append(float("inf"))
        except Exception:
            # Fallback to sequential if the environment blocks multiprocessing
            for _ in range(runs):
                res = run_ga(settings, board_size, time_limit_s=time_limit_s, verbose=False, dynamic_time=dynamic_time)
                durations.append(res.duration)
                total_generations += res.generations
                total_fitness += res.best.fitness()
                if res.best.number_of_attacks == 0:
                    successes += 1
    med = float(median(durations)) if durations else float("inf")
    max_pairs = board_size * (board_size - 1) // 2
    avg_attacks = max(0.0, (max_pairs * runs - total_fitness) / runs) if runs else float("inf")
    return {
        "median_duration": med,
        "success_rate": successes / runs if runs else 0.0,
        "avg_generations": total_generations / runs if runs else 0.0,
        "avg_fitness": total_fitness / runs if runs else 0.0,
        "avg_attacks": avg_attacks,
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
    kilo: int = 0.1,
    batch_runs: int = 34,
    explore_prob: float = EXPLORE_PROB_START,
    time_limit_s: Optional[float] = None,
    workers: Optional[int] = None,
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
        current_metrics = evaluate_settings_over_runs(current, board_size, batch_runs, time_limit_s, workers=workers)

    runs_done = 0
    # Prepare CSV training log per board size
    log_path = PARAM_STORE_PATH.parent / f"training_log_n{board_size}.csv"
    PARAM_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "timestamp", "runs_done", "accept", "median_s", "best_median_s", "success_rate", "avg_attacks",
                "pop", "gens", "elitism", "tourn_k", "cross", "mut"
            ])
    accepted = 0
    try:
        while runs_done < total_budget:
            # Occasionally explore a new candidate
            if random.random() < explore_prob:
                candidate = _perturb(current)
            else:
                candidate = current  # re-measure stability

            cand_metrics = evaluate_settings_over_runs(candidate, board_size, batch_runs, time_limit_s, workers=workers)
            runs_done += batch_runs
            store["total_runs"] = store.get("total_runs", 0) + batch_runs
            entry["trained_runs"] = entry.get("trained_runs", 0) + batch_runs

            # Accept if success not worse and median duration improved sufficiently
            improved = cand_metrics["median_duration"] < current_metrics["median_duration"] * (1 - ACCEPT_MEDIAN_IMPROVEMENT)
            not_worse_success = cand_metrics["success_rate"] >= current_metrics.get("success_rate", 0.0)
            # Also consider average attacks: accept if attacks reduced sufficiently and success not worse
            attacks_improved = cand_metrics.get("avg_attacks", 1e9) < current_metrics.get("avg_attacks", 1e9) * (1 - ACCEPT_ATTACKS_IMPROVEMENT)
            accept = (improved and not_worse_success) or (attacks_improved and not_worse_success)
            # With small probability, accept equal performance to keep exploration
            if not accept and random.random() < 0.05 and not_worse_success:
                accept = True

            if accept:
                current = candidate
                current_metrics = cand_metrics
                accepted += 1
                # small reward: briefly increase exploration for next step
                explore_prob = min(EXPLORE_PROB_MAX, explore_prob + EXPLORE_PROB_REWARD)
                # persist improvement only if it beats best-so-far
                best_metrics_prior = entry.get("best_metrics", {})
                improved_vs_best = (
                    cand_metrics["median_duration"] < best_metrics_prior.get("median_duration", float("inf"))
                    or cand_metrics.get("avg_attacks", 1e9) < best_metrics_prior.get("avg_attacks", 1e9)
                ) and not_worse_success
                if improved_vs_best:
                    entry["best_settings"] = current.to_dict()
                    entry["best_metrics"] = cand_metrics
                    _save_store(store)
            else:
                # cool exploration a bit
                explore_prob = max(EXPLORE_PROB_MIN, explore_prob - EXPLORE_PROB_COOL)

            best_med = (entry.get("best_metrics") or {}).get("median_duration", current_metrics["median_duration"])
            print(
                f"Trained {runs_done}/{total_budget} runs | accept={accept} | med={current_metrics['median_duration']:.4f}s "
                f"best_med={best_med:.4f}s | success={current_metrics['success_rate']:.2f} | "
                f"avg_attacks={current_metrics.get('avg_attacks', float('nan')):.2f} | accepted={accepted}"
            )
            # Append to CSV log
            try:
                with log_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        runs_done,
                        int(bool(accept)),
                        f"{current_metrics['median_duration']:.6f}",
                        f"{best_med:.6f}",
                        f"{current_metrics['success_rate']:.4f}",
                        f"{current_metrics.get('avg_attacks', float('nan')):.3f}",
                        current.population_size,
                        current.generations,
                        current.elitism,
                        current.tournament_k,
                        f"{current.crossover_rate:.3f}",
                        f"{current.mutation_rate:.3f}",
                    ])
            except Exception:
                pass

            # Periodic checkpoint even without acceptance
            batches_done = runs_done // batch_runs if batch_runs else 0
            if STORE_CHECKPOINT_BATCHES and batches_done % max(1, int(STORE_CHECKPOINT_BATCHES)) == 0:
                _save_store(store)
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving current best and progress...")
    finally:
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
    parser.add_argument("--train-kilo", type=int, default=None, help="How many thousands of runs to train (0 = skip)")
    parser.add_argument("--train-batch", type=int, default=None, help="Runs per evaluation batch (statistical check)")
    parser.add_argument("--repeats", type=int, default=None, help="Repeat solve N times and summarize metrics")
    parser.add_argument("--train-multi", type=str, default=None, help="Comma-separated or range list of board sizes to train (e.g. '4-10,16,18')")
    parser.add_argument("--no-run", action="store_true", help="Parse and exit")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers (CPU cores) to use")
    parser.add_argument("--dynamic-time", action="store_true", help="Enable dynamic time window (extends on improvements)")

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

    # Determine per-n run defaults (used for training). For normal single solve,
    # the default is to run once and print the board unless --repeats is provided.
    policy = get_run_defaults(n)
    repeats_solve = int(args.repeats) if args.repeats is not None else 1
    train_batch = int(args.train_batch) if args.train_batch is not None else int(policy["train_batch"])
    train_kilo = int(args.train_kilo) if args.train_kilo is not None else int(policy["train_kilo"])

    # Helper to parse multi-n strings like "4-10,16,18"
    def _parse_sizes(spec: str) -> list[int]:
        out: list[int] = []
        for part in (spec or "").split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    lo = int(a.strip()); hi = int(b.strip())
                except ValueError:
                    continue
                if hi < lo:
                    lo, hi = hi, lo
                out.extend(range(lo, hi + 1))
            else:
                try:
                    out.append(int(part))
                except ValueError:
                    pass
        # unique + sorted
        return sorted(set([n for n in out if n >= 4]))

    multi_spec = args.train_multi if args.train_multi is not None else TRAIN_MULTI_DEFAULT
    if multi_spec:
        boards = _parse_sizes(multi_spec)
        if not boards:
            print("No valid board sizes parsed for --train-multi.")
            return
        for bn in boards:
            pol = get_run_defaults(bn)
            print(f"Training n={bn} | kilo={pol['train_kilo']} | batch={pol['train_batch']}")
            train_parameters(
                board_size=bn,
                kilo=max(1, int(pol['train_kilo'] or 1)),
                batch_runs=max(10, int(pol['train_batch'] or 34)),
                time_limit_s=args.time_limit if args.time_limit is not None else None,
                workers=max(1, int(args.workers or 1)),
            )
        return

    # Only start training when explicitly requested via CLI flags
    training_requested = bool(
        args.train_params or (args.train_kilo is not None and args.train_kilo > 0)
    )
    if training_requested:
        # Resolve effective training defaults now that training is requested
        train_kilo_eff = int(args.train_kilo) if args.train_kilo is not None else int(policy["train_kilo"])
        train_batch_eff = int(args.train_batch) if args.train_batch is not None else int(policy["train_batch"])
        print(f"Training n={n} | kilo={train_kilo_eff} | batch={train_batch_eff}")
        train_parameters(
            board_size=n,
            kilo=max(1, train_kilo_eff or 1),
            batch_runs=max(10, train_batch_eff),
            time_limit_s=args.time_limit if args.time_limit is not None else None,
            workers=max(1, int(args.workers or 1)),
        )
        return

    # Solve once or multiple times and summarize
    if repeats_solve and repeats_solve > 1:
        metrics = evaluate_settings_over_runs(
            settings, n, repeats_solve, args.time_limit,
            workers=max(1, int(args.workers or 1)),
            dynamic_time=bool(args.dynamic_time or DYN_TIME_ENABLED_DEFAULT),
        )
        print(
            f"Repeated {repeats_solve} runs | median={metrics['median_duration']:.4f}s "
            f"success={metrics['success_rate']:.2f} | avg_gen={metrics['avg_generations']:.2f} | avg_fit={metrics['avg_fitness']:.2f} | avg_attacks={metrics.get('avg_attacks', float('nan')):.2f}"
        )
        # Auto-persist if this configuration beats stored best for this n
        try:
            store = _load_store()
            entry = store.setdefault("boards", {}).setdefault(str(n), {"trained_runs": 0})
            current_best = entry.get("best_metrics") or {}
            # Improvement if median time is lower OR avg_attacks lower with non-worse success
            better_time = metrics["median_duration"] < (current_best.get("median_duration", float("inf")))
            better_attacks = metrics.get("avg_attacks", float("inf")) < current_best.get("avg_attacks", float("inf"))
            not_worse_success = metrics["success_rate"] >= current_best.get("success_rate", 0.0)
            if better_time or (better_attacks and not_worse_success):
                entry["best_settings"] = settings.to_dict()
                entry["best_metrics"] = metrics
                entry["trained_runs"] = entry.get("trained_runs", 0) + repeats_solve
                store["total_runs"] = store.get("total_runs", 0) + repeats_solve
                _save_store(store)
                print("Saved improved parameters to params/param_store.json for n=", n)
        except Exception as e:
            print("Warning: could not persist improved parameters:", e)
    else:
        result = run_ga(
            settings=settings,
            board_size=n,
            time_limit_s=args.time_limit,
            verbose=True,
            dynamic_time=bool(args.dynamic_time or DYN_TIME_ENABLED_DEFAULT),
        )
        view.print_board_box(result.best, title="\nBest board:")


if __name__ == "__main__":
    main()
