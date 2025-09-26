import argparse

import json
import csv
import os
import concurrent.futures as cf
import multiprocessing as mp
import time
import random
from statistics import median
from typing import List, Tuple, Optional
from pathlib import Path
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
    DYN_TIME_MAX_PER_BOARD,
    DYN_TIME_BASE_PER_BOARD,
    dyn_time_window_for,
    SOLVE_GUARANTEE_DEFAULT,
    SOLVE_MAX_RESTARTS,
    SOLVE_FINAL_STEPS,
    PROBE_PLATEAU_TRIGGER,
    PROBE_BATCH_RUNS,
    PROBE_COARSE_SCALE,
    PROBE_RATE_STEP,
    CONFIRM_BATCH_MIN,
    default_schedule,
    recommend_schedule,
    AGGR_ENABLE,
    AGGR_PLATEAU_L1,
    AGGR_PLATEAU_L2,
    AGGR_PROBE_SCALE,
    AGGR_RATE_STEP,
    AGGR_TIE_MARGIN,
    AGGR_MACRO_RESEED,
    AGGR_MACRO_CAND,
    AGGR_AUTO_ENABLE,
    AGGR_AUTO_RUNS,
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


def _conflict_columns(board: Board) -> List[int]:
    n = board.size
    cols = []
    for i in range(n):
        qi = board.state[i]
        conflict = False
        for j in range(n):
            if i == j:
                continue
            qj = board.state[j]
            if qj == qi or qj == qi + (j - i) or qj == qi - (j - i):
                conflict = True
                break
        if conflict:
            cols.append(i)
    return cols


def greedy_finish(board: Board, max_iters: int) -> bool:
    """Try to reach 0 attacks via min-conflicts style local search."""
    for _ in range(max_iters):
        if board.number_of_attacks == 0:
            return True
        cols = _conflict_columns(board)
        if not cols:
            board.calculate_number_of_attacks()
            return board.number_of_attacks == 0
        col = random.choice(cols)
        current_row = board.state[col]
        best_row = current_row
        best_att = board.number_of_attacks
        n = board.size
        for r in range(n):
            if r == current_row:
                continue
            old = board.state[col]
            board.state[col] = r
            board.calculate_number_of_attacks()
            att = board.number_of_attacks
            if att < best_att:
                best_att = att
                best_row = r
            board.state[col] = old
            board.calculate_number_of_attacks()
        # Move to best found (or random tweak if no improvement)
        if best_row != current_row:
            board.state[col] = best_row
            board.calculate_number_of_attacks()
        else:
            # random nudge to escape plateau
            board.state[col] = random.randint(0, n - 1)
            board.calculate_number_of_attacks()
    return board.number_of_attacks == 0


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
    trace_path: Optional[str] = None,
    phase_summary: Optional[dict] = None,
    schedule: Optional[dict] = None,
) -> GAResult:
    start = time.perf_counter()
    population = init_population(board_size, settings.population_size)
    best = max(population, key=lambda b: b.fitness()).clone()
    max_fit = best.max_non_attacking_pairs()
    best_generation = 0

    if verbose:
        print(

            f"Initial best fitness: {best.fitness()} / {max_fit} (attacks={best.number_of_attacks})"
        )

    last_generation = 0
    no_improve = 0
    # Resolve schedule knobs (use overrides if provided)
    sched = default_schedule()
    if schedule:
        sched.update(schedule)
    adapt_every = int(sched.get("adapt_every", ADAPT_EVERY_GEN))
    near_thr = float(sched.get("near_thr", ADAPT_NEAR_THRESHOLD))
    far_thr = float(sched.get("far_thr", ADAPT_FAR_THRESHOLD))
    mut_up = float(sched.get("mut_step_up", ADAPT_MUT_STEP_UP))
    mut_dn = float(sched.get("mut_step_down", ADAPT_MUT_STEP_DOWN))
    immigr_frac = float(sched.get("immigrant_frac", IMMIGRANT_FRAC))
    repair_base = int(sched.get("repair_base_steps", REPAIR_BASE_STEPS))
    repair_max_frac = float(sched.get("repair_max_frac", REPAIR_MAX_FRAC))
    patience_mult = float(sched.get("patience_multiplier", PATIENCE_MULTIPLIER))
    max_elit_frac = float(sched.get("max_elitism_frac", MAX_ELITISM_FRAC))
    # Operator weights for simple bandit-style crossover selection
    op_w_one, op_w_uni = 0.5, 0.5
    bandit_alpha = 0.1

    patience_limit = max(adapt_every, int(adapt_every * patience_mult))

    # Dynamic time window handling
    if dynamic_time:
        # Resolve dynamic window using per-n heuristic (with overrides)
        base_default, max_default = dyn_time_window_for(board_size)
        base_cap = time_limit_s if time_limit_s is not None else base_default
        deadline = start + base_cap
        min_deadline = start + DYN_TIME_MIN_S
        max_deadline = start + max_default

    # Optional tracing setup
    trace_file = None
    if trace_path:
        tp = Path(trace_path)
        tp.parent.mkdir(parents=True, exist_ok=True)
        trace_file = open(tp, "w", encoding="utf-8")
        trace_file.write(
            "\t".join([
                "gen", "best_attacks", "best_fitness", "mutation_rate", "tournament_k",
                "repair_steps", "op_w_one", "op_w_uni", "immigrants", "remaining_ratio", "dyn_deadline_s",
            ]) + "\n"
        )

    # Phase summary accumulators
    phase_acc = {
        "far": {"mut": 0.0, "rep": 0.0, "cnt": 0},
        "mid": {"mut": 0.0, "rep": 0.0, "cnt": 0},
        "near": {"mut": 0.0, "rep": 0.0, "cnt": 0},
    }
    last_repair_steps = 0
    last_remaining_ratio = 0.0

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
        elit_cap = max(0, int(max_elit_frac * settings.population_size))
        elit_count = min(settings.elitism, elit_cap, len(population))
        elites = [population[i].clone() for i in range(elit_count)]

        new_pop: List[Board] = elites[:]
        immigrants_injected = False
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
            max_repair = max(0, int(repair_max_frac * board_size))
            repair_steps = min(max_repair, repair_base + int(remaining_ratio * max_repair))
            local_improve(c1, steps=repair_steps)
            last_repair_steps = repair_steps
            last_remaining_ratio = remaining_ratio
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
        if immigr_frac > 0 and no_improve >= max(1, adapt_every // 2):
            k = max(1, int(immigr_frac * settings.population_size))
            new_pop.sort(key=lambda b: b.fitness())  # ascending: worst first
            for i in range(min(k, len(new_pop))):
                new_pop[i] = Board(board_size)
            immigrants_injected = True

        population = new_pop


        # Simple dynamic hyperparameter adaptation based on progress
        if (gen % adapt_every == 0) or (no_improve >= adapt_every):
            # Estimate difficulty by remaining attacks ratio
            max_pairs = best.max_non_attacking_pairs()
            remaining_ratio = best.number_of_attacks / max(1, max_pairs)
            # Adjust mutation: more if far from solution, less if close
            if remaining_ratio > far_thr:
                settings.mutation_rate = min(0.5, settings.mutation_rate + mut_up)
                settings.tournament_k = max(2, settings.tournament_k - 1)
            elif remaining_ratio < near_thr:
                settings.mutation_rate = max(0.02, settings.mutation_rate - mut_dn)
                settings.tournament_k = min(max(3, settings.tournament_k + 1), settings.population_size)
            # Optionally shrink dynamic deadline a bit on stagnation
            if dynamic_time and DYN_TIME_SHRINK_S > 0 and no_improve >= adapt_every:
                deadline = max(deadline - DYN_TIME_SHRINK_S, min_deadline)
            # Reset stagnation counter after adapting
            no_improve = 0

        # Phase accumulation based on last observed remaining_ratio/repair_steps
        phase = "mid"
        if last_remaining_ratio > ADAPT_FAR_THRESHOLD:
            phase = "far"
        elif last_remaining_ratio < ADAPT_NEAR_THRESHOLD:
            phase = "near"
        phase_acc[phase]["mut"] += settings.mutation_rate
        phase_acc[phase]["rep"] += last_repair_steps
        phase_acc[phase]["cnt"] += 1

        # Per-generation trace output
        if trace_file is not None:
            dyn_left = (deadline - now) if dynamic_time else float("nan")
            trace_file.write(
                "\t".join(
                    [
                        str(gen),
                        str(best.number_of_attacks),
                        str(best.fitness()),
                        f"{settings.mutation_rate:.6f}",
                        str(settings.tournament_k),
                        str(last_repair_steps),
                        f"{op_w_one:.6f}",
                        f"{op_w_uni:.6f}",
                        str(int(immigrants_injected)),
                        f"{last_remaining_ratio:.6f}",
                        f"{dyn_left:.6f}",
                    ]
                )
                + "\n"
            )

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

    # Close trace and emit phase summary if requested
    try:
        if trace_file is not None:
            trace_file.close()
    except Exception:
        pass
    if phase_summary is not None:
        for key in ("far", "mid", "near"):
            cnt = phase_acc[key]["cnt"] or 1
            phase_summary[f"{key}_mut_avg"] = phase_acc[key]["mut"] / cnt
            phase_summary[f"{key}_rep_avg"] = phase_acc[key]["rep"] / cnt

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
        schedule=payload.get("schedule"),
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
    schedule = payload.get("schedule")
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
            schedule=schedule,
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


_POOL_FALLBACK_WARNED = False


def evaluate_settings_over_runs(
    settings: GASettings,
    board_size: int,
    runs: int,
    time_limit_s: Optional[float],
    workers: Optional[int] = None,
    dynamic_time: bool = False,
    schedule: Optional[dict] = None,
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
            res = run_ga(settings, board_size, time_limit_s=time_limit_s, verbose=False, dynamic_time=dynamic_time, schedule=schedule)
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
            "schedule": schedule,
        }
        try:
            # Use a short-lived pool per call (will be replaced by persistent one in training)
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
            global _POOL_FALLBACK_WARNED
            if not _POOL_FALLBACK_WARNED:
                print("[warn] Multiprocessing unavailable; falling back to sequential evaluation.")
                _POOL_FALLBACK_WARNED = True
            # Fallback to sequential if the environment blocks multiprocessing
            for _ in range(runs):
                res = run_ga(settings, board_size, time_limit_s=time_limit_s, verbose=False, dynamic_time=dynamic_time, schedule=schedule)
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


def _eval_runs_wrapper(
    settings: GASettings,
    board_size: int,
    runs: int,
    time_limit_s: Optional[float],
    workers: Optional[int],
    dynamic_time: bool,
    schedule: Optional[dict],
    executor: Optional[cf.ProcessPoolExecutor],
) -> dict:
    """Compatibility wrapper that falls back if evaluate_settings_over_runs doesn't
    accept newer keyword arguments in certain environments."""
    try:
        return evaluate_settings_over_runs(
            settings,
            board_size,
            runs,
            time_limit_s,
            workers=workers,
            dynamic_time=dynamic_time,
            schedule=schedule,
            executor=executor,
        )
    except TypeError:
        # Older signature without executor/schedule: retry with minimal args
        return evaluate_settings_over_runs(
            settings,
            board_size,
            runs,
            time_limit_s,
            workers=workers,
            dynamic_time=dynamic_time,
        )


def _perturb(settings: GASettings) -> GASettings:
    s = GASettings.from_dict(settings.to_dict())
    s.population_size = max(50, int(s.population_size + random.randint(-int(0.1 * s.population_size), int(0.1 * s.population_size))))
    s.generations = max(200, int(s.generations + random.randint(-int(0.1 * s.generations), int(0.1 * s.generations))))
    s.elitism = max(1, min(s.elitism + random.randint(-1, 2), s.population_size - 1))
    s.tournament_k = max(2, min(s.tournament_k + random.randint(-1, 2), s.population_size))
    s.crossover_rate = max(0.6, min(0.95, s.crossover_rate + random.uniform(-0.05, 0.05)))
    s.mutation_rate = max(0.02, min(0.5, s.mutation_rate + random.uniform(-0.05, 0.05)))
    return s


def _perturb_schedule(sched: dict) -> dict:
    s = dict(sched)
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))
    # Coarse random tweaks
    if random.random() < 0.5:
        s["adapt_every"] = int(clamp(int(s["adapt_every"]) + random.randint(-10, 10), 10, 200))
    if random.random() < 0.5:
        s["mut_step_up"] = clamp(float(s["mut_step_up"]) + random.uniform(-0.01, 0.01), 0.0, 0.2)
    if random.random() < 0.5:
        s["mut_step_down"] = clamp(float(s["mut_step_down"]) + random.uniform(-0.01, 0.01), 0.0, 0.2)
    if random.random() < 0.5:
        s["near_thr"] = clamp(float(s["near_thr"]) + random.uniform(-0.01, 0.01), 0.0, 0.2)
    if random.random() < 0.5:
        s["far_thr"] = clamp(float(s["far_thr"]) + random.uniform(-0.02, 0.02), 0.05, 0.5)
    if random.random() < 0.5:
        s["immigrant_frac"] = clamp(float(s["immigrant_frac"]) + random.uniform(-0.02, 0.02), 0.0, 0.5)
    if random.random() < 0.5:
        s["repair_base_steps"] = int(clamp(int(s["repair_base_steps"]) + random.choice([-1, 0, 1]), 0, 3))
    if random.random() < 0.5:
        s["repair_max_frac"] = clamp(float(s["repair_max_frac"]) + random.uniform(-0.05, 0.05), 0.0, 1.0)
    if random.random() < 0.5:
        s["patience_multiplier"] = clamp(float(s["patience_multiplier"]) + random.uniform(-0.5, 0.5), 1.0, 5.0)
    if random.random() < 0.5:
        s["max_elitism_frac"] = clamp(float(s["max_elitism_frac"]) + random.uniform(-0.02, 0.02), 0.02, 0.3)
    return s

def _systematic_candidates(
    base: GASettings,
    fine_rates: bool = False,
    coarse_scale: Optional[float] = None,
    rate_step_override: Optional[float] = None,
) -> List[tuple[GASettings, str]]:
    """Generate coarse candidates by systematic parameter changes with labels."""
    cands: List[tuple[GASettings, str]] = []
    def clamp(val, lo, hi):
        return max(lo, min(hi, val))

    cs = coarse_scale if coarse_scale is not None else PROBE_COARSE_SCALE
    # Population ±coarse scale
    for scale, tag in ((1 + cs, "+pop"), (1 - cs, "-pop")):
        s = GASettings.from_dict(base.to_dict())
        s.population_size = max(50, int(s.population_size * scale))
        s.elitism = clamp(int(s.elitism), 1, max(1, int(MAX_ELITISM_FRAC * s.population_size)))
        s.tournament_k = clamp(int(s.tournament_k), 2, s.population_size)
        cands.append((s, tag))
    # Generations ±coarse scale
    for scale, tag in ((1 + cs, "+gens"), (1 - cs, "-gens")):
        s = GASettings.from_dict(base.to_dict())
        s.generations = max(200, int(s.generations * scale))
        cands.append((s, tag))
    # Mutation ±step (and optional half-step for fine sweeps)
    step0 = rate_step_override if rate_step_override is not None else PROBE_RATE_STEP
    rate_steps = [step0]
    if fine_rates:
        rate_steps.append(PROBE_RATE_STEP / 2)
    for step in rate_steps:
        for delta, tag in ((step, "+mut"), (-step, "-mut")):
            s = GASettings.from_dict(base.to_dict())
            s.mutation_rate = clamp(s.mutation_rate + delta, 0.02, 0.5)
            cands.append((s, tag))
    # Crossover ±step (and optional half-step for fine sweeps)
    for step in rate_steps:
        for delta, tag in ((step, "+cross"), (-step, "-cross")):
            s = GASettings.from_dict(base.to_dict())
            s.crossover_rate = clamp(s.crossover_rate + delta, 0.6, 0.95)
            cands.append((s, tag))
    # Selection tweaks: elitism ±2, tournament ±1
    for delt_e, delt_t, tag in ((+2, -1, "+elit/-tk"), (-2, +1, "-elit/+tk")):
        s = GASettings.from_dict(base.to_dict())
        s.elitism = clamp(s.elitism + delt_e, 1, max(1, int(MAX_ELITISM_FRAC * s.population_size)))
        s.tournament_k = clamp(s.tournament_k + delt_t, 2, s.population_size)
        cands.append((s, tag))
    return cands


def _midpoint_candidates(current: GASettings, prev1: dict, prev2: dict) -> List[tuple[GASettings, str]]:
    """Generate midpoint exploration candidates based on the last two accepted
    baselines (prev1 is most-recent, prev2 is before that).

    For each of crossover_rate, mutation_rate (rates), and population_size,
    generations (counts): compute step = |prev1 - prev2| / 2. If the step is
    too small, fall back to a sensible minimum (rates use PROBE_RATE_STEP/2,
    counts use 5% of current). Then propose current ± step for that parameter.
    """
    cands: List[tuple[GASettings, str]] = []
    cur = current.to_dict()

    def clamp(val, lo, hi):
        return max(lo, min(hi, val))

    # Rates
    for key, lo, hi, min_step in (
        ("crossover_rate", 0.6, 0.95, PROBE_RATE_STEP / 2),
        ("mutation_rate", 0.02, 0.5, PROBE_RATE_STEP / 2),
    ):
        try:
            d = abs(float(prev1.get(key, cur[key])) - float(prev2.get(key, cur[key]))) / 2.0
        except Exception:
            d = 0.0
        d = max(d, min_step)
        for sign, tag in ((+1, f"mid+{key}"), (-1, f"mid-{key}")):
            s = GASettings.from_dict(cur)
            v = clamp(float(cur[key]) + sign * d, lo, hi)
            setattr(s, key, v)
            cands.append((s, tag))

    # Counts
    for key, lo, hi in (
        ("population_size", 50, max(50, int(cur["population_size"]) * 10)),
        ("generations", 200, max(200, int(cur["generations"]) * 10)),
    ):
        try:
            d = abs(int(prev1.get(key, cur[key])) - int(prev2.get(key, cur[key]))) // 2
        except Exception:
            d = 0
        min_frac = max(1, int(int(cur[key]) * 0.05))
        d = max(d, min_frac)
        for sign, tag in ((+1, f"mid+{key}"), (-1, f"mid-{key}")):
            s = GASettings.from_dict(cur)
            v = clamp(int(cur[key]) + sign * d, lo, hi)
            setattr(s, key, int(v))
            cands.append((s, tag))

    return cands


def train_parameters(
    board_size: int,
    kilo: int = 1,
    batch_runs: int = 34,
    explore_prob: float = EXPLORE_PROB_START,
    time_limit_s: Optional[float] = True,
    workers: Optional[int] = True,
    dynamic_time: bool = True,
    trace_every: int = 0,
    mp_start: Optional[str] = True,
) -> True:
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
        current_sched = entry.get("best_schedule") or recommend_schedule(board_size)
        current_metrics = entry.get("best_metrics", {})
    else:
        current = GASettings.from_dict(recommend_params(board_size))
        current_sched = recommend_schedule(board_size)
        current_metrics = {}

    runs_done = 0
    # Prepare CSV training log per board size
    log_path = PARAM_STORE_PATH.parent / f"training_log_n{board_size}.csv"
    PARAM_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "timestamp", "runs_done", "accept", "accepted_reason", "is_probe", "probe_kind", "plateau_batches",
                "median_s", "best_median_s", "success_rate", "avg_attacks", "explore_prob",
                "pop", "gens", "elitism", "tourn_k", "cross", "mut",
                "mid_dx_cross", "mid_dx_mut", "mid_dx_pop", "mid_dx_gens",
            ])
    # Prepare TSV alongside CSV for human-friendly viewing
    tsv_path = PARAM_STORE_PATH.parent / f"training_log_n{board_size}.tsv"
    if not tsv_path.exists():
        with tsv_path.open("w", encoding="utf-8") as fh:
            fh.write(
                "\t".join([
                    "timestamp", "runs_done", "accept", "accepted_reason", "is_probe", "probe_kind", "plateau_batches",
                    "median_s", "best_median_s", "success_rate", "avg_attacks", "explore_prob",
                    "pop", "gens", "elitism", "tourn_k", "cross", "mut",
                    "mid_dx_cross", "mid_dx_mut", "mid_dx_pop", "mid_dx_gens",
                ]) + "\n"
            )
    # Prepare fixed-width text-column table for human reading
    # (renamed from .pretty.txt to .text_column.txt)
    pretty_path = PARAM_STORE_PATH.parent / f"training_log_n{board_size}.text_column.txt"
    pretty_headers = [
        "timestamp", "runs_done", "accept", "accepted_reason", "is_probe", "probe_kind", "plateau",
        "median_s", "best_median_s", "success", "avg_attacks", "explore",
        "pop", "gens", "elitism", "tourn_k", "cross", "mut",
        "mid_dx_cross", "mid_dx_mut", "mid_dx_pop", "mid_dx_gens",
    ]
    # Fixed widths (ensure >= header length)
    pretty_widths = {
        "timestamp": 19,
        "runs_done": 7,
        "accept": 1,
        "accepted_reason": 7,
        "is_probe": 1,
        "probe_kind": 9,
        "plateau": 6,
        "median_s": 9,
        "best_median_s": 12,
        "success": 7,
        "avg_attacks": 11,
        "explore": 8,
        "pop": 6,
        "gens": 7,
        "elitism": 7,
        "tourn_k": 7,
        "cross": 5,
        "mut": 5,
        "mid_dx_cross": 12,
        "mid_dx_mut": 12,
        "mid_dx_pop": 12,
        "mid_dx_gens": 12,
    }
    if not pretty_path.exists():
        with pretty_path.open("w", encoding="utf-8") as fh:
            # Header line
            line = " ".join([f"{h:<{max(len(h), pretty_widths[h])}}" for h in pretty_headers])
            fh.write(line + "\n")
            # Separator
            fh.write(" ".join(["-" * max(len(h), pretty_widths[h]) for h in pretty_headers]) + "\n")
    accepted = 0
    plateau_batches = 0
    last_accept_runs = 0
    auto_aggr_printed = False
    accepted_history: List[dict] = []
    executor = None
    try:
        # Persistent process pool for this training session
        if workers and workers > 1:
            try:
                ctx = mp.get_context(mp_start or 'fork')
            except Exception:
                ctx = None
            try:
                if ctx is not None:
                    executor = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
                    print(f"[pool] Process pool active with {workers} workers (start={ctx.get_start_method()})")
                else:
                    executor = cf.ProcessPoolExecutor(max_workers=workers)
                    print(f"[pool] Process pool active with {workers} workers")
            except Exception as e:
                executor = None
                print(f"[pool] Unable to start process pool; using sequential evaluation ({e})")
        while runs_done < total_budget:
            if not current_metrics:
                current_metrics = _eval_runs_wrapper(
                    current,
                    board_size,
                    batch_runs,
                    time_limit_s,
                    workers,
                    dynamic_time,
                    current_sched,
                    executor,
                )
            # Occasionally explore a new candidate
            if random.random() < explore_prob:
                candidate = _perturb(current)
                candidate_sched = _perturb_schedule(current_sched)
            else:
                candidate = current  # re-measure stability
                candidate_sched = dict(current_sched)

            cand_metrics = _eval_runs_wrapper(
                candidate,
                board_size,
                batch_runs,
                time_limit_s,
                workers,
                dynamic_time,
                candidate_sched,
                executor,
            )
            runs_done += batch_runs
            store["total_runs"] = store.get("total_runs", 0) + batch_runs
            entry["trained_runs"] = entry.get("trained_runs", 0) + batch_runs
            runs_since_accept = max(0, runs_done - last_accept_runs)

            # Accept if success not worse and median duration improved sufficiently
            improved = cand_metrics["median_duration"] < current_metrics["median_duration"] * (1 - ACCEPT_MEDIAN_IMPROVEMENT)
            not_worse_success = cand_metrics["success_rate"] >= current_metrics.get("success_rate", 0.0)
            # Also consider average attacks: accept if attacks reduced sufficiently and success not worse
            attacks_improved = cand_metrics.get("avg_attacks", 1e9) < current_metrics.get("avg_attacks", 1e9) * (1 - ACCEPT_ATTACKS_IMPROVEMENT)
            accept = (improved and not_worse_success) or (attacks_improved and not_worse_success)
            # With small probability, accept near-ties only (avoid clear regressions)
            if not accept and random.random() < 0.05 and not_worse_success:
                # Escalate tie margin on longer plateaus (more aggressive exploration)
                tie_base = 0.01
                if AGGR_ENABLE and plateau_batches >= AGGR_PLATEAU_L1:
                    tie_base = max(tie_base, AGGR_TIE_MARGIN)
                tie_margin = max(1e-9, current_metrics["median_duration"] * tie_base)
                median_tie = abs(cand_metrics["median_duration"] - current_metrics["median_duration"]) <= tie_margin
                attacks_not_worse = cand_metrics.get("avg_attacks", 1e9) <= current_metrics.get("avg_attacks", 1e9)
                if median_tie and attacks_not_worse:
                    accept = True

            # Classify reason for acceptance (for logs)
            accepted_reason = ""
            if accept:
                if improved and not_worse_success:
                    accepted_reason = "median"
                elif attacks_improved and not_worse_success:
                    accepted_reason = "attacks"
                else:
                    accepted_reason = "tie"

            if accept:
                current = candidate
                current_sched = candidate_sched
                current_metrics = cand_metrics
                accepted += 1
                plateau_batches = 0
                last_accept_runs = runs_done
                try:
                    accepted_history.append({"runs": runs_done, "settings": current.to_dict()})
                    if len(accepted_history) > 10:
                        accepted_history.pop(0)
                except Exception:
                    pass
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
                    entry["best_schedule"] = dict(current_sched)
                    entry["best_metrics"] = cand_metrics
                    _save_store(store)
            else:
                # cool exploration a bit
                explore_prob = max(EXPLORE_PROB_MIN, explore_prob - EXPLORE_PROB_COOL)
                plateau_batches += 1

            # Systematic probe when plateau persists
            is_probe = 0
            probe_kind = ""
            # Adaptive probe trigger: earlier if success is low
            probe_trigger = PROBE_PLATEAU_TRIGGER
            try:
                if current_metrics.get("success_rate", 1.0) < 0.95:
                    probe_trigger = max(3, PROBE_PLATEAU_TRIGGER // 2)
            except Exception:
                pass
            # Auto-aggression by total runs without acceptance
            auto_aggr = bool(AGGR_AUTO_ENABLE and runs_since_accept >= AGGR_AUTO_RUNS)
            if auto_aggr and not auto_aggr_printed:
                print(f"[aggr] Auto-aggression enabled after {runs_since_accept} runs without acceptance")
                auto_aggr_printed = True
            if plateau_batches >= probe_trigger or auto_aggr:
                best_probe = None
                best_probe_metrics = None
                # Evaluate coarse candidates quickly
                fine_rates = bool(max(batch_runs, CONFIRM_BATCH_MIN) >= 50)
                # Escalate probe step sizes on long plateaus
                coarse = PROBE_COARSE_SCALE
                rate_step = PROBE_RATE_STEP
                if AGGR_ENABLE:
                    if auto_aggr or plateau_batches >= AGGR_PLATEAU_L2:
                        coarse = max(coarse, AGGR_PROBE_SCALE)
                        rate_step = max(rate_step, AGGR_RATE_STEP)
                    elif plateau_batches >= AGGR_PLATEAU_L1:
                        coarse = max(coarse, (PROBE_COARSE_SCALE + AGGR_PROBE_SCALE) / 2)
                        rate_step = max(rate_step, (PROBE_RATE_STEP + AGGR_RATE_STEP) / 2)
                # Midpoint exploration using last two accepted baselines
                midpoint_list: List[tuple[GASettings, str]] = []
                if len(accepted_history) >= 2:
                    prev1 = accepted_history[-1]["settings"]
                    prev2 = accepted_history[-2]["settings"]
                    midpoint_list = _midpoint_candidates(current, prev1, prev2)
                for cand, tag in (_systematic_candidates(current, fine_rates=fine_rates, coarse_scale=coarse, rate_step_override=rate_step) + midpoint_list):
                    # For systematic probes, do not change schedule initially
                    m = _eval_runs_wrapper(cand, board_size, PROBE_BATCH_RUNS, time_limit_s, workers, dynamic_time, current_sched, executor)
                    # Prefer fewer attacks, then faster median
                    better = False
                    if m.get("avg_attacks", 1e9) < current_metrics.get("avg_attacks", 1e9):
                        better = True
                    elif m["median_duration"] < current_metrics["median_duration"] * (1 - ACCEPT_MEDIAN_IMPROVEMENT):
                        better = True
                    if better and (best_probe_metrics is None or m["median_duration"] < best_probe_metrics["median_duration"]):
                        best_probe = (cand, tag)
                        best_probe_metrics = m
                if best_probe is not None:
                    # Confirm with full batch before acceptance
                    cand, tag = best_probe
                    confirm_runs = max(batch_runs, CONFIRM_BATCH_MIN)
                    cm = _eval_runs_wrapper(cand, board_size, confirm_runs, time_limit_s, workers, dynamic_time, current_sched, executor)
                    improved_c = cm["median_duration"] < current_metrics["median_duration"] * (1 - ACCEPT_MEDIAN_IMPROVEMENT)
                    attacks_c = cm.get("avg_attacks", 1e9) < current_metrics.get("avg_attacks", 1e9)
                    not_worse_s = cm["success_rate"] >= current_metrics.get("success_rate", 0.0)
                    if (improved_c or attacks_c) and not_worse_s:
                        current = cand
                        current_metrics = cm
                        accepted += 1
                        plateau_batches = 0
                        last_accept_runs = runs_done
                        is_probe = 1
                        probe_kind = tag
                        # Persist only if beats best-so-far
                        best_metrics_prior = entry.get("best_metrics", {})
                        improved_vs_best = (
                            cm["median_duration"] < best_metrics_prior.get("median_duration", float("inf"))
                            or cm.get("avg_attacks", 1e9) < best_metrics_prior.get("avg_attacks", 1e9)
                        ) and not_worse_s
                        if improved_vs_best:
                            entry["best_settings"] = current.to_dict()
                            entry["best_schedule"] = dict(current_sched)
                            entry["best_metrics"] = cm
                            _save_store(store)
                # Macro reseed of baseline if plateau extremely long
                if AGGR_ENABLE and (plateau_batches >= AGGR_MACRO_RESEED or auto_aggr):
                    macro_best = None
                    macro_m = None
                    # Try recommend_params baseline and a couple of perturbations
                    seeds = [GASettings.from_dict(recommend_params(board_size))]
                    for _ in range(max(1, AGGR_MACRO_CAND - 1)):
                        seeds.append(_perturb(seeds[0]))
                    for seed in seeds:
                        m2 = _eval_runs_wrapper(seed, board_size, PROBE_BATCH_RUNS, time_limit_s, workers, dynamic_time, current_sched, executor)
                        if macro_m is None or m2["median_duration"] < macro_m["median_duration"]:
                            macro_best, macro_m = seed, m2
                    if macro_best is not None:
                        current = macro_best
                        current_metrics = macro_m
                        plateau_batches = 0
                        last_accept_runs = runs_done
                        print("[aggr] Macro reseed of baseline after long plateau")

            best_med = (entry.get("best_metrics") or {}).get("median_duration", current_metrics["median_duration"])
            # Compute midpoint deltas for logging (if at least two acceptances this session)
            mid_dx_cross = mid_dx_mut = mid_dx_pop = mid_dx_gens = "-"
            try:
                if len(accepted_history) >= 2:
                    prev1 = accepted_history[-1]["settings"]
                    prev2 = accepted_history[-2]["settings"]
                    # Rates
                    step_cross = abs(float(prev1.get("crossover_rate", current.crossover_rate)) - float(prev2.get("crossover_rate", current.crossover_rate))) / 2.0
                    step_mut = abs(float(prev1.get("mutation_rate", current.mutation_rate)) - float(prev2.get("mutation_rate", current.mutation_rate))) / 2.0
                    step_cross = max(step_cross, PROBE_RATE_STEP / 2)
                    step_mut = max(step_mut, PROBE_RATE_STEP / 2)
                    mid_dx_cross = f"{step_cross:.6f}"
                    mid_dx_mut = f"{step_mut:.6f}"
                    # Counts
                    step_pop = abs(int(prev1.get("population_size", current.population_size)) - int(prev2.get("population_size", current.population_size))) // 2
                    step_gens = abs(int(prev1.get("generations", current.generations)) - int(prev2.get("generations", current.generations))) // 2
                    min_pop = max(1, int(current.population_size * 0.05))
                    min_gens = max(1, int(current.generations * 0.05))
                    step_pop = max(step_pop, min_pop)
                    step_gens = max(step_gens, min_gens)
                    mid_dx_pop = str(step_pop)
                    mid_dx_gens = str(step_gens)
            except Exception:
                pass
            print(
                f"Trained {runs_done}/{total_budget} runs | accept={accept} ({accepted_reason or '-'}) | med={current_metrics['median_duration']:.4f}s "
                f"best_med={best_med:.4f}s | success={current_metrics['success_rate']:.2f} | "
                f"avg_attacks={current_metrics.get('avg_attacks', float('nan')):.2f} | plateau={plateau_batches} | accepted={accepted} | "
                f"probe={is_probe}:{probe_kind or '-'} | mid(cross={mid_dx_cross}, mut={mid_dx_mut}, pop={mid_dx_pop}, gens={mid_dx_gens})"
            )
            # Append to CSV log
            try:
                with log_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        runs_done,
                        int(bool(accept)),
                        accepted_reason,
                        is_probe,
                        probe_kind or "-",
                        plateau_batches,
                        f"{current_metrics['median_duration']:.6f}",
                        f"{best_med:.6f}",
                        f"{current_metrics['success_rate']:.4f}",
                        f"{current_metrics.get('avg_attacks', float('nan')):.3f}",
                        f"{explore_prob:.4f}",
                        current.population_size,
                        current.generations,
                        current.elitism,
                        current.tournament_k,
                        f"{current.crossover_rate:.3f}",
                        f"{current.mutation_rate:.3f}",
                        mid_dx_cross,
                        mid_dx_mut,
                        mid_dx_pop,
                        mid_dx_gens,
                    ])
                # Mirror to TSV for readability
                with tsv_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        "\t".join([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            str(runs_done),
                            str(int(bool(accept))),
                            accepted_reason or "-",
                            str(is_probe),
                            probe_kind or "-",
                            str(plateau_batches),
                            f"{current_metrics['median_duration']:.6f}",
                            f"{best_med:.6f}",
                            f"{current_metrics['success_rate']:.4f}",
                            f"{current_metrics.get('avg_attacks', float('nan')):.3f}",
                            f"{explore_prob:.4f}",
                            str(current.population_size),
                            str(current.generations),
                            str(current.elitism),
                            str(current.tournament_k),
                            f"{current.crossover_rate:.3f}",
                            f"{current.mutation_rate:.3f}",
                            mid_dx_cross,
                            mid_dx_mut,
                            mid_dx_pop,
                            mid_dx_gens,
                        ]) + "\n"
                    )
                # Mirror to pretty fixed-width table
                try:
                    data_map = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "runs_done": str(runs_done),
                        "accept": str(int(bool(accept))),
                        "accepted_reason": accepted_reason or "-",
                        "is_probe": str(is_probe),
                        "probe_kind": probe_kind or "-",
                        "plateau": str(plateau_batches),
                        "median_s": f"{current_metrics['median_duration']:.6f}",
                        "best_median_s": f"{best_med:.6f}",
                        "success": f"{current_metrics['success_rate']:.4f}",
                        "avg_attacks": f"{current_metrics.get('avg_attacks', float('nan')):.3f}",
                        "explore": f"{explore_prob:.4f}",
                        "pop": str(current.population_size),
                        "gens": str(current.generations),
                        "elitism": str(current.elitism),
                        "tourn_k": str(current.tournament_k),
                        "cross": f"{current.crossover_rate:.3f}",
                        "mut": f"{current.mutation_rate:.3f}",
                        "mid_dx_cross": mid_dx_cross,
                        "mid_dx_mut": mid_dx_mut,
                        "mid_dx_pop": mid_dx_pop,
                        "mid_dx_gens": mid_dx_gens,
                    }
                    row = " ".join(
                        [f"{data_map[h]:<{max(len(h), pretty_widths[h])}}" for h in pretty_headers]
                    )
                    with pretty_path.open("a", encoding="utf-8") as pfh:
                        pfh.write(row + "\n")
                except Exception:
                    pass
            except Exception:
                pass

            # Periodic checkpoint even without acceptance
            batches_done = runs_done // batch_runs if batch_runs else 0
            if STORE_CHECKPOINT_BATCHES and batches_done % max(1, int(STORE_CHECKPOINT_BATCHES)) == 0:
                _save_store(store)
            # Optional per-generation trace every N batches
            if trace_every and batches_done > 0 and batches_done % max(1, int(trace_every)) == 0:
                trace_dir = PARAM_STORE_PATH.parent / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d-%H%M%S")
                trace_file = trace_dir / f"n{board_size}_batch{batches_done}_{ts}.trace.tsv"
                _ = run_ga(
                    current,
                    board_size,
                    time_limit_s=time_limit_s,
                    verbose=False,
                    dynamic_time=dynamic_time,
                    trace_path=str(trace_file),
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving current best and progress...")
    finally:
        try:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
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
    parser.add_argument("--trace-run", action="store_true", help="Write per-generation trace for a single solve")
    parser.add_argument("--trace-every", type=int, default=0, help="During training, write a per-generation trace every N batches (0=off)")
    parser.add_argument(
        "--mp-start",
        choices=["fork", "spawn", "forkserver"],
        default=None,
        help="Multiprocessing start method for worker pool (default: fork on Unix, spawn on macOS sandbox)",
    )

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
    dyn_flag = bool(args.dynamic_time or DYN_TIME_ENABLED_DEFAULT)
    # One-liner banner for visibility
    # Compute dynamic time caps for banner
    base_default, max_default = dyn_time_window_for(n)
    base_cap = args.time_limit if args.time_limit is not None else base_default
    if training_requested:
        # Resolve effective training defaults now that training is requested
        train_kilo_eff = int(args.train_kilo) if args.train_kilo is not None else int(policy["train_kilo"])
        train_batch_eff = int(args.train_batch) if args.train_batch is not None else int(policy["train_batch"])
        sched_disp = recommend_schedule(n)
        workers_eff = max(1, int(args.workers or 1))
        print(
            f"Config | n={n} | dyn_time={'on' if dyn_flag else 'off'} base={base_cap}s max={max_default}s | "
            f"GA pop={settings.population_size} gens={settings.generations} elit={settings.elitism} tk={settings.tournament_k} "
            f"cr={settings.crossover_rate:.2f} mr={settings.mutation_rate:.2f} | "
            f"sched adapt={sched_disp['adapt_every']} near={sched_disp['near_thr']:.3f} far={sched_disp['far_thr']:.3f} "
            f"mut±=({sched_disp['mut_step_up']:.3f}/{sched_disp['mut_step_down']:.3f}) imm={sched_disp['immigrant_frac']:.2f} "
            f"rep=({sched_disp['repair_base_steps']}/{sched_disp['repair_max_frac']:.3f}) elit_frac={sched_disp['max_elitism_frac']:.2f} | "
            f"train batch={train_batch_eff} kilo={train_kilo_eff} pool={workers_eff}"
        )
        print(f"Training n={n} | kilo={train_kilo_eff} | batch={train_batch_eff}")
        train_parameters(
            board_size=n,
            kilo=max(1, train_kilo_eff or 1),
            batch_runs=max(10, train_batch_eff),
            time_limit_s=args.time_limit if args.time_limit is not None else None,
            workers=max(1, int(args.workers or 1)),
            dynamic_time=dyn_flag,
            trace_every=max(0, int(args.trace_every or 0)),
            mp_start=args.mp_start,
        )
        return
    # Banner for normal solve
    sched_disp = recommend_schedule(n)
    workers_eff = max(1, int(args.workers or 1))
    print(
        f"Config | n={n} | dyn_time={'on' if dyn_flag else 'off'} base={base_cap}s max={max_default}s | "
        f"GA pop={settings.population_size} gens={settings.generations} elit={settings.elitism} tk={settings.tournament_k} "
        f"cr={settings.crossover_rate:.2f} mr={settings.mutation_rate:.2f} | "
        f"sched adapt={sched_disp['adapt_every']} near={sched_disp['near_thr']:.3f} far={sched_disp['far_thr']:.3f} "
        f"mut±=({sched_disp['mut_step_up']:.3f}/{sched_disp['mut_step_down']:.3f}) imm={sched_disp['immigrant_frac']:.2f} "
        f"rep=({sched_disp['repair_base_steps']}/{sched_disp['repair_max_frac']:.3f}) elit_frac={sched_disp['max_elitism_frac']:.2f} | "
        f"repeats={repeats_solve} pool={workers_eff}"
    )

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
        # Normal single solve with optional guarantee to reach zero attacks
        dyn = bool(args.dynamic_time or DYN_TIME_ENABLED_DEFAULT)
        # Optional per-generation trace for single solve
        trace_path = None
        if args.trace_run:
            trace_dir = PARAM_STORE_PATH.parent / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            trace_path = str(trace_dir / f"n{n}_{ts}.trace.tsv")

        result = run_ga(
            settings=settings,
            board_size=n,
            time_limit_s=args.time_limit,
            verbose=True,
            dynamic_time=dyn,
            trace_path=trace_path,
        )
        solved = result.best.number_of_attacks == 0
        if not solved and SOLVE_GUARANTEE_DEFAULT:
            # Try greedy finish first
            if greedy_finish(result.best, SOLVE_FINAL_STEPS):
                solved = True
            else:
                # Attempt a few restarts with slightly higher mutation and dyn time
                best_overall = result
                for _ in range(max(0, SOLVE_MAX_RESTARTS)):
                    tweaked = GASettings.from_dict(settings.to_dict())
                    tweaked.mutation_rate = min(0.5, tweaked.mutation_rate + 0.05)
                    tweaked.generations = int(tweaked.generations * 1.25)
                    r2 = run_ga(
                        settings=tweaked,
                        board_size=n,
                        time_limit_s=args.time_limit,
                        verbose=False,
                        dynamic_time=True,
                    )
                    if r2.best.number_of_attacks < best_overall.best.number_of_attacks:
                        best_overall = r2
                    if r2.best.number_of_attacks > 0:
                        greedy_finish(r2.best, SOLVE_FINAL_STEPS)
                    if r2.best.number_of_attacks == 0:
                        result = r2
                        solved = True
                        break
                if not solved:
                    result = best_overall
        view.print_board_box(result.best, title="\nBest board:")


if __name__ == "__main__":
    main()
