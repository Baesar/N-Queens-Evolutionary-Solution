import argparse
import os
import time
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
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
)
import view


@dataclass
class GASettings:
    population_size: int = POPULATION_SIZE
    generations: int = GENERATIONS
    elitism: int = ELITISM
    tournament_k: int = TOURNAMENT_K
    crossover_rate: float = CROSSOVER_RATE
    mutation_rate: float = MUTATION_RATE

    def clone(self) -> "GASettings":
        return GASettings(
            population_size=self.population_size,
            generations=self.generations,
            elitism=self.elitism,
            tournament_k=self.tournament_k,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
        )


@dataclass
class GAResult:
    best: Board
    duration: float
    generations: int


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _ensure_valid_settings(settings: GASettings) -> GASettings:
    settings.population_size = max(2, int(settings.population_size))
    settings.generations = max(1, int(settings.generations))
    settings.elitism = max(0, min(int(settings.elitism), settings.population_size - 1))
    settings.tournament_k = max(2, min(int(settings.tournament_k), settings.population_size))
    settings.crossover_rate = _clamp(float(settings.crossover_rate), 0.0, 1.0)
    settings.mutation_rate = _clamp(float(settings.mutation_rate), 0.0, 1.0)
    return settings


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

def local_improve(board: Board, steps: int = 2) -> None:
    """Small greedy repair: try a few random columns and move to a row that
    reduces conflicts the most. Keeps it cheap but helpful for larger n."""
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
            board.state[col] = old
            board.calculate_number_of_attacks()
        board.state[col] = best_row
        board.calculate_number_of_attacks()

def init_population(board_size: int, population_size: int) -> List[Board]:
    return [Board(board_size) for _ in range(population_size)]


def run_ga(
    settings: Optional[GASettings] = None,
    board_size: int = BOARD_SIZE,
    time_limit_s: Optional[float] = TIME_LIMIT_S,
    verbose: bool = True,
) -> GAResult:
    params = _ensure_valid_settings(settings.clone() if settings else GASettings())

    start = time.perf_counter()
    population = init_population(board_size, params.population_size)
    best = max(population, key=lambda b: b.fitness()).clone()
    max_fit = best.max_non_attacking_pairs()
    best_generation = 0

    if verbose:
        print(
            f"Initial best fitness: {best.fitness()} / {max_fit} "
            f"(attacks={best.number_of_attacks})"
        )

    last_generation = 0
    for gen in range(1, params.generations + 1):
        last_generation = gen
        if time_limit_s is not None and (time.perf_counter() - start) >= time_limit_s:
            if verbose:
                print(f"\nTime limit reached at generation {gen}.")
            break

        population.sort(key=lambda b: b.fitness(), reverse=True)
        if population[0].fitness() > best.fitness():
            best = population[0].clone()
            best_generation = gen

        if best.number_of_attacks == 0:
            if verbose:
                print(
                    f"\nSolved at generation {gen} "
                    f"in {time.perf_counter() - start:.3f}s."
                )
            break

        elites = [population[i].clone() for i in range(min(params.elitism, len(population)))]

        new_pop: List[Board] = elites[:]
        while len(new_pop) < params.population_size:
            p1 = tournament_select(population, params.tournament_k)
            p2 = tournament_select(population, params.tournament_k)
            c1, c2 = crossover(p1, p2, params.crossover_rate)
            c1.mutate(params.mutation_rate)
            # small local improvement helps larger n
            local_improve(c1, steps=max(0, board_size // 8))
            if len(new_pop) < params.population_size:
                new_pop.append(c1)
            c2.mutate(params.mutation_rate)
            local_improve(c2, steps=max(0, board_size // 8))
            if len(new_pop) < params.population_size:
                new_pop.append(c2)

        population = new_pop

        if verbose and (gen % 100 == 0 or gen == 1):
            print(
                f"Gen {gen:5d} | best fit = {best.fitness():3d}/{max_fit} "
                f"| attacks={best.number_of_attacks}"
            )

    duration = time.perf_counter() - start
    if verbose:
        print(
            f"\nFinished in {duration:.3f}s | Best fitness: {best.fitness()}/{max_fit} "
            f"| attacks={best.number_of_attacks} | best gen={best_generation}"
        )
    return GAResult(best=best, duration=duration, generations=last_generation)


# ---------------------------
# Hyperparameter GA support
# ---------------------------

POPULATION_SIZE_MIN, POPULATION_SIZE_MAX = 50, 300
GENERATIONS_MIN, GENERATIONS_MAX = 200, 800
ELITISM_MIN, ELITISM_MAX = 1, 10
TOURNAMENT_MIN, TOURNAMENT_MAX = 2, 10
CROSSOVER_MIN, CROSSOVER_MAX = 0.6, 0.95
MUTATION_MIN, MUTATION_MAX = 0.05, 0.5


@dataclass
class HyperparameterBounds:
    pop_min: int
    pop_max: int
    gens_min: int
    gens_max: int
    elitism_min: int
    elitism_max: int
    tourn_min: int
    tourn_max: int
    crossover_min: float
    crossover_max: float
    mutation_min: float
    mutation_max: float


def bounds_for_board(board_size: int) -> HyperparameterBounds:
    # Scale ranges by board size; fallback to global minima
    pop_min = max(POPULATION_SIZE_MIN, 12 * board_size)
    pop_max = max(POPULATION_SIZE_MAX, 40 * board_size)
    gens_min = max(GENERATIONS_MIN, 30 * board_size)
    gens_max = max(GENERATIONS_MAX, 120 * board_size)
    elitism_min = 1
    elitism_max = max(ELITISM_MAX, max(2, board_size // 2))
    tourn_min = 2
    tourn_max = min(max(TOURNAMENT_MAX, 8), max(3, pop_min))
    crossover_min = 0.65
    crossover_max = 0.95
    mutation_min = 0.05
    mutation_max = 0.4
    return HyperparameterBounds(
        pop_min, pop_max, gens_min, gens_max,
        elitism_min, elitism_max, tourn_min, tourn_max,
        crossover_min, crossover_max, mutation_min, mutation_max,
    )


@dataclass
class HyperparameterGenome:
    population_size: int
    generations: int
    elitism: int
    tournament_k: int
    crossover_rate: float
    mutation_rate: float

    @classmethod
    def random(cls, b: HyperparameterBounds) -> "HyperparameterGenome":
        genome = cls(
            population_size=random.randint(b.pop_min, b.pop_max),
            generations=random.randint(b.gens_min, b.gens_max),
            elitism=random.randint(b.elitism_min, b.elitism_max),
            tournament_k=random.randint(b.tourn_min, b.tourn_max),
            crossover_rate=random.uniform(b.crossover_min, b.crossover_max),
            mutation_rate=random.uniform(b.mutation_min, b.mutation_max),
        )
        genome.enforce_bounds(b)
        return genome

    def clone(self) -> "HyperparameterGenome":
        return HyperparameterGenome(
            population_size=self.population_size,
            generations=self.generations,
            elitism=self.elitism,
            tournament_k=self.tournament_k,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
        )

    def enforce_bounds(self, b: HyperparameterBounds) -> None:
        self.population_size = int(_clamp(self.population_size, b.pop_min, b.pop_max))
        self.generations = int(_clamp(self.generations, b.gens_min, b.gens_max))
        self.elitism = int(_clamp(self.elitism, b.elitism_min, min(b.elitism_max, self.population_size - 1)))
        self.elitism = max(1, self.elitism)
        self.tournament_k = int(_clamp(self.tournament_k, b.tourn_min, min(b.tourn_max, self.population_size)))
        self.crossover_rate = _clamp(self.crossover_rate, b.crossover_min, b.crossover_max)
        self.mutation_rate = _clamp(self.mutation_rate, b.mutation_min, b.mutation_max)

    def to_settings(self) -> GASettings:
        return GASettings(
            population_size=self.population_size,
            generations=self.generations,
            elitism=self.elitism,
            tournament_k=self.tournament_k,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
        )


@dataclass
class HyperparameterEvaluation:
    score: float
    avg_fitness: float
    avg_duration: float
    avg_generations: float
    success_rate: float
    best_result: GAResult


@dataclass
class HyperparameterHistoryRecord:
    generation: int
    rank: int
    score: float
    success_rate: float
    avg_fitness: float
    avg_duration: float
    avg_generations: float
    genome: "HyperparameterGenome"


@dataclass
class HyperparameterOptimizationResult:
    best_settings: GASettings
    best_genome: "HyperparameterGenome"
    best_evaluation: HyperparameterEvaluation
    history: List[HyperparameterHistoryRecord] = field(default_factory=list)


def hyperparameter_crossover(
    p1: HyperparameterGenome, p2: HyperparameterGenome
) -> Tuple[HyperparameterGenome, HyperparameterGenome]:
    def pick(a: float, b: float) -> float:
        return a if random.random() < 0.5 else b

    child1 = HyperparameterGenome(
        population_size=int(pick(p1.population_size, p2.population_size)),
        generations=int(pick(p1.generations, p2.generations)),
        elitism=int(pick(p1.elitism, p2.elitism)),
        tournament_k=int(pick(p1.tournament_k, p2.tournament_k)),
        crossover_rate=float(pick(p1.crossover_rate, p2.crossover_rate)),
        mutation_rate=float(pick(p1.mutation_rate, p2.mutation_rate)),
    )
    child2 = HyperparameterGenome(
        population_size=int(pick(p1.population_size, p2.population_size)),
        generations=int(pick(p1.generations, p2.generations)),
        elitism=int(pick(p1.elitism, p2.elitism)),
        tournament_k=int(pick(p1.tournament_k, p2.tournament_k)),
        crossover_rate=float(pick(p1.crossover_rate, p2.crossover_rate)),
        mutation_rate=float(pick(p1.mutation_rate, p2.mutation_rate)),
    )
    # Bounds are enforced by the caller after mutation when bounds are available
    return child1, child2


def hyperparameter_mutate(
    genome: HyperparameterGenome,
    bounds: HyperparameterBounds,
    rate: float = 0.25,
) -> HyperparameterGenome:
    child = genome.clone()
    # Scale mutation steps relative to bounds
    if random.random() < rate:
        step = max(1, int(0.1 * (bounds.pop_max - bounds.pop_min)))
        child.population_size += random.randint(-step, step)
    if random.random() < rate:
        step = max(1, int(0.1 * (bounds.gens_max - bounds.gens_min)))
        child.generations += random.randint(-step, step)
    if random.random() < rate:
        child.elitism += random.randint(-2, 2)
    if random.random() < rate:
        child.tournament_k += random.randint(-2, 2)
    if random.random() < rate:
        child.crossover_rate += random.uniform(-0.08, 0.08)
    if random.random() < rate:
        child.mutation_rate += random.uniform(-0.08, 0.08)
    child.enforce_bounds(bounds)
    return child


def _meta_tournament_pick(entries, k: int) -> HyperparameterGenome:
    competitors = random.sample(entries, k)
    best_entry = max(competitors, key=lambda item: item[0])
    return best_entry[1].clone()


def evaluate_hyperparameters(
    genome: HyperparameterGenome,
    board_size: int,
    evaluation_runs: int,
    time_limit_s: Optional[float],
) -> HyperparameterEvaluation:
    total_fitness = 0.0
    total_duration = 0.0
    total_generations = 0.0
    successes = 0
    best_result: Optional[GAResult] = None

    for _ in range(evaluation_runs):
        result = run_ga(
            settings=genome.to_settings(),
            board_size=board_size,
            time_limit_s=time_limit_s,
            verbose=False,
        )
        total_fitness += result.best.fitness()
        total_duration += result.duration
        total_generations += result.generations
        if result.best.number_of_attacks == 0:
            successes += 1
        if best_result is None or result.best.fitness() > best_result.best.fitness():
            best_result = result

    # Fallback to ensure a result is always returned
    if best_result is None:
        best_result = run_ga(
            settings=genome.to_settings(),
            board_size=board_size,
            time_limit_s=time_limit_s,
            verbose=False,
        )
        total_fitness += best_result.best.fitness()
        total_duration += best_result.duration
        total_generations += best_result.generations
        evaluation_runs += 1

    avg_fitness = total_fitness / evaluation_runs
    avg_duration = total_duration / evaluation_runs
    avg_generations = total_generations / evaluation_runs
    success_rate = successes / evaluation_runs

    # Heavier weight on success for larger n
    success_weight = 100 + board_size * 12
    score = (
        avg_fitness
        + success_rate * success_weight
        - (avg_duration * 2.0)
        - (avg_generations * 0.05)
    )

    return HyperparameterEvaluation(
        score=score,
        avg_fitness=avg_fitness,
        avg_duration=avg_duration,
        avg_generations=avg_generations,
        success_rate=success_rate,
        best_result=best_result,
    )


def optimize_parameters(
    board_size: int = BOARD_SIZE,
    meta_population_size: int = 8,
    meta_generations: int = 6,
    evaluation_runs: int = 2,
    time_limit_s: Optional[float] = TIME_LIMIT_S,
    verbose: bool = True,
    record_history: bool = False,
    history_top_k: int = 3,
) -> HyperparameterOptimizationResult:
    if meta_population_size < 2:
        raise ValueError("meta_population_size must be at least 2")

    bnds = bounds_for_board(board_size)
    population = [HyperparameterGenome.random(bnds) for _ in range(meta_population_size)]
    best_overall: Optional[Tuple[HyperparameterGenome, HyperparameterEvaluation]] = None
    history_records: List[HyperparameterHistoryRecord] = []

    for generation in range(1, meta_generations + 1):
        scored = []
        for genome in population:
            evaluation = evaluate_hyperparameters(
                genome,
                board_size=board_size,
                evaluation_runs=evaluation_runs,
                time_limit_s=time_limit_s,
            )
            scored.append((evaluation.score, genome.clone(), evaluation))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score, top_genome, top_eval = scored[0]

        if best_overall is None or top_eval.score > best_overall[1].score:
            best_overall = (top_genome.clone(), top_eval)

        if verbose:
            print(
                f"Meta gen {generation:2d} | score={top_score:7.2f} "
                f"| success={top_eval.success_rate:.2f} | avg fitness={top_eval.avg_fitness:.2f}"
            )

        if record_history:
            for rank, (_, genome_snapshot, evaluation_snapshot) in enumerate(
                scored[: max(1, history_top_k)],
                start=1,
            ):
                history_records.append(
                    HyperparameterHistoryRecord(
                        generation=generation,
                        rank=rank,
                        score=evaluation_snapshot.score,
                        success_rate=evaluation_snapshot.success_rate,
                        avg_fitness=evaluation_snapshot.avg_fitness,
                        avg_duration=evaluation_snapshot.avg_duration,
                        avg_generations=evaluation_snapshot.avg_generations,
                        genome=genome_snapshot.clone(),
                    )
                )

        next_population = [top_genome.clone()]
        while len(next_population) < meta_population_size:
            parent1 = _meta_tournament_pick(scored, k=min(3, len(scored)))
            parent2 = _meta_tournament_pick(scored, k=min(3, len(scored)))
            child1, child2 = hyperparameter_crossover(parent1, parent2)
            child1 = hyperparameter_mutate(child1, bnds)
            child2 = hyperparameter_mutate(child2, bnds)
            next_population.append(child1)
            if len(next_population) < meta_population_size:
                next_population.append(child2)

        if random.random() < 0.2:
            next_population[-1] = HyperparameterGenome.random(bnds)

        population = next_population

    assert best_overall is not None
    best_genome, best_eval = best_overall
    return HyperparameterOptimizationResult(
        best_settings=best_genome.to_settings(),
        best_genome=best_genome.clone(),
        best_evaluation=best_eval,
        history=history_records if record_history else [],
    )


def run_gap_report(
    board_min: int,
    board_max: int,
    meta_population: int,
    meta_generations: int,
    evaluation_runs: int,
    time_limit_s: Optional[float],
    history_top_k: int,
    output_dir: Path,
) -> Path:
    if board_min < 4:
        board_min = 4
    if board_max < board_min:
        board_max = board_min

    meta_population = max(2, meta_population)
    meta_generations = max(1, meta_generations)
    evaluation_runs = max(1, evaluation_runs)
    history_top_k = max(1, history_top_k)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"gap_report_{timestamp}.txt"

    header_lines = [
        "GAP hyperparameter optimization report",
        f"Board sizes: {board_min}-{board_max}",
        f"Meta population: {meta_population}",
        f"Meta generations: {meta_generations}",
        f"Evaluation runs per genome: {evaluation_runs}",
        f"History top-k: {history_top_k}",
        f"Time limit per GA run: {time_limit_s if time_limit_s is not None else 'None'}",
        "",
    ]

    with report_path.open("w", encoding="ascii", errors="ignore") as fh:
        for line in header_lines:
            fh.write(line + "\n")

        for board_size in range(board_min, board_max + 1):
            print(f"Running GAP optimization for board size {board_size}...")
            optimization = optimize_parameters(
                board_size=board_size,
                meta_population_size=meta_population,
                meta_generations=meta_generations,
                evaluation_runs=evaluation_runs,
                time_limit_s=time_limit_s,
                verbose=False,
                record_history=True,
                history_top_k=history_top_k,
            )

            fh.write("=" * 72 + "\n")
            fh.write(f"Board size n={board_size}\n")

            fh.write("Best settings:\n")
            for key, value in asdict(optimization.best_settings).items():
                fh.write(f"  {key}: {value}\n")

            summary = optimization.best_evaluation
            fh.write("Best evaluation summary:\n")
            fh.write(f"  score: {summary.score:.3f}\n")
            fh.write(f"  success_rate: {summary.success_rate:.3f}\n")
            fh.write(f"  avg_fitness: {summary.avg_fitness:.3f}\n")
            fh.write(f"  avg_duration: {summary.avg_duration:.4f}s\n")
            fh.write(f"  avg_generations: {summary.avg_generations:.2f}\n")

            best_ga_result = summary.best_result
            best_board = best_ga_result.best
            fh.write("Best sampled board (column -> row):\n")
            fh.write(f"  state: {best_board.state}\n")
            fh.write(
                f"  attacks: {best_board.number_of_attacks}, fitness: {best_board.fitness()} / {best_board.max_non_attacking_pairs()}\n"
            )
            fh.write(f"  duration: {best_ga_result.duration:.4f}s, generations: {best_ga_result.generations}\n")

            if optimization.history:
                fh.write("Generation history (top performers):\n")
                current_generation = None
                for entry in optimization.history:
                    if entry.generation != current_generation:
                        current_generation = entry.generation
                        fh.write(f"  Generation {entry.generation}:\n")
                    genome = entry.genome
                    fh.write(
                        "    Rank {rank}: score={score:.3f}, success={success:.3f}, avg_fit={avg_fit:.3f}, "
                        "avg_dur={avg_dur:.4f}s, avg_gen={avg_gen:.2f}\n".format(
                            rank=entry.rank,
                            score=entry.score,
                            success=entry.success_rate,
                            avg_fit=entry.avg_fitness,
                            avg_dur=entry.avg_duration,
                            avg_gen=entry.avg_generations,
                        )
                    )
                    fh.write(
                        "      genome: population={pop}, generations={gens}, elitism={elitism}, "
                        "tournament_k={tk}, crossover_rate={cr:.3f}, mutation_rate={mr:.3f}\n".format(
                            pop=genome.population_size,
                            gens=genome.generations,
                            elitism=genome.elitism,
                            tk=genome.tournament_k,
                            cr=genome.crossover_rate,
                            mr=genome.mutation_rate,
                        )
                    )

            fh.write("\n")

    return report_path


def solve_with_restarts(
    settings: GASettings,
    board_size: int,
    restarts: int,
    time_limit_s: Optional[float],
    verbose: bool = True,
) -> GAResult:
    best_result: Optional[GAResult] = None
    for i in range(max(1, restarts)):
        result = run_ga(settings=settings, board_size=board_size, time_limit_s=time_limit_s, verbose=verbose)
        if result.best.number_of_attacks == 0:
            return result
        if best_result is None or result.best.fitness() > best_result.best.fitness():
            best_result = result
    assert best_result is not None
    return best_result


# ---------------------------
# Entry point
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="N-Queens genetic algorithm.")
    parser.add_argument("--optimize-params", action="store_true", help="Run the hyperparameter GA before solving.")
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE, help="Board size to solve.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT_S if TIME_LIMIT_S is not None else None,
        help="Optional time limit for each GA run (seconds).",
    )
    parser.add_argument("--population", type=int, help="Override population size for the solver GA.")
    parser.add_argument("--generations", type=int, help="Override number of generations for the solver GA.")
    parser.add_argument("--elitism", type=int, help="Override elitism count for the solver GA.")
    parser.add_argument("--tournament-k", type=int, help="Override tournament size for the solver GA.")
    parser.add_argument("--crossover-rate", type=float, help="Override crossover rate for the solver GA.")
    parser.add_argument("--mutation-rate", type=float, help="Override mutation rate for the solver GA.")
    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="How many independent GA runs to try; stops early on success.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not execute any GA; parse args and exit (useful for IDEs).",
    )
    parser.add_argument("--meta-population", type=int, default=8, help="Hyperparameter GA population size.")
    parser.add_argument("--meta-generations", type=int, default=6, help="Hyperparameter GA generations.")
    parser.add_argument(
        "--meta-evaluations",
        type=int,
        default=2,
        help="Base GA runs per hyperparameter evaluation.",
    )
    parser.add_argument(
        "--gap-report",
        action="store_true",
        help="Generate a GAP report across a range of board sizes instead of solving once.",
    )
    parser.add_argument(
        "--gap-board-min",
        type=int,
        default=4,
        help="Smallest board size to include in the GAP report.",
    )
    parser.add_argument(
        "--gap-board-max",
        type=int,
        default=16,
        help="Largest board size to include in the GAP report.",
    )
    parser.add_argument(
        "--gap-output",
        type=str,
        default="reports",
        help="Directory where GAP report files will be written.",
    )
    parser.add_argument(
        "--gap-history-top",
        type=int,
        default=3,
        help="Number of top genomes to log per meta generation in the GAP report.",
    )
    parser.add_argument(
        "--gap-evaluations",
        type=int,
        default=34,
        help="Number of GA runs per genome when building the GAP report.",
    )

    args = parser.parse_args()

    time_limit = args.time_limit if args.time_limit is not None else None

    # Global kill-switch via flag or environment variable
    if args.no_run or os.getenv("NQUEENS_DISABLE_RUN") == "1":
        print("Execution disabled (--no-run or NQUEENS_DISABLE_RUN=1). Exiting.")
        return

    if args.gap_report:
        report_path = run_gap_report(
            board_min=int(args.gap_board_min),
            board_max=int(args.gap_board_max),
            meta_population=args.meta_population,
            meta_generations=args.meta_generations,
            evaluation_runs=args.gap_evaluations,
            time_limit_s=time_limit,
            history_top_k=args.gap_history_top,
            output_dir=Path(args.gap_output),
        )
        print(f"\nGAP report written to {report_path}")
        return

    board_size = max(4, int(args.board_size))

    # Seed defaults from constants.recommend_params for the chosen board size
    rec = recommend_params(board_size)
    settings = _ensure_valid_settings(
        GASettings(
            population_size=rec["population_size"],
            generations=rec["generations"],
            elitism=rec["elitism"],
            tournament_k=rec["tournament_k"],
            crossover_rate=rec["crossover_rate"],
            mutation_rate=rec["mutation_rate"],
        )
    )

    overrides = {
        "population_size": args.population,
        "generations": args.generations,
        "elitism": args.elitism,
        "tournament_k": args.tournament_k,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(settings, field_name, value)
    settings = _ensure_valid_settings(settings)

    # board_size already computed above

    if args.optimize_params:
        optimization = optimize_parameters(
            board_size=board_size,
            meta_population_size=args.meta_population,
            meta_generations=args.meta_generations,
            evaluation_runs=max(1, args.meta_evaluations),
            time_limit_s=time_limit,
            verbose=True,
            record_history=False,
            history_top_k=args.gap_history_top,
        )
        best_settings = optimization.best_settings
        summary = optimization.best_evaluation
        print("\nBest hyperparameters found:")
        for key, value in asdict(best_settings).items():
            print(f"  {key}: {value}")
        settings = best_settings
        settings = _ensure_valid_settings(settings)
        result = solve_with_restarts(
            settings=settings,
            board_size=board_size,
            restarts=max(1, args.restarts),
            time_limit_s=time_limit,
            verbose=True,
        )
        print(
            f"\nHyperparameter GA summary: score={summary.score:.2f}, "
            f"success_rate={summary.success_rate:.2f}, avg_fitness={summary.avg_fitness:.2f}"
        )
        view.print_board_box(result.best, title="\nBest board after hyperparameter GA:")
    else:
        result = solve_with_restarts(
            settings=settings,
            board_size=board_size,
            restarts=max(1, args.restarts),
            time_limit_s=time_limit,
            verbose=True,
        )
        view.print_board_box(result.best, title="\nBest board:")


if __name__ == "__main__":
    main()
