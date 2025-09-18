import time
import random
from typing import List, Tuple

from board import Board
from constants import * 
import view

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

def crossover(p1: Board, p2: Board) -> Tuple[Board, Board]:
    if random.random() > CROSSOVER_RATE:
        return p1.clone(), p2.clone()
    if random.random() < 0.5:
        return one_point_crossover(p1, p2)
    else:
        return uniform_crossover(p1, p2)

# ---------------------------
# GA main loop
# ---------------------------

def init_population() -> List[Board]:
    return [Board(BOARD_SIZE) for _ in range(POPULATION_SIZE)]

def run_ga() -> Board:
    start = time.perf_counter()
    population = init_population()
    best = max(population, key=lambda b: b.fitness()).clone()
    max_fit = best.max_non_attacking_pairs()

    print(f"Initial best fitness: {best.fitness()} / {max_fit} (attacks={best.number_of_attacks})")

    for gen in range(1, GENERATIONS + 1):
        if TIME_LIMIT_S is not None and (time.perf_counter() - start) >= TIME_LIMIT_S:
            print(f"\nTime limit reached at generation {gen}.")
            break

        population.sort(key=lambda b: b.fitness(), reverse=True)
        if population[0].fitness() > best.fitness():
            best = population[0].clone()

        if best.number_of_attacks == 0:
            print(f"\nSolved at generation {gen} in {time.perf_counter() - start:.3f}s.")
            break

        elites = [population[i].clone() for i in range(ELITISM)]

        new_pop: List[Board] = elites[:]
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_select(population, TOURNAMENT_K)
            p2 = tournament_select(population, TOURNAMENT_K)
            c1, c2 = crossover(p1, p2)
            c1.mutate(MUTATION_RATE)
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(c1)
            c2.mutate(MUTATION_RATE)
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(c2)

        population = new_pop

        if gen % 100 == 0 or gen == 1:
            print(f"Gen {gen:5d} | best fit = {best.fitness():3d}/{max_fit} | attacks={best.number_of_attacks}")

    duration = time.perf_counter() - start
    print(f"\nFinished in {duration:.3f}s | Best fitness: {best.fitness()}/{max_fit} | attacks={best.number_of_attacks}")
    return best

# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    best = run_ga()
    view.print_board_box(best, title="\nBest board:")
