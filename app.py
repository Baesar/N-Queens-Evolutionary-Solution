from random import randint

import board as board_module
import evolutionary
from constants import *


total_generations = 0

for rep in range(REPETITIONS):
    problem_solved = False
    generation = 0

    # Initialize first generation's samples
    boards = [board_module.Board(BOARD_SIZE) for x in range(POPULATION_SIZE)]

    while not problem_solved:
        generation += 1

        for board in boards:
            board.calculate_number_of_attacks()

        # Select best solutions
        boards.sort(key=lambda board: board.number_of_attacks)

        if not generation % 50:
            print(f"Genereration {generation}:")
            print(
                f"Best solution: {boards[0].state} {boards[0].number_of_attacks}\n")

        if boards[0].number_of_attacks == 0:
            total_generations += generation
            problem_solved = True
            print(
                f"Solution found at after {generation} generations: {boards[0].state}")
            break

        selected_boards = evolutionary.roulette_wheel_selection(
            boards, SELECTION_SIZE)

        n = len(selected_boards)
        cross_over_parent_pairs = []
        while len(cross_over_parent_pairs) < POPULATION_SIZE / 2:
            parent_1 = selected_boards[randint(0, n - 1)]
            parent_2 = selected_boards[randint(0, n - 1)]
            if parent_2 == parent_1:
                continue
            cross_over_parent_pairs.append((parent_1, parent_2))

        child_states = []

        # Cross-over and mutate
        for parent_1, parent_2 in cross_over_parent_pairs:
            child_1_state, child_2_state = evolutionary.one_point_crossover(
                parent_1, parent_2)
            if len(child_states) == POPULATION_SIZE - 1:
                child_states.append(child_1_state)
            else:
                child_states.extend([child_1_state, child_2_state])

        children = [board_module.Board(BOARD_SIZE, child_state)
                    for child_state in child_states]

        children = [evolutionary.mutate(child, MUTATION_RATE)
                    for child in children]

        boards = children

    if not problem_solved:
        print(f"Repetition {rep + 1}: {generation} generations\n\n")

if REPETITIONS > 1:
    print(f"Average generation: {total_generations / REPETITIONS}")
