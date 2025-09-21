from random import randint, random

import board as board_module
from Evolutionary import selection, genetic_operators
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
            board.update_number_of_attacks()

        # Sort solutions (best first)
        boards.sort(key=lambda board: board.number_of_attacks)

        if not generation % 50:
            print(f"Genereration {generation}:")
            print(
                f"Best solution: {boards[0].state} {boards[0].number_of_attacks}\n")

        # Check if solution is found
        if boards[0].number_of_attacks == 0:
            total_generations += generation
            problem_solved = True
            print(
                f"Solution found at after {generation} generations: {boards[0].state}")
            break

        # Select boards to be used for recombination
        selected_boards = selection.roulette_wheel_selection(
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
            child_1_state, child_2_state = genetic_operators.one_point_crossover(
                parent_1, parent_2)
            if len(child_states) == POPULATION_SIZE - 1:
                child_states.append(child_1_state)
            else:
                child_states.extend([child_1_state, child_2_state])

        children = [board_module.Board(BOARD_SIZE, child_state)
                    for child_state in child_states]

        children = [genetic_operators.mutate(child) if random() < MUTATION_RATE else child
                    for child in children]

        boards = children

    if not problem_solved:
        print(f"Repetition {rep + 1}: {generation} generations\n\n")

if REPETITIONS > 1:
    print(f"Average generation: {total_generations / REPETITIONS}")
