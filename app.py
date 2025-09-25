from random import randint, random
from time import perf_counter

import board as board_module
from Evolutionary import selection, genetic_operators
from constants import *


total_time = 0
total_generations = 0

for rep in range(REPETITIONS):
    print(rep + 1)
    start = perf_counter()
    problem_solved = False
    generation = 0

    # Initialize first generation's samples.
    boards = [board_module.Board(BOARD_SIZE)
              for x in range(POPULATION_SIZE)]

    while not problem_solved:
        generation += 1

        for board in boards:
            board.update_number_of_attacks()

        # Sort solutions (best first).
        boards.sort(key=lambda board: board.number_of_attacks)

        # if not generation % 50:
        #     print(f"Genereration {generation}:")
        #     print(
        #        f"Best solution: {boards[0].state} {boards[0].number_of_attacks}\n")

        # Check if solution is found.
        if boards[0].number_of_attacks == 0:
            end = perf_counter()

            total_time += end - start
            total_generations += generation
            problem_solved = True
            # print(
            #     f"Solution found at after {generation} generations: {boards[0].state}")
            # print(f"It took {end - start:.6f} seconds")
            # break

        # Select boards to be used for recombination.
        selected_boards = selection.roulette_wheel_selection(
            boards, SELECTION_SIZE)

        # Create parent pairs.
        cross_over_parent_pairs = genetic_operators.pair_parents(
            selected_boards, int(POPULATION_SIZE / 2 + 1))

        # Generate new solutions with cross-over.
        child_states = []
        for parent_1, parent_2 in cross_over_parent_pairs:
            child_1_state, child_2_state = genetic_operators.one_point_crossover(
                parent_1, parent_2)

            # Only add the first child if the there is only 1 space left in the population. This happens if the POPOLATION_SIZE is odd.
            if len(child_states) == POPULATION_SIZE - 1:
                child_states.append(child_1_state)
            else:
                child_states.extend([child_1_state, child_2_state])

        children = [board_module.Board(BOARD_SIZE, child_state)
                    for child_state in child_states]

        # Mutate children.
        children = [genetic_operators.mutate(child) if random() < MUTATION_RATE else child
                    for child in children]

        # Assign new population of solution to the boards variable.
        boards = children

    if not problem_solved:
        print(f"Repetition {rep + 1}: {generation} generations\n\n")

if REPETITIONS > 1:
    print(f"\nAverage generations: {total_generations / REPETITIONS}")
    print(f"Average time: {total_time / REPETITIONS:.6f}")
