import board
import evolutionary
import sys
from constants import *
problem_solved = False
generation = 0
maximum_generations = BOARD_SIZE**3

# Initialize first generation's samples
boards = [board.Board(BOARD_SIZE) for x in range(POPULATION_SIZE)]

while not problem_solved:
    # Select best solutions
    boards.sort(key=lambda board: board.number_of_attacks)

    for board in boards:
        print(board.state, f"No. Attacks: {board.number_of_attacks}",
              f"No. Non-attacks: {board.number_of_non_attacks}")

    if boards[0].number_of_attacks == 0:
        problem_solved = True
        break

    evolutionary.roulette_wheel_selection(boards)

    # Cross-over and mutate

    # Populate new generation
    problem_solved = True
