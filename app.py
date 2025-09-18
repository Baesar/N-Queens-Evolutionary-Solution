import board
import evolutionary

# Problem constants
board_size = 8

# Evolutionary constants
population_size = 10  # board_size**2
mutation_rate = 0.1

problem_solved = False
generation = 0
maximum_generations = board_size**3

# Initialize first generation's samples
boards = [board.Board(board_size) for x in range(population_size)]

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
