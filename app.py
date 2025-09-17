import board

# Problem constants
board_size = 8

# Evolutionary constants
population_size = 10
mutation_rate = 0.1

# Initialize first generation's samples
boards = [board.Board(board_size) for x in range(population_size)]

for board in boards:
    print(board.state, board.number_of_attacks)
