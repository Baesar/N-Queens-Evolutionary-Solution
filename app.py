import board

# Problem constants
BOARD_SIZE = 8

# Evolutionary constants
population_size = 10
mutation_rate = 0.1

# Initialize first generation's samples
boards = [board.Board(BOARD_SIZE) for x in range(population_size)]

for board in boards:
    print(board.state, board.number_of_attacks)


def fill_table():
    table = []
    for i in range(BOARD_SIZE):
        # Create a new list for each row
        row = [" "] * BOARD_SIZE
        table.append(row)
    return table
visual_table = fill_table()


def get_table (board, table):
  
    for i in range(BOARD_SIZE):
        table[board.state[i]][i] = "Q"
    return table
def print_console_table (table):

    for i in range(BOARD_SIZE):
        print(table[i])

visual_table = get_table(boards[1], visual_table)
print_console_table(visual_table)
            

 