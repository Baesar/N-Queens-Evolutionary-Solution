from constants import BOARD_SIZE
from app import boards
###############################################################
## INIT FUNCTIONS
## Fills the table with
def fill_table():
    table = []
    for i in range(BOARD_SIZE):
        # Create a new list for each row
        row = [" "] * BOARD_SIZE
        table.append(row)
    return table
## reads the board and writes queens where its required 
def get_table (board, table):
  
    for i in range(BOARD_SIZE):
        table[board.state[i]][i] = "Q"
    return table
#####################################################################################
## Primitive visuals 
## Cheks the visual board in the console 
def print_console_table (table):

    for i in range(BOARD_SIZE):
        print(table[i])
#####################################################################################


visual_table = fill_table()
visual_table = get_table(boards[1], visual_table)
print_console_table(visual_table)
            