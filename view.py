from constants import BOARD_SIZE
from app import boards
import numpy as np 
import matplotlib as plt 
import matplotlib.pyplot as pyplot 
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
## Use of matplotlib
## We use black background, white tiles all over and inside white tiles with queens inside we put a black Q inside 
def print_matplotlib(table):
    fig , ax = pyplot.subplots(figsize=(BOARD_SIZE, BOARD_SIZE))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if table[i][j] == "Q":
                ax.text(j,i,'Q', fontsize=24, ha='center', va='center', color='black', weight='bold')
    ax.set_xticks(np.arange(BOARD_SIZE))
    ax.set_yticks(np.arange(BOARD_SIZE))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which='both', length = 0)
    for i in range(BOARD_SIZE+1):
        ax.axhline(i-0.5, color='gray', linewidth=1)
        ax.axvline(i-0.5, color='gray', linewidth=1)
    plt.title(f'{n}-Queens Solution', color='white', fontsize=16)
    plt.tight_layout()
    plt.show()  




#####################################################################################





visual_table = fill_table()
visual_table = get_table(boards[1], visual_table)
print_console_table(visual_table)
print_matplotlib(visual_table)