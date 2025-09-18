import board
import sys
from constants import * 


# Initialize first generation's samples
boards = [board.Board(BOARD_SIZE) for x in range(POPULATION_SIZE)]

for board in boards:    
    print(board.state, board.number_of_attacks)




 