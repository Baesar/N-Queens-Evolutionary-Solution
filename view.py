from constants import BOARD_SIZE
from app import boards
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import time
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
        for j in range(BOARD_SIZE):
            if table.state[j] == i:
                print("|W|", end="")
            else:
                print("| |" , end="")
        print(" ")
#####################################################################################
## Use of matplotlib
## We use black background, white tiles all over and inside white tiles with queens inside we put a black Q inside 
def print_matplotlib(table):
    ##Create figure with black background
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('grey')
    ax.set_facecolor('grey')
    
    ##Create a chessboard 
    chessboard = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            # Create alternating pattern (like a chessboard)
            if (i + j) % 2 == 0:
                chessboard[i, j] = 1  # White tile
    
    ##Display the chessboard
    ax.imshow(chessboard, cmap='binary', interpolation='nearest')
    
    #Add the queens
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if table.state[j] == i:
                #Only place W white on black tiles (where chessboard value is 1)
                if (i + j) % 2 == 0:
                    ax.text(j, i, 'W', fontsize= (200 / BOARD_SIZE), ha='center', va='center', 
                            color='white', weight='bold')
                else:
                    # For white tiles, we need a black W
                    ax.text(j, i, 'W', fontsize=200 / BOARD_SIZE, ha='center', va='center', 
                            color='black', weight='bold')
            else :
                ##Rewrite to empty
                ax.text(j, i, '', fontsize=20, ha='center', va='center', 
                            color='black', weight = 'bold')

    

    plt.title(f'{BOARD_SIZE}-Queens Solution', color='white', fontsize=16)
    plt.tight_layout()
    plt.show()




    
#####################################################################################
def print_matplotlib_timer(tables , milisec):
    ##Same as before but now we try again but this time update board every time timer goes off
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('grey')
    ax.set_facecolor('grey')

    chessboard = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if (i + j) % 2 == 0:
                chessboard[i, j] = 1  # White tile
                ##Put in the queens
    for t in tables:
        ##adds delay here
        ax.clear()
        ax.imshow(chessboard, cmap='binary', interpolation='nearest')
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if t.state[j] == i:
                    if (i + j) % 2 == 0:
                        ax.text(j, i, 'W', fontsize=200 / BOARD_SIZE, ha='center', va='center', 
                            color='white', weight='bold')
                    else:
                    ##For white tiles, we need a black W
                        ax.text(j, i, 'W', fontsize=200 / BOARD_SIZE, ha='center', va='center', 
                            color='black', weight='bold')
                else :
                ##Rewrite to empty
                    ax.text(j, i, '', fontsize=20, ha='center', va='center', 
                            color='black', weight = 'bold')
        plt.title(f'Boards heuristic value is : {t.number_of_attacks}')
        plt.draw()
        plt.pause(milisec)  # 2 second delay





                
            
            
    

#####################################################################################
## Heuristic and minumum steps required sorting : 
#Step 1: After each generation save the ones with the highest heuristic values (usually 1 or 2) or just save the 5 with lowest heuristic value 
#step 2: After finding a soluton, rotate the solution to make a table list of all possible solutions (If you rotate solution you get a new solution)
#Step 3: Along with the heuristic value , the later generations we look at how many steps were they away from their closest solution
#Step 4: In the presentation using a graph we can show how the best 5 of each generations heuristic value AND their closest minumum solution steps
# Note : This allows us to see how close the program gets and where the program takes steps away and steps closer 

#han snackar om när en person inte gör piss men ändå säger att han är skitviktig och att han gjorde allt med zaki 
#HMmmm... let låter som någon vi känner


#####################################################################################



#####################################################################################

#####################################################################################

#####################################################################################

#####################################################################################







visual_table = boards[1]
## Old version ## visual_table = get_table(boards[1], visual_table)
## Old version ## print_console_table(visual_table)
print_matplotlib_timer(boards, 1)

