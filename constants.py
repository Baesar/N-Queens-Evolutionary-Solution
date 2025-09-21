#####################
# Problem constants
#####################
BOARD_SIZE = 8


#####################
# Evolutionary constants
#####################

# The amount of solutions per generation
POPULATION_SIZE = 150

# Number of generations before algorithm terminates
MAXIMUM_GENERATIONS = BOARD_SIZE**3

# The proportion of solutions to proceed to recombination
SELECTION_SIZE = 0.3

# The rate at which solutions mutate
MUTATION_RATE = 0.4


#####################
# Testing
#####################

# The amount of times the algorithm is run
REPETITIONS = 100

# wall-clock time limit (in seconds)
TIME_LIMIT_S = None

if BOARD_SIZE < 4:
    print("BOARD_SIZE must be greater than 4")
    exit()

if int(POPULATION_SIZE * SELECTION_SIZE) < 2:
    print("POPULATION_SIZE * SELECTION_SIZE must be greater than 2")
    print(f"POPULATION_SIZE = {POPULATION_SIZE}")
    print(f"SELECTION_SIZE = {SELECTION_SIZE}")
    print(
        f"POPULATION_SIZE * SELECTION_SIZE = {POPULATION_SIZE * SELECTION_SIZE}")
    exit()

if SELECTION_SIZE > 1:
    print("SELECTION_SIZE cannot be greater than 1")
    exit()

if MUTATION_RATE > 1:
    print("MUTATION_RATE cannot be greater than 1")
    exit()

if REPETITIONS == 0:
    print("REPETITIONS cannot equal 0")
    exit()

if BOARD_SIZE < 0 or POPULATION_SIZE < 0 or MAXIMUM_GENERATIONS < 0 or SELECTION_SIZE < 0 or MUTATION_RATE < 0 or REPETITIONS < 0:
    print("No negative values for constants are allowed")
    exit()
