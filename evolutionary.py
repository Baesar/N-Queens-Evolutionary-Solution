from random import random


def roulette_wheel_selection(boards):
    sum = 0
    for board in boards:
        sum += board.number_of_attacks

    proportions = [0 for i in range(len(boards))]

    for i, board in enumerate(boards):
        print(f"i = {i}")
        proportions[i] = board.number_of_attacks / \
            sum + (proportions[i - 1] if i != 0 else 0)

    print(proportions)
    for proportion in proportions:
        print(f"{proportion}, {proportion / sum}")


def crossover(parent_1, parent_2):
    pass


def mutate():
    pass
