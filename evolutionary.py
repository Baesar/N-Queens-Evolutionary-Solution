from random import random


def roulette_wheel_selection(boards, selection_size):
    boards_copy = boards
    selected_boards = []

    for x in range((len(boards_copy) * selection_size).__floor__):
        sum = 0
        for board in boards_copy:
            sum += board.number_of_non_attacks

        proportions = [0 for i in range(len(boards_copy))]

        for i, board in enumerate(boards_copy):
            proportions[i] = board.number_of_non_attacks / \
                sum + (proportions[i - 1] if i != 0 else 0)

        print(proportions)

        random_number = random()

        for i, state in enumerate(proportions):
            if state >= random_number:
                selected_boards.append(boards_copy[i])
                boards_copy.pop(i)
                break


def crossover(parent_1, parent_2):
    pass


def mutate():
    pass
