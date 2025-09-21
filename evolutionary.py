from random import random, randint


def roulette_wheel_selection(boards, selection_size):
    boards_copy = boards
    selected_boards = []

    for i in range(int((len(boards) * selection_size))):
        random_number = (random() ** 5) * (len(boards_copy))
        selected_boards.append(boards_copy.pop(int(random_number)))

    return selected_boards


def one_point_crossover(parent_1, parent_2):
    crossover_point = randint(1, parent_1.size - 2)

    child_1_state = parent_1.state[:crossover_point] + \
        parent_2.state[crossover_point:]
    child_2_state = parent_2.state[:crossover_point] + \
        parent_1.state[crossover_point:]

    return child_1_state, child_2_state


def mutate(child, rate):
    if rate >= random():
        index = randint(0, child.size - 1)
        old_value = child.state[index]
        new_value = old_value
        while new_value == old_value:
            new_value = randint(0, child.size - 1)
        child.state[index] = new_value
    return child
