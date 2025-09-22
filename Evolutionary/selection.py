from random import random


def roulette_wheel_selection(boards, selection_size):
    """Select, randomly (but with bias to low indices), a proportion of elements from a list.

    Args:
        boards: List of boards to be culled through.
        selection_size: A float (0 <= selection_size <= 1) determining the proportion to be selected.

    Returns:
            A list of the selected elements.
    """

    boards_copy = boards
    selected_boards = []

    for i in range(int((len(boards) * selection_size))):
        random_number = (random() ** 5) * (len(boards_copy))
        selected_boards.append(boards_copy.pop(int(random_number)))

    return selected_boards
