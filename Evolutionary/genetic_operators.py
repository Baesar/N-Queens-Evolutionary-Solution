from random import randint


def one_point_crossover(parent_1, parent_2):
    """Combines two parents by choosing a random cross-over point.

    Args:
        parent_1: First parent.
        parent_2: Second parent.

    Returns:
        Two new children
    """

    crossover_point = randint(1, parent_1.size - 2)

    child_1_state = parent_1.state[:crossover_point] + \
        parent_2.state[crossover_point:]
    child_2_state = parent_2.state[:crossover_point] + \
        parent_1.state[crossover_point:]

    return child_1_state, child_2_state


def mutate(child):
    """Mutate child by altering one if it's elements.

    Args:
        child: The child to be mutated.

    Returns:
        The mutated child.
    """

    index = randint(0, child.size - 1)

    old_value = child.state[index]
    new_value = old_value

    while new_value == old_value:
        new_value = randint(0, child.size - 1)

    child.state[index] = new_value

    return child
