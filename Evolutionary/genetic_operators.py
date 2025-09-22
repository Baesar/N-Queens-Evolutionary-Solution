from random import randint


def pair_parents(parents, amount):
    """Creates an amount of parent pairs.

    Args:
        parents: A list of parents.
        amount: The amount of pairs to be created.

    Returns:
        A list of parent pairs.
    """

    parent_pairs = []

    while len(parent_pairs) < amount:
        parent_2 = parents[randint(0, len(parents) - 1)]
        parent_1 = parents[randint(0, len(parents) - 1)]
        if parent_2 == parent_1:
            continue
        parent_pairs.append((parent_1, parent_2))

    return parent_pairs


def one_point_crossover(parent_1, parent_2):
    """Combines two parents by choosing a random cross-over point.

    Args:
        parent_1: First parent.
        parent_2: Second parent.

    Returns:
        Two new children.
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
