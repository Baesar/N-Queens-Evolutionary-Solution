from random import randint, random
from typing import List

class Board:
    def __init__(self, size: int, state: List[int] | None = None):
        self.size = size
        if state is None:
            # Random placement: one queen per column in a random row
            self.state = [randint(0, self.size - 1) for _ in range(self.size)]
        else:
            self.state = state[:]
        self.calculate_number_of_attacks()

    def calculate_number_of_attacks(self):
        attack_count = 0
        for i, queen in enumerate(self.state):
            # Same row →
            for j in range(i + 1, self.size):
                if self.state[j] == queen:
                    attack_count += 1
            # Upper-right ↗
            for j in range(i + 1, self.size):
                if self.state[j] == queen + (j - i):
                    attack_count += 1
            # Lower-right ↘
            for j in range(i + 1, self.size):
                if self.state[j] == queen - (j - i):
                    attack_count += 1
        self.number_of_attacks = attack_count

    def max_non_attacking_pairs(self) -> int:
        n = self.size
        return n * (n - 1) // 2

    def fitness(self) -> int:
        """Higher is better (non-attacking pairs)."""
        return self.max_non_attacking_pairs() - self.number_of_attacks

    def clone(self) -> "Board":
        return Board(self.size, self.state)

    def mutate(self, p_gene: float):
        """Mutation: with probability p, move queen in a column to a random row."""
        changed = False
        for c in range(self.size):
            if random() < p_gene:
                self.state[c] = randint(0, self.size - 1)
                changed = True
        if changed:
            self.calculate_number_of_attacks()
