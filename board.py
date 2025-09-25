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
        # O(n) counting using rows and diagonals
        row_counts = {}
        diag1_counts = {}  # r - c
        diag2_counts = {}  # r + c
        for c, r in enumerate(self.state):
            row_counts[r] = row_counts.get(r, 0) + 1
            d1 = r - c
            d2 = r + c
            diag1_counts[d1] = diag1_counts.get(d1, 0) + 1
            diag2_counts[d2] = diag2_counts.get(d2, 0) + 1

        def comb2(x: int) -> int:
            return x * (x - 1) // 2 if x > 1 else 0

        attack_count = 0
        for cnt in row_counts.values():
            attack_count += comb2(cnt)
        for cnt in diag1_counts.values():
            attack_count += comb2(cnt)
        for cnt in diag2_counts.values():
            attack_count += comb2(cnt)
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
