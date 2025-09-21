from random import randint


class Board:
    def update_number_of_attacks(self):
        attack_count = 0
        non_attack_count = 0
        total_attack_count = 0
        for i, queen in enumerate(self.state):
            attack_count = 0

            # Queens to the right →
            for j in range(i + 1, self.size):
                if self.state[j] == queen:
                    attack_count += 1

            # Queens to the upper right ↗
            for j in range(i + 1, (self.size - queen + i) if i < queen else (self.size)):
                if self.state[j] == queen + j - i:
                    attack_count += 1

            # Queens to the lower right ↘
            for j in range(i + 1, (queen + i + 1) if queen < self.size - i - 1 else self.size):
                if self.state[j] == queen - j + i:
                    attack_count += 1

            total_attack_count += attack_count
            non_attack_count += (self.size - 1 - i - attack_count)

        self.number_of_attacks = total_attack_count
        self.number_of_non_attacks = (
            (self.size - 1) * (self.size / 2)) - total_attack_count

    def __init__(self, size, state=None):
        self.size = size
        self.state = state if state != None else [
            randint(0, self.size - 1) for _ in range(size)]
        self.update_number_of_attacks()
