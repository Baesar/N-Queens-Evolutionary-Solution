from random import randint


class Board:
    def calculate_number_of_attacks(self):
        attack_count = 0
        for i, queen in enumerate(self.state):

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

        self.number_of_attacks = attack_count

    def __init__(self, size=8):
        self.size = size
        self.state = [randint(0, self.size - 1) for x in range(self.size)]
        self.calculate_number_of_attacks()
