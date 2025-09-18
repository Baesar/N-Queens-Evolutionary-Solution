def print_board_box(board, title=None):
    """Pretty console board using box-drawing characters."""
    n = board.size
    if title:
        print("\n" + title)
    # top border
    print("┌" + "───┬" * (n - 1) + "───┐")
    for r in range(n):
        row_cells = []
        for c in range(n):
            row_cells.append("👑 " if board.state[c] == r else "   ")
        print("│" + "│".join(row_cells) + "│")
        if r != n - 1:
            print("├" + "───┼" * (n - 1) + "───┤")
    # bottom border
    print("└" + "───┴" * (n - 1) + "───┘")
    print(f"Attacks: {board.number_of_attacks} | Fitness: {board.fitness()} / {board.max_non_attacking_pairs()}")
