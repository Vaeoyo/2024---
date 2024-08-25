def print_board(board):
    for i in range(3):
        print(board[i * 3 : i * 3 + 3])
    print()


def check_winner(board):
    # 所有可能的胜利组合
    win_combinations = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # 行
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],  # 列
        [0, 4, 8],
        [2, 4, 6],  # 对角线
    ]
    for combo in win_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != 0:
            return board[combo[0]]
    return 0


def is_full(board):
    return all(spot != 0 for spot in board)


def minimax(board, depth, is_maximizing, player):
    winner = check_winner(board)
    if winner != 0:
        return winner * player
    if is_full(board):
        return 0

    if is_maximizing:
        max_eval = float("-inf")
        for i in range(len(board)):
            if board[i] == 0:
                board[i] = player
                eval = minimax(board, depth + 1, False, player)
                board[i] = 0
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for i in range(len(board)):
            if board[i] == 0:
                board[i] = -player
                eval = minimax(board, depth + 1, True, player)
                board[i] = 0
                min_eval = min(min_eval, eval)
        return min_eval


def best_move(board, player):
    winner = check_winner(board)
    if winner != 0 or is_full(board):
        return (winner, -1)

    # 如果是空白棋盘，直接返回中心位置
    # if all(spot == 0 for spot in board):
    #     return (0, 4)

    best_val = float("-inf")
    move = -1
    for i in range(len(board)):
        if board[i] == 0:
            board[i] = player
            move_val = minimax(board, 0, False, player)
            board[i] = 0
            if move_val > best_val:
                best_val = move_val
                move = i

    winner = check_winner(board)
    if winner != 0 or is_full(board):
        return (winner, move)

    return (0, move)


def get_next_move(board, player):
    """
    board = [(0, -1), (1, -1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 0), (7, 0), (8, -1)]
    """
    board.sort(key=lambda item: item[0])
    formatted_board = [spot[1] for spot in board]
    return best_move(formatted_board, player)


if __name__ == "__main__":
    # 示例调用
    board = [(0, -1), (1, -1), (2, 1),
              (3, -1), (4, 1), (5, 0),
                (6, 1), (7, 0), (8, 0)]
    player = 1  # 1表示白棋，-1表示黑棋
    result = get_next_move(board, player)
    print(f"结果: {result}")
