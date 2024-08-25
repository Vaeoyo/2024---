from eye import calibrate, detect, chessManager, detectContiue
from game_engine import get_next_move
import device_contrl
import time
import config
from ui import uiModel


def print_board(board):
    formatted_board = [spot[1] for spot in board]
    print()
    for i in range(3):
        print(formatted_board[i * 3 : i * 3 + 3])
    print()


def take_and_play_chess(player, grid_index=None):
    chess_board = detect()
    print_board(chess_board)
    # 1表示白棋，-1表示黑棋
    waiting_piece = chessManager.getWaitingPiece(player)
    if waiting_piece is None:
        print("没有剩余棋子")
        return

    print(f"取棋子 {waiting_piece}")
    device_contrl.move_to(waiting_piece)
    time.sleep(1)
    device_contrl.placePiece(True)
    time.sleep(0.5)
    device_contrl.init_pos()
    time.sleep(0.5)
    if grid_index is None:
        winner, next_step = get_next_move(board=chess_board, player=player)
        if winner != 0:
            print(f"胜负: {winner}")
        else:
            print(f"在({next_step+1})放棋子")
            device_contrl.move_to(chessManager.gridIndexToCond(next_step))
            time.sleep(1)
            device_contrl.placePiece(False)
    else:
        print(f"在({grid_index+1})放棋子")
        device_contrl.move_to(chessManager.gridIndexToCond(grid_index))
        time.sleep(1)
        device_contrl.placePiece(False)

    time.sleep(0.5)
    device_contrl.init_pos()
    time.sleep(0.5)
    detect()


def check_error_chess(player):
    chess_board = detect()
    print_board(chess_board)
    diff_list = chessManager.chessLog.diffChessBoard()
    from_index, to_index = -1, -1
    for diff in diff_list:
        index, c, p = diff
        if c == player and p == 0:
            from_index = index
        elif c == 0 and p == player:
            to_index = index

    if from_index != -1 and to_index != -1:
        print(f"纠正错误: 将({from_index+1}) 放回({to_index+1}))")
        print(f"在({from_index+1})取棋子")
        device_contrl.init_pos()
        time.sleep(0.5)
        device_contrl.move_to(chessManager.gridIndexToCond(from_index))
        time.sleep(1)
        device_contrl.placePiece(True)
        time.sleep(0.5)
        print(f"在({to_index+1})放棋子")
        device_contrl.move_to(chessManager.gridIndexToCond(to_index))
        time.sleep(1)
        device_contrl.placePiece(False)
        time.sleep(0.5)
        device_contrl.init_pos()
        time.sleep(1)
        detect()
        return True

    return False


device_contrl.init_pos()
device_contrl.placePiece(0)
time.sleep(1)
calibrate()

while True:
    # detectContiue()
    if uiModel.needCalibrate:
        uiModel.needCalibrate = False
        calibrate()
    elif uiModel.nextStep or device_contrl.keyState == 0:
        print("next")
        try:
            device_contrl.setLight(0)
            if not check_error_chess(uiModel.player):
                take_and_play_chess(uiModel.player)
        except Exception as e:
            print(e)
        finally:
            uiModel.nextStep = False
            device_contrl.keyState = 1
            device_contrl.setLight(1)
    elif uiModel.testNextGrid is not None:
        take_and_play_chess(uiModel.testNextGrid[1], uiModel.testNextGrid[0])
        uiModel.testNextGrid = None
