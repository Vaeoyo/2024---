import tkinter as tk
import threading
from tkinter import messagebox


class UIModel:
    def __init__(self) -> None:
        self.needCalibrate = False
        self.player = 0
        self.nextStep = False
        self.testNextGrid = None


uiModel = UIModel()


def calibrate():
    uiModel.needCalibrate = True


def next_step():
    print("点击下一步按钮")
    uiModel.nextStep = True


def choose_black():
    uiModel.player = -1


def choose_white():
    uiModel.player = 1


def on_grid_button_click(value):
    choice = messagebox.askyesno(
        "棋子选择", f"选择在 {value} 下棋，选择”是“下黑棋，选择”否“下白棋"
    )
    if choice:
        uiModel.testNextGrid = (int(value) - 1, -1)
    else:
        uiModel.testNextGrid = (int(value) - 1, 1)


def UIThread():
    root = tk.Tk()
    root.title("棋盘对弈")

    # 创建按钮
    calibrate_button = tk.Button(root, text="校准", command=calibrate)
    calibrate_button.pack(pady=5)

    next_step_button = tk.Button(root, text="对弈下一步", command=next_step)
    next_step_button.pack(pady=5)

    # 选择棋手
    black_button = tk.Button(root, text="选择黑方", command=choose_black)
    black_button.pack(side=tk.LEFT, padx=10, pady=10)

    white_button = tk.Button(root, text="选择白方", command=choose_white)
    white_button.pack(side=tk.RIGHT, padx=10, pady=10)

    # 创建九宫格
    grid_frame = tk.Frame(root)
    grid_frame.pack(pady=5)
    for i in range(3):
        for j in range(3):
            btn = tk.Button(
                grid_frame,
                text=f"{i*3+j+1}",
                width=10,
                height=5,
                command=lambda value=i * 3 + j + 1: on_grid_button_click(value),
            )
            btn.grid(row=i, column=j)

    root.mainloop()


gui_thread = threading.Thread(target=UIThread)
gui_thread.start()
