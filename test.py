def is_left_or_right(A, B, P):
    # 提取点的坐标
    x1, y1 = A
    x2, y2 = B
    x, y = P

    # 计算向量 AB 和 AP 的分量
    AB_x = x2 - x1
    AB_y = y2 - y1
    AP_x = x - x1
    AP_y = y - y1

    # 计算叉积
    cross_product = AB_x * AP_y - AB_y * AP_x

    # 判断叉积的符号
    if cross_product > 0:
        return "left"
    elif cross_product < 0:
        return "right"
    else:
        return "on the line"

# 示例
A = (1, 1)
B = (4, 4)
P = (2, 3)
print(is_left_or_right(A, B, P))  # 输出 "left"

P = (3, 2)
print(is_left_or_right(A, B, P))  # 输出 "right"

P = (2, 2)
print(is_left_or_right(A, B, P))  # 输出 "on the line"
