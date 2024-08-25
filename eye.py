from collections import defaultdict
import platform
import cv2
import numpy as np
import time
import math
import config


class CornerManager:
    def __init__(self) -> None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.isChecked = False
        self.basePoint = (0, 0)
        self.cornerPoint1 = (0, 0)
        self.cornerPoint2 = (0, 0)
        self.perp_slope = None
        self.axes_point = [(1, 1), (1, 1)]
        self.cornerLength = config.CornerLength  # 两个标记中心点之间的毫米距离

    def detectArmTag(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        center_x, center_y = None, None

        if ids is None:
            print("detectArmTag done, not found tag")
            return center_x, center_y, image

        for i, corner in enumerate(corners):
            print("Detected markers:", ids[i])
            if ids[i][0] == 44:
                corner = corner.reshape((4, 2))
                top_left, top_right, bottom_right, bottom_left = corner

                # 计算轮廓的中心点
                center_x = int(
                    (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
                )
                center_y = int(
                    (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4
                )

                cv2.circle(image, (center_x, center_y), 10, (0, 255, 255), -1)

        return center_x, center_y, image

    def detect(self, image):
        image = image.copy()
        pointList = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        if ids is None:
            print("detect done, not found tag")
            return image

        # cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for i, corner in enumerate(corners):
            print("Detected markers:", ids)
            if ids[i] not in [[41], [42]]:
                continue

            # 提取标记的四个角
            corner = corner.reshape((4, 2))
            top_left, top_right, bottom_right, bottom_left = corner

            # 计算轮廓的中心点
            center_x = int(
                (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
            )
            center_y = int(
                (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4
            )

            cv2.circle(image, (center_x, center_y), 7, (0, 0, 255), -1)
            cv2.putText(
                image,
                f"{ids[i]}-{center_x}, {center_y}",
                (center_x - 20, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            pointList.append([center_x, center_y, ids[i]])

        if len(pointList) != 2:
            return image

        self.isChecked = True
        self.computeAxes(pointList)
        self.drawLine(image=image)
        return image

    def computeAxes(self, pointList):
        def midpoint(point1, point2):
            x1, y1, _ = point1
            x2, y2, _ = point2
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            return (int(mid_x), int(mid_y))

        def perpendicular_line(point, slope):
            if slope is not None:
                perp_slope = -1 / slope  # 垂直线的斜率
            else:
                perp_slope = None
            return perp_slope

        p1 = pointList[0]
        p2 = pointList[1]
        mid_point = midpoint(p1, p2)
        self.basePoint = mid_point
        self.cornerPoint1 = p1
        self.cornerPoint2 = p2
        # 计算直线的斜率
        if p2[0] - p1[0] != 0:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            self.perp_slope = perpendicular_line(mid_point, slope)
        else:
            slope = None
            self.perp_slope = None  # 垂直线的斜率是无穷大的

        # 计算垂线上的点
        if self.perp_slope is not None:
            x1, y1 = int(mid_point[0] - 50), int(mid_point[1] - 50 * self.perp_slope)
            x2, y2 = int(mid_point[0] + 50), int(mid_point[1] + 50 * self.perp_slope)
        else:
            x1, y1 = int(mid_point[0]), int(mid_point[1] - 50)
            x2, y2 = int(mid_point[0]), int(mid_point[1] + 50)

        self.axes_point = [(x1, y1), (x2, y2)]

    def drawLine(self, image):
        point1, point2, mid_point = self.cornerPoint1, self.cornerPoint2, self.basePoint
        point1 = (point1[0], point1[1])
        point2 = (point2[0], point2[1])
        # 绘制图像
        color = (0, 255, 0)
        thickness = 2
        cv2.line(image, point1, point2, color, thickness)  # 绘制连线
        cv2.circle(image, point1, 5, (0, 0, 255), -1)  # 绘制点1
        cv2.circle(image, point2, 5, (0, 0, 255), -1)  # 绘制点2
        cv2.circle(
            image, (int(mid_point[0]), int(mid_point[1])), 5, (255, 0, 0), -1
        )  # 绘制中点

        # 绘制垂线
        x1, y1 = self.axes_point[0]
        x2, y2 = self.axes_point[1]
        if self.perp_slope is not None:
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        else:
            cv2.line(
                image,
                (int(mid_point[0]), int(mid_point[1] - 50)),
                (int(mid_point[0]), int(mid_point[1] + 50)),
                (255, 255, 0),
                2,
            )

    def transformCoordinates(self, point):
        # 计算连线的斜率和角度
        dx = self.cornerPoint2[0] - self.cornerPoint1[0]
        dy = self.cornerPoint2[1] - self.cornerPoint1[1]
        angle = np.arctan2(dy, dx)
        original_length = np.sqrt(dx**2 + dy**2)

        # 构建旋转矩阵
        rotation_matrix = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )

        # 将原始点移至中点
        shifted_points = np.array(point) - np.array(self.basePoint)

        # 应用旋转矩阵
        transformed_point = np.dot(shifted_points, rotation_matrix.T)

        # 计算缩放因子
        scale_factor = self.cornerLength / original_length

        # 应用缩放
        scaled_points = transformed_point * scale_factor

        # 提取点的坐标
        x1, y1 = self.axes_point[0]
        x2, y2 = self.axes_point[1]
        x, y = point

        # 计算向量 AB 和 AP 的分量
        AB_x = x2 - x1
        AB_y = y2 - y1
        AP_x = x - x1
        AP_y = y - y1

        # 计算叉积
        cross_product = AB_x * AP_y - AB_y * AP_x

        # 判断叉积的符号
        # if cross_product > 0:
        #     "left"
        # elif cross_product < 0:
        #      "right"
        # else:
        #      "on the line"

        negative = -1 if cross_product > 0 else 1

        return (negative * abs(int(scaled_points[0])), abs(int(scaled_points[1])))


class Transformer:
    camera_matrix = np.array(
        [
            [845.42529841, 0.0, 286.27475729],
            [0.0, 842.95051514, 254.15748931],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.array(
        [
            7.41719414e-02,
            -1.11614687e00,
            2.04187891e-03,
            -1.18444984e-02,
            5.28974031e00,
        ],
        dtype=np.float32,
    )

    def __init__(self):
        self.M = None
        self.h = None
        self.w = None
        self.matrix = None
        self.matrix_roi = None

    def distortionCorrection(self, image):
        if self.matrix is None:
            self.h, self.w = image.shape[:2]
            self.matrix, self.matrix_roi = cv2.getOptimalNewCameraMatrix(
                Transformer.camera_matrix,
                Transformer.dist_coeffs,
                (self.w, self.h),
                1,
                (self.w, self.h),
            )

        undistorted_img = cv2.undistort(
            image,
            Transformer.camera_matrix,
            Transformer.dist_coeffs,
            None,
            self.matrix,
        )
        x, y, w1, h1 = self.matrix_roi
        undistorted_img = undistorted_img[y : y + h1, x : x + w1]
        return undistorted_img

    def warpPerspective(self, image, cornerManager: CornerManager):
        if self.M is None and cornerManager.isChecked:
            self.h, self.w = image.shape[:2]
            dst_pts = np.array(
                [[0, 0], [self.w - 1, 0], [self.w - 1, self.h - 1], [0, self.h - 1]],
                dtype="float32",
            )
            src_points = np.array(
                [
                    cornerManager.pointList[3][:2],
                    cornerManager.pointList[0][:2],
                    cornerManager.pointList[1][:2],
                    cornerManager.pointList[2][:2],
                ],
                dtype=np.float32,
            )
            # 计算透视变换矩阵
            self.M = cv2.getPerspectiveTransform(src_points, dst_pts)

        warped = cv2.warpPerspective(image, self.M, (self.w, self.h))
        return warped

    def detectChess(self, image):
        image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("opening", opening)

        contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        height, width = image.shape[:2]
        min_area = height * width * 0.03
        max_area = height * width * 0.15

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  # 获取矩形的四个顶点
            box = np.intp(box)  # 转换为整数
            center, (width, height), angle = rect
            area = width * height
            if area < min_area or area > max_area:
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) != 4:
                continue

            # Calculate the bounding box of the detected rectangle
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(cnt)
            roi = gray[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w]

            # Detect corners in the ROI
            corners = cv2.goodFeaturesToTrack(
                roi, maxCorners=25, qualityLevel=0.1, minDistance=20
            )
            num_corners = 0
            if corners is not None:
                corners = np.intp(corners)
                for corner in corners:
                    num_corners += 1
                    x, y = corner.ravel()
                    # cv2.circle(
                    #     image, (rect_x + x, rect_y + y), 5, 255, -1
                    # )  # Draw corners
            if num_corners < 9:
                continue

            # 绘制矩形
            cv2.drawContours(image, [box], 0, (0, 255, 0), 1)  # 绿色矩形，线宽为2

            center = np.array(center)
            angle = -angle  # OpenCV's angle is in the clockwise direction
            grid_rows = 3
            grid_cols = 3
            cell_width = width / grid_cols
            cell_height = height / grid_rows

            # Calculate rotation matrix
            M = cv2.getRotationMatrix2D(tuple(center), angle, 1)
            grids = []
            grid_index = 0
            for i in range(grid_rows):
                for j in range(grid_cols):
                    grid_index += 1
                    cell_center_x = center[0] - (width / 2) + (j + 0.5) * cell_width
                    cell_center_y = center[1] - (height / 2) + (i + 0.5) * cell_height
                    cell_center = np.dot(M, np.array([cell_center_x, cell_center_y, 1]))
                    grids.append((cell_center, (cell_width, cell_height), angle))
                    cv2.circle(
                        image,
                        (int(cell_center[0]), int(cell_center[1])),
                        2,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        image,
                        str(grid_index),
                        (int(cell_center[0]), int(cell_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            return True, image, grids

        return False, image, []


class Checkerboard:
    def __init__(self, cell_center, index=0):
        self.index = index
        self.cell_center = cell_center
        """
        state:
            -1 black
            0 empty
            1 white
        """
        self.current_state = 0


class ChessLog:
    def __init__(self) -> None:
        self.logList = []

    def lastLog(self):
        return self.logList[-1]

    def addLog(self, chessBoard: list[Checkerboard]):
        log = []
        for checkerboard in chessBoard:
            log.append((checkerboard.index, checkerboard.current_state))

        log.sort(key=lambda item: item[0])
        # if len(self.logList) == 0 or log != self.logList[-1]:
        print("记录日志",log)
        self.logList.append(log)

    def diffChessBoard(self):
        if len(self.logList) < 2:
            return []

        c = self.logList[-1]
        p = self.logList[-2]

        diffList = []
        for i in range(len(c)):
            current_state = c[i][1]
            previous_state = p[i][1]
            if current_state != previous_state:
                print(f"第{i+1}格子 由({previous_state})变为了({current_state}) ")
                diffList.append((i, current_state, previous_state))
        return diffList


class ChessManager:
    def __init__(self):
        self.chessBoard: list[Checkerboard] = []
        self.waitingPieces: list[Checkerboard] = []
        self.chessLog = ChessLog()

        self.cell_width = 0
        self.cell_height = 0
        self.angle = 0

        self.isInited = False

        self.circle_count = defaultdict(int)
        self.missed_count = defaultdict(int)
        self.spatial_threshold = 5  # 空间位置阈值
        self.radius_threshold = 5  # 半径阈值
        self.min_frame_count = 10  # 需要出现的最小帧数

    def setChessBoard(self, image, grids):
        self.chessBoardImage = image.copy()
        self.chessBoard = [
            Checkerboard(cell_center=grid[0], index=i) for i, grid in enumerate(grids)
        ]
        self.cell_width, self.cell_height = grids[0][1]
        self.angle = grids[0][2]

        self.isInited = True

    def updateChessBoard(self, x, y, newState):
        cell_radius = self.cell_width / 2
        in_grid = False
        for checkerboard in self.chessBoard:
            distance_squared = (x - checkerboard.cell_center[0]) ** 2 + (
                y - checkerboard.cell_center[1]
            ) ** 2
            if distance_squared <= cell_radius**2:
                in_grid = True
                checkerboard.current_state = newState
                break

        if not in_grid:
            checkerboard = Checkerboard((x, y), 0)
            checkerboard.current_state = newState
            self.waitingPieces.append(checkerboard)

    def getWaitingPiece(self, state):
        basePoint = np.array(cornerManager.basePoint)
        points = np.array(
            [
                item.cell_center
                for item in self.waitingPieces
                if item.current_state == state
            ]
        )
        if len(points) == 0:
            return None

        # 计算每个点与A的距离
        distances = np.linalg.norm(points - basePoint, axis=1)
        # 对点按照距离进行排序
        sorted_indices = np.argsort(distances)
        sorted_points = points[sorted_indices]
        # 打印结果
        # for i, idx in enumerate(sorted_indices):
        #     print(f"点 {points[idx]} 的距离为 {distances[idx]:.2f}")
        return cornerManager.transformCoordinates(sorted_points[0])

    def calculate_black_ratio(self, image, x, y, r):
        # 确保图像是二值图像
        assert len(image.shape) == 2 and (
            image.dtype == np.uint8 or image.dtype == np.bool
        ), "Image must be a binary image"

        # 创建一个与输入图像相同大小的掩膜
        mask = np.zeros_like(image, dtype=np.uint8)

        # 在掩膜上绘制圆形
        cv2.circle(mask, (x, y), r, (255), thickness=-1)

        # 使用掩膜提取圆形区域
        masked_image = cv2.bitwise_and(image, mask)

        # 计算圆形区域内的黑色像素
        black_pixels = np.sum((masked_image == 0) & (mask == 255))
        total_pixels = np.sum(mask == 255)

        if total_pixels == 0:
            return 0.0  # 防止除零错误

        # 计算黑色像素占比
        black_ratio = black_pixels / total_pixels
        return black_ratio

    def detectPieces(self, image):
        image = image.copy()
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 定义绿色的HSV范围
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        # 创建掩模
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # 应用掩模
        green_area = cv2.bitwise_and(image, image, mask=mask)
        # 转换掩模为灰度图像
        gray = cv2.cvtColor(green_area, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("hsv", opening)

        contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_circles = []

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) > 6:  # 需要至少6个点来拟合圆形
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                radius = int(radius)
                if radius > 7 and radius < 20:  # 根据需要调整半径范围
                    area = cv2.contourArea(cnt)
                    circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2)
                    if circularity > 0.8:  # 根据需要调整圆形度阈值
                        detected_circles.append(((int(x), int(y)), radius))

        # 记录当前帧中出现的圆
        current_circles = set()

        for (x, y), radius in detected_circles:
            matched = False
            for key in list(self.circle_count.keys()):
                (hx, hy, hr) = key
                if (
                    abs(x - hx) < self.spatial_threshold
                    and abs(y - hy) < self.spatial_threshold
                    and abs(radius - hr) < self.radius_threshold
                ):
                    self.circle_count[key] += 1
                    self.missed_count[key] = 0  # 重置未出现计数
                    matched = True
                    current_circles.add(key)
                    break
            if not matched:
                self.circle_count[(x, y, radius)] = 1
                self.missed_count[(x, y, radius)] = 0
                current_circles.add((x, y, radius))

        # 更新未出现的圆的计数
        to_remove = []
        for key in self.circle_count.keys():
            if key not in current_circles:
                self.missed_count[key] += 1
                if self.missed_count[key] >= self.min_frame_count:
                    to_remove.append(key)
                    (x, y, radius) = key
                    self.updateChessBoard(x, y, newState=0)

        # 移除不再出现的圆
        for key in to_remove:
            del self.circle_count[key]
            del self.missed_count[key]

        # 更新当前棋盘上的棋子
        self.waitingPieces.clear()
        for key, count in self.circle_count.items():
            if count >= self.min_frame_count:
                (x, y, radius) = key
                # 创建掩模以检测圆心像素
                mask_circle = np.zeros_like(gray)
                cv2.circle(mask_circle, (x, y), radius, 255, -1)
                # 使用蒙版提取圆形区域的像素
                masked_image = cv2.bitwise_and(image, image, mask=mask_circle)
                # 将图像转换为HSV颜色空间
                hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
                # 检查颜色是否为白色
                lower_white = np.array([78, 12, 99])
                upper_white = np.array([170, 52, 255])
                # 创建白色和黑色的掩码
                white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
                # 计算白色和黑色像素的数量
                num_white_pixels = cv2.countNonZero(white_mask)
                num_total_pixels = cv2.countNonZero(mask_circle)
                # 计算白色和黑色像素的比例
                white_ratio = num_white_pixels / num_total_pixels

                # 根据比例设置颜色和状态
                if white_ratio > 0.5:
                    color = (0, 255, 255)
                    newState = 1
                else:
                    color = (130, 130, 130)
                    newState = -1

                cv2.circle(image, (x, y), radius, color, 2)
                self.updateChessBoard(x, y, newState=newState)

        return image

    def drawGrid(self, image):
        for checkerboard in self.chessBoard:
            cv2.circle(
                image,
                (int(checkerboard.cell_center[0]), int(checkerboard.cell_center[1])),
                2,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                image,
                str(checkerboard.index + 1),
                (int(checkerboard.cell_center[0]), int(checkerboard.cell_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def gridIndexToCond(self, index):
        for checkerboard in self.chessBoard:
            if checkerboard.index == index:
                x, y = cornerManager.transformCoordinates(checkerboard.cell_center)
                return x, y
        return (0, 0)


def initCamera():
    is_pi = platform.system() == "Linux"
    if is_pi:
        camera_index = 0
        camera = cv2.VideoCapture(camera_index)
        while not camera.isOpened():
            print(f"打开摄像头失败 {camera_index}")
            camera_index += 1
            time.sleep(1)
            camera = cv2.VideoCapture(camera_index)
        return camera
    else:
        camera = cv2.VideoCapture(700)
        return camera


cornerManager = CornerManager()
imageTransformer = Transformer()
chessManager = ChessManager()
camera = initCamera()


# 定义回调函数，处理鼠标点击事件
def click_event(event, x, y, flags, param):
    from device_contrl import inverse_kinematics, move_to, init_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        new_points = cornerManager.transformCoordinates((x, y))
        print(f"({x}, {y}) --> ({new_points[0]}, {new_points[1]})")
        move_to(new_points)


cv2.namedWindow("chess")
cv2.setMouseCallback("chess", click_event)


def towPointDis(point1, point2):
    # 计算x和y方向的差值
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    distance = math.sqrt(dx**2 + dy**2)
    return dx, dy, distance


def calibrate():
    cornerManager.isChecked = False
    chessManager.isInited = False
    while True:
        cv2.waitKey(1)

        (grabbed, image) = camera.read()
        image = imageTransformer.distortionCorrection(image)
        # cv2.imshow("distortionCorrection", image)

        if not cornerManager.isChecked:
            corner_detect = cornerManager.detect(image=image)
            cv2.imshow("corner detect", corner_detect)
            continue

        if not chessManager.isInited:
            success, grid_image, grid = imageTransformer.detectChess(image)
            cv2.imshow("grid", grid_image)
            if success:
                chessManager.setChessBoard(grid_image, grid)
                return True
            else:
                continue


def detectContiue():
    cv2.waitKey(1)
    (grabbed, image) = camera.read()
    image = imageTransformer.distortionCorrection(image)

    chessImage = chessManager.detectPieces(image)
    chessManager.drawGrid(chessImage)
    cornerManager.drawLine(chessImage)
    cv2.imshow("chess", chessImage)


def detect():
    # 持续检测一秒
    for i in range(15):
        detectContiue()

    chessManager.chessLog.addLog(chessBoard=chessManager.chessBoard)
    return chessManager.chessLog.lastLog()


# camera.release()
# cv2.destroyAllWindows()
