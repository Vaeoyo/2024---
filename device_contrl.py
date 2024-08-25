import math
import config
import platform
import time

is_pi = platform.system() == "Linux"
if is_pi:
    import pigpio
else:
    print("当前平台不是Pi")


def inverse_kinematics(x, y, l1, l2):
    # 计算目标点到原点的距离
    D = math.sqrt(x**2 + y**2)

    # 检查目标点是否在机械臂的可达范围内
    if D > l1 + l2:
        print("目标点超出机械臂的最大伸展范围")
        return False, 0, 0

    # 计算第二个关节的角度 theta2
    cos_theta2 = (D**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = math.acos(cos_theta2)

    # 计算第一个关节的角度 theta1
    theta1 = math.atan2(y, x) - math.atan2(
        l2 * math.sin(theta2), l1 + l2 * math.cos(theta2)
    )

    # 将角度转换为度数
    theta1_deg = math.degrees(theta1)
    theta2_deg = math.degrees(theta2)

    return True, theta1_deg, theta2_deg


def placePiece(isFetch):
    angle = 130
    pulse_width = 500 + (angle / 180.0) * 2000  # 将角度转换为脉冲宽度
    pi.set_servo_pulsewidth(config.joint_pin_3, pulse_width)
    # 开启磁力
    time.sleep(1)
    pi.write(config.electromagnet_pin, isFetch)
    time.sleep(1)
    angle = 50
    pulse_width = 500 + (angle / 180.0) * 2000  # 将角度转换为脉冲宽度
    pi.set_servo_pulsewidth(config.joint_pin_3, pulse_width)


def setServoAngle(servo_pin, angle):
    if not is_pi:
        print("is not pi")
        return
    # 舵机角度范围0-270，占空比范围500-2500微秒
    pulse_width = 500 + (angle / 270.0) * 2000  # 将角度转换为脉冲宽度
    pi.set_servo_pulsewidth(servo_pin, pulse_width)


def move_to(point):
    x, y = point
    x += config.center_offset[0]
    y += config.center_offset[1]
    success, theta1_deg, theta2_deg = inverse_kinematics(x, y, config.L1, config.L2)
    theta1_deg_fix = config.JointBase1 + theta1_deg
    theta2_deg_fix = config.JointBase2 - theta2_deg
    print(
        f"关节1:{theta1_deg} - 加偏后:{theta1_deg_fix}, 关节2:{theta2_deg} - 加偏后:{theta2_deg_fix}"
    )
    if not success:
        init_pos()
        return

    setServoAngle(config.joint_pin_1, theta1_deg_fix)
    setServoAngle(config.joint_pin_2, theta2_deg_fix)


def setLight(isLight):
    pi.write(config.led_pin, not isLight)


def init_pos():
    setServoAngle(config.joint_pin_1, 0)
    setServoAngle(config.joint_pin_2, 0)


# 定义回调函数，当 GPIO 引脚状态变化时调用
def gpio_callback(gpio, level, tick):
    print(
        f"GPIO {gpio} state changed to {'HIGH' if level == 1 else 'LOW'} at tick {tick}"
    )
    global keyState
    keyState = level


pi = None
keyState = 0
if is_pi:
    pi = pigpio.pi()
    pi.set_mode(config.electromagnet_pin, pigpio.OUTPUT)
    pi.set_mode(config.input_pin, pigpio.INPUT)
    pi.set_pull_up_down(config.input_pin, pigpio.PUD_DOWN)

    pi.set_mode(config.led_pin, pigpio.OUTPUT)

    pi.callback(config.input_pin, pigpio.EITHER_EDGE, gpio_callback)
    if not pi.connected:
        raise Exception("Failed to connect to pigpio daemon")

if __name__ == "__main__":
    # 示例使用
    x = 9
    y = 230
    import time

    # angle = 150
    # pulse_width = 500 + (angle / 180.0) * 2000  # 将角度转换为脉冲宽度
    # pi.set_servo_pulsewidth(config.joint_pin_3, pulse_width)
    # setServoAngle(config.joint_pin_1, 0)
    # setServoAngle(config.joint_pin_2, 0)

    # pi.write(config.electromagnet_pin, 0)
    # init_pos()
    # time.sleep(1)
    # move_to((x, y))
    # placePiece(1)
    placePiece(0)

    """
    2,237
    9,230
    +7, -7
    """
