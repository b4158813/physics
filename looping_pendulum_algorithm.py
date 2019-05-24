import matplotlib.pyplot as plt
# from pyecharts import Line
import seaborn as sns
import numpy as np

sns.set()


def draw_shaft(R):  # 画一个圆轴的方法
    theta = np.linspace(0, 2 * np.pi, 100)
    x, y = R * np.cos(theta), R * np.sin(theta)
    plt.plot(x, y, linewidth=2.0)


# 初始条件
theta0 = 2 / 3 * np.pi  # 初始的theta值
omega0 = 0.  # 初始角速度
L = 1.  # 初始绳长
R = 0.01  # 初始轴的半径
H0 = 0.01  # 初始重物的y坐标
s0 = L - R * theta0 - H0  # 计算出初始吊着轻负载的绳长
m = 0.1  # 轻负载质量
M = 0.2  # 重负载质量
mu = 0.4  # 绳和杆的滑动摩擦因数
g = 9.8  # 重力加速度

# 造一堆列表便于画图
T_list = []
a_list = []
epsilon_list = []
omega_list = []
theta_list = []
H_list = []
s_list = []
v_list = []
v_light_list = []
t_list = []

# 轻重负载的坐标
x_list_light = []
y_list_light = []
x_list_heavy = [R]
y_list_heavy = [-H0]

# 初值
delta_H = 0.  # 重负载y坐标增量初始值
theta = theta0  # theta初始值
omega = omega0  # 角速度初始值
H = H0  # 重负载坐标初始值
s = s0  # 轻负载到“轴-线”切线距离初始值
v = 0.  # 初速度
t = 0.
dt = 0.001  # 时间间隔


def core_ini(m1=0.1, M1=0.2, mu1=0.1):
    global m, M, mu
    m = m1  # 轻负载质量
    M = M1  # 重负载质量
    mu = mu1  # 绳和杆的滑动摩擦因数


def operate_algorithm():
    global T_list, a_list, epsilon_list, omega_list, theta_list, H_list, s_list, v_list, \
        v_light_list, t_list, x_list_light, y_list_light, x_list_heavy, y_list_heavy, R
    global omega, s, theta, v, H, t
    # 开始算法前前清空所有的原始数据
    T_list = []
    a_list = []
    epsilon_list = []
    omega_list = []
    theta_list = []
    H_list = []
    s_list = []
    v_list = []
    v_light_list = []
    t_list = []

    # 轻重负载的坐标
    x_list_light = []
    y_list_light = []
    x_list_heavy = [R]
    y_list_heavy = [-H0]

    # 初值
    delta_H = 0.  # 重负载y坐标增量初始值
    theta = theta0
    omega = omega0
    H = H0
    s = s0
    v = 0
    t = 0.
    dt = 0.01  # 时间间隔
    while True:
        T = m * omega ** 2 * s - m * g * np.cos(theta)  # 轻负载所受拉力
        a = g - T * np.e ** (mu * theta) / M
        if v >= 0 and a >= 0 and s > 0 and H < 1 and s0 > R * (theta - theta0):
            epsilon = abs(g * np.sin(theta) / s)
            a = g - T * np.e ** (mu * theta) / M
            v = v + a * dt
            delta_H = v * dt
            H += delta_H
            s = L - R * theta - H
            # 把新的数值存入列表
            t_list.append(t)
            T_list.append(T)
            a_list.append(a)
            epsilon_list.append(epsilon)
            omega_list.append(omega)
            theta_list.append(theta)
            H_list.append(H)
            s_list.append(s)
            v_list.append(v)
            v_light_list.append(omega * s)
            # 计算新的坐标值并存入列表
            x = -s * np.sin(theta)
            y = s * np.cos(theta) + R * np.sin(theta)
            x_list_light.append(x)
            y_list_light.append(y)
            x_list_heavy.append(R)  # 重负载x坐标一直是轴的半径
            y_list_heavy.append(-H)  # 重负载坐标是负的
            # 更新新的值
            omega += epsilon * dt
            theta += omega * dt
            # 下一个dt
            t += dt

        elif v >= 0 and s > 0 and H < 1 and s0 > R * (theta - theta0):
            epsilon = abs(g * np.sin(theta) / s)
            a = 0
            v = v + a * dt
            delta_H = v * dt
            H += delta_H
            s = L - R * theta - H
            # 把新的数值存入列表
            t_list.append(t)
            T_list.append(T)
            a_list.append(a)
            epsilon_list.append(epsilon)
            omega_list.append(omega)
            theta_list.append(theta)
            H_list.append(H)
            s_list.append(s)
            v_list.append(v)
            v_light_list.append(omega * s)
            # 计算新的坐标值并存入列表
            x = -s * np.sin(theta)
            y = s * np.cos(theta) + R * np.sin(theta)
            x_list_light.append(x)
            y_list_light.append(y)
            x_list_heavy.append(R)  # 重负载x坐标一直是轴的半径
            y_list_heavy.append(-H)  # 重负载坐标是负的
            # 更新新的值
            omega += epsilon * dt
            theta += omega * dt
            # 下一个dt
            t += dt

        elif t < 2 and v < 0:
            v = -0.0000000001
            epsilon = abs(g * np.sin(theta) / s)
            s = L - R * theta - H
            t_list.append(t)
            T_list.append(T)
            a_list.append(a)
            epsilon_list.append(epsilon)
            omega_list.append(omega)
            theta_list.append(theta)
            H_list.append(H)
            s_list.append(s)
            v_list.append(v)
            v_light_list.append(omega * s)
            # 计算新的坐标值并存入列表
            x = -s * np.sin(theta)
            y = s * np.cos(theta) + R * np.sin(theta)
            x_list_light.append(x)
            y_list_light.append(y)
            x_list_heavy.append(R)  # 重负载x坐标一直是轴的半径
            y_list_heavy.append(-H)  # 重负载坐标是负的
            # 更新新的值
            omega += epsilon * dt
            theta += omega * dt
            # 下一个dt
            t += dt
        else:
            break


def print_params():
    print('T:', T_list)
    print('epsilon:', epsilon_list)
    print('a', a_list)
    print('omega:', omega_list)
    print('theta:', theta_list)
    print('H:', H_list)
    print('s:', s_list)
    print('v:', v_list)
    print('v_light:', v_light_list)
    print('t:', t_list)


def draw_pic():
    plt.subplot(221)
    core_ini(0.1,0.2,0.1)
    operate_algorithm()
    print_params()
    draw_shaft(R)  # 画一个轴
    plt.plot(x_list_light, y_list_light, label='light_object')  # 轻负载坐标绘制
    plt.plot(x_list_heavy, y_list_heavy, label='heavy_object')  # 重负载坐标绘制
    plt.xlabel('x')
    plt.ylabel('y')
    plt.text(-0.8, -0.3, 'μ=0.1\nR=0.01m\nL=1m\nθ=120°')
    plt.legend()

    plt.subplot(222)
    core_ini(0.1,0.2,0.2)
    operate_algorithm()
    print_params()
    draw_shaft(R)  # 画一个轴
    plt.plot(x_list_light, y_list_light, label='light_object')  # 轻负载坐标绘制
    plt.plot(x_list_heavy, y_list_heavy, label='heavy_object')  # 重负载坐标绘制
    plt.xlabel('x')
    plt.ylabel('y')
    plt.text(-0.8, -0.3, 'μ=0.2\nR=0.01m\nL=1m\nθ=120°')
    plt.legend()

    plt.subplot(223)
    core_ini(0.1, 0.2, 0.3)
    operate_algorithm()
    print_params()
    draw_shaft(R)  # 画一个轴
    plt.plot(x_list_light, y_list_light, label='light_object')  # 轻负载坐标绘制
    plt.plot(x_list_heavy, y_list_heavy, label='heavy_object')  # 重负载坐标绘制
    plt.xlabel('x')
    plt.ylabel('y')
    plt.text(-0.8, -0.3, 'μ=0.3\nR=0.01m\nL=1m\nθ=120°')
    plt.legend()

    plt.subplot(224)
    core_ini(0.1, 0.2, 0.4)
    operate_algorithm()
    print_params()
    draw_shaft(R)  # 画一个轴
    plt.plot(x_list_light, y_list_light, label='light_object')  # 轻负载坐标绘制
    plt.plot(x_list_heavy, y_list_heavy, label='heavy_object')  # 重负载坐标绘制
    plt.xlabel('x')
    plt.ylabel('y')
    plt.text(-0.8, -0.3, 'μ=0.4\nR=0.01m\nL=1m\nθ=120°')
    plt.legend()

    plt.show()


draw_pic()
