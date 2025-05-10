import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
import sys
import pytest


# 常量设置
a = 1  # 圆环半径
q = 1  # 电荷参数，对应总电荷 Q = 4*pi*eps0*q
C = q / (2 * np.pi)  # 电势积分中的常量部分


# 计算电势的被积函数
def integrand(phi, x, y, z):
    """
    计算电势积分中的被积函数。
    参数：
        phi: 积分变量
        x, y, z: 场点坐标
    返回：
        被积函数的值
    """
    R = np.sqrt((x - a * np.cos(phi)) ** 2 + (y - a * np.sin(phi)) ** 2 + z ** 2)
    # 处理数值稳定性问题，避免R接近0
    R = np.where(R < 1e-10, 1e-10, R)
    return C / R


# 计算电势函数
def calculate_potential(x, y, z):
    """
    计算空间中一点 (x, y, z) 处的电势。
    参数：
        x, y, z: 场点坐标
    返回：
        该点的电势值
    """
    result, _ = quad(integrand, 0, 2 * np.pi, args=(x, y, z))
    return result


# 计算电场函数（通过数值微分）
def calculate_electric_field(x, y, z):
    """
    通过数值微分计算电场。
    参数：
        x, y, z: 场点坐标
    返回：
        Ex, Ey, Ez: 电场在x, y, z方向的分量
    """
    h = 1e-6  # 微分步长
    Vx_plus = calculate_potential(x + h, y, z)
    Vx_minus = calculate_potential(x - h, y, z)
    Vy_plus = calculate_potential(x, y + h, z)
    Vy_minus = calculate_potential(x, y - h, z)
    Vz_plus = calculate_potential(x, y, z + h)
    Vz_minus = calculate_potential(x, y, z - h)
    Ex = -(Vx_plus - Vx_minus) / (2 * h)
    Ey = -(Vy_plus - Vy_minus) / (2 * h)
    Ez = -(Vz_plus - Vz_minus) / (2 * h)
    return Ex, Ey, Ez


# 可视化部分
# 选择yz平面进行可视化
y_range = np.linspace(-2 * a, 2 * a, 100)
z_range = np.linspace(-2 * a, 2 * a, 100)
Y, Z = np.meshgrid(y_range, z_range)
V = np.zeros_like(Y)
Ey = np.zeros_like(Y)
Ez = np.zeros_like(Y)

for i in range(len(y_range)):
    for j in range(len(z_range)):
        V[j, i] = calculate_potential(0, y_range[i], z_range[j])
        Ey[j, i], _, Ez[j, i] = calculate_electric_field(0, y_range[i], z_range[j])


# 测试函数
def test_calculate_potential():
    """测试电势计算函数"""
    # 简单测试点
    x, y, z = 0, 0, 1
    potential = calculate_potential(x, y, z)
    assert isinstance(potential, float)


def test_calculate_electric_field():
    """测试电场计算函数"""
    # 简单测试点
    x, y, z = 0, 0, 1
    Ex, Ey, Ez = calculate_electric_field(x, y, z)
    assert isinstance(Ex, float)
    assert isinstance(Ey, float)
    assert isinstance(Ez, float)


# 主函数用于绘图
def main():
    # 绘制等势线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    contour = plt.contour(Y, Z, V, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('yz平面上的等势线 (x = 0)')
    plt.grid(True)

    # 绘制电场矢量分布
    plt.subplot(1, 2, 2)
    plt.streamplot(Y, Z, Ey, Ez, density=1.5, color='b')
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('yz平面上的电场线 (x = 0)')
    plt.grid(True)

    # 保存图片而不是显示，避免在无图形界面环境报错
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, 'charged_ring_plot.png'))
    plt.close()


if __name__ == "__main__":
    if '--test' in sys.argv:
        pytest.main([__file__])
    else:
        main()
