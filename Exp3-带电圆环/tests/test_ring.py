import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- 常量定义 ---
a = 1.0  # 圆环半径 (m)
q = 1.0  # 电荷参数 (对应电荷 Q = 4*pi*eps0*q)
# V(x,y,z) = q/(2*pi) * integral(...)
# 这里 C 对应 q/(2*pi)
C = q / (2 * np.pi)

# --- 计算函数 ---

def calculate_potential_on_grid(y_coords, z_coords):
    """
    在 yz 平面 (x=0) 的网格上计算电势 V(0, y, z)。
    使用 numpy 的向量化和 trapz 进行数值积分。

    参数:
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组

    返回:
        V (np.ndarray): 在 (y, z) 网格上的电势值
        y_grid (np.ndarray): y 坐标网格
        z_grid (np.ndarray): z 坐标网格
    """
    # 创建 y, z, phi 三维网格
    z_grid, y_grid, phi_grid = np.mgrid[
        z_coords.min():z_coords.max():complex(0, len(z_coords)),
        y_coords.min():y_coords.max():complex(0, len(y_coords)),
        0:2*np.pi:100j  # phi 方向积分点数
    ]

    # 计算场点到圆环上各点的距离 R
    R = np.sqrt((a * np.cos(phi_grid))**2 + (y_grid - a * np.sin(phi_grid))**2 + z_grid**2)
    
    # 处理 R=0 的情况，避免除零错误
    R = np.maximum(R, 1e-10)

    # 计算电势微元并对 phi 积分
    dV = C / R
    V = np.trapezoid(dV, dx=phi_grid[0,0,1]-phi_grid[0,0,0], axis=-1)
    
    return V, y_grid[:,:,0], z_grid[:,:,0]

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    根据电势 V 计算 yz 平面上的电场 E = -∇V。
    使用 np.gradient 进行数值微分。

    参数:
        V (np.ndarray): 电势网格 (z 维度优先)
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组

    返回:
        Ey (np.ndarray): 电场的 y 分量
        Ez (np.ndarray): 电场的 z 分量
    """
    # 计算梯度步长
    dz = z_coords[1] - z_coords[0]
    dy = y_coords[1] - y_coords[0]

    # 计算电势梯度，注意 V 的维度顺序是 (z, y)
    grad_z, grad_y = np.gradient(-V, dz, dy)

    # 电场分量 = -梯度分量
    Ez = grad_z
    Ey = grad_y
    
    return Ey, Ez

# --- 可视化函数 ---

def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制 yz 平面上的等势线和电场线。

    参数:
        y_coords, z_coords: 定义网格的坐标
        V: 电势网格
        Ey, Ez: 电场分量网格
        y_grid, z_grid: 绘图用的二维网格坐标
    """
    fig = plt.figure('带电圆环的电势和电场分布 (yz平面, x=0)', figsize=(14, 6))
    plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']  # 确保中文显示

    # 1. 绘制等势线 (填充图)
    ax1 = plt.subplot(1, 2, 1)
    # 使用对数归一化处理电势的大范围变化
    norm = LogNorm(vmin=V[V>0].min(), vmax=V.max())
    
    # 绘制填充等势线
    contourf_plot = plt.contourf(
        y_grid, z_grid, V, 
        levels=20, 
        cmap='viridis',
        norm=norm
    )
    plt.colorbar(contourf_plot, label='电势 V (单位: q/(2πε₀))')
    
    # 绘制等势线轮廓
    contour_plot = plt.contour(
        y_grid, z_grid, V, 
        levels=contourf_plot.levels, 
        colors='white', 
        linewidths=0.5
    )
    plt.clabel(contour_plot, inline=True, fontsize=8, fmt='%.2f')  # 标注等势线值
    
    # 标记圆环位置
    circle = plt.Circle((0, 0), a, fill=False, color='red', linestyle='--', linewidth=2)
    ax1.add_patch(circle)
    plt.text(0, 0, '圆环', ha='center', va='center', color='red', fontweight='bold')
    
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('yz平面等势线分布')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. 绘制电场线 (流线图)
    ax2 = plt.subplot(1, 2, 2)
    # 计算电场强度用于着色
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    
    # 绘制电场流线图
    stream_plot = plt.streamplot(
        y_grid, z_grid, Ey, Ez,
        color=E_magnitude,
        cmap='plasma',
        linewidth=1,
        density=1.5,
        arrowstyle='->',
        arrowsize=1.5
    )
    plt.colorbar(stream_plot.lines, label='电场强度 |E|')
    
    # 标记圆环位置
    circle = plt.Circle((0, 0), a, fill=False, color='blue', linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    plt.scatter([a, -a], [0, 0], color='blue', s=50, zorder=5)  # 圆环与yz平面交点
    plt.text(a, 0, '圆环', ha='left', va='bottom', color='blue', fontweight='bold')
    
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('yz平面电场线分布')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域 (yz 平面, x=0)
    y_range = np.linspace(-2*a, 2*a, 100)  # 增加点数提高精度
    z_range = np.linspace(-2*a, 2*a, 100)  # 增加点数提高精度

    # 1. 计算电势
    print("正在计算电势分布...")
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    print("电势计算完成.")

    # 2. 计算电场
    print("正在计算电场分布...")
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    print("电场计算完成.")

    # 3. 可视化
    print("正在生成可视化结果...")
    plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
    print("可视化完成.")    
