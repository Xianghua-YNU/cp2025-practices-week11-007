import numpy as np
import matplotlib.pyplot as plt

# --- 常量定义 ---
a = 1.0  # 圆环半径 (m)
q = 1.0  # q 参数 (对应电荷 Q = 4*pi*eps0*q)
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
    """
    # 创建 y, z, phi 网格
    # 注意：mgrid 的索引顺序是 'ij' (笛卡尔) 还是 'xy' (矩阵)
    # 这里 z 是第一个维度，y 是第二个维度
    z_grid, y_grid, phi_grid = np.mgrid[z_coords.min():z_coords.max():complex(0, len(z_coords)),
                                        y_coords.min():y_coords.max():complex(0, len(y_coords)),
                                        0:2*np.pi:100j] # phi 方向积分点数增加到100

    # 计算到圆环上各点的距离 R
    # 圆环在 xy 平面，方程 x=a*cos(phi), y=a*sin(phi), z=0
    # 场点 P = (0, y, z)
    # 环上点 Q = (a*cos(phi), a*sin(phi), 0)
    # R = |P - Q| = sqrt( (0 - a*cos(phi))^2 + (y - a*sin(phi))^2 + (z - 0)^2 )
    R = np.sqrt((a * np.cos(phi_grid))**2 + (y_grid - a * np.sin(phi_grid))**2 + z_grid**2)

    # 处理 R=0 的情况（虽然在 yz 平面且 z!=0 或 y!=a 时不会发生）
    R[R < 1e-10] = 1e-10 # 避免除零错误

    # 计算电势微元 dV
    dV = C / R

    # 对 phi 进行积分 (使用梯形法则)
    # np.trapezoid 默认沿最后一个轴积分
    V = np.trapezoid(dV, dx=phi_grid[0,0,1]-phi_grid[0,0,0], axis=-1)
    return V, y_grid[:,:,0], z_grid[:,:,0] # 返回 V 和对应的 y, z 网格

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
    dz = z_coords[1] - z_coords[0]
    dy = y_coords[1] - y_coords[0]

    # 计算梯度，注意 V 的维度顺序是 (z, y)
    # gradient 返回值顺序与 V 的维度顺序一致: (dV/dz, dV/dy)
    grad_z, grad_y = np.gradient(-V, dz, dy)

    # E = -∇V，所以 Ez = -dV/dz, Ey = -dV/dy
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
    fig = plt.figure('Potential and Electric Field of Charged Ring (yz plane, x=0)', figsize=(12, 6))


    # 1. 绘制等势线 (填充图)
    plt.subplot(1, 2, 1)
    # 使用 contourf 绘制填充等势线图
    # levels = np.linspace(V.min(), V.max(), 15) # 自动或手动设置等势线级别
    contourf_plot = plt.contourf(y_grid, z_grid, V, levels=20, cmap='viridis')
    plt.colorbar(contourf_plot, label='Potential V (units: q/(2πε₀))') # 修改标签为英文
    # 使用 contour 绘制等势线线条
    contour_plot = plt.contour(y_grid, z_grid, V, levels=contourf_plot.levels, colors='white', linewidths=0.5)
    # plt.clabel(contour_plot, inline=True, fontsize=8) # 在等势线上标示数值
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Equipotential Lines (yz plane)') # 修改标题为英文
    plt.gca().set_aspect('equal', adjustable='box') # 保持纵横比为1
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. 绘制电场线 (流线图)
    plt.subplot(1, 2, 2)
    # 计算电场强度用于着色（可选）
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    # 限制流线图密度和长度
    stream_plot = plt.streamplot(y_grid, z_grid, Ey, Ez,
                                 color=E_magnitude,
                                 cmap='autumn',
                                 linewidth=1,
                                 density=1.5,
                                 arrowstyle='->',
                                 arrowsize=1.0)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Electric Field Lines (yz plane)') # 修改标题为英文
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)
    # 标记圆环在 yz 平面的截面位置 (y=±a, z=0)
    plt.plot([-1, 1], [0, 0], 'ro', markersize=5, label='Ring Cross-section') # 修改标签为英文
    plt.legend()

    plt.tight_layout() # 调整子图布局
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域 (yz 平面, x=0)
    # 范围以圆环半径 a 为单位
    y_range = np.linspace(-2*a, 2*a, 40) # y 方向点数
    z_range = np.linspace(-2*a, 2*a, 40) # z 方向点数

    # 1. 计算电势
    print("正在计算电势...")
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    print("电势计算完成.")

    # 2. 计算电场
    print("正在计算电场...")
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    print("电场计算完成.")

    # 3. 可视化
    print("正在绘图...")
    plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
    print("绘图完成.")
