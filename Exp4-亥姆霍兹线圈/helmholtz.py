import numpy as np
import matplotlib.pyplot as plt

# --- 物理和线圈参数 ---
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (T*m/A)
I = 1.0  # 电流 (A) - 假设为1A，实际计算中常数因子可以合并

def Helmholtz_coils(r_low, r_up, d):
    '''
    计算亥姆霍兹线圈（或两个不同半径线圈）的磁场。
    线圈平行于xy平面，圆心在z轴。
    下方线圈半径 r_low，位于 z = -d/2。
    上方线圈半径 r_up，位于 z = +d/2。

    输入:
        r_low (float): 下方线圈的半径 (m)
        r_up (float): 上方线圈的半径 (m)
        d (float): 两线圈中心之间的距离 (m)
    返回:
        Y_plot (np.ndarray): 用于绘图的 Y 坐标网格 (通常是 Y[:,:,0])
        Z_plot (np.ndarray): 用于绘图的 Z 坐标网格 (通常是 Z[:,:,0])
        By (np.ndarray): y方向的磁场分量 (T)
        Bz (np.ndarray): z方向的磁场分量 (T)
    '''
    print(f"开始计算磁场: r_low={r_low}, r_up={r_up}, d={d}")

    # 1. 定义积分角度 phi 和空间网格 y, z
    max_r = max(r_low,r_up)
    phi_angles = np.linspace(0,2*np.pi,20)
    y_coords = np.linspace(-2*max_r, 2*max_r, 25)
    z_coords = np.linspace(-1.5*d, 1.5*d, 25)

    # 2. 创建三维网格 Y, Z, Phi (用于后续计算)
    Y,Z,Phi = np.meshgrid(y_coords, z_coords, phi_angles)

    # 3. 计算到下方线圈 (r_low, 中心在 z=-d/2) 上各电流元的距离 dist1
    dist1 = np.sqrt((r_low * np.cos(Phi))**2 + (Y - r_low * np.sin(Phi))**2 + (Z - (-d/2))**2)

    # 4. 计算到上方线圈 (r_up, 中心在 z=+d/2) 上各电流元的距离 dist2
    dist2 = np.sqrt((r_up * np.cos(Phi))**2 + (Y - r_up * np.sin(Phi))**2 + (Z - (d/2))**2)

    # 5. 计算磁场贡献的被积函数 dBy_integrand 和 dBz_integrand
    dBy_integrand = r_low*(Z - (-d/2))*np.sin(Phi)/dist1**3 + r_up*(Z - (d/2))*np.sin(Phi)/dist2**3                
    dBz_integrand = r_low*(r_low - Y*np.sin(Phi))/dist1**3 + r_up*(r_up - Y*np.sin(Phi))/dist2**3

    # 6. 对 phi_angles 进行数值积分 (例如使用 np.trapezoid)
    By_unscaled = np.trapz(dBy_integrand,x=phi_angles,axis=-1)
    Bz_unscaled = np.trapz(dBz_integrand,x=phi_angles,axis=-1)

    # 7. 引入物理常数因子得到真实的磁场值 (单位 T)
    By = (MU0*I)/(4*np.pi)*By_unscaled
    Bz = (MU0*I)/(4*np.pi)*Bz_unscaled
    
    print("磁场计算完成.")
    # 返回用于绘图的2D网格 (取一个phi切片) 和计算得到的磁场分量
    return Y[:,:,0],Z[:,:,0],By,Bz


def plot_magnetic_field_streamplot(r_coil_1, r_coil_2, d_coils):
    """
    调用 Helmholtz_coils 计算磁场，并使用流线图可视化。
    """
    print(f"开始绘图准备: r_coil_1={r_coil_1}, r_coil_2={r_coil_2}, d_coils={d_coils}")
    # 1. 调用 Helmholtz_coils 函数获取磁场数据
    Y_plot,Z_plot,By_field,Bz_field = Helmholtz_coils(r_coil_1,r_coil_2,d_coils)

    if Y_plot is None: # 检查计算是否成功
        print("磁场数据未计算，无法绘图。")
        return

    plt.figure(figsize=(8, 7))

    # 2. (可选) 定义流线图的起始点，可以参考solution或自行设置
    #    y_start_coords = np.arange(-0.8*max(r_coil_1, r_coil_2), 0.85*max(r_coil_1, r_coil_2), 0.1*max(r_coil_1, r_coil_2))
    #    sy, sz = np.meshgrid(y_start_coords, 0) # 例如从z=0平面开始
    #    start_points = np.vstack([sy.ravel(), sz.ravel()]).T
    # 学生代码开始
    start_points = None # 占位符, 如果为None，streamplot会自动选择起点
    # 学生代码结束

    # 3. 使用 plt.streamplot 绘制磁场流线图
    plt.streamplot(Y_plot,Z_plot,By_field,Bz_field,
                   density=1.5,color='k',linewidth=1.0,
                   arrowstyle='->',arrowsize=1.0,start_points=start_points)

    # 4. 绘制线圈的截面位置 (用于参考)
    #下方线圈 (r_coil_1, z=-d_coils/2)
    plt.plot([-r_coil_1, -r_coil_1], [-d_coils/2-0.02, -d_coils/2+0.02], 'b-', linewidth=3) # 左边缘
    plt.plot([r_coil_1, r_coil_1], [-d_coils/2-0.02, -d_coils/2+0.02], 'b-', linewidth=3)   # 右边缘
    plt.text(0, -d_coils/2 - 0.1*max(r_coil_1,r_coil_2,d_coils), f'Coil 1 (R={r_coil_1})', color='blue', ha='center')
    #上方线圈 (r_coil_2, z=+d_coils/2)
    plt.plot([-r_coil_2, -r_coil_2], [d_coils/2-0.02, d_coils/2+0.02], 'r-', linewidth=3)
    plt.plot([r_coil_2, r_coil_2], [d_coils/2-0.02, d_coils/2+0.02], 'r-', linewidth=3)
    plt.text(0, d_coils/2 + 0.1*max(r_coil_1,r_coil_2,d_coils), f'Coil 2 (R={r_coil_2})', color='red', ha='center')

    # 5. 设置图形属性
    plt.xlabel('y / m')
    plt.ylabel('z / m')
    plt.title(f'Magnetic Field Lines (R1={r_coil_1}, R2={r_coil_2}, d={d_coils})')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    # 定义线圈参数 - 学生可以修改这些值进行测试
    # 标准亥姆霍兹线圈: r1 = r2 = R, d = R
    radius_1 = 0.5  # 下方线圈半径 (m)
    radius_2 = 0.5  # 上方线圈半径 (m)
    distance_between_coils = 0.5  # 两线圈中心距离 (m)

    # 调用绘图函数，该函数内部会调用计算函数
    plot_magnetic_field_streamplot(radius_1, radius_2, distance_between_coils)

    # 额外的测试用例 (可选)
    # print("\nTesting with different parameters (e.g., non-Helmholtz):")
    # plot_magnetic_field_streamplot(0.5, 0.5, 0.8)
    # plot_magnetic_field_streamplot(0.3, 0.7, 0.6)
