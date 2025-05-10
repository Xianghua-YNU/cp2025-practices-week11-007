# -*- coding: utf-8 -*-
"""
学生代码模板：计算伽马函数 Gamma(a)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial, sqrt, pi

# --- Task 1: 绘制被积函数 ---

def integrand_gamma(x, a):
    """
    伽马函数的原始被积函数: f(x, a) = x^(a-1) * exp(-x)
    使用对数技巧提高数值稳定性，尤其当 x 或 a 较大时。
    f = exp((a-1)*log(x) - x)
    """
    # 处理 x=0 的情况
    if x == 0:
        if a > 1:
            return 0.0
        elif a == 1:
            # 当 a=1, f(x,1) = exp(-x), 在 x=0 时为 1
            return 1.0
        else: # a < 1
            # 当 a<1, x^(a-1) 在 x=0 处发散
            return np.inf
    # 处理 x > 0 的情况
    elif x > 0:
        # 防止 log(0) 或负数
        try:
            # 使用对数避免直接计算大数的幂
            log_f = (a - 1) * np.log(x) - x
            return np.exp(log_f)
        except ValueError:
            # 如果 x 非常小导致 log(x) 问题（理论上不应发生，因已处理x=0）
            return 0.0 # 或根据情况返回 np.nan
    # 处理 x < 0 的情况 (积分区间是 [0, inf)，理论上不应输入负数)
    else:
        return 0.0 # 或者抛出错误

def plot_integrands():
    """绘制 a=2, 3, 4 时的被积函数图像"""
    x_vals = np.linspace(0.01, 10, 400) # 从略大于0开始，到10以看清下降趋势
    plt.figure(figsize=(10, 6))

    for a_val in [2, 3, 4]:
        # 计算 y 值，处理可能的 inf 或 nan
        y_vals = np.array([integrand_gamma(x, a_val) for x in x_vals])
        # 过滤掉 inf 值以便绘图
        valid_indices = np.isfinite(y_vals)
        plt.plot(x_vals[valid_indices], y_vals[valid_indices], label=f'$a = {a_val}$')

        # 标记理论峰值位置 x = a-1
        peak_x = a_val - 1
        if peak_x > 0: # 仅当峰值在绘制范围内时计算y值
             peak_y = integrand_gamma(peak_x, a_val)
             # 添加一个点标记峰值
             plt.plot(peak_x, peak_y, 'o', ms=5, label=f'Peak at x={peak_x}' if a_val==2 else None)


    plt.xlabel("$x$")
    plt.ylabel("$f(x, a) = x^{a-1} e^{-x}$")
    plt.title("Integrand of the Gamma Function")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Y轴从0开始
    plt.xlim(left=0)   # X轴从0开始
    plt.show()

# --- Task 2 & 3: 解析推导 (在注释中说明) ---
# Task 2: 峰值位置
# f(x, a) = x^(a-1) * exp(-x)
# ln(f) = (a-1)ln(x) - x
# d(ln(f))/dx = (a-1)/x - 1
# 令导数为 0: (a-1)/x - 1 = 0  => x = a-1 (假设 a > 1)
# 二阶导数 d^2(ln(f))/dx^2 = -(a-1)/x^2 < 0 (若 a > 1), 确认是最大值。

# Task 3: 变量代换 z = x/(c+x)
# 1. 若 z=1/2: 1/2 = x/(c+x) => c+x = 2x => x = c.
# 2. 使峰值 x=a-1 映射到 z=1/2: 需要 c = x = a-1.
# 这个代换和 c 的选择主要对 a > 1 的情况最有意义，此时峰值在 x > 0。

# --- Task 4: 实现伽马函数计算 ---

def transformed_integrand_gamma(z, a):
    """
    变换后的被积函数 g(z, a) = f(x(z), a) * dx/dz
    其中 x = cz / (1-z) 和 dx/dz = c / (1-z)^2, 且 c = a-1
    假设 a > 1
    """
    c = a - 1.0
    # 确保 c > 0，因为此变换是基于 a > 1 推导的
    if c <= 0:
        # 如果 a <= 1, 这个变换的推导基础（峰值在 a-1 > 0）不成立
        # 理论上应使用其他方法或原始积分。这里返回0或NaN，让外部处理。
        # 或者可以尝试用一个小的正数c，但这偏离了原意。
        # 返回 0 比较安全，避免在积分器中产生问题。
        return 0.0 # 或者 raise ValueError("Transformation assumes a > 1")

    # 处理 z 的边界情况
    if z < 0 or z > 1: # 积分区间外
        return 0.0
    if z == 1: # 对应 x = inf, 极限应为 0
        return 0.0
    if z == 0: # 对应 x = 0
        # 使用原始被积函数在 x=0 的行为
        return integrand_gamma(0, a) * c # dx/dz 在 z=0 时为 c

    # 计算 x 和 dx/dz
    x = c * z / (1.0 - z)
    dxdz = c / ((1.0 - z)**2)

    # 计算 f(x, a) * dx/dz
    # 使用原始被积函数（带对数优化）计算 f(x,a)
    val_f = integrand_gamma(x, a)

    # 检查计算结果是否有效
    if not np.isfinite(val_f) or not np.isfinite(dxdz):
        # 如果出现 inf 或 nan，可能表示数值问题或 a<=1 的情况处理不当
        return 0.0 # 返回0避免破坏积分

    return val_f * dxdz

def gamma_function(a):
    """
    计算 Gamma(a)
    - 如果 a > 1, 使用变量代换 z = x/(c+x) 和 c=a-1 进行数值积分。
    - 如果 a <= 1, 直接对原始被积函数进行积分（因为变换推导不适用）。
    使用 scipy.integrate.quad 进行积分。
    """
    if a <= 0:
        print(f"警告: Gamma(a) 对 a={a} <= 0 无定义 (或为复数)。")
        return np.nan

    try:
        if a > 1.0:
            # 使用变换后的积分，区间 [0, 1]
            result, error = quad(transformed_integrand_gamma, 0, 1, args=(a,))
        else:
            # 对于 a <= 1 (例如 a=1.5/2=0.75, 或 a=1), 变换的 c<=0, 推导失效
            # 直接积分原始函数，区间 [0, inf]
            # quad 对 x=0 处的奇异点 (当 a<1 时) 有较好的处理能力
            result, error = quad(integrand_gamma, 0, np.inf, args=(a,))

        # 可以检查一下积分误差 `error`，如果过大则给出警告
        # print(f"Integration error estimate for a={a}: {error}")
        return result

    except Exception as e:
        print(f"计算 Gamma({a}) 时发生错误: {e}")
        return np.nan

# --- 主程序 ---
def test_gamma():
    """测试伽马函数的计算结果"""
    # 测试Γ(3/2)
    a_test = 1.5
    result = gamma_function(a_test) # 使用 gamma_function 而不是 gamma
    expected = np.sqrt(np.pi) / 2  # 更精确的期望值
    relative_error = abs(result - expected) / expected if expected != 0 else 0
    print(f"Γ({a_test}) = {result:.8f} (精确值: {expected:.8f}, 相对误差: {relative_error:.2e})")

    # 测试整数值
    test_values = [3, 6, 10]
    print("\n测试整数值：")
    print("-" * 60)
    print("a\t计算值 Γ(a)\t精确值 (a-1)!\t相对误差")
    print("-" * 60)
    for a in test_values:
        result = gamma_function(a) # 使用 gamma_function 而不是 gamma
        # 使用 math.factorial 而不是 np.math.factorial
        factorial_val = float(factorial(a-1)) # 转换为浮点数以便计算误差
        relative_error = abs(result - factorial_val) / factorial_val if factorial_val != 0 else 0
        print(f"{a}\t{result:<12.6e}\t{factorial_val:<12.0f}\t{relative_error:.2e}")
    print("-" * 60)

def main():
    # 绘制原始被积函数
    plot_integrands() # 使用 plot_integrands 而不是 plot_integrand

    # 运行测试
    test_gamma()

if __name__ == '__main__':
    main()
