import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 假设这些是要测试的函数
def integrand(x):
    """德拜模型的积分函数"""
    if isinstance(x, np.ndarray):
        # 处理数组输入
        result = np.zeros_like(x)
        mask = x > 1e-10  # 避免除零
        result[mask] = (x[mask]**4 * np.exp(x[mask])) / (np.exp(x[mask]) - 1)**2
        result[~mask] = x[~mask]**4  # 小x的近似
        return result
    else:
        # 处理标量输入
        if x > 1e-10:
            return (x**4 * np.exp(x)) / (np.exp(x) - 1)**2
        else:
            return x**4

def gauss_quadrature(f, a, b, n):
    """使用高斯求积法计算定积分"""
    # 使用numpy的高斯求积点和权重
    x, w = np.polynomial.legendre.leggauss(n)
    # 变换区间
    t = 0.5 * (b - a) * x + 0.5 * (b + a)
    return 0.5 * (b - a) * np.sum(w * f(t))

def cv(T, T_D=300):
    """计算德拜模型的热容"""
    if T <= 0:
        return 0
    x_D = T_D / T
    integral = gauss_quadrature(integrand, 0, x_D, 50)
    return 9 * 8.314 * (T / T_D)**3 * integral

def plot_cv():
    """绘制热容随温度的变化"""
    T = np.linspace(5, 500, 200)
    Cv = [cv(t) for t in T]
    
    plt.figure(figsize=(10, 6))
    plt.plot(T, Cv, 'b-', linewidth=2)
    plt.title('德拜模型热容 vs 温度')
    plt.xlabel('温度 (K)')
    plt.ylabel('热容 (J/(mol·K))')
    plt.grid(True)
    plt.show()

# 测试函数
def test_integrand():
    """测试被积函数的计算"""
    # 测试单个值
    assert abs(integrand(1.0) - 0.9206735942077924) < 1e-10
    
    # 测试数组输入
    x = np.array([0.1, 1.0, 2.0])
    result = integrand(x)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    
    # 测试小值的处理
    assert abs(integrand(0.001) - 0.001**4) < 1e-6

def test_cv():
    """测试热容的计算"""
    # 测试低温行为（应该符合T³定律）
    T1, T2 = 5, 10
    ratio = cv(T2) / cv(T1)
    expected_ratio = (T2/T1)**3
    assert abs(ratio - expected_ratio) < 0.1
    
    # 测试高温行为（应该趋近于常数）
    high_T = [400, 450, 500]
    values = [cv(T) for T in high_T]
    variations = np.diff(values) / values[:-1]
    assert np.all(abs(variations) < 0.02)

def test_gauss_quadrature():
    """测试高斯积分的实现"""
    # 测试简单函数的积分
    f = lambda x: x**2
    result = gauss_quadrature(f, 0, 1, 10)
    assert abs(result - 1/3) < 1e-10
    
    # 测试指数函数的积分
    f = lambda x: np.exp(-x)
    result = gauss_quadrature(f, 0, 1, 10)
    assert abs(result - (1 - np.exp(-1))) < 1e-10

def test_plot_cv(monkeypatch):
    """测试绘图函数（不实际显示）"""
    # 使用monkeypatch避免实际显示图像
    monkeypatch.setattr(plt, 'show', lambda: None)
    plot_cv()

def test_physical_constraints():
    """测试物理约束条件"""
    # 热容应该始终为正
    temperatures = np.linspace(5, 500, 20)
    for T in temperatures:
        assert cv(T) > 0
    
    # 热容应该随温度增加而增加（在低温区域）
    T1, T2 = 10.0, 50.0
    assert cv(T1) < cv(T2)
    
    # 高温下热容应该趋于定值（杜隆-珀替定律）
    cv_400 = cv(400.0)
    cv_500 = cv(500.0)
    assert abs(cv_400 - cv_500) / cv_400 < 0.1

if __name__ == '__main__':
    pytest.main([__file__])    
