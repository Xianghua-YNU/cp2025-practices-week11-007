import os
import sys
import json
import subprocess
import pytest
from pathlib import Path
import shutil
import platform

# 更新测试配置以匹配当前5个实验项目
TESTS = [
    {"name": "实验一: 德拜热容", 
     "file": "Exp1-德拜热容/tests/test_debye.py", 
     "points": 10},
    {"name": "实验二: 伽马函数积分", 
     "file": "Exp2-伽马函数/tests/test_gamma_function.py", 
     "points": 10},
    {"name": "实验三: 带电圆环电势", 
     "file": "Exp3-带电圆环/tests/test_ring.py", 
     "points": 10},
    {"name": "实验四: 亥姆霍兹线圈", 
     "file": "Exp4-亥姆霍兹线圈/tests/test_helmholtz.py", 
     "points": 10},
    {"name": "实验五: 均匀薄片引力", 
     "file": "Exp5-均匀薄片引力/tests/test_gravity.py", 
     "points": 10}
]

def run_test(test_file):
    """运行单个测试文件并返回结果"""
    print(f"执行测试文件: {test_file}")
    
    # 创建临时目录存储测试结果
    test_dir = Path(test_file).parent
    temp_dir = test_dir / "temp_test_results"
    temp_dir.mkdir(exist_ok=True)
    
    # 构建pytest命令
    junit_file = temp_dir / "test-results.xml"
    cmd = [
        "pytest", 
        "-v", 
        "--junitxml", str(junit_file),
        "--html", str(temp_dir / "test-report.html"),
        test_file
    ]
    
    # 执行测试
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 解析测试结果
    passed = result.returncode == 0
    
    # 打印测试输出
    print("标准输出:")
    print(result.stdout)
    
    if result.stderr:
        print("标准错误:")
        print(result.stderr)
    
    # 清理临时目录
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"警告: 无法清理临时目录 {temp_dir}: {e}")
    
    return passed

def calculate_score():
    """计算总分并生成结果报告"""
    total_points = 0
    max_points = 0
    results = []
    
    print("\n开始执行测试...\n")
    
    for test in TESTS:
        max_points += test["points"]
        test_file = test["file"]
        test_name = test["name"]
        points = test["points"]
        
        print(f"运行测试: {test_name}")
        
        # 检查测试文件是否存在
        if not Path(test_file).exists():
            print(f"错误: 测试文件 {test_file} 不存在")
            status = "测试文件缺失"
            results.append({
                "name": test_name,
                "status": status,
                "points": 0,
                "max_points": points,
                "details": f"测试文件 {test_file} 不存在"
            })
            print(f"  状态: {status}")
            print(f"  得分: 0/{points}")
            print()
            continue
        
        passed = run_test(test_file)
        
        if passed:
            total_points += points
            status = "通过"
        else:
            status = "失败"
        
        results.append({
            "name": test_name,
            "status": status,
            "points": points if passed else 0,
            "max_points": points
        })
        
        print(f"  状态: {status}")
        print(f"  得分: {points if passed else 0}/{points}")
        print()
    
    # 生成总结
    print(f"总分: {total_points}/{max_points}")
    
    # 生成GitHub Actions兼容的输出
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY', 'score_summary.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# 自动评分结果\n\n")
        f.write("| 实验名称 | 状态 | 得分 |\n")
        f.write("|----------|------|------|\n")
        
        for result in results:
            f.write(f"| {result['name']} | {result['status']} | {result['points']}/{result['max_points']} |\n")
        
        f.write(f"\n## 总分: {total_points}/{max_points}\n")
    
    # 生成分数JSON文件
    score_data = {
        "score": total_points,
        "max_score": max_points,
        "tests": results
    }
    
    with open('score.json', 'w', encoding='utf-8') as f:
        json.dump(score_data, f, indent=2, ensure_ascii=False)
    
    return total_points, max_points

def setup_environment():
    """设置测试环境"""
    print("设置测试环境...")
    
    # 检查操作系统
    os_name = platform.system()
    print(f"运行环境: {os_name}")
    
    # 确保工作目录是项目根目录
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 安装依赖
    print("安装依赖...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("依赖安装成功")
    except subprocess.CalledProcessError as e:
        print(f"错误: 依赖安装失败: {e}")
        sys.exit(1)
    
    # 验证pytest安装
    try:
        import pytest
        print(f"pytest版本: {pytest.__version__}")
    except ImportError:
        print("错误: pytest未安装")
        sys.exit(1)

if __name__ == "__main__":
    # 设置环境
    setup_environment()
    
    # 运行测试并计算分数
    print("\n开始评分...\n")
    total, maximum = calculate_score()
    
    # 设置GitHub Actions输出变量
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"points={total}\n")
    
    # 退出代码 - 如果所有测试通过返回0，否则返回1
    exit_code = 0 if total == maximum else 1
    print(f"\n评分完成，退出代码: {exit_code}")
    sys.exit(exit_code)    
