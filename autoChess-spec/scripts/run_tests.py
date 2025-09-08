#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本
"""

import sys
import subprocess
import os
from pathlib import Path
import argparse


def run_pytest(test_path: Path, coverage: bool = True, verbose: bool = True):
    """运行pytest测试"""
    cmd = [sys.executable, "-m", "pytest"]
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-v")
    
    cmd.append(str(test_path))
    
    print(f"运行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"错误: 运行测试失败: {e}")
        return False


def run_type_check(src_path: Path):
    """运行类型检查"""
    print("运行mypy类型检查...")
    cmd = [sys.executable, "-m", "mypy", str(src_path)]
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("✓ 类型检查通过")
            return True
        else:
            print("✗ 类型检查失败")
            return False
    except Exception as e:
        print(f"错误: 类型检查失败: {e}")
        return False


def run_code_style_check(src_path: Path, test_path: Path):
    """运行代码风格检查"""
    print("运行代码风格检查...")
    
    # Black格式检查
    black_cmd = [sys.executable, "-m", "black", "--check", str(src_path), str(test_path)]
    print(f"运行Black检查: {' '.join(black_cmd)}")
    
    try:
        result = subprocess.run(black_cmd, check=False)
        if result.returncode != 0:
            print("✗ Black格式检查失败")
            return False
    except Exception as e:
        print(f"错误: Black检查失败: {e}")
        return False
    
    # Flake8检查
    flake8_cmd = [sys.executable, "-m", "flake8", str(src_path), str(test_path)]
    print(f"运行Flake8检查: {' '.join(flake8_cmd)}")
    
    try:
        result = subprocess.run(flake8_cmd, check=False)
        if result.returncode != 0:
            print("✗ Flake8检查失败")
            return False
    except Exception as e:
        print(f"错误: Flake8检查失败: {e}")
        return False
    
    print("✓ 代码风格检查通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行测试和代码检查")
    parser.add_argument("--no-coverage", action="store_true", help="禁用覆盖率报告")
    parser.add_argument("--no-style", action="store_true", help="跳过代码风格检查")
    parser.add_argument("--no-type", action="store_true", help="跳过类型检查")
    parser.add_argument("--test-path", type=str, help="指定测试路径")
    parser.add_argument("--unit-only", action="store_true", help="仅运行单元测试")
    parser.add_argument("--integration-only", action="store_true", help="仅运行集成测试")
    
    args = parser.parse_args()
    
    print("中国象棋智能对弈助手 - 测试运行器")
    print("=" * 50)
    
    # 项目路径
    root_dir = Path(__file__).parent.parent
    src_path = root_dir / "src"
    test_path = root_dir / "tests"
    
    if args.test_path:
        test_path = Path(args.test_path)
    elif args.unit_only:
        test_path = root_dir / "tests" / "unit"
    elif args.integration_only:
        test_path = root_dir / "tests" / "integration"
    
    success = True
    
    # 运行测试
    if test_path.exists():
        print(f"运行测试: {test_path}")
        if not run_pytest(test_path, coverage=not args.no_coverage):
            success = False
    else:
        print(f"警告: 测试路径不存在: {test_path}")
    
    # 类型检查
    if not args.no_type and src_path.exists():
        if not run_type_check(src_path):
            success = False
    
    # 代码风格检查
    if not args.no_style:
        if not run_code_style_check(src_path, root_dir / "tests"):
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有检查通过!")
        return 0
    else:
        print("✗ 部分检查失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())