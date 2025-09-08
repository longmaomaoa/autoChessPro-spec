#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化环境设置脚本 - 解决编码和兼容性问题
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("错误: 需要Python 3.8+")
        return False
    elif version >= (3, 13):
        print("警告: Python 3.13较新，某些包可能不兼容，建议使用Python 3.10-3.11")
    
    return True

def create_virtual_env(venv_path: Path):
    """创建虚拟环境"""
    if venv_path.exists():
        print(f"虚拟环境已存在: {venv_path}")
        return True
    
    print(f"创建虚拟环境: {venv_path}")
    try:
        subprocess.run([
            sys.executable, "-m", "venv", str(venv_path)
        ], check=True)
        print("虚拟环境创建成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"创建虚拟环境失败: {e}")
        return False

def get_pip_path(venv_path: Path):
    """获取pip路径"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "pip.exe"
    else:  # Unix-like
        return venv_path / "bin" / "pip"

def install_minimal_deps(pip_path: Path):
    """安装最小化依赖"""
    print("升级pip...")
    try:
        subprocess.run([
            str(pip_path), "install", "--upgrade", "pip"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"升级pip失败: {e}")
    
    # 核心依赖列表
    core_packages = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0", 
        "PyQt6>=6.5.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
        "loguru>=0.7.0",
        "psutil>=5.9.0"
    ]
    
    # Windows专用包
    if platform.system() == "Windows":
        core_packages.extend([
            "pywin32>=306",
            "dxcam>=0.0.5"
        ])
    
    print("安装核心依赖...")
    success_count = 0
    
    for package in core_packages:
        package_name = package.split(">=")[0]
        print(f"  安装 {package_name}...")
        
        try:
            subprocess.run([
                str(pip_path), "install", package, "--no-cache-dir"
            ], check=True, timeout=300)
            print(f"  成功: {package_name}")
            success_count += 1
        except subprocess.TimeoutExpired:
            print(f"  超时: {package_name}")
        except subprocess.CalledProcessError as e:
            print(f"  失败: {package_name} - {e}")
    
    print(f"核心依赖安装完成: {success_count}/{len(core_packages)} 成功")
    return success_count >= len(core_packages) * 0.8  # 80%成功率

def install_pytorch(pip_path: Path):
    """安装PyTorch"""
    print("安装PyTorch...")
    
    # 检查GPU
    has_cuda = False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            has_cuda = True
            print("检测到NVIDIA GPU")
    except FileNotFoundError:
        print("未检测到NVIDIA GPU，安装CPU版本")
    
    try:
        if has_cuda:
            # GPU版本
            subprocess.run([
                str(pip_path), "install", 
                "torch>=2.0.0", "torchvision>=0.15.0",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True, timeout=600)
        else:
            # CPU版本
            subprocess.run([
                str(pip_path), "install",
                "torch>=2.0.0", "torchvision>=0.15.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], check=True, timeout=600)
        
        # YOLOv8
        subprocess.run([
            str(pip_path), "install", "ultralytics>=8.0.0"
        ], check=True, timeout=300)
        
        print("PyTorch安装成功")
        return True
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"PyTorch安装失败: {e}")
        return False

def verify_installation(venv_path: Path):
    """验证安装"""
    print("验证安装...")
    
    python_path = venv_path / ("Scripts/python.exe" if os.name == 'nt' else "bin/python")
    
    test_script = '''
try:
    import cv2
    print("OpenCV: OK")
except ImportError as e:
    print(f"OpenCV: Failed - {e}")

try:
    import numpy
    print("NumPy: OK")
except ImportError as e:
    print(f"NumPy: Failed - {e}")

try:
    import PyQt6
    print("PyQt6: OK")
except ImportError as e:
    print(f"PyQt6: Failed - {e}")

try:
    import yaml
    print("PyYAML: OK")
except ImportError as e:
    print(f"PyYAML: Failed - {e}")

import sys
if sys.platform == "win32":
    try:
        import dxcam
        print("DXcam: OK")
    except ImportError as e:
        print(f"DXcam: Failed - {e}")
    
    try:
        import win32api
        print("PyWin32: OK")
    except ImportError as e:
        print(f"PyWin32: Failed - {e}")

try:
    import torch
    print("PyTorch: OK")
except ImportError as e:
    print(f"PyTorch: Failed - {e}")

try:
    import ultralytics
    print("YOLOv8: OK")
except ImportError as e:
    print(f"YOLOv8: Failed - {e}")
'''
    
    try:
        subprocess.run([str(python_path), "-c", test_script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("验证失败")
        return False

def create_activate_script(venv_path: Path, root_dir: Path):
    """创建激活脚本"""
    if os.name == 'nt':
        activate_script = root_dir / "activate.bat"
        with open(activate_script, 'w', encoding='utf-8') as f:
            f.write(f'''@echo off
echo 激活中国象棋智能对弈助手开发环境...
call "{venv_path}\\Scripts\\activate.bat"
echo 环境已激活!
echo.
echo 常用命令:
echo   python -m chess_ai.main           # 运行应用
echo   python tests/test_basic.py        # 运行基础测试
echo   pytest tests/ -v                  # 运行完整测试
echo.
''')
        print(f"创建激活脚本: {activate_script}")

def main():
    """主函数"""
    print("中国象棋智能对弈助手 - 简化环境设置")
    print("=" * 50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    # 检查Python版本
    if not check_python_version():
        return 1
    
    # 项目根目录
    root_dir = Path(__file__).parent.parent
    venv_path = root_dir / ".venv"
    
    print(f"项目目录: {root_dir}")
    print(f"虚拟环境: {venv_path}")
    
    # 创建虚拟环境
    if not create_virtual_env(venv_path):
        return 1
    
    # 获取pip路径
    pip_path = get_pip_path(venv_path)
    if not pip_path.exists():
        print(f"pip不存在: {pip_path}")
        return 1
    
    # 安装核心依赖
    if not install_minimal_deps(pip_path):
        print("核心依赖安装失败")
        return 1
    
    # 安装PyTorch (可选)
    install_pytorch(pip_path)
    
    # 验证安装
    verify_installation(venv_path)
    
    # 创建便捷脚本
    create_activate_script(venv_path, root_dir)
    
    print("\n" + "=" * 50)
    print("环境设置完成!")
    print(f"\n虚拟环境位置: {venv_path}")
    
    if os.name == 'nt':
        print("\n激活环境:")
        print("  方法1: 双击 activate.bat")
        print(f"  方法2: {venv_path}\\Scripts\\activate")
    else:
        print(f"\n激活环境: source {venv_path}/bin/activate")
    
    print("\n测试安装:")
    print("  python tests/test_screen_capture_basic.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())