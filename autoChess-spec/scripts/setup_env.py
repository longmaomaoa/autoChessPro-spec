#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋智能对弈助手 - 智能环境设置脚本
"""

import sys
import subprocess
import os
import platform
import time
from pathlib import Path
from typing import List, Tuple, Optional


# 核心依赖包 (按安装顺序)
CORE_PACKAGES = [
    ("setuptools>=65.0", "Python打包工具"),
    ("wheel>=0.40.0", "打包格式支持"),
    ("numpy>=1.24.0", "科学计算基础库"),
    ("opencv-python>=4.8.0", "OpenCV图像处理库"),
    ("PyQt6>=6.5.0", "GUI框架"),
    ("pyyaml>=6.0", "YAML配置文件处理"),
    ("pillow>=10.0.0", "图像格式支持"),
    ("loguru>=0.7.0", "现代化日志库"),
]

# Windows专用依赖
WINDOWS_PACKAGES = [
    ("pywin32>=306", "Windows API支持"),
    ("dxcam>=0.0.5", "高性能屏幕捕获"),
]

# AI相关依赖
AI_PACKAGES = [
    ("torch>=2.0.0", "PyTorch深度学习框架"),
    ("torchvision>=0.15.0", "计算机视觉模型库"),
    ("ultralytics>=8.0.0", "YOLOv8目标检测框架"),
]

# 系统监控依赖
SYSTEM_PACKAGES = [
    ("psutil>=5.9.0", "系统资源监控"),
]

# 测试依赖
TEST_PACKAGES = [
    ("pytest>=7.0.0", "Python测试框架"),
    ("pytest-cov>=4.0.0", "测试覆盖率"),
    ("pytest-mock>=3.10.0", "Mock测试对象"),
    ("pytest-qt>=4.2.0", "PyQt测试支持"),
]

# 开发工具依赖
DEV_PACKAGES = [
    ("black>=23.0.0", "代码格式化工具"),
    ("flake8>=6.0.0", "代码风格检查"),
    ("mypy>=1.0.0", "静态类型检查"),
    ("isort>=5.12.0", "import语句排序"),
]


def print_banner():
    """显示横幅"""
    print("🏁 中国象棋智能对弈助手 - 智能环境设置")
    print("=" * 60)
    print(f"🖥️  操作系统: {platform.system()} {platform.release()}")
    print(f"🐍 Python版本: {sys.version.split()[0]}")
    print(f"📁 工作目录: {os.getcwd()}")
    print("=" * 60)


def check_system_requirements() -> bool:
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor} (需要 >= 3.8)")
        return False
    elif version >= (3, 12):
        print(f"⚠️  Python版本较新: {version.major}.{version.minor} (某些包可能不兼容)")
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    
    # 检查操作系统
    if platform.system() != "Windows":
        print(f"⚠️  当前系统: {platform.system()} (项目主要针对Windows优化)")
    
    # 检查是否为64位系统
    if platform.machine() not in ["AMD64", "x86_64"]:
        print(f"⚠️  系统架构: {platform.machine()} (推荐64位系统)")
    
    return True


def create_virtual_env(venv_path: Path) -> bool:
    """创建虚拟环境"""
    if venv_path.exists():
        print(f"✅ 虚拟环境已存在: {venv_path}")
        return True
    
    print(f"🔨 创建虚拟环境: {venv_path}")
    try:
        subprocess.run([
            sys.executable, "-m", "venv", str(venv_path)
        ], check=True, capture_output=True, text=True)
        print("✅ 虚拟环境创建成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 创建虚拟环境失败: {e}")
        print(f"stderr: {e.stderr}")
        return False


def get_pip_path(venv_path: Path) -> Path:
    """获取pip路径"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "pip.exe"
    else:  # Unix-like
        return venv_path / "bin" / "pip"


def upgrade_pip(pip_path: Path) -> bool:
    """升级pip"""
    print("📦 升级pip和基础工具...")
    try:
        subprocess.run([
            str(pip_path), "install", "--upgrade",
            "pip", "setuptools", "wheel"
        ], check=True, capture_output=True, text=True)
        print("✅ pip升级成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  pip升级失败: {e}")
        return False


def install_package_group(pip_path: Path, packages: List[Tuple[str, str]], 
                         group_name: str, optional: bool = False) -> bool:
    """安装包组"""
    print(f"📦 安装 {group_name}...")
    
    success_count = 0
    for package, description in packages:
        package_name = package.split(">=")[0].split("==")[0]
        print(f"  ⏳ 安装 {package_name}: {description}")
        
        try:
            result = subprocess.run([
                str(pip_path), "install", package, "--no-cache-dir"
            ], check=True, capture_output=True, text=True, timeout=300)
            
            print(f"  ✅ {package_name} 安装成功")
            success_count += 1
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {package_name} 安装超时")
            if not optional:
                return False
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {package_name} 安装失败: {e.returncode}")
            if e.stdout:
                print(f"     stdout: {e.stdout.strip()}")
            if e.stderr:
                print(f"     stderr: {e.stderr.strip()}")
            
            if not optional:
                return False
    
    if optional:
        print(f"✅ {group_name} 安装完成: {success_count}/{len(packages)} 成功")
        return True
    else:
        return success_count == len(packages)


def install_pytorch_with_options(pip_path: Path) -> bool:
    """智能安装PyTorch"""
    print("🤖 安装PyTorch (AI框架)...")
    
    # 检查是否有NVIDIA GPU
    has_cuda = False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            has_cuda = True
            print("🎮 检测到NVIDIA GPU，安装CUDA版本PyTorch")
        else:
            print("💻 未检测到NVIDIA GPU，安装CPU版本PyTorch")
    except FileNotFoundError:
        print("💻 未安装NVIDIA驱动，安装CPU版本PyTorch")
    
    try:
        if has_cuda:
            # GPU版本
            subprocess.run([
                str(pip_path), "install", 
                "torch>=2.0.0", "torchvision>=0.15.0",
                "--index-url", "https://download.pytorch.org/whl/cu118",
                "--no-cache-dir"
            ], check=True, timeout=600)
        else:
            # CPU版本
            subprocess.run([
                str(pip_path), "install",
                "torch>=2.0.0", "torchvision>=0.15.0", 
                "--index-url", "https://download.pytorch.org/whl/cpu",
                "--no-cache-dir"
            ], check=True, timeout=600)
        
        # 安装YOLOv8
        subprocess.run([
            str(pip_path), "install", "ultralytics>=8.0.0", "--no-cache-dir"
        ], check=True, timeout=300)
        
        print("✅ PyTorch和相关AI库安装成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch安装失败: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("⏰ PyTorch安装超时，请检查网络连接")
        return False


def verify_installation(venv_path: Path) -> bool:
    """验证安装"""
    print("🔍 验证安装...")
    
    if os.name == 'nt':
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    
    # 验证核心包
    verification_script = """
import sys
import traceback

def test_import(module_name, display_name):
    try:
        __import__(module_name)
        print(f'✅ {display_name}: OK')
        return True
    except ImportError as e:
        print(f'❌ {display_name}: {e}')
        return False
    except Exception as e:
        print(f'⚠️  {display_name}: {e}')
        return False

print('核心库验证:')
results = []
results.append(test_import('cv2', 'OpenCV'))
results.append(test_import('numpy', 'NumPy'))
results.append(test_import('PyQt6', 'PyQt6'))
results.append(test_import('yaml', 'PyYAML'))
results.append(test_import('PIL', 'Pillow'))
results.append(test_import('loguru', 'Loguru'))

if sys.platform == 'win32':
    print('\\nWindows专用库:')
    results.append(test_import('win32api', 'PyWin32'))
    results.append(test_import('dxcam', 'DXcam'))

print('\\nAI库验证:')
results.append(test_import('torch', 'PyTorch'))
results.append(test_import('torchvision', 'TorchVision'))
results.append(test_import('ultralytics', 'YOLOv8'))

print('\\n系统库验证:')
results.append(test_import('psutil', 'PSUtil'))

success_count = sum(results)
total_count = len(results)
print(f'\\n验证结果: {success_count}/{total_count} 成功')

if success_count >= total_count * 0.8:  # 80%成功率
    print('✅ 验证通过!')
    sys.exit(0)
else:
    print('❌ 验证失败!')
    sys.exit(1)
"""
    
    try:
        result = subprocess.run([
            str(python_path), "-c", verification_script
        ], check=True, text=True, timeout=60)
        return True
    except subprocess.CalledProcessError:
        print("❌ 安装验证失败")
        return False
    except subprocess.TimeoutExpired:
        print("⏰ 验证超时")
        return False


def create_activation_scripts(venv_path: Path, root_dir: Path):
    """创建激活脚本"""
    print("📝 创建便捷脚本...")
    
    # Windows激活脚本
    if os.name == 'nt':
        activate_script = root_dir / "activate.bat"
        with open(activate_script, 'w', encoding='utf-8') as f:
            f.write(f'''@echo off
echo 🏁 激活中国象棋智能对弈助手开发环境...
call "{venv_path}\\Scripts\\activate.bat"
echo ✅ 环境已激活!
echo.
echo 📖 常用命令:
echo   python -m chess_ai.main           # 运行应用
echo   python tests/test_basic.py        # 运行基础测试
echo   pytest tests/ -v                  # 运行完整测试
echo   black src/                        # 格式化代码
echo.
''')
        print(f"✅ 创建激活脚本: {activate_script}")
    
    # 项目启动脚本
    start_script = root_dir / ("start.bat" if os.name == 'nt' else "start.sh")
    if os.name == 'nt':
        with open(start_script, 'w', encoding='utf-8') as f:
            f.write(f'''@echo off
echo 🚀 启动中国象棋智能对弈助手...
call "{venv_path}\\Scripts\\activate.bat"
python -m chess_ai.main
pause
''')
    else:
        with open(start_script, 'w', encoding='utf-8') as f:
            f.write(f'''#!/bin/bash
echo "🚀 启动中国象棋智能对弈助手..."
source "{venv_path}/bin/activate"
python -m chess_ai.main
''')
        start_script.chmod(0o755)
    
    print(f"✅ 创建启动脚本: {start_script}")


def main():
    """主函数"""
    print_banner()
    time.sleep(1)
    
    # 项目根目录
    root_dir = Path(__file__).parent.parent
    venv_path = root_dir / ".venv"  # 使用.venv而不是venv
    
    # 检查系统要求
    if not check_system_requirements():
        print("❌ 系统要求检查失败")
        return 1
    
    time.sleep(1)
    
    # 创建虚拟环境
    if not create_virtual_env(venv_path):
        print("❌ 虚拟环境创建失败")
        return 1
    
    # 获取pip路径
    pip_path = get_pip_path(venv_path)
    if not pip_path.exists():
        print(f"❌ pip不存在: {pip_path}")
        return 1
    
    # 升级pip
    upgrade_pip(pip_path)
    time.sleep(1)
    
    # 分组安装依赖
    installation_steps = [
        (CORE_PACKAGES, "核心依赖", False),
        (SYSTEM_PACKAGES, "系统监控库", False),
    ]
    
    # Windows专用依赖
    if platform.system() == "Windows":
        installation_steps.append((WINDOWS_PACKAGES, "Windows专用库", False))
    
    # 安装各组依赖
    for packages, name, optional in installation_steps:
        if not install_package_group(pip_path, packages, name, optional):
            if not optional:
                print(f"❌ {name} 安装失败")
                return 1
        time.sleep(0.5)
    
    # 智能安装PyTorch
    if not install_pytorch_with_options(pip_path):
        print("⚠️  PyTorch安装失败，但继续安装其他依赖...")
    time.sleep(1)
    
    # 安装测试和开发依赖 (可选)
    install_package_group(pip_path, TEST_PACKAGES, "测试框架", optional=True)
    install_package_group(pip_path, DEV_PACKAGES, "开发工具", optional=True)
    
    # 验证安装
    if verify_installation(venv_path):
        print("✅ 所有核心组件验证通过")
    else:
        print("⚠️  部分组件验证失败，但可以继续使用")
    
    # 创建便捷脚本
    create_activation_scripts(venv_path, root_dir)
    
    # 完成提示
    print("\n" + "🎉" * 20)
    print("✅ 中国象棋智能对弈助手环境设置完成!")
    print("🎉" * 20)
    
    print(f"\n📁 虚拟环境位置: {venv_path}")
    print("\n🚀 快速开始:")
    if os.name == 'nt':
        print("  1. 双击 activate.bat 激活环境")
        print("  2. 双击 start.bat 启动应用")
        print("  或手动激活:")
        print(f"     {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    
    print("\n📖 常用命令:")
    print("  python -m chess_ai.main           # 运行应用")
    print("  python tests/test_basic.py        # 运行基础测试")
    print("  pytest tests/ -v                  # 运行完整测试")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())