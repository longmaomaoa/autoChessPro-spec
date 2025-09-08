# 中国象棋智能对弈助手 - 安装指南

本文档提供详细的环境配置和依赖安装步骤。

## 系统要求

### 硬件要求
- **处理器**: Intel i5-8400 或 AMD Ryzen 5 2600 及以上
- **内存**: 8GB RAM (推荐 16GB)
- **显卡**: 支持DirectX 11的独立显卡 (可选，用于加速)
- **存储**: 2GB 可用空间
- **显示器**: 支持1920x1080分辨率

### 软件要求
- **操作系统**: Windows 10/11 (64位) - **必需**
- **Python**: 3.8 - 3.11 (推荐 3.10)
- **Visual C++**: Microsoft Visual C++ 14.0 或更高版本

## 快速安装

### 1. 安装Python
从 [python.org](https://www.python.org/downloads/) 下载Python 3.10：
```bash
# 验证Python安装
python --version
pip --version
```

### 2. 克隆项目
```bash
git clone <项目地址>
cd autoChess-spec
```

### 3. 创建虚拟环境
```bash
# 使用项目提供的脚本
python scripts/setup_env.py

# 或手动创建
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 4. 安装依赖
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装核心依赖
pip install -r requirements.txt

# 验证安装
python -c "import cv2, numpy, dxcam; print('Core dependencies OK')"
```

## 详细安装步骤

### Step 1: Python环境准备

#### 1.1 安装Python 3.10
```bash
# 下载并安装Python 3.10.x from python.org
# 安装时勾选 "Add Python to PATH"
# 安装时勾选 "Install pip"

# 验证安装
python --version    # 应该显示 Python 3.10.x
pip --version      # 应该显示pip版本
```

#### 1.2 安装Visual C++构建工具 (Windows)
```bash
# 下载 Microsoft Visual C++ Build Tools
# 或安装 Visual Studio Community (包含构建工具)
# 某些Python包(如dxcam)编译时需要
```

### Step 2: 项目环境配置

#### 2.1 创建项目目录
```bash
cd C:\Users\<用户名>\
mkdir Projects
cd Projects
git clone <项目仓库地址> autoChess-spec
cd autoChess-spec
```

#### 2.2 自动环境设置
```bash
# 使用项目提供的自动化脚本
python scripts/setup_env.py
```

#### 2.3 手动环境设置 (可选)
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 升级pip
python -m pip install --upgrade pip

# 安装wheel (提高安装速度)
pip install wheel
```

### Step 3: 依赖安装

#### 3.1 核心依赖安装
```bash
# 确保虚拟环境已激活
.venv\Scripts\activate

# 安装核心依赖 (顺序安装以避免冲突)
pip install numpy>=1.24.0
pip install opencv-python>=4.8.0
pip install PyQt6>=6.5.0
pip install pyyaml>=6.0
pip install pillow>=10.0.0
```

#### 3.2 屏幕捕获依赖
```bash
# Windows专用屏幕捕获库
pip install dxcam>=0.0.5
pip install pywin32>=306

# 验证dxcam安装
python -c "import dxcam; print('DXcam installed successfully')"
```

#### 3.3 AI和机器学习依赖
```bash
# PyTorch (选择适合的版本)
# CPU版本 (较小)
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# GPU版本 (如果有NVIDIA GPU)
# pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# YOLOv8目标检测
pip install ultralytics>=8.0.0
```

#### 3.4 系统监控依赖
```bash
pip install psutil>=5.9.0
pip install loguru>=0.7.0
```

#### 3.5 测试框架依赖
```bash
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0
pip install pytest-mock>=3.10.0
pip install pytest-qt>=4.2.0
```

#### 3.6 开发工具依赖 (可选)
```bash
pip install black>=23.0.0
pip install flake8>=6.0.0
pip install mypy>=1.0.0
pip install isort>=5.12.0
```

#### 3.7 一键安装所有依赖
```bash
# 安装完整依赖列表
pip install -r requirements.txt

# 如果遇到安装失败，可以分组安装
pip install -r requirements.txt --no-deps  # 跳过依赖检查
```

### Step 4: 验证安装

#### 4.1 基础功能验证
```bash
# 激活环境
.venv\Scripts\activate

# 验证核心库
python -c "
import cv2
import numpy as np  
import dxcam
import yaml
from PyQt6 import QtCore
print('All core dependencies imported successfully!')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')
print(f'PyQt6: {QtCore.PYQT_VERSION_STR}')
"
```

#### 4.2 屏幕捕获验证
```bash
# 测试屏幕捕获功能
python -c "
import dxcam
camera = dxcam.create()
if camera:
    frame = camera.grab()
    if frame is not None:
        print(f'Screen capture successful! Frame shape: {frame.shape}')
        camera.release()
    else:
        print('Failed to capture frame')
else:
    print('Failed to create DXcam instance')
"
```

#### 4.3 运行测试套件
```bash
# 运行基础测试
python tests/test_screen_capture_basic.py

# 运行完整测试 (需要完整环境)
pytest tests/ -v
```

## 常见问题解决

### 问题1: DXcam安装失败
```bash
# 解决方案1: 升级pip和setuptools
python -m pip install --upgrade pip setuptools wheel

# 解决方案2: 使用预编译版本
pip install dxcam --prefer-binary

# 解决方案3: 从源码安装
pip install git+https://github.com/ra1nty/DXcam.git
```

### 问题2: PyTorch安装缓慢
```bash
# 使用国内镜像源
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或使用清华镜像
pip install torch torchvision -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题3: PyQt6安装问题
```bash
# 确保系统有必要的C++运行库
# 安装Visual C++ Redistributable

# 尝试不同版本
pip install PyQt6==6.5.0
```

### 问题4: 权限问题
```bash
# 以管理员权限运行命令提示符
# 或使用用户级安装
pip install --user <包名>
```

### 问题5: 内存不足
```bash
# 分批安装大包
pip install torch --no-cache-dir
pip install torchvision --no-cache-dir
```

## 性能优化建议

### GPU加速配置
```bash
# 检查NVIDIA GPU支持
nvidia-smi

# 安装CUDA版本PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 内存优化
```bash
# 设置环境变量减少内存使用
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set OPENCV_IO_MAX_IMAGE_PIXELS=1073741824
```

## 开发环境设置

### IDE配置
推荐使用以下IDE之一：
- **VS Code** + Python扩展
- **PyCharm Professional**
- **Visual Studio** + Python Tools

### VS Code配置
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true
}
```

### Git配置
```bash
# 配置Git忽略文件
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".pytest_cache/" >> .gitignore
```

## 部署准备

### 创建可执行文件
```bash
# 安装打包工具
pip install pyinstaller

# 创建可执行文件
pyinstaller --onefile --windowed src/chess_ai/main.py
```

### 创建安装包
```bash
# 安装NSIS或Inno Setup
# 创建Windows安装程序
```

## 故障排除

### 日志调试
```bash
# 启用详细日志
set PYTHONPATH=src
python -m chess_ai.main --debug

# 查看日志文件
type logs\chess_ai.log
```

### 环境重置
```bash
# 完全重置环境
rmdir /s .venv
python scripts/setup_env.py
```

### 联系支持
如果遇到无法解决的问题：
1. 查看项目Wiki
2. 提交GitHub Issue
3. 查看常见问题FAQ

---

**注意**: 本项目主要针对Windows系统优化，Linux/macOS支持有限。