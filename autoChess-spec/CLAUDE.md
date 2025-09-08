# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个中国象棋智能对弈助手项目，使用Python开发，具备AI分析、计算机视觉识别和桌面用户界面功能。项目采用分层架构设计，结合深度学习和传统象棋引擎。

### 当前开发状态 (2025-09-08)
- ✅ **基础架构**: 项目结构、数据模型、配置系统 (任务1-3完成)
- ✅ **屏幕捕获**: DXcam高性能捕获、多显示器支持 (任务4-5完成)  
- ✅ **棋盘检测**: 多算法检测、透视变换、坐标映射 (任务6完成)
- 🚧 **计算机视觉**: 棋子识别、状态监控 (任务7-9开发中)
- 📋 **AI引擎**: Pikafish集成、走法分析 (任务10-13待开发)
- 📋 **用户界面**: PyQt6界面开发 (任务14-17待开发)

### 项目结构
```
autoChess-spec/
├── src/chess_ai/           # 主要源码目录
│   ├── ai/                 # AI算法模块(新增，未在架构图中体现)
│   ├── ai_engine/          # AI引擎层 - pikafish引擎集成
│   ├── config/             # 配置管理层 - YAML配置处理  
│   ├── core/               # 核心业务层 - 象棋规则和棋局状态
│   ├── data/               # 数据管理层 - 棋谱数据处理
│   ├── ui/                 # 用户界面层 - PyQt6桌面界面
│   ├── utils/              # 工具类层 - 性能监控和公用工具
│   ├── vision/             # 视觉识别层 - 屏幕捕获和棋盘识别
│   └── main.py             # 应用程序入口点
├── tests/                  # 测试目录
├── scripts/                # 脚本目录(环境设置、测试运行等)
├── docs/                   # 文档目录
├── .claude/                # Claude开发配置
├── config/                 # 应用配置文件(YAML格式)
├── pyproject.toml          # Python项目配置(包含pytest、mypy、black配置)
├── setup.py                # 安装脚本
├── requirements.txt        # 依赖定义
└── activate.bat            # Windows环境激活脚本
```

## 常用开发命令

### 环境设置

#### 开发要求
- 基于 `.claude/design.md` `.claude/requirements.md ` 开发整个项目
- 在完成task之后，需要将当前开发进度同步到`.claude/tasks.md`文档中
- 在测试完成之后需要将测试结果同步更新到`docs/testResult.md`中
- 在引入新的依赖之前需要先进行验证当前环境是否有相关依赖，如果没有则再引入，同事更新到`requirements.txt`中

#### 自动化安装 (推荐)
```bash
# 一键智能环境设置
python scripts/setup_env.py

# Windows便捷脚本 (安装后可用)
activate.bat        # 激活开发环境
start.bat          # 启动应用程序
```

#### 手动安装
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖 (完整版)
pip install -r requirements.txt

# 安装依赖 (最小版)
pip install -r requirements-minimal.txt

# 开发模式安装
pip install -e .
```

#### 依赖说明
- `requirements.txt` - 完整依赖包，包含AI、测试、开发工具等
- `requirements-minimal.txt` - 最小化依赖，仅运行核心功能
- 屏幕捕获功能需要Windows系统和DXcam库
- AI功能需要PyTorch和Ultralytics (YOLOv8)
- 详细安装指南见 `docs/INSTALL.md`

### 运行应用
```bash
# 通过入口点运行
chess-ai

# 或直接运行主模块  
python -m chess_ai.main
```

### 测试命令
```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term

# 运行特定类型的测试
pytest -m unit      # 单元测试
pytest -m integration  # 集成测试
pytest -m slow      # 慢速测试

# 运行单个测试文件
pytest tests/unit/test_specific.py

# 使用测试运行器脚本
python scripts/run_tests.py
```

### 代码质量检查
```bash
# 代码格式化 (自动修复)
black src/ tests/

# 类型检查 (严格模式,配置在pyproject.toml中)
mypy src/

# 代码风格检查
flake8 src/ tests/

# import语句排序
isort src/ tests/

# 安全漏洞检查
bandit -r src/

# 完整代码质量检查 (组合命令)
black src/ tests/ && isort src/ tests/ && mypy src/ && flake8 src/ tests/
```

## 核心架构

### 分层架构设计
项目采用分层架构模式，各层职责清晰：

```
src/chess_ai/
├── main.py         # 应用程序入口点 - 应用启动和初始化
├── ui/             # 用户界面层 - PyQt6桌面界面
├── ai_engine/      # AI引擎层 - pikafish引擎集成，UCCI协议
├── ai/             # AI算法层 - 独立AI算法实现(实际存在但未在原架构中体现)
├── vision/         # 视觉识别层 - 屏幕捕获和棋盘识别  
├── core/           # 核心业务层 - 象棋规则和棋局状态管理
├── data/           # 数据管理层 - 棋谱数据处理
├── config/         # 配置管理层 - YAML配置处理
└── utils/          # 工具类层 - 性能监控和公用工具
```

### 关键模块依赖关系
- **main.py** → **core.application** (应用主控制器)
- **ui层** ↔ **core层** (界面与业务逻辑交互)  
- **ai_engine层** ↔ **core层** (AI引擎与棋局状态交互)
- **vision层** → **core层** (视觉识别结果传递给核心层)
- **所有层** → **config层** (统一配置管理)
- **所有层** → **utils层** (公共工具和监控)

### 关键组件说明

- **Application 类**: MVC模式的主控制器，协调各个模块
- **BoardState**: 象棋棋局状态管理，实现象棋规则逻辑
- **AI引擎接口**: 与pikafish等引擎通信，遵循UCCI协议
- **视觉识别系统**: 使用OpenCV + YOLO进行屏幕捕获和棋子识别
- **PyQt6界面**: 主窗口 + 分析面板 + 控制面板 + 状态栏

### 技术栈核心

- **GUI框架**: PyQt6 6.5.0+ - 桌面用户界面
- **计算机视觉**: OpenCV 4.8.0+ - 图像处理和棋盘识别
- **AI/深度学习**: PyTorch 2.0.0+ + Ultralytics 8.0.0+ (YOLO)
- **屏幕捕获**: dxcam 0.0.5+ - 高性能屏幕捕获
- **日志系统**: loguru 0.7.0+ - 现代化日志处理

### 配置管理
- **格式**: YAML格式，存放在项目根目录 `config/` 目录
- **管理**: 通过 `src/chess_ai/config/` 模块统一管理应用配置
- **功能**: 支持不同环境的配置切换、运行时配置更新
- **配置项**: 包含UI设置、AI引擎参数、视觉识别参数、性能监控设置等

### 打包和分发
- **项目配置**: 使用 `pyproject.toml` 作为主配置文件(现代Python标准)
- **兼容配置**: `setup.py` 提供传统安装方式兼容
- **依赖管理**: `requirements.txt` 定义完整依赖列表
- **安装模式**: 支持开发模式安装 (`pip install -e .`)
- **入口点**: 通过 `chess-ai` 命令启动应用

### 测试策略
- **覆盖率要求**: 单元测试覆盖率 > 80%，关键模块 > 90%
- **测试框架**: pytest + pytest-cov + pytest-mock + pytest-qt + pytest-benchmark
- **测试分类**: 支持 unit/integration/slow 三种标记，可选择性运行
- **配置管理**: 所有pytest配置集成在 `pyproject.toml` 中，无需单独配置文件
- **测试结果**: 自动生成HTML覆盖率报告，测试结果记录在 `docs/testResult.md`
- **Mock策略**: Mock外部依赖（AI引擎、屏幕捕获、系统API）确保测试稳定性
- **性能测试**: 使用pytest-benchmark进行性能基准测试

### 已完成模块详情
#### 屏幕捕获模块 (ScreenCaptureModule)
- **性能**: 支持1-240Hz帧率，<50ms捕获延迟
- **兼容性**: 基于Windows Desktop Duplication API，支持多显示器
- **可靠性**: 自动错误恢复、重试机制、线程安全设计
- **监控**: 完整性能统计、健康检查、诊断功能
- **测试**: 28个单元测试、8个集成测试，100%功能验证通过

#### 棋盘检测模块 (ChessBoardDetector)
- **算法**: 多检测算法支持 (Hough直线、轮廓检测、自适应阈值)
- **精度**: 透视变换校正、9x10网格坐标系统、像素级精确定位
- **鲁棒性**: 置信度评估、几何验证、长宽比检查
- **性能**: 检测统计监控、算法性能对比、自适应参数调整
- **测试**: 27个单元测试全部通过，完整的数据类和算法验证

## 开发注意事项

### 代码规范
- **代码风格**: 遵循Google Java Style思路，但适配Python最佳实践
- **格式化工具**: Black (行长度88字符) + isort (import排序) 
- **类型检查**: MyPy严格模式，所有函数必须有类型注解
- **代码质量**: Flake8检查 + Bandit安全扫描
- **文档要求**: 类和方法必须有完整的中文注释和文档字符串
- **配置集中**: 所有工具配置统一在 `pyproject.toml` 中管理

### 性能要求
- 界面响应延迟必须 < 50ms（符合车企级实时性要求）
- AI分析结果展示延迟 < 200ms
- 屏幕捕获和识别频率可配置，默认30fps

### 错误处理
- 所有外部调用（AI引擎、屏幕捕获）必须有兜底策略
- 异常信息记录到日志，避免程序崩溃
- 用户界面显示友好的错误提示

### 安全要求
- 不记录敏感的棋局位置信息到日志
- AI引擎通信使用本地进程，避免网络传输
- 配置文件不包含硬编码的敏感信息