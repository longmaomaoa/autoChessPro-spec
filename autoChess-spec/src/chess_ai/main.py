#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋智能对弈助手主入口
"""

import sys
import os
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from chess_ai.config import ConfigManager
from chess_ai.core import Application


def setup_logging(config: ConfigManager) -> None:
    """配置日志系统"""
    log_config = config.get_section("logging")
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        level=log_config.get("console_level", "INFO"),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 添加文件处理器
    if log_config.get("file_enabled", True):
        log_file = Path(log_config.get("file_path", "logs/chess_ai.log"))
        log_file.parent.mkdir(exist_ok=True)
        
        logger.add(
            str(log_file),
            level=log_config.get("file_level", "DEBUG"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=log_config.get("rotation", "10 MB"),
            retention=log_config.get("retention", "30 days"),
            encoding="utf-8"
        )


def main() -> int:
    """主程序入口"""
    try:
        # 初始化配置管理器
        config = ConfigManager()
        config.load_config()
        
        # 设置日志
        setup_logging(config)
        logger.info("中国象棋智能对弈助手启动")
        
        # 创建并运行主应用
        app = Application(config)
        return app.run()
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        return 0
    except Exception as e:
        logger.exception(f"程序发生未预期的错误: {e}")
        return 1
    finally:
        logger.info("中国象棋智能对弈助手退出")


if __name__ == "__main__":
    sys.exit(main())