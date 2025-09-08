#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋智能对弈助手安装脚本
"""

from setuptools import setup, find_packages
import os

def read_requirements():
    """读取requirements.txt文件"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
        return requirements

setup(
    name="chinese-chess-ai-assistant",
    version="1.0.0",
    description="中国象棋智能对弈助手",
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    author="AI Assistant",
    author_email="ai@example.com",
    url="https://github.com/example/chinese-chess-ai-assistant",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'chess-ai=chess_ai.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'chess_ai': ['config/*.yaml', 'models/*.pt'],
    },
)