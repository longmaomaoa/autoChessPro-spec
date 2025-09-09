@echo off
echo 🏁 激活中国象棋智能对弈助手开发环境...
call "C:\Users\sibin.long\AI-Coding\aiCoding\autoChessPro-spec\autoChess-spec\.venv\Scripts\activate.bat"
echo ✅ 环境已激活!
echo.
echo 📖 常用命令:
echo   python -m chess_ai.main           # 运行应用
echo   python tests/test_basic.py        # 运行基础测试
echo   pytest tests/ -v                  # 运行完整测试
echo   black src/                        # 格式化代码
echo.
