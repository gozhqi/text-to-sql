#!/bin/bash

# 启动 Text-to-SQL 服务

echo "=========================================="
echo "🚀 启动 Text-to-SQL RAG 服务"
echo "=========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
pip3 install -q fastapi uvicorn python-multipart 2>/dev/null || pip install -q fastapi uvicorn python-multipart 2>/dev/null

# 创建必要的目录
mkdir -p data/knowledge
mkdir -p data/vector

# 启动服务
echo ""
echo "📍 访问地址: http://localhost:8000"
echo "📚 知识库管理: http://localhost:8000 (知识库 Tab)"
echo ""
echo "按 Ctrl+C 停止服务"
echo "=========================================="

python3 run_server.py