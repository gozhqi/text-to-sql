#!/bin/bash

# Text-to-SQL 项目 Git 初始化脚本
# 运行此脚本前，请确保已安装 Git 并配置好 GitHub 认证

set -e

echo "=========================================="
echo "Text-to-SQL 项目 Git 初始化"
echo "=========================================="

# 进入项目目录
cd "$(dirname "$0")"

# 初始化 Git 仓库
echo "📦 初始化 Git 仓库..."
git init

# 添加所有文件
echo "📝 添加文件到暂存区..."
git add .

# 提交
echo "💾 创建初始提交..."
git commit -m "🎉 Initial commit: Text-to-SQL智能查询系统

功能特性:
- ✅ 单轮自然语言转SQL
- ✅ 多轮对话上下文理解
- ✅ Schema智能检索
- ✅ SQL安全校验
- ✅ Web API接口
- ✅ Web UI界面

技术栈:
- FastAPI + Uvicorn
- OpenAI GPT-4
- ChromaDB
- Sentence Transformers
"

echo ""
echo "✅ Git 仓库初始化完成！"
echo ""
echo "=========================================="
echo "下一步：创建 GitHub 仓库"
echo "=========================================="
echo ""
echo "方式一：使用 GitHub CLI (推荐)"
echo "--------------------------------------"
echo "1. 安装 GitHub CLI:"
echo "   brew install gh  # macOS"
echo "   或访问 https://cli.github.com/"
echo ""
echo "2. 登录 GitHub:"
echo "   gh auth login"
echo ""
echo "3. 创建仓库并推送:"
echo "   gh repo create text-to-sql --public --source=. --push"
echo ""
echo "方式二：手动创建"
echo "--------------------------------------"
echo "1. 访问 https://github.com/new"
echo "2. 创建新仓库: text-to-sql"
echo "3. 不要勾选 'Add a README file'"
echo "4. 创建后运行以下命令:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/text-to-sql.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=========================================="