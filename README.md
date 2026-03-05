# Text-to-SQL 智能查询系统

> 基于大语言模型的企业级 Text-to-SQL 系统，支持训练、推理和完整的前端界面

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 功能特性

### 核心功能
- ✅ **多模型支持**: OpenAI GPT-4o, Anthropic Claude, 本地模型 (CodeLlama, DeepSeek, Qwen)
- ✅ **LoRA/QLoRA 微调**: 高效的模型训练和优化
- ✅ **复杂度自适应**: 根据查询复杂度动态选择处理策略
- ✅ **多级缓存**: L1 内存缓存 + L2 Redis 缓存
- ✅ **流式响应**: 支持 SSE 实时流式输出
- ✅ **WebSocket**: 实时双向通信接口
- ✅ **多轮对话**: 支持上下文管理和问题改写
- ✅ **Schema 管理**: 自动提取和智能检索相关表结构

### 前端界面
- 🎨 现代化 UI 设计，支持深色主题
- 📊 SQL 预览和语法高亮
- 📈 查询结果可视化
- 📜 历史记录管理
- ⌨️ 快捷键支持 (Ctrl+Enter 提交)

## 项目结构

```
text-to-sql/
├── app/                          # 原有应用模块
│   ├── main.py                   # FastAPI 主入口
│   ├── config.py                 # 配置管理
│   ├── models/                   # 数据模型
│   ├── core/                     # 核心功能
│   │   ├── schema_retriever.py   # Schema 检索
│   │   ├── prompt_builder.py     # Prompt 构建
│   │   ├── sql_generator.py      # SQL 生成
│   │   ├── sql_validator.py      # SQL 校验
│   │   └── context_manager.py    # 上下文管理
│   └── services/                 # 业务服务
│       └── pipeline.py           # 处理流水线
│
├── training/                     # 训练模块 (新增)
│   ├── data_preparation.py       # 数据准备
│   └── lora_trainer.py           # LoRA/QLoRA 训练器
│
├── inference/                    # 推理模块 (新增)
│   ├── text2sql_service.py       # 推理服务
│   └── api_server.py             # FastAPI 服务器
│
├── frontend-v2/                  # 前端界面 (新增)
│   └── index.html                # 完整的 Web UI
│
├── docs/                         # 文档 (新增)
│   ├── COMPREHENSIVE_RESEARCH.md  # 学术研究汇总
│   ├── COMPLETE_GUIDE.md         # 完整开发指南
│   ├── TEXT_TO_SQL_DEVELOPMENT_PATH.md  # 开发路径
│   └── ANALYSIS_AND_SOLUTIONS.md # 方案分析
│
├── examples/                     # 示例代码
│   ├── agent_sql_generator.py    # Agent 方案
│   ├── finetuning_sql_generator.py  # 微调方案
│   └── rag_sql_generator.py      # RAG 方案
│
├── requirements.txt              # 依赖列表
├── .env.example                  # 环境变量模板
└── README.md                     # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
cd /home/admin/.openclaw/workspace/text-to-sql

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装核心依赖
pip install -r requirements.txt

# 可选：训练相关依赖
pip install peft bitsandbytes transformers torch
```

### 2. 配置环境变量

```bash
cp .env.example .env
nano .env
```

配置示例：
```env
# OpenAI API
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o

# Redis 缓存 (可选)
REDIS_URL=redis://localhost:6379

# 数据库
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=postgres
DB_PASSWORD=your_password
```

### 3. 启动服务

```bash
# 启动 API 服务
python -m inference.api_server

# 或使用 uvicorn
uvicorn inference.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问界面

打开浏览器访问：`http://localhost:8000`

## 训练指南

### 数据准备

```bash
# 1. 从数据库提取 Schema
python training/data_preparation.py \
    --action extract \
    --db-type postgresql \
    --db-host localhost \
    --db-name your_database \
    --db-user postgres \
    --db-password your_password

# 2. 构建训练数据
python training/data_preparation.py \
    --action build \
    --db-name your_database \
    --spider-path /path/to/spider

# 3. 数据增强
python training/data_preparation.py \
    --action augment \
    --db-name your_database
```

### LoRA/QLoRA 微调

```bash
# QLoRA 微调 (约 10GB VRAM)
python training/lora_trainer.py \
    --train-path data/training/train.jsonl \
    --val-path data/training/val.jsonl \
    --output-dir outputs/text2sql_qlora \
    --mode qlora \
    --epochs 3

# LoRA 微调 (约 24GB VRAM)
python training/lora_trainer.py \
    --train-path data/training/train.jsonl \
    --val-path data/training/val.jsonl \
    --output-dir outputs/text2sql_lora \
    --mode lora \
    --lora-rank 64
```

## API 文档

### RESTful API

#### 查询接口

**POST** `/api/v1/query`

```bash
curl -X POST http://localhost:8000/api/v1/query \
    -H "Content-Type: application/json" \
    -d '{
        "question": "查询销售额最高的前10个产品",
        "db_name": "sales_db"
    }'
```

**响应**:
```json
{
    "success": true,
    "sql": "SELECT p.product_name, SUM(...) ...",
    "explanation": "这个查询统计了每个产品的销售总额...",
    "confidence": 0.92,
    "complexity": "medium",
    "tables_used": ["products", "order_items"]
}
```

#### 流式查询

**POST** `/api/v1/query/stream`

支持 Server-Sent Events (SSE) 流式响应。

#### Schema 查询

**GET** `/api/v1/schemas/{db_name}`

获取数据库的完整 Schema 信息。

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
    ws.send(JSON.stringify({
        message: "查询销售额最高的产品",
        db_name: "sales_db"
    }));
};

ws.onmessage = (event) => {
    console.log(JSON.parse(event.data));
};
```

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端界面                              │
│  自然语言输入 / SQL 预览 / 结果可视化 / 历史记录             │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI 服务层                            │
│  RESTful API / WebSocket / 认证授权 / 限流                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Text-to-SQL 引擎                          │
│  复杂度评估 / 缓存管理 / Schema 检索 / SQL 生成             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  OpenAI API  │  Anthropic   │  本地模型     │  微调模型    │
│   GPT-4o     │   Claude     │  CodeLlama   │  LoRA/QLoRA  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### 核心组件

| 组件 | 功能 | 技术栈 |
|------|------|--------|
| **API 服务** | RESTful/WebSocket | FastAPI |
| **SQL 生成** | 调用 LLM 生成 SQL | OpenAI/Local |
| **Schema 管理** | 自动提取和检索 | SQLAlchemy |
| **缓存** | 多级缓存策略 | Redis + 内存 |
| **训练** | LoRA/QLoRA 微调 | PEFT + Transformers |

## 性能对比

| 配置 | 准确率 | 响应时间 | 显存占用 |
|------|--------|----------|----------|
| OpenAI GPT-4o | ~85% | 2-5s | - |
| LoRA (r=64) | ~83% | 1-3s | 24GB |
| QLoRA (4-bit) | ~81% | 1-3s | 10GB |
| 基座模型 (34B) | ~75% | 1-2s | 40GB |

## 文档

- [学术研究汇总](docs/COMPREHENSIVE_RESEARCH.md) - 论文和业界最佳实践
- [完整开发指南](docs/COMPLETE_GUIDE.md) - 从零到一的完整教程
- [开发路径分析](docs/TEXT_TO_SQL_DEVELOPMENT_PATH.md) - 技术方案对比
- [架构分析](docs/ANALYSIS_AND_SOLUTIONS.md) - 当前架构分析

## 示例方案

项目包含三种实现方案的示例：

1. **RAG 方案** (`examples/rag_sql_generator.py`) - 基于 Vanna.ai 风格
2. **Agent 方案** (`examples/agent_sql_generator.py`) - 基于 LangChain SQL Agent
3. **微调方案** (`examples/finetuning_sql_generator.py`) - 基于 LoRA/QLoRA

## 配置说明

### 环境变量

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_PROVIDER` | 模型提供商 (openai/anthropic/local) | openai |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_MODEL` | 模型名称 | gpt-4o |
| `REDIS_URL` | Redis 连接 URL | - |
| `DB_TYPE` | 数据库类型 | postgresql |
| `DB_HOST` | 数据库地址 | localhost |
| `DB_PORT` | 数据库端口 | 5432 |
| `DB_NAME` | 数据库名称 | - |
| `DB_USER` | 数据库用户 | - |
| `DB_PASSWORD` | 数据库密码 | - |

## 部署

### Docker 部署

```bash
# 构建镜像
docker build -t text2sql:latest .

# 运行容器
docker run -d \
    --name text2sql \
    -p 8000:8000 \
    --env-file .env \
    text2sql:latest
```

### Docker Compose

```bash
docker-compose up -d
```

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 许可证

MIT License

## 参考资料

### 论文
- [RAT-SQL](https://github.com/Microsoft/rat-sql) - 关系感知的 Schema 编码
- [PICARD](https://arxiv.org/abs/2109.05093) - 增量约束解析
- [Spider Dataset](https://yale-lily.github.io/spider) - 跨域 Text-to-SQL 基准
- [BIRD Benchmark](https://bird-bench.github.io/) - 大规模企业级评估

### 开源项目
- [Vanna.ai](https://github.com/vanna-ai/vanna) - RAG 驱动的 Text-to-SQL
- [LangChain SQL](https://python.langchain.com/docs/use_cases/sql) - SQL Agent
- [DB-GPT-Hub](https://gitee.com/googx/DB-GPT-Hub) - LoRA/QLoRA 实现

---

**版本**: v2.0 | **更新时间**: 2025-03-05
