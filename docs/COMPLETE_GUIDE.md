# Text-to-SQL 完整开发指南

> 从零到一构建生产级 Text-to-SQL 系统
>
> **版本**: v2.0 | **更新时间**: 2025-03-05

---

## 目录

1. [系统概述](#系统概述)
2. [快速开始](#快速开始)
3. [训练指南](#训练指南)
4. [部署指南](#部署指南)
5. [API 文档](#api-文档)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)

---

## 系统概述

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端界面                              │
│  自然语言输入 / SQL 预览 / 结果可视化 / 历史记录             │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 服务                            │
│  RESTful API / WebSocket 流式响应 / 限流 / CORS             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Text-to-SQL 引擎                           │
│  复杂度评估 / 缓存管理 / Schema 检索 / SQL 生成             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  OpenAI API  │  Anthropic   │  本地模型     │  微调模型    │
│   GPT-4o     │   Claude     │  CodeLlama   │  LoRA/QLoRA  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### 核心特性

| 特性 | 描述 | 状态 |
|------|------|------|
| **多模型支持** | OpenAI、Anthropic、本地模型、微调模型 | ✅ |
| **复杂度自适应** | 根据查询复杂度选择处理策略 | ✅ |
| **多级缓存** | L1 内存缓存 + L2 Redis 缓存 | ✅ |
| **流式响应** | 支持 SSE 实时流式输出 | ✅ |
| **WebSocket** | 实时双向通信 | ✅ |
| **Schema 管理** | 自动提取和检索相关表结构 | ✅ |
| **多轮对话** | 支持上下文管理 | ✅ |
| **LoRA/QLoRA** | 高效模型微调 | ✅ |

---

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+ (本地模型)
- Redis (可选，用于缓存)

### 安装

```bash
# 克隆项目
cd /home/admin/.openclaw/workspace/text-to-sql

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 额外依赖（按需安装）
pip install -r requirements-optional.txt
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置
nano .env
```

`.env` 文件示例：

```env
# API 配置
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# LLM 配置 (选择一种)
LLM_PROVIDER=openai  # openai, anthropic, local
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 或使用本地模型
# LOCAL_MODEL_PATH=/models/codellama-34b
# LORA_PATH=/outputs/text2sql_lora

# Redis 缓存
REDIS_URL=redis://localhost:6379

# 数据库 (用于 Schema 提取)
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sales_db
DB_USER=postgres
DB_PASSWORD=your_password
```

### 运行服务

```bash
# 启动 API 服务
python -m inference.api_server

# 或使用 uvicorn
uvicorn inference.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 访问前端

打开浏览器访问：`http://localhost:8000`

---

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
    --db-password your_password \
    --output-dir data/schemas

# 2. 构建训练数据 (从 Spider 数据集)
python training/data_preparation.py \
    --action build \
    --db-name your_database \
    --spider-path /path/to/spider \
    --output-dir data/training

# 3. 数据增强
python training/data_preparation.py \
    --action augment \
    --db-name your_database \
    --output-dir data/training
```

### LoRA 微调

```bash
# QLoRA 微调 (低显存，约 10GB VRAM)
python training/lora_trainer.py \
    --train-path data/training/your_database_text2sql_augmented_train.jsonl \
    --val-path data/training/your_database_text2sql_augmented_val.jsonl \
    --output-dir outputs/text2sql_qlora \
    --base-model codellama/CodeLlama-34b-hf \
    --mode qlora \
    --epochs 3 \
    --batch-size 2 \
    --learning-rate 1e-4

# LoRA 微调 (高性能，约 24GB VRAM)
python training/lora_trainer.py \
    --train-path data/training/your_database_text2sql_augmented_train.jsonl \
    --val-path data/training/your_database_text2sql_augmented_val.jsonl \
    --output-dir outputs/text2sql_lora \
    --base-model codellama/CodeLlama-34b-hf \
    --mode lora \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --lora-rank 64
```

### 从 Checkpoint 恢复

```bash
python training/lora_trainer.py \
    --train-path data/training/train.jsonl \
    --val-path data/training/val.jsonl \
    --output-dir outputs/text2sql_lora \
    --resume outputs/text2sql_lora/checkpoint-1000
```

---

## 部署指南

### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "inference.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

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

```yaml
# docker-compose.yml
version: '3.8'

services:
  text2sql:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./outputs:/app/outputs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./frontend-v2:/usr/share/nginx/html
    depends_on:
      - text2sql

volumes:
  redis_data:
```

### 生产环境配置

```python
# 生产环境配置 (production.py)
import uvicorn

config = uvicorn.Config(
    "inference.api_server:app",
    host="0.0.0.0",
    port=8000,
    workers=4,  # CPU 核心数
    log_level="info",
    access_log=True,
    reload=False,
    limit_concurrency=100,
    limit_max_requests=1000,
    timeout_keep_alive=30,
)

# 使用 gunicorn + uvicorn workers
# gunicorn inference.api_server:app \
#     --workers 4 \
#     --worker-class uvicorn.workers.UvicornWorker \
#     --bind 0.0.0.0:8000 \
#     --timeout 120 \
#     --access-logfile - \
#     --error-logfile -
```

---

## API 文档

### RESTful API

#### 1. 查询接口

**POST** `/api/v1/query`

```bash
curl -X POST http://localhost:8000/api/v1/query \
    -H "Content-Type: application/json" \
    -d '{
        "question": "查询销售额最高的前10个产品",
        "db_name": "sales_db",
        "session_id": "optional-session-id"
    }'
```

响应：
```json
{
    "success": true,
    "sql": "SELECT p.product_name, SUM(...) ...",
    "explanation": "这个查询统计了每个产品的销售总额...",
    "confidence": 0.92,
    "execution_time": 1.23,
    "complexity": "medium",
    "tables_used": ["products", "order_items", "orders"]
}
```

#### 2. 流式查询

**POST** `/api/v1/query/stream`

```bash
curl -N http://localhost:8000/api/v1/query/stream \
    -H "Content-Type: application/json" \
    -d '{
        "question": "查询销售额最高的产品",
        "db_name": "sales_db"
    }'
```

#### 3. 获取 Schema

**GET** `/api/v1/schemas/{db_name}`

```bash
curl http://localhost:8000/api/v1/schemas/sales_db
```

#### 4. 用户反馈

**POST** `/api/v1/feedback`

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
    -H "Content-Type: application/json" \
    -d '{
        "query_id": "abc123",
        "rating": 5,
        "sql_correct": true,
        "result_correct": true
    }'
```

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
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

---

## 最佳实践

### 1. Schema 设计

**原则**：
- 使用清晰的表名和列名
- 添加详细的注释
- 合理使用主外键关系
- 避免过度规范化

**示例**：
```sql
-- 好的命名
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    status VARCHAR(20) COMMENT '订单状态: pending/completed/cancelled',
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 避免的命名
CREATE TABLE t1 (
    id INT,
    c1 VARCHAR(100),
    c2 INT
);
```

### 2. Prompt 优化

**关键要素**：
1. 清晰的系统提示词
2. 相关的 Schema 信息
3. 高质量的 Few-shot 示例
4. 明确的输出格式

### 3. 缓存策略

```python
# 三级缓存策略
1. L1: 内存缓存 (热点数据)
2. L2: Redis 缓存 (共享缓存)
3. L3: 向量相似度检索
```

### 4. 错误处理

```python
# 优雅降级策略
1. 模型失败 → 降级到规则引擎
2. 执行失败 → 提供修复建议
3. 超时 → 返回部分结果
```

### 5. 安全措施

```python
# 安全检查清单
□ SQL 注入防护
□ 权限控制 (只读账号)
□ 限流和配额
□ 敏感数据脱敏
□ 审计日志
□ 危险操作拦截
```

---

## 故障排除

### 常见问题

#### 1. 模型加载失败

```
Error: CUDA out of memory
```

**解决方案**：
- 使用 QLoRA 替代 LoRA
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用更小的模型

#### 2. SQL 生成不准确

**排查步骤**：
1. 检查 Schema 是否完整
2. 增加 Few-shot 示例
3. 调整温度参数
4. 考虑使用更强的模型

#### 3. 响应缓慢

**优化方案**：
1. 启用 Redis 缓存
2. 使用流式响应
3. 减小上下文大小
4. 使用本地模型

---

## 扩展开发

### 添加新的模型后端

```python
# inference/text2sql_service.py

class CustomModelBackend(ModelBackend):
    async def generate(self, prompt: str, **kwargs) -> str:
        # 实现您的模型调用逻辑
        pass
```

### 自定义 Schema 检索

```python
# app/core/schema_retriever.py

class CustomSchemaRetriever:
    async def retrieve(self, question: str, db_name: str):
        # 实现自定义检索逻辑
        pass
```

---

## 参考资源

### 论文
- [RAT-SQL](https://github.com/Microsoft/rat-sql)
- [PICARD](https://arxiv.org/abs/2109.05093)
- [Spider Dataset](https://yale-lily.github.io/spider)
- [BIRD Benchmark](https://bird-bench.github.io/)

### 开源项目
- [Vanna.ai](https://github.com/vanna-ai/vanna)
- [LangChain SQL](https://python.langchain.com/docs/use_cases/sql)
- [DB-GPT-Hub](https://gitee.com/googx/DB-GPT-Hub)

### 工具
- [PEFT](https://github.com/huggingface/peft) - LoRA/QLoRA
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - 量化
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理

---

*文档版本: v2.0*
*最后更新: 2025-03-05*
