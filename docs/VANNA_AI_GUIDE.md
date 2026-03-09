# Vanna.ai 2.0 - 开源 SQL Agent 框架

> **官网**: https://vanna.ai/  
> **GitHub**: https://github.com/vanna-ai/vanna  
> **版本**: Vanna 2.0

---

## 一、概述

Vanna 是一个基于 LLM 的开源 SQL 生成框架，采用 RAG (Retrieval-Augmented Generation) 技术，让用户可以用自然语言查询数据库。

### 核心特性
- ✅ **开源免费**: MIT 许可证
- ✅ **多数据库支持**: PostgreSQL, MySQL, Snowflake, BigQuery 等
- ✅ **多轮对话**: 支持上下文管理和 follow-up 问题
- ✅ **访问控制**: 集成企业级权限管理
- ✅ **自我学习**: 从用户反馈中持续改进

---

## 二、Vanna 2.0 新特性

### 2.1 Lifecycle Hooks
```python
# 在请求生命周期的关键点添加自定义逻辑
- 配额检查
- 自定义日志
- 内容过滤
```

### 2.2 LLM Middlewares
```python
# LLM 调用的中间件
- 缓存层
- Prompt 工程
- 成本追踪
```

### 2.3 Conversation Storage
- 持久化对话历史
- 按用户检索历史
- 支持多会话管理

### 2.4 Observability
- 内置追踪和指标
- 与监控系统集成
- 调试和性能分析

### 2.5 Context Enrichers
- RAG 增强
- 文档注入
- 自定义上下文

---

## 三、核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│              (Streamlit / Flask / Custom UI)                │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Vanna Core                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Question    │  │  SQL         │  │  Execution   │      │
│  │  Understanding│→ │  Generation  │→ │  & Validation│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↑                  ↑                  │             │
│  ┌──────────────────────────────────────────────┐          │
│  │              RAG Layer                        │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │          │
│  │  │ DDL Docs │  │ SQL Docs │  │ Q-A Pairs │   │          │
│  │  └──────────┘  └──────────┘  └──────────┘   │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Database Layer                          │
│         PostgreSQL / MySQL / Snowflake / BigQuery          │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、核心概念

### 4.1 RAG 训练数据

Vanna 使用三种类型的训练数据：

| 类型 | 内容 | 用途 |
|------|------|------|
| **DDL** | 表结构定义 | Schema 理解 |
| **Documentation** | 表/列说明 | 语义理解 |
| **SQL** | 问题-SQL 对 | 示例学习 |

### 4.2 向量存储

```python
# 支持多种向量数据库
- ChromaDB (默认)
- Pinecone
- Weaviate
- 自定义
```

### 4.3 LLM 集成

```python
# 支持多种 LLM 提供商
- OpenAI (GPT-4)
- Anthropic (Claude)
- Azure OpenAI
- 本地模型 (Ollama)
- 自定义
```

---

## 五、使用示例

### 5.1 快速开始

```python
import vanna
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': 'sk-...', 'model': 'gpt-4'})

# 训练
vn.train(ddl="CREATE TABLE users (id INT, name VARCHAR(100))")
vn.train(documentation="users 表存储用户信息")
vn.train(question="查询所有用户", sql="SELECT * FROM users")

# 生成 SQL
sql = vn.generate_sql("查询名为张三的用户")
print(sql)
```

### 5.2 连接数据库

```python
vn.connect_to_postgres(
    host="localhost",
    dbname="mydb",
    user="user",
    password="password"
)

# 执行并获取结果
df = vn.run_sql(sql)
```

### 5.3 Flask 应用

```python
from vanna.flask import VannaFlaskApp

app = VannaFlaskApp(vn)
app.run()
```

---

## 六、与本项目对比

| 功能 | Vanna.ai 2.0 | 本项目 |
|------|-------------|--------|
| **RAG 架构** | ✅ Agentic RAG | ⚠️ 简单检索 |
| **Schema RAG** | ✅ 自动提取 | ⚠️ 手动导入 |
| **执行验证** | ✅ 自动执行+修正 | ❌ 无 |
| **对话存储** | ✅ 持久化 | ❌ 无 |
| **中间件** | ✅ 可扩展 | ❌ 无 |
| **可观测性** | ✅ 内置 | ❌ 无 |
| **向量存储** | ✅ 多种选择 | ⚠️ TF-IDF |

---

## 七、改进建议

基于 Vanna.ai 的最佳实践，本项目应优先实现：

1. **执行反馈循环**: 生成 SQL 后执行，根据错误修正
2. **真正的向量存储**: 替换 TF-IDF 为语义嵌入
3. **训练数据管理**: 支持 DDL/文档/SQL 三种类型
4. **对话上下文**: 多轮对话支持
5. **Middleware 架构**: 可扩展的请求处理管道

---

**更新时间**: 2026-03-09