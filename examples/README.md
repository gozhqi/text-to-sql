# Text-to-SQL 实现方案对比

本目录包含三种主流的 Text-to-SQL 实现方案，基于 2024-2025 年最新研究成果。

## 📚 方案概览

| 方案 | 文件 | 核心理念 | 适用场景 |
|------|------|----------|----------|
| **RAG 方案** | `rag_sql_generator.py` | SQL 即知识，检索增强生成 | 有大量历史 SQL |
| **Agent 方案** | `agent_sql_generator.py` | ReAct 推理循环，动态探索 | 复杂查询、未知 Schema |
| **Fine-tuning 方案** | `finetuning_sql_generator.py` | 模型微调，专项训练 | 有大量标注数据 |

## 🔬 理论基础

### 方案一：RAG（Vanna.ai 风格）

**核心论文**：
- Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Text-to-SQL (EMNLP 2018)
- RAG-based approaches for Text-to-SQL (2024-2025)

**工作原理**：
1. **训练阶段**：将 DDL、业务文档、历史 SQL 存入向量库
2. **检索阶段**：根据问题检索相关的 DDL 和 SQL 示例
3. **生成阶段**：使用检索到的上下文构建 Prompt 生成 SQL

**优势**：
- ✅ 利用历史知识，准确率高
- ✅ 可以持续学习（添加新示例）
- ✅ 冷启动后效果越来越好

**劣势**：
- ❌ 冷启动问题（需要初始数据）
- ❌ 对全新问题效果较差

### 方案二：Agent（LangChain SQL Agent 风格）

**核心论文**：
- ReAct: Synergizing Reasoning and Acting in Language Models (ICLR 2023)
- DB-Surfer: Multi-Agent Collaboration for Text-to-SQL
- SQL-of-Thought (August 2025)

**工作原理**：
1. **思考**：分析当前状态
2. **行动**：调用工具（list_tables, get_schema, search_columns 等）
3. **观察**：获得执行结果
4. **循环**：直到可以生成 SQL

**优势**：
- ✅ 可以处理未知 Schema
- ✅ 自我纠错能力强
- ✅ 适合复杂、多步查询

**劣势**：
- ❌ LLM 调用次数多（成本高）
- ❌ 不确定性（可能循环）
- ❌ 延迟高

### 方案三：Fine-tuning

**核心论文**：
- RESDSQL: Instruction Learning for Text-to-SQL
- Alpha-SQL: Monte Carlo Tree Search for Text-to-SQL (ICML 2025)
- Fine-tuning methods for Text-to-SQL (2024-2025)

**工作原理**：
1. **准备数据**：收集 Question-SQL 对
2. **微调模型**：使用 LoRA/QLoRA 进行参数高效微调
3. **推理使用**：加载微调后的模型生成 SQL

**优势**：
- ✅ 最高准确率
- ✅ 推理速度快
- ✅ 领域适配能力强

**劣势**：
- ❌ 需要大量训练数据
- ❌ 训练成本高（GPU 资源）
- ❌ 需要定期重新训练

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_examples.txt
```

### 2. 运行演示界面

```bash
streamlit run demo_app.py
```

### 3. 使用代码示例

#### RAG 方案

```python
from examples.rag_sql_generator import (
    RAGSQLGenerator,
    DDLDocument,
    SQLExample
)

# 创建生成器
generator = await RAGSQLGenerator().init()

# 训练 DDL
await generator.train_ddl(DDLDocument(
    table_name="orders",
    ddl="CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL)",
    description="订单表"
))

# 训练 SQL 示例
await generator.train_sql(SQLExample(
    question="查询销售额最高的产品",
    sql="SELECT product_id, SUM(amount) FROM orders GROUP BY product_id ORDER BY SUM(amount) DESC"
))

# 生成 SQL
result = await generator.generate_sql(
    question="查询上个月销售额最高的5个产品",
    db_name="sales_db"
)
```

#### Agent 方案

```python
from examples.agent_sql_generator import SQLAgent

# 创建 Agent
agent = SQLAgent(llm_client=client)

# 运行
result = await agent.run(
    question="查询每个地区的季度销售额趋势",
    db_name="sales_db",
    max_iterations=10
)
```

#### Fine-tuning 方案

```python
from examples.finetuning_sql_generator import (
    SQLFineTuner,
    FineTuningConfig,
    TrainingExample
)

# 准备数据
examples = [
    TrainingExample(
        question="查询所有用户",
        sql="SELECT * FROM users"
    ),
    # ... 更多示例
]

# 配置并训练
config = FineTuningConfig(
    base_model="codellama/CodeLlama-7b-Instruct-hf",
    method=FineTuningMethod.LORA
)

tuner = SQLFineTuner(config)
tuner.train(examples)
```

## 📊 性能对比

| 指标 | RAG | Agent | Fine-tuning |
|------|-----|-------|-------------|
| 准确率 | 75-85% | 70-80% | 80-90% |
| 响应时间 | 快 | 慢 | 最快 |
| 训练成本 | 低 | 无 | 高 |
| 推理成本 | 中 | 高 | 低 |
| 数据需求 | 中 | 无 | 高 |

## 🎯 选择建议

```
问题：我应该选择哪种方案？

┌─────────────────────────────────────────────────────┐
│  有大量历史 SQL？                                    │
│     ├── 是 → RAG 方案                                │
│     └── 否 ↓                                         │
│  查询非常复杂？                                      │
│     ├── 是 → Agent 方案                              │
│     └── 否 ↓                                         │
│  有大量标注数据 + GPU 资源？                         │
│     ├── 是 → Fine-tuning 方案                        │
│     └── 否 → RAG 方案（从零开始）                    │
└─────────────────────────────────────────────────────┘
```

## 🔗 参考资源

### 数据集
- [Spider 数据集](https://yale-lily.github.io/spider) - 跨域 Text-to-SQL 基准
- [BIRD 数据集](https://bird-bench.github.io/) - 大规模 Text-to-SQL 基准
- [CSpider](https://github.com/terryyz/CSpider) - 中文 Spider

### 开源项目
- [Vanna.ai](https://github.com/vanna-ai/vanna) - RAG 驱动的 Text-to-SQL
- [LangChain SQL](https://python.langchain.com/docs/use_cases/sql) - Agent 驱动的方案
- [SQL-Coder](https://github.com/defog/sqlcoder) - 微调的 SQL 模型

### 论文
- RAT-SQL (2020) - 关系感知的 Transformer
- DB-Surfer (2024) - Multi-Agent 协作
- SQL-of-Thought (2025) - 思维链 + 动态错误修正

## 📝 许可

本代码基于 MIT 许可开源。
