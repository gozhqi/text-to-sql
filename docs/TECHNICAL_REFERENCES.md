# 技术参考与设计借鉴

> 本文档详细说明项目如何参考 Vanna.ai、LangChain SQL Agent、DB-GPT-Hub 等主流开源项目

---

## 一、参考项目概览

| 项目 | Stars | 许可证 | 核心贡献 | 我们借鉴内容 |
|------|-------|--------|----------|--------------|
| **Vanna.ai** | 20,000+ | MIT | RAG 驱动 SQL 生成 | 知识库设计、训练接口 |
| **LangChain SQL Agent** | 90,000+ | MIT | Agent 推理框架 | ReAct 循环、工具设计 |
| **DB-GPT-Hub** | 5,000+ | Apache 2.0 | 微调训练方案 | LoRA/QLoRA 配置、数据格式 |

---

## 二、Vanna.ai 参考详情

### 2.1 项目简介

Vanna.ai 是一个开源的 Python RAG 框架，专门用于 SQL 生成。

**GitHub**: https://github.com/vanna-ai/vanna

**核心特点**:
- 使用向量数据库存储 SQL 知识
- 支持 DDL、文档、SQL 示例三种训练数据
- 准确率可达 82%+（GPT-4 + 充分训练）

### 2.2 我们借鉴的设计

#### 借鉴 1: 三类知识存储

**Vanna.ai 原始设计**:
```python
# Vanna 的训练接口
vn.train(ddl="CREATE TABLE users (...)")
vn.train(documentation="users 表存储用户信息...")
vn.train(sql="SELECT * FROM users WHERE active = 1")
```

**我们的实现** (`training/data_preparation.py`):
```python
class SQLKnowledgeBase:
    def train_ddl(self, ddl: str, db_name: str = "default"):
        """存储 DDL 知识 - 借鉴自 Vanna"""
        ...

    def train_documentation(self, documentation: str, db_name: str = "default"):
        """存储业务文档 - 借鉴自 Vanna"""
        ...

    def train_sql(self, question: str, sql: str, db_name: str = "default"):
        """存储 SQL 示例 - 借鉴自 Vanna"""
        ...
```

**借鉴说明**:
- Vanna 的三类知识设计非常合理：DDL 定义结构、文档补充语义、SQL 提供模式
- 我们直接采用这个设计，因为它能有效覆盖 SQL 生成的所有知识需求

#### 借鉴 2: 相似度检索优先

**Vanna 原始设计**:
```python
# Vanna 的检索逻辑
similar_questions = vn.get_similar_question_sql(question)
# 返回相似的问题-SQL 对
```

**我们的实现** (`inference/text2sql_service.py`):
```python
class TextToSQLService:
    async def _retrieve_context(self, question: str):
        """检索相关上下文 - 借鉴自 Vanna 的相似度检索"""
        # 1. 检索相似 SQL
        similar_sqls = await self.kb.retrieve_similar_sql(question)

        # 2. 检索相关 DDL
        relevant_ddls = await self.kb.retrieve_relevant_ddl(question)

        return {
            "similar_sqls": similar_sqls,
            "relevant_ddls": relevant_ddls
        }
```

**借鉴说明**:
- Vanna 证明了"相似 SQL 示例"是提升准确率最有效的因素
- 我们将其作为 RAG 检索的第一优先级

#### 借鉴 3: 从反馈中学习

**Vanna 原始设计**:
```python
# Vanna 的反馈学习
vn.train(sql=correct_sql, question=user_question)
```

**我们的实现** (`inference/text2sql_service.py`):
```python
class FeedbackManager:
    """反馈管理器 - 借鉴自 Vanna 的持续学习理念"""

    async def record_success(self, query_id: str, sql: str):
        """记录成功的查询，加入知识库"""
        await self.kb.train_sql(
            question=self._get_question(query_id),
            sql=sql
        )
```

**借鉴说明**:
- Vanna 的核心理念是"每次正确的 SQL 都是新知识"
- 我们实现了自动化的反馈收集和知识库更新

### 2.3 我们改进的部分

| 方面 | Vanna 原设计 | 我们的改进 |
|------|-------------|------------|
| 向量库 | 仅 ChromaDB | 支持 ChromaDB/Qdrant/Milvus |
| 多轮对话 | 无 | 完整的多轮上下文管理 |
| 复杂度评估 | 无 | 动态路由到不同处理路径 |
| 缓存 | 无 | 多级缓存（内存 + Redis） |

---

## 三、LangChain SQL Agent 参考详情

### 3.1 项目简介

LangChain SQL Agent 是 LangChain 框架中专门用于数据库查询的 Agent 实现。

**文档**: https://python.langchain.com/docs/use_cases/sql

**核心特点**:
- 使用 ReAct 推理循环
- 动态探索 Schema
- 自动纠错机制

### 3.2 我们借鉴的设计

#### 借鉴 1: ReAct 推理循环

**LangChain 原始设计**:
```python
# LangChain 的 Agent 执行循环
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
# 内部执行: Thought → Action → Observation → ...
```

**我们的实现** (`examples/agent_sql_generator.py`):
```python
class SQLAgent:
    """SQL Agent - 借鉴 LangChain 的 ReAct 模式"""

    async def run(self, question: str) -> Dict:
        iterations = 0

        while iterations < self.max_iterations:
            # Thought: LLM 思考下一步
            thought = await self._get_next_thought()

            # Action: 执行工具
            action_result = await self._execute_tool(thought.action)

            # Observation: 观察结果
            self._add_observation(action_result)

            # 检查是否完成
            if thought.is_final:
                return self._build_result()

            iterations += 1
```

**借鉴说明**:
- LangChain 的 ReAct 模式非常适合 SQL 生成这种需要多步推理的任务
- 我们完整实现了 Thought → Action → Observation 循环

#### 借鉴 2: 工具集设计

**LangChain 原始设计**:
```python
# LangChain SQL Agent 工具
tools = [
    InfoSQLDatabaseTool(db=db),      # 获取 Schema
    ListSQLDatabaseTool(db=db),      # 列出表
    QuerySQLDataBaseTool(db=db),     # 执行查询
    QuerySQLCheckerTool(db=db),      # 检查 SQL
]
```

**我们的实现** (`examples/agent_sql_generator.py`):
```python
class SQLAgent:
    """工具集 - 借鉴 LangChain 的工具设计"""

    TOOLS = {
        "list_tables": {
            "description": "列出数据库所有表",
            "func": "_list_tables"
        },
        "get_schema": {
            "description": "获取表结构",
            "func": "_get_schema"
        },
        "query_sample": {
            "description": "查询样本数据",
            "func": "_query_sample"
        },
        "validate_sql": {
            "description": "验证 SQL 语法",
            "func": "_validate_sql"
        },
        "execute_sql": {
            "description": "执行 SQL 查询",
            "func": "_execute_sql"
        }
    }
```

**借鉴说明**:
- LangChain 的工具设计清晰、职责单一
- 我们保持了相同的工具划分，增加了 `query_sample` 用于理解数据

#### 借鉴 3: 自动纠错机制

**LangChain 原始设计**:
```python
# LangChain 的 QuerySQLCheckerTool
class QuerySQLCheckerTool(BaseTool):
    def _run(self, query: str) -> str:
        # 使用 LLM 检查并修复 SQL
        return llm.predict(f"检查这个 SQL 是否正确: {query}")
```

**我们的实现** (`examples/agent_sql_generator.py`):
```python
class ValidatorAgent(BaseAgent):
    """验证 Agent - 借鉴 LangChain 的自动纠错"""

    async def run(self, context: Dict) -> Dict:
        sql = context["sql"]
        error = context.get("error")

        # 使用 LLM 分析错误并修复
        prompt = f"""
        SQL: {sql}
        错误: {error}

        请分析错误并返回修复后的 SQL。
        """

        fixed_sql = await self._call_llm(prompt)

        return {"fixed_sql": fixed_sql}
```

**借鉴说明**:
- SQL 执行失败时自动尝试修复是 Agent 的核心优势
- 我们实现了完整的多轮修复机制（最多 3 次）

### 3.3 我们改进的部分

| 方面 | LangChain 原设计 | 我们的改进 |
|------|------------------|------------|
| Agent 数量 | 单 Agent | Multi-Agent 协作 |
| Schema 缓存 | 无 | 智能缓存，减少探索次数 |
| 错误恢复 | 简单重试 | 结构化错误分析 + 修复 |
| 流式输出 | 不支持 | 支持流式返回推理过程 |

---

## 四、DB-GPT-Hub 参考详情

### 4.1 项目简介

DB-GPT-Hub 是一个专注于 Text-to-SQL 微调的开源项目。

**Gitee**: https://gitee.com/googx/DB-GPT-Hub

**核心特点**:
- 完整的微调 Pipeline
- 支持 LoRA/QLoRA
- Spider 数据集处理工具

### 4.2 我们借鉴的设计

#### 借鉴 1: 训练数据格式

**DB-GPT-Hub 原始设计**:
```json
{
  "db_id": "financial",
  "question": "查询余额超过10000的账户",
  "query": "SELECT * FROM accounts WHERE balance > 10000",
  "schema": {
    "tables": ["accounts"],
    "columns": ["account_id", "balance", "customer_id"]
  }
}
```

**我们的实现** (`training/data_preparation.py`):
```python
@dataclass
class TrainingSample:
    """训练样本格式 - 借鉴 DB-GPT-Hub"""
    db_id: str
    question: str
    query: str
    schema: Dict[str, List[str]]
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)
```

**借鉴说明**:
- DB-GPT-Hub 的数据格式经过 Spider 数据集验证
- 我们直接采用这个格式，确保与主流数据集兼容

#### 借鉴 2: LoRA/QLoRA 配置

**DB-GPT-Hub 原始配置**:
```yaml
# DB-GPT-Hub 的 QLoRA 配置
quantization_bit: 4
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.1
learning_rate: 1e-4
```

**我们的实现** (`training/lora_trainer.py`):
```python
class LoRATrainer:
    """LoRA 训练器 - 借鉴 DB-GPT-Hub 的配置"""

    QLORA_CONFIG = {
        "quantization_bit": 4,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
        # 目标模块 - DB-GPT-Hub 验证过的最佳实践
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }

    LORA_CONFIG = {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"]
    }
```

**借鉴说明**:
- DB-GPT-Hub 提供了经过大量实验验证的参数配置
- 我们直接使用这些配置，减少调参成本

#### 借鉴 3: 评估指标

**DB-GPT-Hub 原始设计**:
```python
# DB-GPT-Hub 的评估函数
def evaluate(model, dataset):
    exact_match = compute_exact_match(predictions, labels)
    execution_acc = compute_execution_accuracy(predictions, labels, db)
    return {"EM": exact_match, "EX": execution_acc}
```

**我们的实现** (`training/lora_trainer.py`):
```python
class EvaluationMetrics:
    """评估指标 - 借鉴 DB-GPT-Hub"""

    @staticmethod
    def exact_match(predicted_sql: str, gold_sql: str) -> float:
        """精确匹配 (EM) - DB-GPT-Hub 标准"""
        normalized_pred = normalize_sql(predicted_sql)
        normalized_gold = normalize_sql(gold_sql)
        return 1.0 if normalized_pred == normalized_gold else 0.0

    @staticmethod
    def execution_accuracy(predicted_sql: str, gold_sql: str, db) -> float:
        """执行准确率 (EX) - DB-GPT-Hub 标准"""
        try:
            pred_result = db.execute(predicted_sql)
            gold_result = db.execute(gold_sql)
            return 1.0 if compare_results(pred_result, gold_result) else 0.0
        except:
            return 0.0
```

**借鉴说明**:
- EM 和 EX 是 Spider 数据集的标准评估指标
- 我们使用相同指标，确保评估结果可比较

### 4.3 我们改进的部分

| 方面 | DB-GPT-Hub 原设计 | 我们的改进 |
|------|-------------------|------------|
| 数据增强 | 基础 | 多策略增强（SQL改写、问题改写） |
| 负样本 | 无 | 自动生成负样本用于对比学习 |
| 训练监控 | 基础 | 完整的 WandB 集成 |
| 模型导出 | 仅 LoRA | 支持合并导出、量化导出 |

---

## 五、综合借鉴总结

### 5.1 架构层面

```
我们最终架构 = Vanna.ai 的 RAG 设计 + LangChain 的 Agent 框架 + DB-GPT-Hub 的微调能力

┌─────────────────────────────────────────────────────────────┐
│                      我们的系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ RAG 层 (借鉴 Vanna.ai)                               │   │
│  │ - 三类知识存储: DDL / 文档 / SQL 示例               │   │
│  │ - 向量检索相似 SQL                                  │   │
│  │ - 持续学习机制                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Agent 层 (借鉴 LangChain)                            │   │
│  │ - ReAct 推理循环                                     │   │
│  │ - 多工具协作                                         │   │
│  │ - 自动纠错机制                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 微调层 (借鉴 DB-GPT-Hub)                             │   │
│  │ - LoRA/QLoRA 配置                                    │   │
│  │ - Spider 数据格式                                    │   │
│  │ - EM/EX 评估指标                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 代码层面借鉴

| 源项目 | 借鉴代码/设计 | 我们的应用位置 |
|--------|---------------|----------------|
| Vanna.ai | `train_ddl/train_documentation/train_sql` | `training/data_preparation.py` |
| Vanna.ai | 相似度检索逻辑 | `inference/text2sql_service.py` |
| LangChain | ReAct Agent 循环 | `examples/agent_sql_generator.py` |
| LangChain | SQL 工具集设计 | `examples/agent_sql_generator.py` |
| DB-GPT-Hub | LoRA 配置参数 | `training/lora_trainer.py` |
| DB-GPT-Hub | 训练数据格式 | `training/data_preparation.py` |

### 5.3 创新点

我们在借鉴基础上做了以下创新：

1. **动态路由**: 根据查询复杂度自动选择 RAG/Agent/微调模型
2. **Multi-Agent**: 引入多 Agent 协作，提升复杂查询处理能力
3. **多级缓存**: 内存 + Redis + 向量相似度三层缓存
4. **数据增强**: SQL 改写、问题改写、Schema 变体多种增强策略
5. **完整前端**: 提供可视化演示界面，便于测试和对比

---

## 六、参考资料链接

### 论文
- **Spider Dataset**: https://yale-lily.github.io/spider
- **RAT-SQL**: https://github.com/Microsoft/rat-sql
- **BIRD Benchmark**: https://bird-bench.github.io/

### 开源项目
- **Vanna.ai**: https://github.com/vanna-ai/vanna
- **LangChain SQL**: https://python.langchain.com/docs/use_cases/sql
- **DB-GPT-Hub**: https://gitee.com/googx/DB-GPT-Hub

### 数据集
- **Spider**: https://yale-lily.github.io/spider
- **CSpider (中文)**: https://github.com/yyyflint2/Cspider
- **BIRD**: https://bird-bench.github.io/

---

*文档版本: v1.0*
*创建时间: 2025-03-05*