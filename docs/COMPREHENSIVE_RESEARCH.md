# Text-to-SQL 综合研究文档

> 基于学术论文和业界最佳实践的完整技术分析
>
> **更新时间**: 2025-03-05 | **版本**: v1.0

---

## 目录

1. [第一部分：学术论文研究](#第一部分学术论文研究)
2. [第二部分：数据集分析](#第二部分数据集分析)
3. [第三部分：业界最佳实践](#第三部分业界最佳实践)
4. [第四部分：技术架构对比](#第四部分技术架构对比)
5. [第五部分：完整方案设计](#第五部分完整方案设计)

---

## 第一部分：学术论文研究

### 1.1 基础数据集与基准

#### Spider 数据集 (2018)

**论文**: *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Text-to-SQL Semantic Parsing*

**关键特性**:
- **规模**: 10,181 个问题，200 个数据库，138 个领域
- **复杂度**: 涵盖单表到多表 JOIN、嵌套查询、聚合等复杂 SQL
- **评估指标**: Exact Match (EM) 和 Execution Accuracy

**影响**: 成为 Text-to-SQL 领域的标准基准

#### Spider 2.0 (2024)

**重要升级**:
- **企业级规模**: 数据库平均包含 800+ 列
- **真实场景**: 包含复杂 SQL 方言和脏数据
- **性能**: SOTA 方法达到 91.2% Strict Recall Rate

#### BIRD Benchmark (2023)

**论文**: *Can LLM Already Serve as A Database Interface? A Big Bench for Large-Scale Database Grounded Text-to-SQLs*

**关键特性**:
- **数据规模**: 33GB 数据，12,751 个查询任务
- **数据库**: 95 个大型真实数据库
- **领域覆盖**: 金融、电力、医疗、零售等 37 个行业
- **独特挑战**:
  - 脏数据处理
  - 外部知识需求
  - SQL 效率评估 (R-VES 指标)

**当前 SOTA** (2025):
1. Agentar-Scale-SQL: 81.67% 执行准确率
2. AskData + GPT-4o: 80.88%
3. LongData-SQL: 77.53%

---

### 1.2 核心模型论文

#### RAT-SQL (ACL 2020)

**论文**: *RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers*

**作者**: Wang et al. (Microsoft Research)

**核心创新**:

1. **Schema Encoding** (模式编码):
```
将数据库结构编码为向量表示:
- 表名 (table names)
- 列名 (column names)
- 列类型 (column types)
- 主键 (primary keys)
- 外键 (foreign keys)
```

2. **Schema Linking** (模式链接):
```
对齐自然语言问题与数据库元素:
- 问题词汇 → 表名
- 问题词汇 → 列名
- 语义消歧
```

3. **Relation-Aware Self-Attention**:
```python
# 修改的自注意力机制
# 同时处理:
# - 问题 tokens
# - 表 entities
# - 列 entities
# - 显式关系 (主外键、语义链接)
```

**Spider 性能**: 65.6% 准确率 (2020年SOTA)

**官方实现**: https://github.com/Microsoft/rat-sql

#### GraPPa (2020)

**论文**: *Grammar-Augmented Pre-Training for Table Semantic Parsing*

**核心思想**: 使用语法增强的预训练

**创新点**:
1. SQL 语法树结构作为预训练目标
2. Schema 信息增强
3. 跨域迁移能力强

#### PICARD (ICLR 2022)

**论文**: *PICARD: Incremental Constraining for Jointly Predicting SQL Queries and Table Unions*

**核心创新**:

1. **增量约束** (Incremental Constraining):
```python
# 在生成过程中逐步应用约束
# 确保:
# - 语法正确性
# - 表名存在性
# - 列名存在性
# - 类型兼容性
```

2. **基于 T5 架构**

**Spider 性能**: 69.8% 准确率

#### SEDS (Schema-Aware Denoising)

**核心思想**: 专门针对 Schema 理解的去噪预训练

**方法**:
- Schema 感知的掩码策略
- 结构化去噪任务
- 提升模型对数据库结构的理解

#### NAT (Non-Autoregressive)

**核心思想**: 非自回归生成，提升推理速度

**优势**:
- 并行生成
- 更快的推理速度
- 适合低延迟场景

---

### 1.3 最新进展 (2024-2025)

#### Multi-Agent 架构

**代表**: DB-Surfer

**核心思想**: 多 Agent 协作解决复杂查询

```
Orchestrator (总控)
    ├── Schema Explorer Agent (探索 Schema)
    ├── SQL Generator Agent (生成 SQL)
    └── Validator Agent (验证和修复)
```

#### Schema Linking 优化

**代表**: AutoLink

**成就**: 91.2% Schema Recall (提升 27.2%)

**三种链接类型**:
1. **Table Linking**: "查询订单" → orders 表
2. **Column Linking**: "销售额" → orders.total_amount
3. **Value Linking**: "上个月" → DATE_SUB(NOW(), INTERVAL 1 MONTH)

#### Chain-of-Thought SQL

**核心思想**: 在生成 SQL 前先生成分析步骤

**效果**: Spider 2.0 上 SOTA 性能

---

## 第二部分：数据集分析

### 2.1 数据集对比

| 数据集 | 发布年份 | 问题数量 | 数据库数量 | 复杂度 | 主要用途 |
|--------|----------|----------|------------|--------|----------|
| **WikiSQL** | 2017 | 80,654 | 单表 | 低 | 基础评估 |
| **Spider** | 2018 | 10,181 | 200 | 高 | 跨域基准 |
| **CSpider** | 2020 | ~10,000 | 200 | 高 | 中文评估 |
| **BIRD** | 2023 | 12,751 | 95 | 极高 | 企业级评估 |
| **Spider 2.0** | 2024 | - | 大型 | 极高 | 生产环境 |

### 2.2 数据集统计

#### Spider 数据集查询模式分布:
```
简单查询 (WHERE, ORDER BY): 25%
聚合查询 (GROUP BY, HAVING): 30%
JOIN 查询: 20%
嵌套/子查询: 15%
集合操作 (UNION, INTERSECT): 10%
```

#### BIRD 数据集领域分布:
```
金融: 15%
医疗: 12%
零售: 18%
教育: 10%
政府: 8%
其他: 37%
```

---

## 第三部分：业界最佳实践

### 3.1 Vanna.ai - RAG 驱动方案

**GitHub**: https://github.com/vanna-ai/vanna (20,000+ stars)

**核心架构**:

```
训练阶段:
DDL + 业务文档 + 正确SQL → 向量化 → 知识库

查询阶段:
用户问题 → 检索相似SQL → Prompt → LLM → SQL
```

**关键设计**:

1. **三类知识存储**:
```python
# 1. DDL 文档
vn.train(ddl="CREATE TABLE orders (...)")

# 2. 业务文档
vn.train(documentation="订单表存储所有销售订单...")

# 3. SQL 示例
vn.train(sql="SELECT * FROM orders WHERE status = 'completed'")
```

2. **准确率优化**:
| 配置 | 准确率 |
|------|--------|
| 默认参数 | ~3% |
| 温度=0.5 + 静态示例 | ~40% |
| 温度=0.3 + GPT-4 + 动态示例 | ~82% |

3. **支持的技术栈**:
- **LLM**: OpenAI, Anthropic, Gemini, 本地模型
- **向量库**: ChromaDB, Pinecone, PGVector
- **数据库**: PostgreSQL, MySQL, Snowflake 等

### 3.2 LangChain SQL Agent

**核心思想**: LLM 作为推理引擎，动态调用工具

**ReAct 循环**:
```
Thought → Action → Observation → Thought → ...
```

**可用工具**:
- `sql_db_list_tables`: 列出所有表
- `sql_db_schema`: 获取表结构
- `sql_db_query`: 执行 SQL
- `sql_db_query_checker`: 验证 SQL

**工作流程示例**:
```
1. 用户: "查询销售额"
2. Agent: 我需要找到相关表 → list_tables()
3. Agent: 看到 orders 表 → get_schema("orders")
4. Agent: 找到 amount 字段 → generate_sql()
5. Agent: 不确定语法 → query_checker()
6. Agent: 验证通过 → query_sql()
7. 返回结果
```

### 3.3 微调方案

#### LoRA vs QLoRA 性能对比 (7B模型):

| 方法 | 准确率 | 显存占用 | 训练成本 |
|------|--------|----------|----------|
| 全量微调 | 87.2% | 80GB | $320 |
| LoRA (r=32) | 85.1% | 24GB | $48 |
| QLoRA (4-bit) | 83.7% | 10GB | $12 |

#### 推荐配置:

**QLoRA** (入门):
```yaml
quantization_bit: 4
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
learning_rate: 1e-4
```

**LoRA** (生产):
```yaml
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
learning_rate: 2e-4
```

#### 支持的基座模型:
- CodeLlama 34B (SQL 能力强)
- DeepSeek Coder (推理能力强)
- SQLCoder (专门针对 SQL)
- Qwen (通义千问)

---

## 第四部分：技术架构对比

### 4.1 四种主流架构

#### 架构 1: Pipeline 模式

```
Question → Schema检索 → Prompt构建 → LLM → SQL校验 → 执行
```

**优点**: 流程清晰，易于理解，易于扩展
**缺点**: 串行执行，错误传播

#### 架构 2: RAG + Generation 模式

```
Question → RAG检索相似SQL → Prompt构建 → LLM → SQL
```

**优点**: 利用历史知识，准确率高，可增量学习
**缺点**: 冷启动问题，需维护知识库

#### 架构 3: Agent 模式

```
Question → Agent → [工具调用循环] → Result
```

**优点**: 灵活，自我纠错，支持多步推理
**缺点**: 不确定性高，成本难以控制

#### 架构 4: Multi-Agent 模式

```
Orchestrator
    ├── Schema Agent
    ├── SQL Agent
    └── Validator Agent
```

**优点**: 处理复杂问题，专业分工
**缺点**: 实现复杂，协调成本高

### 4.2 性能对比

| 架构 | 准确率 | 响应时间 | 训练成本 | 推理成本 | 数据需求 |
|------|--------|----------|----------|----------|----------|
| Pipeline | 70-75% | 快 (2-5s) | 低 | 低 | 中 |
| RAG | 75-85% | 快 (2-5s) | 低 | 中 | 中 |
| Agent | 70-80% | 慢 (10-30s) | 无 | 高 | 无 |
| Multi-Agent | 80-90% | 慢 (15-40s) | 高 | 高 | 高 |
| Fine-tuning | 80-90% | 最快 (1-3s) | 高 | 低 | 高 |

---

## 第五部分：完整方案设计

### 5.1 系统架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        用户交互层                            │
│  Web UI / API / 第三方集成                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      对话管理层                              │
│  会话管理 / 多轮上下文 / 意图识别 / 问题改写                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    复杂度评估层                              │
│  简单查询 → 单次生成                                        │
│  中等查询 → CoT + 校验                                      │
│  复杂查询 → Multi-Agent                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Schema 理解层                             │
│  Schema 检索 / Schema Linking / 动态示例检索                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    SQL 生成层                               │
│  Prompt 构建 / CoT 推理 / 自我修正                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   安全执行层                                │
│  SQL 校验 / 权限控制 / 执行 / 结果格式化                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 数据准备阶段

#### Schema 提取和存储

```python
# 自动提取 Schema
async def extract_schema(db_connection):
    """从数据库自动提取完整 Schema"""
    schemas = {}

    # 表信息
    tables = await db_connection.get_tables()

    for table in tables:
        # 列信息
        columns = await db_connection.get_columns(table)

        # 主键/外键
        primary_keys = await db_connection.get_primary_keys(table)
        foreign_keys = await db_connection.get_foreign_keys(table)

        # 表注释
        comment = await db_connection.get_table_comment(table)

        schemas[table] = TableSchema(
            table_name=table,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            comment=comment
        )

    return schemas
```

#### 训练数据构造

**数据来源**:
1. **Spider 数据集**: 10,181 个高质量 Question-SQL 对
2. **BIRD 数据集**: 12,751 个企业级查询
3. **自定义数据**: 业务分析师编写的 SQL

**数据格式**:
```json
{
  "id": "spider_001",
  "question": "查询销售额最高的前10个产品",
  "sql": "SELECT p.product_name, SUM(o.amount) as total FROM products p JOIN order_items o ON p.id = o.product_id GROUP BY p.product_name ORDER BY total DESC LIMIT 10",
  "schema": {
    "tables": ["products", "order_items"],
    "columns": ["product_name", "amount"]
  },
  "difficulty": "medium",
  "tags": ["join", "aggregate", "order_by"]
}
```

#### 数据增强策略

```python
class DataAugmenter:
    """数据增强器"""

    def augment(self, question_sql_pair):
        """生成多样化的训练样本"""
        augmented = []

        # 1. SQL 等价改写
        augmented.extend(self._rewrite_sql(question_sql_pair))

        # 2. 问题改写
        augmented.extend(self._rewrite_question(question_sql_pair))

        # 3. Schema 变体
        augmented.extend(self._schema_variants(question_sql_pair))

        return augmented

    def _rewrite_sql(self, pair):
        """SQL 等价改写"""
        # JOIN 顺序改变
        # 子查询 ↔ JOIN
        # CASE WHEN 语法变体
        pass

    def _rewrite_question(self, pair):
        """问题改写"""
        # 同义词替换
        # 句式变换
        # 简化/详细化
        pass
```

#### 负样本构造

```python
class NegativeSampleGenerator:
    """负样本生成器"""

    def generate(self, positive_pair):
        """生成负样本"""
        return [
            # 1. SQL 语法错误
            self._introduce_syntax_error(positive_pair),

            # 2. Schema 不匹配
            self._wrong_schema_reference(positive_pair),

            # 3. 语义偏离
            self._semantic_drift(positive_pair)
        ]
```

### 5.3 模型训练阶段

#### 预训练模型选择

**决策树**:
```
是否有大量标注数据?
├── 是 → 微调方案
│   ├── 有 GPU → LoRA/QLoRA
│   └── 无 GPU → API 微调
└── 否 → RAG + Prompt Engineering
```

**推荐模型**:
- **商业 API**: GPT-4o, Claude 3.5 Sonnet
- **开源微调**: CodeLlama 34B, DeepSeek Coder, Qwen

#### LoRA/QLoRA 参数配置

```yaml
# QLoRA 配置 (低显存)
qlora_config:
  base_model: "codellama/CodeLlama-34b-hf"
  quantization_bit: 4
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  learning_rate: 1e-4
  warmup_ratio: 0.1
  gradient_accumulation_steps: 4
  per_device_train_batch_size: 2

# LoRA 配置 (高性能)
lora_config:
  base_model: "codellama/CodeLlama-34b-hf"
  lora_rank: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  learning_rate: 2e-4
  warmup_ratio: 0.05
  gradient_accumulation_steps: 2
  per_device_train_batch_size: 4
```

#### 训练流程

```python
# 完整训练流程
async def train_model(config):
    # 1. 数据准备
    dataset = prepare_dataset(config.dataset_path)
    train_set, val_set = split_dataset(dataset, ratio=0.9)

    # 2. 模型加载
    model, tokenizer = load_model(config.base_model)

    # 3. LoRA 配置
    lora_config = get_lora_config(config)
    model = apply_lora(model, lora_config)

    # 4. 训练
    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=val_set,
        args=config.training_args
    )

    # 5. Checkpoint 管理
    callbacks = [
        SaveCheckpointCallback(save_strategy="epoch"),
        EarlyStoppingCallback(patience=3),
        EvaluationCallback(evaluate_every=1000)
    ]

    # 6. 训练执行
    trainer.train(callbacks=callbacks)

    # 7. 模型保存
    save_model(model, tokenizer, config.output_dir)

    return trainer
```

#### 评估指标

```python
class EvaluationMetrics:
    """Text-to-SQL 评估指标"""

    @staticmethod
    def exact_match(predicted_sql, gold_sql):
        """精确匹配"""
        # SQL 规范化后比较
        normalized_pred = normalize_sql(predicted_sql)
        normalized_gold = normalize_sql(gold_sql)
        return normalized_pred == normalized_gold

    @staticmethod
    def execution_accuracy(predicted_sql, gold_sql, db_connection):
        """执行准确率"""
        # 执行两个 SQL，比较结果
        pred_result = db_connection.execute(predicted_sql)
        gold_result = db_connection.execute(gold_sql)
        return compare_results(pred_result, gold_result)

    @staticmethod
    def schema_recall(predicted_sql, gold_schema):
        """Schema 召回率"""
        # 检查是否使用了正确的表和列
        used_tables = extract_tables(predicted_sql)
        used_columns = extract_columns(predicted_sql)

        table_recall = len(set(used_tables) & set(gold_schema.tables)) / len(gold_schema.tables)
        column_recall = len(set(used_columns) & set(gold_schema.columns)) / len(gold_schema.columns)

        return (table_recall + column_recall) / 2
```

### 5.4 推理部署阶段

#### 在线服务架构

```python
class TextToSQLService:
    """Text-to-SQL 在线服务"""

    def __init__(self):
        self.model = self._load_model()
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter()

    async def query(self, request: QueryRequest):
        # 1. 缓存检查
        cache_key = self._get_cache_key(request)
        if cached := await self.cache.get(cache_key):
            return cached

        # 2. 限流检查
        if not await self.rate_limiter.check(request.user_id):
            raise RateLimitError()

        # 3. 复杂度评估
        complexity = self._assess_complexity(request.question)

        # 4. 路由决策
        if complexity < 0.3:
            result = await self._simple_path(request)
        elif complexity < 0.7:
            result = await self._standard_path(request)
        else:
            result = await self._agent_path(request)

        # 5. 缓存结果
        await self.cache.set(cache_key, result, ttl=3600)

        return result
```

#### 缓存策略

```python
class CacheManager:
    """多级缓存管理"""

    def __init__(self):
        self.l1_cache = {}  # 内存缓存 (热点数据)
        self.l2_cache = Redis()  # Redis 缓存
        self.vector_index = VectorIndex()  # 向量相似度检索

    async def get(self, key):
        # L1 缓存
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2 缓存
        if value := await self.l2_cache.get(key):
            self.l1_cache[key] = value
            return value

        # 向量相似度检索
        similar = await self.vector_index.search(key, top_k=3)
        if similar and similar[0].score > 0.9:
            return similar[0].value

        return None
```

#### 延迟优化

```python
class LatencyOptimizer:
    """延迟优化器"""

    async def optimize_inference(self, question):
        # 1. 并行执行
        schema_retrieval, example_retrieval = await asyncio.gather(
            self._retrieve_schema(question),
            self._retrieve_examples(question)
        )

        # 2. 批处理
        if self._batch_ready():
            results = await self._batch_infer()
        else:
            results = await self._single_infer(question)

        # 3. 流式输出
        if self._should_stream():
            return self._stream_results(results)

        return results
```

### 5.5 前端交互设计

#### 查询输入界面

```typescript
interface QueryInputProps {
  placeholder?: string;
  suggestions?: string[];
  onQuery: (question: string) => void;
}

function QueryInput({ placeholder, suggestions, onQuery }: QueryInputProps) {
  const [value, setValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  return (
    <div className="query-input-container">
      <Autocomplete
        value={value}
        onChange={setValue}
        suggestions={suggestions}
        placeholder={placeholder || "输入您的查询问题..."}
        onSubmit={onQuery}
      />
      <QuickActions>
        <ActionButton icon="table" hint="查看表结构" />
        <ActionButton icon="history" hint="历史查询" />
        <ActionButton icon="template" hint="查询模板" />
      </QuickActions>
    </div>
  );
}
```

#### SQL 预览和编辑

```typescript
interface SQLPreviewProps {
  sql: string;
  readonly?: boolean;
  onEdit?: (sql: string) => void;
  onExecute?: () => void;
}

function SQLPreview({ sql, readonly, onEdit, onExecute }: SQLPreviewProps) {
  return (
    <div className="sql-preview">
      <div className="sql-header">
        <span className="label">生成的 SQL</span>
        <div className="actions">
          <Button icon="copy" onClick={copyToClipboard} />
          {!readonly && <Button icon="edit" onClick={() => setEditing(true)} />}
          <Button icon="play" onClick={onExecute} />
        </div>
      </div>
      <SQLEditor
        value={sql}
        onChange={onEdit}
        readonly={readonly}
        language="sql"
      />
      <SQLValidationStatus sql={sql} />
    </div>
  );
}
```

#### 结果可视化

```typescript
interface ResultVisualizationProps {
  results: QueryResult[];
  queryType: 'select' | 'aggregate' | 'join';
}

function ResultVisualization({ results, queryType }: ResultVisualizationProps) {
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table');

  return (
    <div className="result-visualization">
      <ViewToggle value={viewMode} onChange={setViewMode} />

      {viewMode === 'table' && (
        <DataTable data={results} sortable filterable />
      )}

      {viewMode === 'chart' && (
        <AutoChart data={results} queryType={queryType} />
      )}

      <ResultSummary results={results} />
    </div>
  );
}
```

### 5.6 后端 API 设计

#### RESTful API

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI(title="Text-to-SQL API")

class QueryRequest(BaseModel):
    question: str
    db_name: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    sql: str
    explanation: str
    results: List[Dict]
    execution_time: float

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest, current_user: User = Depends(get_current_user)):
    """生成并执行 SQL 查询"""
    pipeline = await get_pipeline()
    result = await pipeline.process(
        question=request.question,
        db_name=request.db_name,
        session_id=request.session_id
    )
    return result

@app.post("/api/v1/chat")
async def chat(message: ChatMessage):
    """多轮对话接口"""
    # WebSocket 或长轮询
    pass

@app.get("/api/v1/schema/{db_name}")
async def get_schema(db_name: str):
    """获取数据库 Schema"""
    pipeline = await get_pipeline()
    return await pipeline.get_schema_info(db_name)

@app.post("/api/v1/feedback")
async def submit_feedback(feedback: Feedback):
    """用户反馈收集"""
    await feedback_service.save(feedback)
    return {"status": "success"}
```

#### WebSocket 实时交互

```python
from fastapi import WebSocket

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())

    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()

            # 发送处理状态
            await websocket.send_json({
                "type": "status",
                "message": "正在分析问题..."
            })

            # 处理查询
            result = await process_query(data["question"], session_id)

            # 发送结果
            await websocket.send_json({
                "type": "result",
                "data": result
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
```

### 5.7 持续优化

#### 用户反馈收集

```python
class FeedbackCollector:
    """用户反馈收集器"""

    async def collect_feedback(self, query_id, feedback):
        """收集用户反馈"""
        await self.feedback_db.insert({
            "query_id": query_id,
            "rating": feedback.rating,  # 1-5 星
            "sql_correct": feedback.sql_correct,
            "result_correct": feedback.result_correct,
            "user_edited_sql": feedback.edited_sql,
            "comment": feedback.comment,
            "timestamp": datetime.now()
        })

    async def analyze_feedback(self):
        """分析反馈数据"""
        # 计算满意度
        avg_rating = await self.feedback_db.avg("rating")

        # 找出常见错误模式
        error_patterns = await self.feedback_db.aggregate([
            {"$match": {"sql_correct": False}},
            {"$group": {"_id": "$error_type", "count": {"$sum": 1}}}
        ])

        return {
            "average_rating": avg_rating,
            "error_patterns": error_patterns
        }
```

#### A/B 测试

```python
class ABTestManager:
    """A/B 测试管理器"""

    def __init__(self):
        self.experiments = self._load_experiments()

    async def route_request(self, user_id, request):
        """路由请求到不同实验组"""
        experiment = self.experiments["sql_generation"]

        # 一致性哈希分配
        group = hash(user_id) % len(experiment.groups)
        config = experiment.groups[group]

        return await self._execute_with_config(request, config)

    async def evaluate_experiment(self, experiment_id):
        """评估实验结果"""
        # 收集各组的指标
        metrics = await self._collect_metrics(experiment_id)

        # 统计显著性检验
        winner = self._statistical_test(metrics)

        return winner
```

#### 模型迭代更新

```python
class ModelLifecycleManager:
    """模型生命周期管理"""

    async def should_retrain(self):
        """判断是否需要重新训练"""
        # 1. 检查性能下降
        recent_accuracy = await self._get_recent_accuracy()
        if recent_accuracy < self.baseline_accuracy - 0.05:
            return True

        # 2. 检查新数据量
        new_data_count = await self._get_new_data_count()
        if new_data_count > 1000:
            return True

        # 3. 定期重新训练
        days_since_last_training = (datetime.now() - self.last_training_date).days
        if days_since_last_training > 30:
            return True

        return False

    async def retrain_model(self):
        """重新训练模型"""
        # 1. 收集新数据
        new_data = await self._collect_new_data()

        # 2. 合并历史数据
        training_data = self._merge_with_historical(new_data)

        # 3. 数据增强
        augmented_data = self._augment_data(training_data)

        # 4. 训练新模型
        new_model = await self._train_model(augmented_data)

        # 5. 评估新模型
        metrics = await self._evaluate_model(new_model)

        # 6. 灰度发布
        if metrics.accuracy > self.current_model_accuracy:
            await self._gradual_rollout(new_model)
```

---

## 参考资料

### 数据集
- [Spider Dataset](https://yale-lily.github.io/spider)
- [BIRD Benchmark](https://bird-bench.github.io/)
- [CSpider 中文数据集](https://github.com/yyyflint2/Cspider)

### 论文
- **RAT-SQL** (ACL 2020): https://github.com/Microsoft/rat-sql
- **PICARD** (ICLR 2022): https://arxiv.org/abs/2109.05093
- **DB-Surfer**: Multi-Agent 协作架构

### 开源项目
- [Vanna.ai](https://github.com/vanna-ai/vanna) - MIT 许可，20,000+ stars
- [LangChain SQL](https://python.langchain.com/docs/use_cases/sql)
- [DB-GPT-Hub](https://gitee.com/googx/DB-GPT-Hub) - LoRA/QLoRA 实现

### 工具和框架
- **ChromaDB**: 轻量级向量数据库
- **Qdrant**: 高性能向量数据库
- **Milvus**: 大规模向量数据库
- **vLLM**: 高性能推理引擎

---

*文档版本: v1.0*
*创建时间: 2025-03-05*
*基于项目: text-to-sql*
