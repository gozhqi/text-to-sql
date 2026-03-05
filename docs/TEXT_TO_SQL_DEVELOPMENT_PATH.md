# Text-to-SQL 系统开发路径指南

> 基于项目深度分析 + 业界 SOTA 研究 + Spider 2.0 Benchmark 的完整开发路线图
>
> **更新时间**: 2025-03-05 | **文档版本**: v2.0

---

## 执行摘要

> **核心结论（2024-2025 业界共识）**：直接 Text-to-SQL 在生产环境是"高风险、低收益"方案。企业级系统必须包含 Semantic Layer。

**关键发现：**
- Spider 2.0 数据集显示，企业级数据库平均有 **800+ 字段**
- AutoLink 方法证明，Schema Linking 能提升 **27.2%** 的召回率
- Multi-Agent 架构在复杂查询上比单次生成高 **20%** 以上

---

## 第一部分：技术讨论与深度分析

### 1.1 当前项目架构深度分析

#### 核心组件评估

| 组件 | 当前实现 | 优点 | 缺点 | 改进方向 |
|------|----------|------|------|----------|
| **Schema Retriever** | 向量检索+关键词混合 | 召回率较好 | 缺少Schema Linking、字段级映射 | 增加AutoLink风格的语义链接 |
| **SQL Generator** | 单次LLM调用 | 简单直接 | 无思维链、无自我修正 | 加入CoT和Refinement循环 |
| **Context Manager** | 会话+轮次管理 | 支持多轮 | 缺少状态累积、条件继承 | 添加filter state管理 |
| **SQL Validator** | 规则-based校验 | 安全性好 | 无语法校验、无修复能力 | 集成SQL解析器 |
| **Prompt Builder** | 模板化构建 | 结构清晰 | Few-shot静态、无示例检索 | 动态示例检索 |

#### 架构层面的问题

```
当前架构：Linear Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question → Schema检索 → Prompt构建 → LLM → 校验 → 执行
           ↓              ↓            ↓
        ChromaDB      静态模板      单次调用

问题分析：
1. ❌ 缺少 Semantic Layer - 用户需要了解表名/字段名
2. ❌ 缺少 Few-shot 动态检索 - 只有3个硬编码示例
3. ❌ 缺少自我修正 - SQL错误后无法自动修复
4. ❌ 缺少 Schema Linking - "销售额"→哪个字段？
5. ❌ 单点故障 - Schema检索错误→SQL全错

业界SOTA架构对比：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DB-Surfer (Spider 2.0: 59.78%)
┌──────────────┐
│ Orchestrator │ ← 总控Agent，任务分解
└──────┬───────┘
       ├────→ Schema Agent (探索Schema)
       ├────→ SQL Agent (生成SQL)
       └────→ Validator Agent (测试修复)

AutoLink (91.2% Schema Recall)
┌──────────────┐
│ Schema Linker│ ← 专门的语义链接模块
└──────┬───────┘
       ├────→ 表名链接
       ├────→ 列名链接
       └────→ 值映射 ("上个月"→SQL表达式)
```

#### 关键差距分析

```
能力维度雷达图：

                 Schema理解
                    △
                    /|\
                   / | \
        多轮对话——●—— | ——●—— SQL准确性
         (65%)     \ | /    (70%)
                   \|/
        安全性——●——— | ———●—— 扩展性
        (90%)        |      (60%)

当前项目 vs 业界SOTA：
- Schema理解: 80% → 91% (AutoLink)
- SQL准确性: 70% → 60% (DB-Surfer执行准确率)
- 多轮对话: 65% → 80% (完善状态管理)
- 安全性: 90% ✓ (已达标准)
- 扩展性: 60% → 85% (Semantic Layer)
```

---

### 1.2 业界SOTA方案深度对比

#### 方案一：Vanna.ai（RAG驱动）

```
核心思想：SQL即知识

┌─────────────────────────────────────────────────────────┐
│                    Vanna.ai 架构                        │
├─────────────────────────────────────────────────────────┤
│  训练阶段                                               │
│  DDL + 业务文档 + 正确SQL → 向量化 → 知识库            │
│                                                         │
│  查询阶段                                               │
│  用户问题 → 检索相似SQL → Prompt → LLM → SQL           │
└─────────────────────────────────────────────────────────┘

关键设计：
1. 将SQL作为"知识"存储，而不仅仅是Schema
2. 用户可以训练：添加正确SQL到知识库
3. 支持SQL-as-you-type：实时反馈

适用场景：
- 有大量历史SQL查询
- 查询模式相对固定
- 业务分析师使用

不适用场景：
- 冷启动（无历史数据）
- 探索性查询（全新问题）
```

#### 方案二：LangChain SQL Agent（工具调用）

```
核心思想：LLM作为推理引擎，动态调用工具

┌─────────────────────────────────────────────────────────┐
│                 LangChain SQL Agent                     │
├─────────────────────────────────────────────────────────┤
│  ReAct循环：                                            │
│  Thought → Action → Observation → Thought → ...        │
│                                                         │
│  可用工具：                                             │
│  - sql_db_list_tables                                  │
│  - sql_db_schema                                       │
│  - sql_db_query                                        │
│  - sql_db_query_checker                                │
│                                                         │
│  工作流程：                                             │
│  1. 我需要查询销售额 → 列出表                          │
│  2. 看到orders表 → 获取schema                          │
│  3. 知道有amount字段 → 生成SQL                         │
│  4. 不确定语法 → 检查SQL                               │
│  5. 修正后执行                                          │
└─────────────────────────────────────────────────────────┘

优势：
✅ 可以处理未知Schema（动态探索）
✅ 自我纠错能力强
✅ 适合复杂、多步查询

劣势：
❌ LLM调用次数多（成本高）
❌ 不确定性（可能循环）
❌ 延迟高
```

#### 方案三：DB-Surfer（Multi-Agent）

```
Spider 2.0 数据集上的 SOTA 方案

┌─────────────────────────────────────────────────────────┐
│                  DB-Surfer 架构                         │
├─────────────────────────────────────────────────────────┤
│  总分 → 协调 → 总控策略                                 │
│                                                         │
│  ┌──────────────────────────────────────────┐          │
│  │           Schema Explorer Agent          │          │
│  │  - 识别相关表                             │          │
│  │  - 理解表关系                             │          │
│  │  - 找到JOIN路径                           │          │
│  └──────────────────────────────────────────┘          │
│                    ↓                                   │
│  ┌──────────────────────────────────────────┐          │
│  │            SQL Generator Agent           │          │
│  │  - 基于Schema生成SQL                     │          │
│  │  - 处理复杂条件                           │          │
│  │  - 优化查询结构                           │          │
│  └──────────────────────────────────────────┘          │
│                    ↓                                   │
│  ┌──────────────────────────────────────────┐          │
│  │           Validator Agent                │          │
│  │  - 语法检查                               │          │
│  │  - 测试执行（dry-run）                    │          │
│  │  - 错误修复                               │          │
│  └──────────────────────────────────────────┘          │
│                                                         │
│  通过Agent间的协作解决复杂问题                          │
└─────────────────────────────────────────────────────────┘

关键创新：
1. "总-分-总"策略：先规划→分步执行→合并结果
2. Agent间通信：共享上下文，协作决策
3. 测试执行：LIMIT 1验证SQL正确性
```

#### 方案四：AutoLink（Schema Linking）

```
专门解决Schema理解问题

┌─────────────────────────────────────────────────────────┐
│                   AutoLink 模块                         │
├─────────────────────────────────────────────────────────┤
│  输入：用户问题 + 完整Schema                            │
│  输出：Schema Link（词汇→Schema元素映射）               │
│                                                         │
│  三种链接：                                             │
│  ┌──────────────────────────────────────────────┐      │
│  │ 1. Table Linking                              │      │
│  │    "查询订单" → orders表                      │      │
│  │                                              │      │
│  │ 2. Column Linking                             │      │
│  │    "销售额" → orders.total_amount            │      │
│  │                                              │      │
│  │ 3. Value Linking                              │      │
│  │    "上个月" → DATE_SUB(NOW(), INTERVAL 1 MONTH)│      │
│  └──────────────────────────────────────────────┘      │
│                                                         │
│  效果：在Spider 2.0上达到91.2%的Schema Recall          │
└─────────────────────────────────────────────────────────┘

为什么重要：
- Schema理解是Text-to-SQL的瓶颈
- 用户词汇与表名/字段名往往不一致
- 值映射（如"上个月"）需要额外逻辑
```

---

### 1.3 核心设计决策讨论

#### 决策1：是否需要Semantic Layer？

**企业级生产：YES**

```
问题：当数据库有100+张表时，用户怎么知道用哪张表？

方案A（当前项目）：直接Text-to-SQL
用户：查询Q1销售额
系统：需要用户知道哪个表存储销售数据 → ❌

方案B：Semantic Layer
用户：查询Q1销售额
系统：解析语义 → revenue + Q1 → SQL → ✅
```

**Semantic Layer定义：**
```python
# 语义层定义（业务视角）
METRICS = {
    "revenue": {
        "name": "销售收入",
        "base_expression": "SUM(order_items.quantity * order_items.price)",
        "required_joins": ["orders", "order_items"],
        "default_filters": ["orders.status = 'completed'"]
    },
    "active_users": {
        "name": "活跃用户",
        "base_expression": "COUNT(DISTINCT user_id)",
        "time_grain": "daily"
    }
}

DIMENSIONS = {
    "time": ["date", "week", "month", "quarter"],
    "region": ["country", "city", "district"],
    "category": ["product_category", "subcategory"]
}

# 查询流程
"查询Q1各地区的销售额"
  ↓
parse: metrics=[revenue], dimensions=[region], filters=[Q1]
  ↓
to_sql: SELECT region, SUM(...) FROM ... WHERE date >= '2024-01-01'
```

**结论：**
- 内部工具/小规模Schema：直接Text-to-SQL
- 企业级/大规模Schema：Semantic Layer + Text-to-SQL混合

#### 决策2：单次生成 vs Multi-Agent？

```
根据查询复杂度动态选择：

┌─────────────────────────────────────────────────────┐
│                   复杂度评估                         │
├─────────────────────────────────────────────────────┤
│  简单（< 0.3）                                      │
│  → 单次生成（快、便宜）                             │
│                                                     │
│  中等（0.3 - 0.7）                                  │
│  → Schema检索 → CoT生成 → 校验                      │
│                                                     │
│  复杂（> 0.7）                                      │
│  → Multi-Agent（Schema Explorer → SQL → Validator）│
└─────────────────────────────────────────────────────┘

复杂度评分因素：
- 需要的表数量（>3张 → 复杂）
- 是否需要子查询
- 是否需要窗口函数
- 条件复杂度（多个AND/OR）
```

#### 决策3：Few-shot策略？

```
问题：硬编码示例 vs 动态检索？

结论：动态检索 + 多样性保证

检索策略：
1. 向量检索：问题 vs 历史Query-SQL对
2. 覆盖模式：确保包含单表、JOIN、聚合、子查询
3. 相同表：优先使用涉及相同表的示例

示例管理：
┌─────────────────────────────────────────────────────┐
│  示例来源                                            │
│  ├─ 冷启动：手工精选高质量示例（10-20个）            │
│  ├─ 用户反馈：用户确认正确的SQL                      │
│  ├─ 数据分析师：专家编写的SQL                        │
│  └─ 自动筛选：执行成功的SQL + 用户满意              │
│                                                     │
│  质量管理                                            │
│  ├─ 定期清理低质量示例                               │
│  ├─ 示例去重（避免重复）                             │
│  └─ 覆盖度监控（确保模式多样性）                     │
└─────────────────────────────────────────────────────┘
```

---

### 1.2 架构模式讨论

**模式 A: Pipeline 模式（当前项目）**

```
Question → Schema检索 → Prompt构建 → LLM生成 → SQL校验 → 执行
                ↑
           向量索引
```

优点：
- 流程清晰，易于理解
- 每个模块可独立测试
- 便于扩展（插入新模块）

缺点：
- 串行执行，延迟累加
- 错误传播（Schema 检错 → SQL 全错）
- 难以动态调整

**模式 B: Agent 模式（LangChain SQL Agent）**

```
Question → Agent → [工具调用循环] → Result
                  ├── get_schema
                  ├── query_checker  
                  ├── query_sql
                  └── ...
```

优点：
- 灵活，可以自我纠错
- 可以处理复杂场景
- 支持多步推理

缺点：
- 不确定性高（可能无限循环）
- 成本难以控制
- 调试困难

**模式 C: RAG + Generation 模式（Vanna.ai）**

```
Question → RAG检索相似SQL → Prompt构建 → LLM生成 → 执行
              ↑
         SQL知识库（历史查询+正确SQL）
```

优点：
- 利用历史知识，准确率高
- 新增数据可以增量学习
- 适合有大量历史查询的场景

缺点：
- 冷启动问题（没有历史数据时）
- 需要维护知识库

**结论：融合模式**

```
┌─────────────────────────────────────────────────────────────┐
│                     Text-to-SQL 系统架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ 用户问题 │───→│  Schema检索  │───→│  相关表+字段     │   │
│  └─────────┘    └──────────────┘    └──────────────────┘   │
│                       │                         │          │
│                       ↓                         ↓          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ RAG:相似SQL  │←─│  Prompt构建  │←─│  Few-shot Examples│  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                  │                               │
│         ↓                  ↓                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    LLM 生成                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │   │
│  │  │ 思维链推理 │→│ SQL生成   │→│ 自我纠错(可选)  │  │   │
│  │  └───────────┘  └───────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│                           ↓                                │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  SQL校验     │───→│  安全检查    │───→│  执行/返回  │  │
│  └──────────────┘    └──────────────┘    └─────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               多轮对话上下文管理                       │  │
│  │  Session → Turns → Context Summary → Question Rewrite │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 1.3 业界方案对比

| 方案 | 核心特点 | 适用场景 | 开源 |
|------|----------|----------|------|
| **Vanna.ai** | RAG + 向量检索，利用历史SQL | 有历史查询数据 | ✅ |
| **LangChain SQL Agent** | Agent 工具调用，自我纠错 | 复杂推理场景 | ✅ |
| **LlamaIndex SQL** | 索引优化，多表连接 | 大规模 Schema | ✅ |
| **Uber QueryGPT** | 内部知识 + 自然语言 | 企业级应用 | ❌ |
| **当前项目** | Pipeline + 多轮对话 | 中小规模应用 | ✅ |

**Vanna.ai 核心思想学习：**

```python
# Vanna 的核心流程
1. 训练阶段：将 DDL + 文档 + 历史SQL 存入向量库
2. 检索阶段：根据问题检索相似的 DDL + 历史SQL
3. 生成阶段：用检索到的上下文构建 Prompt

# 关键创新：SQL 作为「知识」存储
# 而不仅仅是 Schema 信息
```

**可借鉴点：**
- 增加「SQL 知识库」模块
- 支持从历史查询中学习
- 用户反馈机制（正确/错误的 SQL）

---

## 第二部分：标准开发路径

### Level 1: 基础版（MVP - 2-3周）

**目标：** 快速验证核心能力

```
┌─────────────────────────────────────────────────────────┐
│                    MVP 架构                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户问题 → Schema检索 → Prompt → LLM → SQL → 执行    │
│                  ↓            ↓                        │
│              ChromaDB      Few-shot示例                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**实现清单：**

```python
# Week 1: 基础设施
✅ app/core/database.py
   - 多数据库支持（MySQL/PostgreSQL）
   - Schema自动提取
   - 安全执行（LIMIT、超时）

✅ app/core/schema_retriever.py
   - 向量索引构建
   - 混合检索（向量+关键词）

# Week 2: 核心生成
✅ app/core/prompt_builder.py
   - System Prompt模板
   - Schema描述格式化
   - 静态Few-shot示例

✅ app/core/sql_generator.py
   - LLM调用封装
   - 响应解析（JSON/SQL）

# Week 3: 校验与API
✅ app/core/sql_validator.py
   - 危险操作检测
   - SQL注入防护

✅ app/main.py
   - FastAPI接口
   - 错误处理
```

**适用场景：**
- 简单查询（单表、简单JOIN）
- Schema规模小（<20张表）
- 内部工具/原型验证

---

### Level 2: 增强版（生产可用 - 4-6周）

**新增核心能力：**

```
┌─────────────────────────────────────────────────────────┐
│                  增强版架构                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户问题                                              │
│     ↓                                                   │
│  ┌──────────────────┐                                  │
│  │  意图分类        │ ← 新增：决策处理路径            │
│  └────────┬─────────┘                                  │
│           ↓                                             │
│  ┌──────────────────┐                                  │
│  │  Schema Linking  │ ← 新增：字段级语义映射          │
│  └────────┬─────────┘                                  │
│     ┌──────┴──────┐                                     │
│     ↓             ↓                                    │
│  Few-shot检索  动态示例  ← 新增：RAG检索              │
│     ↓             ↓                                    │
│  ┌──────────────────┐                                  │
│  │  CoT SQL生成    │ ← 新增：思维链推理              │
│  └────────┬─────────┘                                  │
│           ↓                                             │
│  ┌──────────────────┐                                  │
│  │  自我修正循环    │ ← 新增：错误自修复              │
│  │  (最多3轮)       │                                  │
│  └────────┬─────────┘                                  │
│           ↓                                             │
│        执行 → 返回                                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              多轮对话上下文                     │   │
│  │  状态累积 + 问题改写 + 意图追踪                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Week 4-5: Schema Linking**

```python
# app/core/schema_linker.py (NEW)

class SchemaLinker:
    """解决"用户词汇"→"数据库字段"的映射问题"""

    def link(self, question: str, schema: TableSchema) -> SchemaLink:
        """
        输入: "查询上个月的销售额"
        输出: SchemaLink(
            table_mentions=["orders", "order_items"],
            column_mappings={
                "销售额": "total_amount",
                "订单": "orders"
            },
            value_mappings={
                "上个月": "DATE_SUB(NOW(), INTERVAL 1 MONTH)"
            },
            join_paths=[
                JoinPath(from="orders", to="order_items", on="order_id")
            ]
        )
        """

    # 实现策略
    def _link_tables(self, question: str, schemas: List[TableSchema]):
        # 1. 关键词匹配（表名、注释）
        # 2. 向量相似度（问题 vs 表描述）
        # 3. 外键传播（选中A表 → 包含关联的B表）

    def _link_columns(self, question: str, table: TableSchema):
        # 1. 字段名模糊匹配
        # 2. 字段注释语义匹配
        # 3. 枚举值匹配（如status="completed"）

    def _link_values(self, question: str):
        # 处理常见值映射
        VALUE_PATTERNS = {
            r"上个月": "DATE_SUB(NOW(), INTERVAL 1 MONTH)",
            r"今年": "YEAR(NOW())",
            r"昨天": "DATE_SUB(NOW(), INTERVAL 1 DAY)",
        }
```

**Week 5-6: 动态Few-shot检索**

```python
# app/core/example_retriever.py (NEW)

class ExampleRetriever:
    """动态检索相似Query-SQL对"""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.seed_examples = self._load_seed_examples()

    async def retrieve(self, question: str, k: int = 3) -> List[Example]:
        # 1. 向量检索最相似的k*2个示例
        candidates = await self.vector_store.search(question, k=k*2)

        # 2. 多样性重排序（避免示例太相似）
        diverse = self._diversify(candidates, k)

        # 3. 确保覆盖不同模式
        return self._ensure_coverage(diverse, patterns=[
            "single_table", "join", "aggregate", "subquery"
        ])

    def _diversify(self, examples: List[Example], k: int) -> List[Example]:
        """使用Max Marginal Relevance算法"""
        selected = []
        while len(selected) < k and examples:
            # 选择与已选示例最不相似的新示例
            best = max(examples, key=lambda ex: self._mmr_score(ex, selected))
            selected.append(best)
            examples.remove(best)
        return selected
```

**Week 6: 自我修正机制**

```python
# app/core/sql_refiner.py (NEW)

class SQLRefiner:
    """SQL生成后的自我修正循环"""

    async def refine(
        self,
        sql: str,
        error: Optional[str],
        schema: TableSchema,
        max_iterations: int = 3
    ) -> RefinementResult:

        for iteration in range(max_iterations):
            # 1. 语法检查
            syntax_valid, syntax_error = await self._check_syntax(sql)
            if not syntax_valid:
                sql = await self._fix_syntax(sql, syntax_error, schema)
                continue

            # 2. Schema验证
            schema_valid, schema_error = self._validate_schema(sql, schema)
            if not schema_valid:
                sql = await self._fix_schema(sql, schema_error, schema)
                continue

            # 3. 测试执行（LIMIT 1）
            try:
                await self.db.execute(sql + " LIMIT 1")
                return RefinementResult(success=True, sql=sql)
            except Exception as e:
                # 分析错误类型，生成修复Prompt
                sql = await self._fix_execution_error(sql, str(e), schema)

        return RefinementResult(success=False, last_sql=sql)

    async def _fix_execution_error(self, sql: str, error: str, schema: TableSchema):
        """基于执行错误修复SQL"""
        fix_prompt = f"""
        SQL执行失败：
        ```sql
        {sql}
        ```
        错误信息：{error}

        请分析错误原因并修复SQL。只返回修复后的SQL。
        """
        # 调用LLM修复
        response = await self.llm.generate(fix_prompt)
        return self._extract_sql(response)
```

---

### Level 3: 企业版（8-12周）

**核心特性：Semantic Layer + Multi-Agent**

```
┌─────────────────────────────────────────────────────────┐
│                 企业版架构                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户问题                                              │
│     ↓                                                   │
│  ┌──────────────────┐                                  │
│  │  Semantic Layer  │ ← NEW: 业务语义抽象             │
│  │  解析            │                                  │
│  └────────┬─────────┘                                  │
│           ↓                                             │
│     ┌─────┴──────┐                                     │
│     ↓            ↓                                     │
│  指标命中?   未命中                                     │
│     ↓            ↓                                     │
│  SQL模板    传统Text2SQL (Level 2)                     │
│     ↓            ↓                                     │
│  ┌────────────────────────────┐                       │
│  │   Multi-Agent协调器        │ ← NEW: 复杂查询处理   │
│   ──┬────────┬────────┬────────┤                       │
│      ↓        ↓        ↓        ↓                       │
│   Schema   SQL    Validator Result                     │
│   Agent   Agent   Agent  Merger                        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              安全治理层                         │   │
│  │  行级权限 + 敏感数据脱敏 + 审计日志              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Week 7-9: Semantic Layer实现**

```python
# app/core/semantic_layer.py (NEW)

class SemanticLayer:
    """业务语义抽象层"""

    def __init__(self, config_path: str):
        self.metrics = self._load_metrics(config_path)
        self.dimensions = self._load_dimensions(config_path)

    def parse_query(self, question: str) -> SemanticQuery:
        """
        将自然语言解析为语义查询

        "查询Q1各地区的销售额"
        → SemanticQuery(
            metrics=["revenue"],
            dimensions=["region"],
            filters={"time": "Q1"},
            aggregation="SUM"
          )
        """
        # 使用LLM进行语义解析
        semantic_prompt = self._build_semantic_prompt(question)
        response = await self.llm.generate(semantic_prompt)
        return SemanticQuery.parse(response)

    def to_sql(self, query: SemanticQuery) -> str:
        """将语义查询转换为SQL"""
        # 获取每个metric的SQL模板
        metric_defs = [self.metrics[m] for m in query.metrics]

        # 构建JOIN
        required_tables = self._collect_required_tables(metric_defs)
        join_clauses = self._build_joins(required_tables)

        # 构建WHERE
        where_clauses = [self._translate_filter(f, query.filters[f])
                        for f in query.filters]

        # 构建GROUP BY
        group_by = [self.dimensions[d].sql_expression
                   for d in query.dimensions]

        # 组装SQL
        sql = f"""
        SELECT
            {', '.join(group_by)},
            {', '.join(m.sql_expression for m in metric_defs)}
        FROM {', '.join(required_tables)}
        {join_clauses}
        WHERE {' AND '.join(where_clauses)}
        GROUP BY {', '.join(group_by)}
        """
        return sql
```

**Metric定义（YAML）：**

```yaml
# config/metrics.yaml
metrics:
  revenue:
    name: 销售收入
    base_expression: "SUM(order_items.quantity * order_items.price)"
    required_tables:
      - order_items
      - orders
    required_joins:
      - target: orders
        condition: "order_items.order_id = orders.order_id"
    default_filters:
      - condition: "orders.status = 'completed'"
        description: "只计算已完成订单"
    dimensions:
      - time:
          column: orders.created_at
          grain: [day, week, month, quarter, year]
      - region:
          column: orders.shipping_region
      - category:
          column: products.category
          join_required: true

  active_users:
    name: 活跃用户数
    base_expression: "COUNT(DISTINCT user_id)"
    required_tables:
      - user_activities
    time_grain: daily
    definition: "有至少一次登录行为的用户"

dimensions:
  time:
    name: 时间
    grains:
      day: "DATE(created_at)"
      week: "DATE_TRUNC('week', created_at)"
      month: "DATE_TRUNC('month', created_at)"
      quarter: "DATE_TRUNC('quarter', created_at)"
      year: "DATE_TRUNC('year', created_at)"

  region:
    name: 地区
    hierarchy: [country, province, city, district]
```

**Week 10-12: Multi-Agent系统**

```python
# app/agents/base.py (NEW)

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Agent(ABC):
    """Agent基类"""

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        pass

    @property
    def tools(self) -> List[AgentTool]:
        """Agent可用的工具列表"""
        return []

# app/agents/orchestrator.py

class OrchestratorAgent(Agent):
    """总控Agent：任务分解与协调"""

    async def execute(self, task: AgentTask) -> AgentResult:
        # 1. 意图识别与复杂度评估
        intent = await self._classify_intent(task.question)
        complexity = await self._assess_complexity(task)

        # 2. 根据复杂度选择处理路径
        if complexity < 0.3:
            return await self._simple_path(task)
        elif complexity < 0.7:
            return await self._standard_path(task)
        else:
            return await self._agent_path(task)

    async def _agent_path(self, task: AgentTask):
        """Multi-Agent处理路径"""

        # 3. 任务分解
        subtasks = await self._decompose_task(task)

        # 4. Agent调度与执行
        results = []
        for subtask in subtasks:
            agent = self._get_agent(subtask.agent_type)
            result = await agent.execute(subtask)

            # Agent间结果传递
            if result.next_agent:
                subtask = self._update_context(subtask, result)
                agent = self._get_agent(result.next_agent)
                result = await agent.execute(subtask)

            results.append(result)

        # 5. 结果合并
        return await self._merge_results(results)

# app/agents/schema_agent.py

class SchemaAgent(Agent):
    """Schema探索Agent"""

    @property
    def tools(self) -> List[AgentTool]:
        return [
            AgentTool(
                name="list_tables",
                description="列出数据库中的所有表",
                function=self._list_tables
            ),
            AgentTool(
                name="get_table_schema",
                description="获取指定表的详细结构",
                function=self._get_table_schema
            ),
            AgentTool(
                name="find_join_path",
                description="找到两个表之间的关联路径",
                function=self._find_join_path
            ),
            AgentTool(
                name="search_columns",
                description="搜索包含特定关键词的列",
                function=self._search_columns
            ),
        ]

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        示例执行流程：
        1. 需要查询"销售额" → search_columns("销售")
        2. 找到orders.total_amount → get_table_schema("orders")
        3. 需要产品信息 → find_join_path("orders", "products")
        4. 返回Schema信息给SQLAgent
        """
        # 使用ReAct模式循环调用工具
        thought = f"需要理解问题：{task.question}"
        observations = []

        while not self._is_sufficient(observations):
            # 决策下一步行动
            action = await self._decide_action(thought, observations)

            # 执行工具
            tool = self.get_tool(action.tool_name)
            observation = await tool.function(**action.params)
            observations.append(observation)

            # 更新思考
            thought = await self._reason(thought, action, observation)

        return AgentResult(data=observations)

# app/agents/sql_agent.py

class SQLAgent(Agent):
    """SQL生成Agent"""

    @property
    def tools(self) -> List[AgentTool]:
        return [
            AgentTool(
                name="generate_sql",
                description="根据描述生成SQL",
                function=self._generate_sql
            ),
            AgentTool(
                name="validate_sql",
                description="验证SQL语法",
                function=self._validate_sql
            ),
            AgentTool(
                name="test_execute",
                description="测试执行SQL",
                function=self._test_execute
            ),
        ]

    async def execute(self, task: AgentTask) -> AgentResult:
        # 基于SchemaAgent提供的Schema信息生成SQL
        schema_context = task.context.get("schema", {})

        # 使用思维链生成SQL
        cot_result = await self._chain_of_thought(
            question=task.question,
            schema=schema_context
        )

        # 自我验证
        validation = await self._validate_sql(cot_result.sql)
        if not validation.is_valid:
            # 修正SQL
            cot_result.sql = await self._fix_sql(
                cot_result.sql,
                validation.error
            )

        return AgentResult(
            sql=cot_result.sql,
            reasoning=cot_result.reasoning
        )
```

**Agent协作示例：**

```
用户问题："查询每个地区的季度销售额趋势"

Orchestrator分析：
- 复杂度：高（多维度、时间序列）
- 决策：使用Multi-Agent

执行流程：
┌────────────────────────────────────────────────────────┐
│ Schema Agent:                                          │
│ 1. search_columns("销售") → 找到orders.total_amount   │
│ 2. search_columns("地区") → 找到orders.region        │
│ 3. get_table_schema("orders")                         │
│ 4. 返回: {"tables": ["orders"], "columns": [...]}    │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│ SQL Agent:                                             │
│ 思考: 需要按地区和时间分组                             │
│ 生成:                                                  │
│   SELECT                                               │
│     region,                                            │
│     DATE_TRUNC('quarter', created_at) as quarter,     │
│     SUM(total_amount) as revenue                      │
│   FROM orders                                          │
│   GROUP BY region, quarter                             │
│   ORDER BY region, quarter                             │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│ Validator Agent:                                       │
│ 1. validate_sql() → 语法正确                          │
│ 2. test_execute(LIMIT 1) → 执行成功                   │
│ 3. 返回: 验证通过                                      │
└────────────────────────────────────────────────────────┘
                         ↓
                     返回结果
```

---

## 第三部分：技术选型决策树

### LLM 选择决策

```
需要考虑的问题：
┌─────────────────────────────────────────────────────┐
│ 1. 成本敏感？                                        │
│    ├── 是 → 开源模型 + 本地部署                      │
│    │   ├── CodeLlama 34B (SQL能力强)                │
│    │   ├── DeepSeek Coder (推理能力强)              │
│    │   └── SQLCoder (专门针对SQL)                   │
│    └── 否 → 商业API                                  │
│        ├── GPT-4o (最全面)                          │
│        ├── Claude 3.5 Sonnet (长文本好)             │
│        └── Gemini Pro (便宜)                        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 2. 中文支持需求？                                    │
│    ├── 强需求                                        │
│    │   ├── Claude 3.5 (中文理解好)                  │
│    │   ├── DeepSeek-V3 (国产模型)                   │
│    │   └── Qwen (通义千问)                          │
│    └── 弱需求 → GPT-4o                               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 3. SQL复杂度？                                       │
│    ├── 简单查询 → GPT-4o-mini / 3.5-turbo          │
│    ├── 中等查询 → GPT-4o                             │
│    └── 复杂查询 → Claude 3.5 Sonnet / o1-preview   │
└─────────────────────────────────────────────────────┘
```

### 向量数据库选择

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **ChromaDB** | 零配置、Python原生 | 性能一般、功能简单 | 开发、原型、小规模 |
| **Qdrant** | 性能好、易部署、Filter强 | 社区较小 | 中等规模生产 |
| **Milvus** | 高性能、功能全、云原生 | 运维复杂、资源占用高 | 大规模生产 |
| **PGVector** | 使用现有PG基础设施 | 性能相对低 | 已有PG、简化架构 |

### Embedding模型选择

```
中文为主：
├── BAAI/bge-m3 (通用、多语言)
├── BAAI/bge-large-zh-v1.5 (中文专注)
└── text2vec-base-chinese (轻量)

英文为主：
├── OpenAI text-embedding-3-small (便宜)
├── OpenAI text-embedding-3-large (质量高)
└── voyage-law-2 (代码/SQL特化)

多语言：
├── paraphrase-multilingual-MiniLM-L12-v2 (轻量)
└── BAAI/bge-m3 (推荐)
```

### 数据库支持矩阵

| 数据库 | Schema提取 | SQL方言支持 | 生产推荐 |
|--------|-----------|------------|---------|
| MySQL | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| PostgreSQL | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| SQLite | ✅ | ⚠️ | ⭐⭐⭐ |
| Snowflake | ✅ | ✅ | ⭐⭐⭐⭐ |
| BigQuery | ✅ | ✅ | ⭐⭐⭐⭐ |
| SQL Server | ⚠️ | ⚠️ | ⭐⭐⭐ |

---

## 第四部分：常见陷阱与解决方案

### 陷阱1: Schema过载

**问题：** 把整个Schema塞给LLM，Token超限且注意力分散

```python
# ❌ BAD
prompt = f"""
完整的数据库Schema：
{all_100_tables_with_800_columns}
请生成SQL...
"""

# ✅ GOOD
relevant_tables = schema_retriever.retrieve(question, top_k=5)
prompt = f"""
相关表结构：
{relevant_tables}  # 只包含必要的5张表
请生成SQL...
"""
```

**解决方案：**
1. 实现Schema检索（向量+关键词）
2. 动态确定需要的表数量
3. 表间关系传播（选中A → 包含关联的B）

### 陷阱2: 硬编码Few-shot示例

**问题：** 示例不匹配时误导模型

```python
# ❌ BAD
EXAMPLES = [
    ("查询销售额", "SELECT SUM(amount) FROM orders"),
    # 只有固定示例，无法适配新场景
]

# ✅ GOOD
examples = await example_retriever.retrieve(question, k=3)
# 动态检索最相似的3个历史示例
# + 确保覆盖不同模式（单表、JOIN、聚合）
```

### 陷阱3: 忽略Schema Linking

**问题：** LLM猜错字段名

```
用户: "查询销售额"
LLM: SELECT sales FROM revenue_table  # ❌ 猜错表名
正确: SELECT total_amount FROM orders  # ✅ 有Schema Linking
```

**解决方案：**
```python
# 添加Schema Linking模块
linker = SchemaLinker()
mapping = linker.link("查询销售额", schema)
# 返回: {"销售额": "total_amount"}

prompt += f"\n# 词汇映射: {mapping}"
```

### 陷阱4: 缺少自我修正

**问题：** 一次生成错误率高

```python
# ❌ BAD
sql = await generate_sql(question)  # 只生成一次
return sql

# ✅ GOOD
for attempt in range(3):
    sql = await generate_sql(question)
    if await validate_and_test(sql):
        break
    sql = await refine_sql(sql, error)
```

### 陷阱5: 多轮对话上下文管理不当

**问题：** "它"指代不明，上下文丢失

```
对话示例：
用户: "查询北京的用户"
系统: [返回100个用户]
用户: "上海的也加上"  # ❌ 当前项目理解不了
```

**解决方案：**
```python
# 状态累积管理
class ConversationState:
    filters_applied: Dict[str, Any] = {}

# "上海的也加上" → 解析为追加条件
new_filter = {"city": "上海"}
state.filters_applied = {"city": ["北京", "上海"]}  # 合并条件
```

### 陷阱6: 时间处理不当

**问题：** "上个月"、"今年Q1"需要转换为SQL表达式

```python
# ❌ BAD
# LLM可能生成错误的时间表达式
sql = "WHERE date = 'last month'"  # 错误

# ✅ GOOD
class ValueMapper:
    TIME_PATTERNS = {
        r"上个月": lambda: "DATE_SUB(NOW(), INTERVAL 1 MONTH)",
        r"今年": lambda: "YEAR(NOW())",
        r"Q1": lambda: "DATE_TRUNC('quarter', NOW()) = DATE_TRUNC('quarter', '2024-01-01')",
    }

    def map_value(self, value: str) -> str:
        for pattern, generator in self.TIME_PATTERNS.items():
            if re.match(pattern, value):
                return generator()
        return value
```

---

## 第五部分：当前项目评估与改进建议

### 3.1 项目架构评分

| 模块 | 完成度 | 评价 | 改进优先级 |
|------|--------|------|------------|
| 数据库连接 | 85% | 支持多数据库，缺少连接池 | 低 |
| Schema 检索 | 80% | 混合检索已实现，缺表关系 | 中 |
| Prompt 构建 | 75% | 有 Few-shot，需优化模板 | 中 |
| SQL 生成 | 70% | 单次调用，缺思维链 | 高 |
| SQL 校验 | 90% | 安全校验完善 | 低 |
| 多轮对话 | 65% | 框架完整，缺状态累积 | 高 |
| 错误处理 | 60% | 需要更友好的错误提示 | 中 |

### 3.2 关键改进建议

#### 优先级 P0：必须实现

1. **多轮状态累积**
   - 当前：只有上下文摘要
   - 改进：维护累积的 WHERE 条件字典
   - 效果：支持"再加一个条件"、"排除某个值"

2. **思维链 SQL 生成**
   - 当前：直接生成 SQL
   - 改进：先生成分析步骤，再生成 SQL
   - 效果：复杂查询准确率提升

#### 优先级 P1：重要优化

3. **SQL 知识库**
   - 收集正确的 SQL 作为 Few-shot
   - 建立问题-SQL 向量索引

4. **错误恢复机制**
   - SQL 执行失败时，自动尝试修正
   - 提供更友好的错误提示和修复建议

#### 优先级 P2：体验提升

5. **结果解释**
   - 自动生成结果摘要
   - 推荐可视化方式

6. **性能优化**
   - Schema 缓存
   - 并行执行（Schema 检索 + RAG 检索）

---

## 第四部分：快速启动指南

### 4.1 最小可行产品（MVP）清单

```
必备功能：
✅ 数据库连接（单数据库）
✅ Schema 提取
✅ 向量检索（Schema）
✅ Prompt 构建（基础模板）
✅ LLM SQL 生成
✅ SQL 安全校验
✅ 执行并返回结果

可选功能（MVP 后）：
⚪ 多轮对话
⚪ SQL 知识库
⚪ 结果解释
```

### 4.2 技术选型建议

```
数据库：
- 开发：SQLite
- 生产：PostgreSQL / MySQL

向量数据库：
- 开发：ChromaDB
- 生产：Milvus / Qdrant

LLM：
- 推荐：GPT-4 / Claude-3
- 开源：Qwen / DeepSeek

框架：
- 推荐：FastAPI（已有）
- 前端：Streamlit / Gradio（快速原型）
```

### 4.3 测试策略

```
单元测试：
├── Schema 检索测试（不同问题类型）
├── SQL 生成测试（不同复杂度）
├── SQL 校验测试（安全、语法）
└── 多轮对话测试（意图识别、改写）

集成测试：
├── 端到端查询流程
├── 错误恢复流程
└── 性能测试（响应时间、并发）

评估指标：
├── 执行成功率：SQL 能否正确执行
├── 结果正确率：结果是否符合预期
├── 用户满意度：用户反馈评分
└── 响应时间：端到端延迟
```

---

## 附录：参考资源

### 论文

1. **Text-to-SQL in the Wild** - 真实场景挑战分析
2. **Spider: A Large-Scale Human-Labeled Dataset** - 数据集基准
3. **RAT-SQL** - 关系感知的 Text-to-SQL

### 开源项目

1. **Vanna.ai** - https://github.com/vanna-ai/vanna
2. **LangChain SQL** - https://python.langchain.com/docs/use_cases/sql
3. **LlamaIndex SQL** - https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo.html

### 数据集

1. **Spider** - 跨域 Text-to-SQL 数据集
2. **CSpider** - 中文 Spider
3. **Spider-Syn** - 同义词变体

---

*文档版本: v1.0*
*创建时间: 2026-03-05*
*基于项目: text-to-sql (1890 行代码)*