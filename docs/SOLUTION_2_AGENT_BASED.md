# 方案二：Agent 驱动的 Text-to-SQL 系统

> 基于 LangChain SQL Agent / DB-GPT-Hub 架构，适合复杂查询场景

---

## 一、方案概述

### 适用场景
- ✅ 查询复杂度高（多表 JOIN、嵌套查询）
- ✅ 需要自我纠错能力
- ✅ Schema 经常变化
- ✅ 需要透明的推理过程

### 核心优势
| 特点 | 说明 |
|------|------|
| **自我纠错** | SQL 执行失败时自动尝试修复 |
| **灵活适应** | 动态探索 Schema，适应变化 |
| **可解释** | 完整的推理过程 |
| **无冷启动** | 不需要预先训练 |

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent 架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户问题                                                    │
│      │                                                      │
│      ↓                                                      │
│  ┌───────────────────────────────────────┐                  │
│  │           ReAct 推理循环               │                  │
│  │                                       │                  │
│  │   Thought ─→ Action ─→ Observation    │                  │
│  │      ↑                      │          │                  │
│  │      └──────────────────────┘          │                  │
│  └───────────────────────────────────────┘                  │
│                      │                                      │
│                      ↓                                      │
│  ┌───────────────────────────────────────┐                  │
│  │             工具集                     │                  │
│  │  ┌─────────┐  ┌─────────┐  ┌────────┐ │                  │
│  │  │Schema   │  │SQL      │  │Query   │ │                  │
│  │  │Explorer │  │Generator│  │Checker │ │                  │
│  │  └─────────┘  └─────────┘  └────────┘ │                  │
│  └───────────────────────────────────────┘                  │
│                      │                                      │
│                      ↓                                      │
│  ┌───────────────────────────────────────┐                  │
│  │           执行与验证                   │                  │
│  │  SQL执行 ─→ 结果检查 ─→ 错误修复      │                  │
│  └───────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、ReAct 推理框架

### 2.1 核心思想

ReAct = Reasoning + Acting

```
用户问题: "查询北京地区销售额最高的客户"

推理过程:
┌────────────────────────────────────────────────────────────┐
│ Thought 1: 我需要先了解数据库有哪些表                       │
│ Action 1: sql_db_list_tables                               │
│ Observation 1: ['customers', 'orders', 'products']          │
│                                                            │
│ Thought 2: 客户信息在 customers 表，订单在 orders 表        │
│ Action 2: sql_db_schema(customers, orders)                 │
│ Observation 2: customers(customer_id, name, city...)       │
│                orders(order_id, customer_id, amount...)    │
│                                                            │
│ Thought 3: 需要按城市过滤并计算总金额，生成 SQL             │
│ Action 3: sql_db_query_checker                             │
│ Observation 3: SQL 语法正确                                 │
│                                                            │
│ Thought 4: 执行 SQL 查询                                   │
│ Action 4: sql_db_query                                     │
│ Observation 4: [{"name": "张三", "total": 150000}]         │
│                                                            │
│ Final Answer: 北京地区销售额最高的客户是张三，总金额15万元  │
└────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 实现

```python
from typing import List, Dict, Optional, Any
from openai import OpenAI
import json
import re

class SQLAgent:
    """SQL 生成 Agent"""

    SYSTEM_PROMPT = """你是一个数据库查询专家，使用 ReAct 框架解决用户问题。

## 可用工具

1. `list_tables` - 列出数据库所有表
   输入: {}
   输出: 表名列表

2. `get_schema` - 获取表结构
   输入: {"table_names": ["table1", "table2"]}
   输出: 表的 DDL 和字段说明

3. `query_sample` - 查询表样本数据
   输入: {"table_name": "table1", "limit": 5}
   输出: 样本数据

4. `generate_sql` - 生成 SQL
   输入: {"question": "用户问题", "schema": "表结构"}
   输出: SQL 语句

5. `validate_sql` - 验证 SQL
   输入: {"sql": "SQL语句"}
   输出: 是否有效 + 错误信息

6. `execute_sql` - 执行 SQL
   输入: {"sql": "SQL语句"}
   输出: 查询结果

## 输出格式

每次思考输出 JSON:
```json
{
  "thought": "当前思考",
  "action": "工具名称",
  "action_input": {}
}
```

最终答案:
```json
{
  "answer": "最终答案",
  "sql": "生成的SQL",
  "explanation": "解释"
}
```

## 规则
1. 最多 10 轮交互
2. SQL 执行失败时尝试修复（最多 3 次）
3. 只生成 SELECT 查询
"""

    def __init__(
        self,
        db_connection,
        model: str = "gpt-4o",
        max_iterations: int = 10
    ):
        self.db = db_connection
        self.client = OpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.conversation_history = []

    # ========== 工具实现 ==========

    async def list_tables(self) -> List[str]:
        """列出所有表"""
        result = await self.db.execute("SHOW TABLES")
        return [list(row.values())[0] for row in result]

    async def get_schema(self, table_names: List[str]) -> str:
        """获取表结构"""
        schemas = []
        for table in table_names:
            # 获取建表语句
            ddl = await self.db.execute(f"SHOW CREATE TABLE {table}")
            schemas.append(ddl[0]["Create Table"])

            # 获取字段注释
            columns = await self.db.execute(f"DESCRIBE {table}")
            schema_info = f"\n-- {table} 表字段:\n"
            for col in columns:
                schema_info += f"-- {col['Field']}: {col['Type']}\n"
            schemas.append(schema_info)

        return "\n".join(schemas)

    async def query_sample(self, table_name: str, limit: int = 5) -> List[Dict]:
        """查询样本数据"""
        return await self.db.execute(f"SELECT * FROM {table_name} LIMIT {limit}")

    async def validate_sql(self, sql: str) -> Dict:
        """验证 SQL"""
        # 语法检查
        dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']
        sql_upper = sql.upper()

        for keyword in dangerous_keywords:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return {
                    "valid": False,
                    "error": f"危险操作: {keyword}"
                }

        if not sql_upper.strip().startswith('SELECT'):
            return {
                "valid": False,
                "error": "只允许 SELECT 查询"
            }

        return {"valid": True, "error": None}

    async def execute_sql(self, sql: str) -> Dict:
        """执行 SQL"""
        try:
            results = await self.db.execute(sql)
            return {
                "success": True,
                "results": results,
                "row_count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # ========== Agent 主循环 ==========

    async def run(self, question: str) -> Dict:
        """
        执行 Agent 推理
        """
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        iterations = 0
        final_result = None

        while iterations < self.max_iterations:
            iterations += 1

            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.1
            )

            assistant_message = response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # 解析响应
            parsed = self._parse_response(assistant_message)

            # 检查是否完成
            if "answer" in parsed:
                final_result = {
                    "success": True,
                    "answer": parsed["answer"],
                    "sql": parsed.get("sql", ""),
                    "explanation": parsed.get("explanation", ""),
                    "iterations": iterations
                }
                break

            # 执行工具
            if "action" in parsed:
                observation = await self._execute_tool(
                    parsed["action"],
                    parsed.get("action_input", {})
                )

                self.conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {json.dumps(observation, ensure_ascii=False)}"
                })

        if final_result is None:
            final_result = {
                "success": False,
                "error": "达到最大迭代次数",
                "iterations": iterations
            }

        return final_result

    def _parse_response(self, content: str) -> Dict:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # 尝试直接解析
        try:
            return json.loads(content)
        except:
            return {"raw": content}

    async def _execute_tool(self, action: str, action_input: Dict) -> Any:
        """执行工具"""
        tool_map = {
            "list_tables": self.list_tables,
            "get_schema": self.get_schema,
            "query_sample": self.query_sample,
            "validate_sql": self.validate_sql,
            "execute_sql": self.execute_sql
        }

        if action not in tool_map:
            return {"error": f"未知工具: {action}"}

        try:
            if action in ["list_tables"]:
                result = await tool_map[action]()
            else:
                result = await tool_map[action](**action_input)
            return result
        except Exception as e:
            return {"error": str(e)}
```

---

## 三、Multi-Agent 协作架构

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────┐                          │
│                    │ Orchestrator│                          │
│                    │   协调器    │                          │
│                    └──────┬──────┘                          │
│                           │                                 │
│           ┌───────────────┼───────────────┐                 │
│           │               │               │                 │
│           ↓               ↓               ↓                 │
│    ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│    │  Schema   │   │    SQL    │   │ Validator │           │
│    │  Agent    │   │  Generator│   │   Agent   │           │
│    │           │   │   Agent   │   │           │           │
│    │ 探索Schema│   │  生成SQL  │   │ 验证修复  │           │
│    └───────────┘   └───────────┘   └───────────┘           │
│                                                             │
│    工作流程:                                                │
│    1. Schema Agent 分析问题，找出相关表和字段               │
│    2. SQL Generator Agent 生成 SQL                         │
│    3. Validator Agent 验证并修复错误                        │
│    4. Orchestrator 协调，直到得到正确结果                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Agent 定义

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, name: str, model: str = "gpt-4o"):
        self.name = name
        self.model = model
        self.client = OpenAI()

    @abstractmethod
    async def run(self, context: Dict) -> Dict:
        """执行 Agent 任务"""
        pass

    async def _call_llm(self, messages: list) -> str:
        """调用 LLM"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content


class SchemaExplorerAgent(BaseAgent):
    """Schema 探索 Agent"""

    SYSTEM_PROMPT = """你是数据库 Schema 分析专家。
根据用户问题，找出相关的表和字段。

输出 JSON 格式:
{
  "relevant_tables": ["table1", "table2"],
  "relevant_columns": {
    "table1": ["col1", "col2"],
    "table2": ["col1"]
  },
  "join_hints": ["table1.col1 = table2.col1"],
  "reasoning": "分析推理过程"
}"""

    def __init__(self, db_connection, model: str = "gpt-4o"):
        super().__init__("schema_explorer", model)
        self.db = db_connection

    async def run(self, context: Dict) -> Dict:
        question = context["question"]

        # 获取所有表
        tables = await self.db.execute("SHOW TABLES")
        table_names = [list(t.values())[0] for t in tables]

        # 获取 Schema
        schema_info = []
        for table in table_names[:10]:  # 限制数量
            columns = await self.db.execute(f"DESCRIBE {table}")
            schema_info.append({
                "table": table,
                "columns": [{"name": c["Field"], "type": c["Type"]} for c in columns]
            })

        # 调用 LLM 分析
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"问题: {question}\n\n数据库Schema:\n{json.dumps(schema_info, ensure_ascii=False)}"}
        ]

        response = await self._call_llm(messages)

        # 解析响应
        try:
            result = json.loads(response)
        except:
            result = {"relevant_tables": [], "relevant_columns": {}}

        return result


class SQLGeneratorAgent(BaseAgent):
    """SQL 生成 Agent"""

    SYSTEM_PROMPT = """你是 SQL 生成专家。
根据问题、Schema 和相关信息，生成准确的 SQL。

规则:
1. 只生成 SELECT 语句
2. 使用标准 SQL 语法
3. 正确处理 JOIN 和聚合

输出 JSON:
{
  "sql": "SELECT ...",
  "explanation": "查询逻辑说明",
  "confidence": 0.9
}"""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__("sql_generator", model)

    async def run(self, context: Dict) -> Dict:
        question = context["question"]
        schema_info = context.get("schema_info", {})
        examples = context.get("examples", [])

        prompt = f"""问题: {question}

相关表: {schema_info.get('relevant_tables', [])}
相关字段: {json.dumps(schema_info.get('relevant_columns', {}), ensure_ascii=False)}
连接提示: {schema_info.get('join_hints', [])}
"""

        if examples:
            prompt += f"\n相似示例:\n{json.dumps(examples, ensure_ascii=False)}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = await self._call_llm(messages)

        try:
            result = json.loads(response)
        except:
            result = {"sql": "", "explanation": response, "confidence": 0}

        return result


class ValidatorAgent(BaseAgent):
    """验证修复 Agent"""

    SYSTEM_PROMPT = """你是 SQL 验证和修复专家。
检查 SQL 的语法和逻辑，修复错误。

输出 JSON:
{
  "is_valid": true/false,
  "errors": ["错误列表"],
  "fixed_sql": "修复后的SQL（如有错误）",
  "fix_explanation": "修复说明"
}"""

    def __init__(self, db_connection, model: str = "gpt-4o"):
        super().__init__("validator", model)
        self.db = db_connection

    async def run(self, context: Dict) -> Dict:
        sql = context["sql"]
        error = context.get("error")

        prompt = f"SQL: {sql}"
        if error:
            prompt += f"\n执行错误: {error}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = await self._call_llm(messages)

        try:
            result = json.loads(response)
        except:
            result = {"is_valid": False, "errors": ["解析失败"]}

        return result


class Orchestrator:
    """协调器"""

    def __init__(self, db_connection, model: str = "gpt-4o"):
        self.schema_agent = SchemaExplorerAgent(db_connection, model)
        self.generator_agent = SQLGeneratorAgent(model)
        self.validator_agent = ValidatorAgent(db_connection, model)
        self.db = db_connection

    async def run(self, question: str) -> Dict:
        """执行完整流程"""
        context = {"question": question}

        # 1. Schema 探索
        schema_info = await self.schema_agent.run(context)
        context["schema_info"] = schema_info

        # 2. SQL 生成
        sql_result = await self.generator_agent.run(context)
        context["sql"] = sql_result.get("sql", "")

        # 3. 验证和修复循环
        max_fixes = 3
        for i in range(max_fixes):
            # 执行 SQL
            try:
                results = await self.db.execute(context["sql"])
                return {
                    "success": True,
                    "sql": context["sql"],
                    "results": results,
                    "explanation": sql_result.get("explanation", "")
                }
            except Exception as e:
                # 验证失败，尝试修复
                validation = await self.validator_agent.run({
                    "sql": context["sql"],
                    "error": str(e)
                })

                if validation.get("fixed_sql"):
                    context["sql"] = validation["fixed_sql"]
                else:
                    return {
                        "success": False,
                        "sql": context["sql"],
                        "error": str(e)
                    }

        return {
            "success": False,
            "sql": context["sql"],
            "error": "多次修复失败"
        }
```

---

## 四、工具集扩展

### 4.1 内置工具

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class SQLDatabaseTool(BaseTool):
    """SQL 数据库工具基类"""
    db_connection: Any

    class Config:
        arbitrary_types_allowed = True


class ListTablesTool(SQLDatabaseTool):
    """列出所有表"""
    name = "list_tables"
    description = "列出数据库中的所有表名"

    def _run(self) -> str:
        tables = await self.db_connection.execute("SHOW TABLES")
        return str([list(t.values())[0] for t in tables])


class GetSchemaTool(SQLDatabaseTool):
    """获取表结构"""
    name = "get_schema"
    description = "获取指定表的结构信息"

    def _run(self, table_names: str) -> str:
        tables = table_names.split(",")
        result = []
        for table in tables:
            columns = await self.db_connection.execute(f"DESCRIBE {table.strip()}")
            result.append(f"Table: {table}\n{json.dumps(columns, indent=2)}")
        return "\n\n".join(result)


class QueryCheckerTool(SQLDatabaseTool):
    """SQL 语法检查"""
    name = "query_checker"
    description = "检查 SQL 语法是否正确"

    def _run(self, sql: str) -> str:
        # 使用 EXPLAIN 验证
        try:
            await self.db_connection.execute(f"EXPLAIN {sql}")
            return "SQL 语法正确"
        except Exception as e:
            return f"SQL 错误: {str(e)}"


class QuerySQLTool(SQLDatabaseTool):
    """执行 SQL"""
    name = "query_sql"
    description = "执行 SQL 查询并返回结果"

    def _run(self, sql: str) -> str:
        results = await self.db_connection.execute(sql)
        return json.dumps(results, indent=2, ensure_ascii=False)
```

---

## 五、使用示例

### 5.1 单 Agent 模式

```python
# 初始化
db = DatabaseConnection("mysql://...")
agent = SQLAgent(db_connection=db, model="gpt-4o")

# 执行查询
result = await agent.run("查询北京地区销售额最高的前5个客户")

print(f"SQL: {result['sql']}")
print(f"结果: {result['answer']}")
```

### 5.2 Multi-Agent 模式

```python
# 初始化
db = DatabaseConnection("mysql://...")
orchestrator = Orchestrator(db_connection=db, model="gpt-4o")

# 执行查询
result = await orchestrator.run("统计每个产品类别的月销售趋势")

print(f"SQL: {result['sql']}")
print(f"结果数量: {len(result['results'])}")
```

---

## 六、性能对比

### 6.1 Agent vs RAG 对比

| 指标 | Agent 方案 | RAG 方案 |
|------|------------|----------|
| 准确率 | 75-85% | 80-90% |
| 响应时间 | 10-30s | 2-5s |
| 自我纠错 | ✅ 支持 | ❌ 不支持 |
| 冷启动 | 无需 | 需要示例 |
| 成本 | 高 | 中 |

### 6.2 适用场景

```
查询复杂度低 + 有历史SQL → RAG 方案
查询复杂度高 + 需要纠错 → Agent 方案
两者结合 → 混合方案
```

---

## 七、优化建议

### 7.1 减少迭代次数

```python
# 添加 Schema 缓存
class CachedSchemaExplorerAgent(SchemaExplorerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema_cache = None

    async def get_all_schema(self):
        if self._schema_cache is None:
            self._schema_cache = await self._fetch_all_schema()
        return self._schema_cache
```

### 7.2 提前终止

```python
# 添加置信度判断
async def run(self, question: str) -> Dict:
    # ...

    if sql_result.get("confidence", 0) > 0.9:
        # 高置信度，跳过验证
        return await self._execute_direct(sql_result["sql"])

    # 低置信度，走验证流程
    return await self._validate_and_fix(sql_result)
```

---

## 八、总结

### 优势
1. **自我纠错** - SQL 执行失败时自动修复
2. **动态探索** - 自动发现 Schema 变化
3. **可解释** - 完整的推理过程
4. **无冷启动** - 不需要预先训练

### 劣势
1. **延迟高** - 多轮 LLM 调用
2. **成本高** - Token 消耗大
3. **不确定性** - 可能进入错误路径

### 适用场景
- ✅ 复杂查询场景
- ✅ Schema 经常变化
- ✅ 需要透明推理过程
- ✅ 容忍较高延迟