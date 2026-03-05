"""
Agent 方案 - 多 Agent 协调器

将复杂任务分解给专门的 Agent 处理
"""
from typing import Dict, Any, List, Optional
from loguru import logger


class Agent:
    """Agent 基类"""

    def __init__(self, name: str, llm_client=None, db_manager=None):
        self.name = name
        self.llm_client = llm_client
        self.db_manager = db_manager

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入"""
        raise NotImplementedError


class SchemaExplorerAgent(Agent):
    """Schema 探索 Agent"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """探索 Schema，找出相关表和字段"""
        question = input_data.get("question", "")
        db_name = input_data.get("db_name", "")

        # 使用工具探索
        from ..tools.base import create_tool_registry

        tool_registry = create_tool_registry(self.db_manager)

        # 1. 列出表
        list_tool = tool_registry.get("list_tables")
        tables_result = await list_tool.execute(db_name=db_name)

        tables = []
        if tables_result.success:
            tables = tables_result.data.get("tables", [])

        # 2. 分析问题找出相关表
        relevant_tables = await self._find_relevant_tables(question, tables)

        # 3. 获取相关表的结构
        schemas = {}
        schema_tool = tool_registry.get("get_schema")

        for table in relevant_tables[:3]:  # 限制最多获取3个表
            schema_result = await schema_tool.execute(
                table_name=table,
                db_name=db_name
            )
            if schema_result.success:
                schemas[table] = schema_result.data

        return {
            "agent": self.name,
            "tables": tables,
            "relevant_tables": relevant_tables,
            "schemas": schemas
        }

    async def _find_relevant_tables(
        self,
        question: str,
        tables: List[str]
    ) -> List[str]:
        """找出相关表"""
        if not tables:
            return []

        question_lower = question.lower()

        # 简单关键词匹配
        relevant = []
        for table in tables:
            if table.lower() in question_lower:
                relevant.append(table)

        # 如果没有直接匹配，检查常见命名模式
        if not relevant:
            keywords = {
                "用户": ["user", "users", "customer", "customers"],
                "订单": ["order", "orders"],
                "产品": ["product", "products", "item", "items"],
                "销售": ["sale", "sales", "revenue"],
                "客户": ["client", "clients", "customer", "customers"]
            }

            for key, patterns in keywords.items():
                if key in question_lower:
                    for table in tables:
                        for pattern in patterns:
                            if pattern in table.lower():
                                relevant.append(table)
                                break

        return relevant if relevant else tables[:3]


class SQLGeneratorAgent(Agent):
    """SQL 生成 Agent"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成 SQL"""
        question = input_data.get("question", "")
        context = input_data.get("context", {})

        schemas = context.get("schemas", {})
        relevant_tables = context.get("relevant_tables", [])

        # 构建 Prompt
        prompt = self._build_prompt(question, schemas, relevant_tables)

        # 调用 LLM
        if self.llm_client:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "你是 SQL 专家。根据提供的 Schema 信息生成准确的 SQL 查询。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            content = response.choices[0].message.content

            # 提取 SQL
            import re
            sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                sql = content.strip()
        else:
            # 模拟返回
            sql = "SELECT * FROM users LIMIT 10"

        return {
            "agent": self.name,
            "sql": sql,
            "tables_used": relevant_tables
        }

    def _build_prompt(
        self,
        question: str,
        schemas: Dict[str, Any],
        relevant_tables: List[str]
    ) -> str:
        """构建 Prompt"""
        parts = [
            f"## 用户问题\n{question}",
            f"\n## 相关表\n{', '.join(relevant_tables)}"
        ]

        if schemas:
            parts.append("\n## 表结构")
            for table_name, schema in schemas.items():
                parts.append(f"\n### {table_name}")
                if isinstance(schema, dict) and schema.get("columns"):
                    for col in schema["columns"]:
                        parts.append(f"- {col.get('name')}: {col.get('type')}")

        parts.append("\n## 请生成 SQL 查询")

        return "\n".join(parts)


class ValidatorAgent(Agent):
    """验证和修复 Agent"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证和修复 SQL"""
        sql = input_data.get("sql", "")
        context = input_data.get("context", {})

        # 验证 SQL
        is_valid, errors = await self._validate(sql)

        if not is_valid:
            # 尝试修复
            fixed_sql = await self._fix_sql(sql, errors, context)
            return {
                "agent": self.name,
                "is_valid": False,
                "original_sql": sql,
                "fixed_sql": fixed_sql,
                "errors": errors
            }

        return {
            "agent": self.name,
            "is_valid": True,
            "sql": sql,
            "errors": []
        }

    async def _validate(self, sql: str) -> tuple[bool, List[str]]:
        """验证 SQL"""
        errors = []

        if not sql or not sql.strip():
            errors.append("SQL 为空")
            return False, errors

        sql_upper = sql.upper()

        if "SELECT" not in sql_upper:
            errors.append("不是 SELECT 语句")

        if "FROM" not in sql_upper:
            errors.append("缺少 FROM 子句")

        # 检查括号匹配
        if sql.count("(") != sql.count(")"):
            errors.append("括号不匹配")

        return len(errors) == 0, errors

    async def _fix_sql(
        self,
        sql: str,
        errors: List[str],
        context: Dict[str, Any]
    ) -> str:
        """尝试修复 SQL"""
        # 简化实现：返回原 SQL
        # 实际可以使用 LLM 进行修复
        return sql


class MultiAgentOrchestrator:
    """多 Agent 协调器"""

    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
        self.agents = {}
        self._init_agents()

    def _init_agents(self):
        """初始化子 Agent"""
        self.agents = {
            "schema_explorer": SchemaExplorerAgent(
                "SchemaExplorer",
                self.llm_client,
                self.db_manager
            ),
            "sql_generator": SQLGeneratorAgent(
                "SQLGenerator",
                self.llm_client,
                self.db_manager
            ),
            "validator": ValidatorAgent(
                "Validator",
                self.llm_client,
                self.db_manager
            )
        }

    async def process(
        self,
        question: str,
        db_name: str,
        use_all_agents: bool = True
    ) -> Dict[str, Any]:
        """
        处理查询

        流程：Schema Explorer → SQL Generator → Validator
        """
        result = {
            "question": question,
            "db_name": db_name,
            "steps": [],
            "success": False
        }

        # Step 1: Schema 探索
        logger.info("Step 1: Schema 探索")
        schema_result = await self.agents["schema_explorer"].process({
            "question": question,
            "db_name": db_name
        })
        result["steps"].append(schema_result)

        context = {
            "tables": schema_result.get("tables", []),
            "relevant_tables": schema_result.get("relevant_tables", []),
            "schemas": schema_result.get("schemas", {})
        }

        # Step 2: SQL 生成
        logger.info("Step 2: SQL 生成")
        sql_result = await self.agents["sql_generator"].process({
            "question": question,
            "context": context
        })
        result["steps"].append(sql_result)

        sql = sql_result.get("sql", "")

        # Step 3: 验证和修复
        if use_all_agents:
            logger.info("Step 3: 验证和修复")
            validation_result = await self.agents["validator"].process({
                "sql": sql,
                "context": context
            })
            result["steps"].append(validation_result)

            result["success"] = validation_result.get("is_valid", False)
            result["sql"] = validation_result.get(
                "fixed_sql",
                validation_result.get("sql", sql)
            )
        else:
            result["success"] = bool(sql)
            result["sql"] = sql

        result["method"] = "multi_agent"

        return result

    async def process_with_retry(
        self,
        question: str,
        db_name: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """带重试的处理"""
        for attempt in range(max_retries):
            result = await self.process(question, db_name)

            if result.get("success"):
                return result

            logger.warning(f"第 {attempt + 1} 次尝试失败，重试...")

        return result


def create_orchestrator(llm_client=None, db_manager=None) -> MultiAgentOrchestrator:
    """创建多 Agent 协调器"""
    return MultiAgentOrchestrator(llm_client, db_manager)
