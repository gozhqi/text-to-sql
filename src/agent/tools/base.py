"""
Agent 方案 - 工具集模块

定义 Agent 可用的工具
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class ToolCategory(Enum):
    """工具分类"""
    SCHEMA = "schema"  # Schema 相关
    QUERY = "query"  # 查询相关
    VALIDATION = "validation"  # 验证相关
    UTILITY = "utility"  # 工具类


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """工具基类"""

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        db_manager=None
    ):
        self.name = name
        self.description = description
        self.category = category
        self.db_manager = db_manager

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass

    def get_signature(self) -> Dict[str, Any]:
        """获取工具签名（用于 LLM 理解）"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self._get_parameters()
        }

    def _get_parameters(self) -> Dict[str, Any]:
        """获取参数定义"""
        return {}


# ==================== Schema 工具 ====================

class ListTablesTool(BaseTool):
    """列出数据库中的所有表"""

    def __init__(self, db_manager=None):
        super().__init__(
            name="list_tables",
            description="列出数据库中的所有表。参数：db_name (数据库名称)",
            category=ToolCategory.SCHEMA,
            db_manager=db_manager
        )

    async def execute(self, db_name: str, **kwargs) -> ToolResult:
        """执行"""
        try:
            if self.db_manager:
                schemas = await self.db_manager.get_table_schemas(db_name)
                tables = list(schemas.keys())

                return ToolResult(
                    success=True,
                    data={
                        "tables": tables,
                        "count": len(tables)
                    },
                    metadata={"db_name": db_name}
                )
            else:
                # 模拟返回
                return ToolResult(
                    success=True,
                    data={
                        "tables": ["users", "orders", "products"],
                        "count": 3
                    }
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "db_name": {
                "type": "string",
                "description": "数据库名称",
                "required": True
            }
        }


class GetSchemaTool(BaseTool):
    """获取指定表的详细结构"""

    def __init__(self, db_manager=None):
        super().__init__(
            name="get_schema",
            description="获取指定表的详细结构，包括字段名、类型、注释。参数：table_name (表名), db_name (数据库名)",
            category=ToolCategory.SCHEMA,
            db_manager=db_manager
        )

    async def execute(self, table_name: str, db_name: str, **kwargs) -> ToolResult:
        """执行"""
        try:
            if self.db_manager:
                schemas = await self.db_manager.get_table_schemas(db_name)
                if table_name in schemas:
                    schema = schemas[table_name]
                    columns = []
                    for col in schema.columns:
                        columns.append({
                            "name": col.name,
                            "type": col.type,
                            "nullable": getattr(col, "nullable", True),
                            "comment": getattr(col, "comment", "")
                        })

                    return ToolResult(
                        success=True,
                        data={
                            "table_name": table_name,
                            "columns": columns
                        }
                    )
                else:
                    return ToolResult(success=False, error=f"表 {table_name} 不存在")
            else:
                # 模拟返回
                return ToolResult(
                    success=True,
                    data={
                        "table_name": table_name,
                        "columns": [
                            {"name": "id", "type": "INT", "comment": "主键"},
                            {"name": "name", "type": "VARCHAR(100)", "comment": "名称"}
                        ]
                    }
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "table_name": {
                "type": "string",
                "description": "表名",
                "required": True
            },
            "db_name": {
                "type": "string",
                "description": "数据库名称",
                "required": True
            }
        }


class SearchColumnsTool(BaseTool):
    """搜索包含特定关键词的列"""

    def __init__(self, db_manager=None):
        super().__init__(
            name="search_columns",
            description="搜索包含特定关键词的列名或注释。参数：keyword (关键词), db_name (数据库名)",
            category=ToolCategory.SCHEMA,
            db_manager=db_manager
        )

    async def execute(self, keyword: str, db_name: str, **kwargs) -> ToolResult:
        """执行"""
        try:
            results = []

            if self.db_manager:
                schemas = await self.db_manager.get_table_schemas(db_name)
                for table_name, schema in schemas.items():
                    for col in schema.columns:
                        if (keyword.lower() in col.name.lower() or
                            (hasattr(col, 'comment') and col.comment and
                             keyword.lower() in col.comment.lower())):
                            results.append({
                                "table": table_name,
                                "column": col.name,
                                "type": col.type,
                                "comment": getattr(col, "comment", "")
                            })
            else:
                # 模拟返回
                results = [
                    {"table": "users", "column": f"{keyword}_id", "type": "INT"}
                ]

            return ToolResult(
                success=True,
                data={
                    "keyword": keyword,
                    "results": results,
                    "count": len(results)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "keyword": {
                "type": "string",
                "description": "搜索关键词",
                "required": True
            },
            "db_name": {
                "type": "string",
                "description": "数据库名称",
                "required": True
            }
        }


class FindJoinPathTool(BaseTool):
    """找到两个表之间的关联路径"""

    def __init__(self, db_manager=None):
        super().__init__(
            name="find_join_path",
            description="找到两个表之间的关联路径（外键关系）。参数：table1 (第一个表名), table2 (第二个表名), db_name (数据库名)",
            category=ToolCategory.SCHEMA,
            db_manager=db_manager
        )

    async def execute(self, table1: str, table2: str, db_name: str, **kwargs) -> ToolResult:
        """执行"""
        try:
            # 简化实现：返回可能的关联
            path = f"{table1}.id = {table2}.{table1}_id"

            # 检查常见的关联模式
            common_patterns = [
                f"{table1}.id = {table2}.{table1}_id",
                f"{table2}.id = {table1}.{table2}_id",
                f"{table1}.id = {table2}.ref_id",
                f"{table2}.id = {table1}.ref_id"
            ]

            return ToolResult(
                success=True,
                data={
                    "table1": table1,
                    "table2": table2,
                    "suggested_joins": common_patterns,
                    "note": "请根据实际表结构选择合适的关联条件"
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "table1": {
                "type": "string",
                "description": "第一个表名",
                "required": True
            },
            "table2": {
                "type": "string",
                "description": "第二个表名",
                "required": True
            },
            "db_name": {
                "type": "string",
                "description": "数据库名称",
                "required": True
            }
        }


# ==================== 查询工具 ====================

class GenerateSQLTool(BaseTool):
    """生成 SQL 查询"""

    def __init__(self, db_manager=None, llm_client=None):
        super().__init__(
            name="generate_sql",
            description="根据收集的信息生成 SQL 查询。参数：question (用户问题), context (收集的上下文信息)",
            category=ToolCategory.QUERY,
            db_manager=db_manager
        )
        self.llm_client = llm_client

    async def execute(self, question: str, context: Dict[str, Any], **kwargs) -> ToolResult:
        """执行"""
        try:
            prompt = self._build_prompt(question, context)

            if self.llm_client:
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是 SQL 专家。根据收集的信息生成准确的 SQL 查询。"
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

                return ToolResult(success=True, data={"sql": sql})
            else:
                # 模拟返回
                return ToolResult(
                    success=True,
                    data={"sql": "SELECT * FROM users LIMIT 10"}
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _build_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """构建 Prompt"""
        parts = [f"用户问题: {question}", "", "收集的信息:"]

        if context.get("tables"):
            parts.append(f"相关表: {', '.join(context['tables'])}")

        if context.get("schemas"):
            parts.append("\n表结构:")
            for table, schema in context["schemas"].items():
                parts.append(f"\n{table}:")
                if isinstance(schema, dict) and schema.get("columns"):
                    for col in schema["columns"]:
                        parts.append(f"  - {col.get('name')}: {col.get('type')}")

        parts.append("\n请根据上述信息生成 SQL 查询。")
        parts.append("只返回 SQL，不要解释。")

        return "\n".join(parts)

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "question": {
                "type": "string",
                "description": "用户问题",
                "required": True
            },
            "context": {
                "type": "object",
                "description": "收集的上下文信息",
                "required": True
            }
        }


# ==================== 验证工具 ====================

class ValidateSQLTool(BaseTool):
    """验证 SQL 语法"""

    def __init__(self, db_manager=None):
        super().__init__(
            name="validate_sql",
            description="验证 SQL 的语法和正确性。参数：sql (SQL 语句)",
            category=ToolCategory.VALIDATION,
            db_manager=db_manager
        )

    async def execute(self, sql: str, **kwargs) -> ToolResult:
        """执行"""
        try:
            if not sql or not sql.strip():
                return ToolResult(success=False, error="SQL 为空")

            # 基本语法检查
            sql_upper = sql.upper()

            if "SELECT" not in sql_upper:
                return ToolResult(success=False, error="SQL 不是 SELECT 语句")

            # 检查括号匹配
            if sql.count("(") != sql.count(")"):
                return ToolResult(success=False, error="括号不匹配")

            # 检查基本结构
            required = ["SELECT", "FROM"]
            missing = [r for r in required if r not in sql_upper]
            if missing:
                return ToolResult(
                    success=False,
                    error=f"缺少必要的子句: {', '.join(missing)}"
                )

            return ToolResult(
                success=True,
                data={
                    "valid": True,
                    "message": "SQL 语法检查通过"
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "sql": {
                "type": "string",
                "description": "SQL 语句",
                "required": True
            }
        }


class TestExecuteTool(BaseTool):
    """测试执行 SQL（使用 LIMIT 1）"""

    def __init__(self, db_manager=None):
        super().__init__(
            name="test_execute",
            description="测试执行 SQL（使用 LIMIT 1），检查是否能正确执行。参数：sql (SQL 语句)",
            category=ToolCategory.VALIDATION,
            db_manager=db_manager
        )

    async def execute(self, sql: str, **kwargs) -> ToolResult:
        """执行"""
        try:
            if self.db_manager:
                # 添加 LIMIT 1
                test_sql = sql
                if "LIMIT" not in sql.upper():
                    test_sql = f"{sql} LIMIT 1"

                results = await self.db_manager.execute_sql(test_sql)

                return ToolResult(
                    success=True,
                    data={
                        "executed": True,
                        "row_count": len(results) if results else 0,
                        "message": f"测试执行成功，返回 {len(results) if results else 0} 条记录（LIMIT 1）"
                    }
                )
            else:
                # 模拟返回
                return ToolResult(
                    success=True,
                    data={
                        "executed": True,
                        "row_count": 1,
                        "message": "测试执行成功（模拟）"
                    }
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"测试执行失败: {str(e)}"
            )

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "sql": {
                "type": "string",
                "description": "SQL 语句",
                "required": True
            }
        }


# ==================== 工具注册表 ====================

class ToolRegistry:
    """工具注册表"""

    def __init__(self, db_manager=None, llm_client=None):
        self.tools = {}
        self.db_manager = db_manager
        self.llm_client = llm_client
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        self.register(ListTablesTool(self.db_manager))
        self.register(GetSchemaTool(self.db_manager))
        self.register(SearchColumnsTool(self.db_manager))
        self.register(FindJoinPathTool(self.db_manager))
        self.register(GenerateSQLTool(self.db_manager, self.llm_client))
        self.register(ValidateSQLTool(self.db_manager))
        self.register(TestExecuteTool(self.db_manager))

    def register(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self, category: ToolCategory = None) -> list:
        """列出工具"""
        if category:
            return [t for t in self.tools.values() if t.category == category]
        return list(self.tools.values())

    def get_tool_descriptions(self) -> str:
        """获取工具描述（用于 Prompt）"""
        descriptions = []
        for tool in self.tools.values():
            desc = f"- {tool.name}: {tool.description}"
            descriptions.append(desc)
        return "\n".join(descriptions)


def create_tool_registry(db_manager=None, llm_client=None) -> ToolRegistry:
    """创建工具注册表"""
    return ToolRegistry(db_manager, llm_client)
