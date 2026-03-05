"""
Agent 方案 - 工具集

提供 Agent 可调用的各种工具
"""
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from abc import ABC, abstractmethod

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy 未安装")

from .react_loop import Tool


@dataclass
class TableInfo:
    """表信息"""
    name: str
    columns: List[Dict[str, Any]]
    row_count: int = 0
    sample_rows: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": self.columns,
            "row_count": self.row_count,
            "sample_rows": self.sample_rows or []
        }


class DatabaseTool(Tool, ABC):
    """数据库工具基类"""

    def __init__(self, engine: Engine):
        super().__init__(
            name=self.__class__.__name__,
            description=self.get_description()
        )
        self.engine = engine

    @abstractmethod
    def get_description(self) -> str:
        pass


class ListTablesTool(DatabaseTool):
    """列出所有表"""

    def get_description(self) -> str:
        return "列出数据库中的所有表。参数：database (可选，数据库名称)"

    async def execute(self, database: str = "") -> str:
        """执行"""
        try:
            inspector = inspect(self.engine)

            # 获取表名列表
            tables = inspector.get_table_names()

            if not tables:
                return "数据库中没有表"

            result = f"数据库包含 {len(tables)} 个表：\n"
            for table in tables:
                # 获取表的行数
                try:
                    with self.engine.connect() as conn:
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        row_count = count_result.scalar()
                        result += f"- {table} ({row_count} 行)\n"
                except:
                    result += f"- {table}\n"

            return result

        except Exception as e:
            logger.error(f"列出表失败: {e}")
            return f"错误：{str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "database": {
                    "type": "string",
                    "description": "数据库名称（可选）",
                    "default": ""
                }
            }
        }


class DescribeTableTool(DatabaseTool):
    """描述表结构"""

    def get_description(self) -> str:
        return "获取指定表的详细结构信息，包括列名、数据类型、是否可空等。参数：table_name (必需，表名)"

    async def execute(self, table_name: str) -> str:
        """执行"""
        try:
            inspector = inspect(self.engine)

            # 获取列信息
            columns = inspector.get_columns(table_name)

            if not columns:
                return f"表 '{table_name}' 不存在或没有列"

            result = f"表 '{table_name}' 的结构：\n\n"
            result += "| 列名 | 类型 | 可空 | 默认值 | 主键 |\n"
            result += "|" + "-" * 50 + "|\n"

            for col in columns:
                primary_key = "是" if col.get("primary_key") else "否"
                nullable = "是" if col.get("nullable") else "否"
                default = str(col.get("default", "")) if col.get("default") is not None else ""

                result += f"| {col['name']} | {col['type']} | {nullable} | {default} | {primary_key} |\n"

            # 获取外键信息
            foreign_keys = inspector.get_foreign_keys(table_name)
            if foreign_keys:
                result += "\n外键关系：\n"
                for fk in foreign_keys:
                    result += f"- {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"

            # 获取索引信息
            indexes = inspector.get_indexes(table_name)
            if indexes:
                result += "\n索引：\n"
                for idx in indexes:
                    unique = "(唯一)" if idx.get("unique") else ""
                    result += f"- {idx['name']}: {idx['column_names']} {unique}\n"

            return result

        except Exception as e:
            logger.error(f"描述表失败: {e}")
            return f"错误：{str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "表名",
                    "required": True
                }
            }
        }


class GetSampleRowsTool(DatabaseTool):
    """获取示例行"""

    def get_description(self) -> str:
        return "获取表的前几行数据作为示例。参数：table_name (必需，表名), limit (可选，行数，默认5)"

    async def execute(self, table_name: str, limit: int = 5) -> str:
        """执行"""
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT * FROM {table_name} LIMIT :limit")
                result = conn.execute(query, {"limit": limit})

                rows = result.fetchall()
                columns = result.keys()

                if not rows:
                    return f"表 '{table_name}' 是空的"

                # 格式化输出
            output = f"表 '{table_name}' 的示例数据（前 {len(rows)} 行）：\n\n"

            # 表头
            output += "| " + " | ".join(columns) + " |\n"
            output += "|" + "|".join(["-" * max(len(str(col)), 5) for col in columns]) + "|\n"

            # 数据行
            for row in rows:
                output += "| " + " | ".join(str(v) if v is not None else "NULL" for v in row) + " |\n"

            return output

        except Exception as e:
            logger.error(f"获取示例行失败: {e}")
            return f"错误：{str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "表名",
                    "required": True
                },
                "limit": {
                    "type": "integer",
                    "description": "返回行数",
                    "default": 5
                }
            }
        }


class ExecuteSQLTool(DatabaseTool):
    """执行 SQL 查询（只读）"""

    def get_description(self) -> str:
        return "执行只读 SQL 查询并返回结果。参数：query (必需，SQL 查询语句)。注意：只允许 SELECT 查询。"

    async def execute(self, query: str) -> str:
        """执行"""
        try:
            # 安全检查
            query_upper = query.upper().strip()
            dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
            if any(kw in query_upper for kw in dangerous_keywords):
                return "错误：不允许执行非 SELECT 查询"

            # 添加限制
            if "LIMIT" not in query_upper:
                query += " LIMIT 100"

            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()

                if not rows:
                    return "查询返回空结果"

                # 格式化输出
                output = f"查询结果（{len(rows)} 行）：\n\n"

                # 表头
                output += "| " + " | ".join(columns) + " |\n"
                output += "|" + "|".join(["-" * max(len(str(col)), 5) for col in columns]) + "|\n"

                # 数据行
                for row in rows[:20]:  # 最多显示 20 行
                    output += "| " + " | ".join(str(v) if v is not None else "NULL" for v in row) + " |\n"

                if len(rows) > 20:
                    output += f"\n... 还有 {len(rows) - 20} 行\n"

                return output

        except Exception as e:
            logger.error(f"执行 SQL 失败: {e}")
            return f"错误：{str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "SQL 查询语句",
                    "required": True
                }
            }
        }


class SearchTableTool(DatabaseTool):
    """搜索表"""

    def get_description(self) -> str:
        return "根据关键词搜索相关的表。参数：keyword (必需，搜索关键词)"

    async def execute(self, keyword: str) -> str:
        """执行"""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            keyword_lower = keyword.lower()
            matching_tables = []

            for table in tables:
                # 检查表名
                if keyword_lower in table.lower():
                    matching_tables.append((table, "表名匹配"))
                    continue

                # 检查列名
                columns = inspector.get_columns(table)
                for col in columns:
                    if keyword_lower in col["name"].lower():
                        matching_tables.append((table, f"列名匹配: {col['name']}"))
                        break

            if not matching_tables:
                return f"没有找到与 '{keyword}' 相关的表"

            result = f"找到 {len(matching_tables)} 个相关的表：\n"
            for table, reason in matching_tables:
                result += f"- {table}: {reason}\n"

            return result

        except Exception as e:
            logger.error(f"搜索表失败: {e}")
            return f"错误：{str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词",
                    "required": True
                }
            }
        }


class GetRelationshipsTool(DatabaseTool):
    """获取表关系"""

    def get_description(self) -> str:
        return "获取表之间的外键关系。参数：table_name (可选，指定表名，不指定则返回所有关系)"

    async def execute(self, table_name: str = "") -> str:
        """执行"""
        try:
            inspector = inspect(self.engine)

            if table_name:
                tables = [table_name]
            else:
                tables = inspector.get_table_names()

            result = "表关系：\n\n"

            for table in tables:
                foreign_keys = inspector.get_foreign_keys(table)
                if foreign_keys:
                    result += f"{table}:\n"
                    for fk in foreign_keys:
                        constrained = ", ".join(fk['constrained_columns'])
                        referred = ", ".join(fk['referred_columns'])
                        result += f"  - {constrained} -> {fk['referred_table']}.{referred}\n"

            if result == "表关系：\n\n":
                return "没有找到表关系"

            return result

        except Exception as e:
            logger.error(f"获取关系失败: {e}")
            return f"错误：{str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "表名（可选）",
                    "default": ""
                }
            }
        }


def create_database_tools(engine: Engine) -> List[Tool]:
    """创建数据库工具集"""
    return [
        ListTablesTool(engine),
        DescribeTableTool(engine),
        GetSampleRowsTool(engine),
        ExecuteSQLTool(engine),
        SearchTableTool(engine),
        GetRelationshipsTool(engine)
    ]


def get_tool_schemas(tools: List[Tool]) -> List[Dict[str, Any]]:
    """获取所有工具的模式"""
    return [tool.get_schema() for tool in tools]


def print_tool_schemas(tools: List[Tool]):
    """打印工具模式"""
    schemas = get_tool_schemas(tools)
    print(json.dumps(schemas, indent=2, ensure_ascii=False))
