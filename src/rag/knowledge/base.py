"""
RAG 方案 - 知识库管理模块

提供 DDL、业务文档、历史 SQL 的存储和管理功能
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from loguru import logger


@dataclass
class DDLDocument:
    """DDL 文档数据类"""
    table_name: str
    ddl: str
    description: str = ""
    business_rules: List[str] = field(default_factory=list)
    database: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "table_name": self.table_name,
            "ddl": self.ddl,
            "description": self.description,
            "business_rules": self.business_rules,
            "database": self.database,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DDLDocument":
        """从字典创建"""
        return cls(
            table_name=data["table_name"],
            ddl=data["ddl"],
            description=data.get("description", ""),
            business_rules=data.get("business_rules", []),
            database=data.get("database", ""),
            metadata=data.get("metadata", {})
        )

    def to_document(self) -> str:
        """转换为文档格式用于向量化"""
        parts = [f"表名: {self.table_name}"]
        if self.database:
            parts.append(f"数据库: {self.database}")
        if self.description:
            parts.append(f"描述: {self.description}")
        if self.business_rules:
            parts.append(f"业务规则: {'; '.join(self.business_rules)}")
        parts.append(f"DDL:\n{self.ddl}")
        return "\n".join(parts)


@dataclass
class SQLExample:
    """SQL 示例数据类"""
    id: str = ""
    question: str = ""
    sql: str = ""
    database: str = ""
    table_names: List[str] = field(default_factory=list)
    difficulty: str = "easy"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "question": self.question,
            "sql": self.sql,
            "database": self.database,
            "table_names": self.table_names,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLExample":
        """从字典创建"""
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            sql=data.get("sql", ""),
            database=data.get("database", ""),
            table_names=data.get("table_names", []),
            difficulty=data.get("difficulty", "easy"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

    def to_document(self) -> str:
        """转换为文档格式用于向量化"""
        parts = [f"问题: {self.question}"]
        if self.database:
            parts.append(f"数据库: {self.database}")
        if self.table_names:
            parts.append(f"涉及表: {', '.join(self.table_names)}")
        parts.append(f"SQL: {self.sql}")
        return "\n".join(parts)


@dataclass
class BusinessDocumentation:
    """业务文档数据类"""
    id: str = ""
    title: str = ""
    content: str = ""
    database: str = ""
    related_tables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "database": self.database,
            "related_tables": self.related_tables,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusinessDocumentation":
        """从字典创建"""
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            database=data.get("database", ""),
            related_tables=data.get("related_tables", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

    def to_document(self) -> str:
        """转换为文档格式用于向量化"""
        parts = [f"标题: {self.title}"]
        if self.database:
            parts.append(f"数据库: {self.database}")
        if self.related_tables:
            parts.append(f"相关表: {', '.join(self.related_tables)}")
        parts.append(f"内容:\n{self.content}")
        return "\n".join(parts)


class KnowledgeBase(ABC):
    """知识库抽象基类"""

    @abstractmethod
    async def add_ddl(self, doc: DDLDocument) -> bool:
        """添加 DDL 文档"""
        pass

    @abstractmethod
    async def add_sql_example(self, example: SQLExample) -> bool:
        """添加 SQL 示例"""
        pass

    @abstractmethod
    async def add_documentation(self, doc: BusinessDocumentation) -> bool:
        """添加业务文档"""
        pass

    @abstractmethod
    async def get_ddl(self, table_name: str) -> Optional[DDLDocument]:
        """获取 DDL 文档"""
        pass

    @abstractmethod
    async def get_sql_examples(self, database: str = "", limit: int = 100) -> List[SQLExample]:
        """获取 SQL 示例列表"""
        pass

    @abstractmethod
    async def get_documentation(self, doc_id: str) -> Optional[BusinessDocumentation]:
        """获取业务文档"""
        pass

    @abstractmethod
    async def delete_ddl(self, table_name: str) -> bool:
        """删除 DDL 文档"""
        pass

    @abstractmethod
    async def delete_sql_example(self, example_id: str) -> bool:
        """删除 SQL 示例"""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识库"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        pass


class FileKnowledgeBase(KnowledgeBase):
    """基于文件的知识库实现"""

    def __init__(self, base_path: str = "./data/knowledge"):
        self.base_path = base_path
        self.ddl_path = os.path.join(base_path, "ddl")
        self.sql_path = os.path.join(base_path, "sql")
        self.doc_path = os.path.join(base_path, "docs")

        # 创建目录
        os.makedirs(self.ddl_path, exist_ok=True)
        os.makedirs(self.sql_path, exist_ok=True)
        os.makedirs(self.doc_path, exist_ok=True)

        # 内存缓存
        self._ddl_cache: Dict[str, DDLDocument] = {}
        self._sql_cache: Dict[str, SQLExample] = {}
        self._doc_cache: Dict[str, BusinessDocumentation] = {}

        # 加载已有数据
        self._load_all()

    def _load_all(self):
        """加载所有数据到缓存"""
        self._load_ddls()
        self._load_sqls()
        self._load_docs()
        logger.info(f"知识库加载完成: DDL={len(self._ddl_cache)}, SQL={len(self._sql_cache)}, Doc={len(self._doc_cache)}")

    def _load_ddls(self):
        """加载 DDL 文档"""
        if not os.path.exists(self.ddl_path):
            return
        for filename in os.listdir(self.ddl_path):
            if filename.endswith(".json"):
                filepath = os.path.join(self.ddl_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        doc = DDLDocument.from_dict(data)
                        self._ddl_cache[doc.table_name] = doc
                except Exception as e:
                    logger.warning(f"加载 DDL 文件失败 {filename}: {e}")

    def _load_sqls(self):
        """加载 SQL 示例"""
        if not os.path.exists(self.sql_path):
            return
        for filename in os.listdir(self.sql_path):
            if filename.endswith(".json"):
                filepath = os.path.join(self.sql_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        example = SQLExample.from_dict(data)
                        self._sql_cache[example.id] = example
                except Exception as e:
                    logger.warning(f"加载 SQL 文件失败 {filename}: {e}")

    def _load_docs(self):
        """加载业务文档"""
        if not os.path.exists(self.doc_path):
            return
        for filename in os.listdir(self.doc_path):
            if filename.endswith(".json"):
                filepath = os.path.join(self.doc_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        doc = BusinessDocumentation.from_dict(data)
                        self._doc_cache[doc.id] = doc
                except Exception as e:
                    logger.warning(f"加载文档文件失败 {filename}: {e}")

    async def add_ddl(self, doc: DDLDocument) -> bool:
        """添加 DDL 文档"""
        try:
            self._ddl_cache[doc.table_name] = doc

            filepath = os.path.join(self.ddl_path, f"{doc.table_name}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"添加 DDL: {doc.table_name}")
            return True
        except Exception as e:
            logger.error(f"添加 DDL 失败: {e}")
            return False

    async def add_sql_example(self, example: SQLExample) -> bool:
        """添加 SQL 示例"""
        try:
            if not example.id:
                import hashlib
                example.id = hashlib.md5(
                    f"{example.question}{example.sql}".encode()
                ).hexdigest()[:16]

            self._sql_cache[example.id] = example

            filepath = os.path.join(self.sql_path, f"{example.id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(example.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"添加 SQL 示例: {example.id}")
            return True
        except Exception as e:
            logger.error(f"添加 SQL 示例失败: {e}")
            return False

    async def add_documentation(self, doc: BusinessDocumentation) -> bool:
        """添加业务文档"""
        try:
            if not doc.id:
                import hashlib
                doc.id = hashlib.md5(
                    f"{doc.title}{doc.content}".encode()
                ).hexdigest()[:16]

            self._doc_cache[doc.id] = doc

            filepath = os.path.join(self.doc_path, f"{doc.id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"添加业务文档: {doc.id}")
            return True
        except Exception as e:
            logger.error(f"添加业务文档失败: {e}")
            return False

    async def get_ddl(self, table_name: str) -> Optional[DDLDocument]:
        """获取 DDL 文档"""
        return self._ddl_cache.get(table_name)

    async def get_sql_examples(
        self,
        database: str = "",
        limit: int = 100
    ) -> List[SQLExample]:
        """获取 SQL 示例列表"""
        examples = list(self._sql_cache.values())

        if database:
            examples = [e for e in examples if e.database == database]

        return examples[:limit]

    async def get_documentation(self, doc_id: str) -> Optional[BusinessDocumentation]:
        """获取业务文档"""
        return self._doc_cache.get(doc_id)

    async def delete_ddl(self, table_name: str) -> bool:
        """删除 DDL 文档"""
        try:
            if table_name in self._ddl_cache:
                del self._ddl_cache[table_name]

                filepath = os.path.join(self.ddl_path, f"{table_name}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)

                logger.info(f"删除 DDL: {table_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除 DDL 失败: {e}")
            return False

    async def delete_sql_example(self, example_id: str) -> bool:
        """删除 SQL 示例"""
        try:
            if example_id in self._sql_cache:
                del self._sql_cache[example_id]

                filepath = os.path.join(self.sql_path, f"{example_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)

                logger.info(f"删除 SQL 示例: {example_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除 SQL 示例失败: {e}")
            return False

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识库"""
        query_lower = query.lower()
        results = []

        # 搜索 DDL
        for table_name, doc in self._ddl_cache.items():
            score = 0
            if query_lower in table_name.lower():
                score += 10
            if query_lower in doc.description.lower():
                score += 5
            for rule in doc.business_rules:
                if query_lower in rule.lower():
                    score += 3

            if score > 0:
                results.append({
                    "type": "ddl",
                    "score": score,
                    "data": doc.to_dict()
                })

        # 搜索 SQL 示例
        for example_id, example in self._sql_cache.items():
            score = 0
            if query_lower in example.question.lower():
                score += 10
            if query_lower in example.sql.lower():
                score += 5
            for tag in example.tags:
                if query_lower in tag.lower():
                    score += 3

            if score > 0:
                results.append({
                    "type": "sql",
                    "score": score,
                    "data": example.to_dict()
                })

        # 搜索业务文档
        for doc_id, doc in self._doc_cache.items():
            score = 0
            if query_lower in doc.title.lower():
                score += 10
            if query_lower in doc.content.lower():
                score += 5

            if score > 0:
                results.append({
                    "type": "doc",
                    "score": score,
                    "data": doc.to_dict()
                })

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "ddl_count": len(self._ddl_cache),
            "sql_count": len(self._sql_cache),
            "doc_count": len(self._doc_cache)
        }

    async def bulk_import(
        self,
        ddl_docs: List[DDLDocument] = None,
        sql_examples: List[SQLExample] = None,
        docs: List[BusinessDocumentation] = None
    ) -> Dict[str, int]:
        """批量导入"""
        stats = {"ddl": 0, "sql": 0, "doc": 0}

        if ddl_docs:
            for doc in ddl_docs:
                if await self.add_ddl(doc):
                    stats["ddl"] += 1

        if sql_examples:
            for example in sql_examples:
                if await self.add_sql_example(example):
                    stats["sql"] += 1

        if docs:
            for doc in docs:
                if await self.add_documentation(doc):
                    stats["doc"] += 1

        logger.info(f"批量导入完成: {stats}")
        return stats


def create_knowledge_base(base_path: str = "./data/knowledge") -> KnowledgeBase:
    """创建知识库实例"""
    return FileKnowledgeBase(base_path)
