"""
RAG 方案 - 完整管道

整合知识库、检索器和生成器，提供完整的 Text-to-SQL 服务
"""
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime
import json

from .knowledge.base import (
    FileKnowledgeBase,
    DDLDocument,
    SQLExample,
    BusinessDocumentation,
    create_knowledge_base
)
from .retrieval.vector_store import (
    VectorStore,
    HybridRetriever,
    create_vector_store,
    create_hybrid_retriever
)
from .generator.sql_generator import (
    SQLGenerator,
    GenerationContext,
    GenerationResult,
    create_sql_generator
)


@dataclass
class RAGConfig:
    """RAG 配置"""
    # 知识库配置
    knowledge_base_path: str = "./data/knowledge"

    # 向量存储配置
    vector_store_path: str = "./data/chroma"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # LLM 配置
    llm_api_key: str = ""
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1

    # 检索配置
    top_k_ddl: int = 3
    top_k_sql: int = 3
    top_k_doc: int = 2

    # 生成配置
    use_cot: bool = False
    max_tokens: int = 2000


@dataclass
class RAGResult:
    """RAG 结果"""
    sql: str
    explanation: str
    confidence: float
    retrieved_context: Dict[str, List[Dict]]
    generation_time: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None


class RAGPipeline:
    """RAG 完整管道"""

    def __init__(self, config: RAGConfig):
        self.config = config

        # 初始化组件
        self.knowledge_base = create_knowledge_base(config.knowledge_base_path)
        self.vector_store = create_vector_store(
            config.vector_store_path,
            config.embedding_model
        )
        self.retriever = None
        self.generator = None

        self._initialized = False

    async def initialize(self):
        """初始化管道"""
        if self._initialized:
            return

        logger.info("初始化 RAG 管道...")

        # 初始化向量存储
        await self.vector_store.init()

        # 初始化混合检索器
        self.retriever = HybridRetriever(
            self.vector_store,
            self.knowledge_base
        )

        # 初始化生成器
        if self.config.llm_api_key:
            self.generator = create_sql_generator(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature
            )

        # 同步知识库数据到向量存储
        await self._sync_to_vector_store()

        self._initialized = True
        logger.info("RAG 管道初始化完成")

    async def _sync_to_vector_store(self):
        """同步知识库数据到向量存储"""
        logger.info("同步知识库到向量存储...")

        # 获取所有 DDL
        stats = await self.knowledge_base.get_stats()
        logger.info(f"知识库统计: {stats}")

        # 同步 DDL
        ddl_docs = await self._get_all_ddls()
        for doc in ddl_docs:
            content = doc.to_document()
            await self.vector_store.add_ddl(
                table_name=doc.table_name,
                content=content,
                metadata={
                    "database": doc.database,
                    "description": doc.description
                }
            )

        # 同步 SQL 示例
        sql_examples = await self.knowledge_base.get_sql_examples(limit=1000)
        for example in sql_examples:
            content = example.to_document()
            await self.vector_store.add_sql_example(
                example_id=example.id,
                content=content,
                metadata={
                    "database": example.database,
                    "difficulty": example.difficulty
                }
            )

        # 同步业务文档
        business_docs = await self._get_all_business_docs()
        for doc in business_docs:
            content = doc.to_document()
            await self.vector_store.add_business_doc(
                doc_id=doc.id,
                content=content,
                metadata={
                    "database": doc.database,
                    "title": doc.title
                }
            )

        logger.info("知识库同步完成")

    async def _get_all_ddls(self) -> List[DDLDocument]:
        """获取所有 DDL"""
        # 从文件系统读取
        ddls = []
        ddl_path = os.path.join(self.config.knowledge_base_path, "ddl")

        if os.path.exists(ddl_path):
            for filename in os.listdir(ddl_path):
                if filename.endswith(".json"):
                    filepath = os.path.join(ddl_path, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            ddls.append(DDLDocument.from_dict(data))
                    except Exception as e:
                        logger.warning(f"加载 DDL 失败 {filename}: {e}")

        return ddls

    async def _get_all_business_docs(self) -> List[BusinessDocumentation]:
        """获取所有业务文档"""
        docs = []
        doc_path = os.path.join(self.config.knowledge_base_path, "docs")

        if os.path.exists(doc_path):
            for filename in os.listdir(doc_path):
                if filename.endswith(".json"):
                    filepath = os.path.join(doc_path, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            docs.append(BusinessDocumentation.from_dict(data))
                    except Exception as e:
                        logger.warning(f"加载文档失败 {filename}: {e}")

        return docs

    async def query(
        self,
        question: str,
        database: str = "",
        use_hybrid: bool = True
    ) -> RAGResult:
        """执行完整的 RAG 查询"""
        start_time = datetime.now()

        try:
            if not self._initialized:
                await self.initialize()

            logger.info(f"处理查询: {question}")

            # 检索相关上下文
            if use_hybrid:
                retrieved = await self.retriever.retrieve(
                    query=question,
                    top_k=max(self.config.top_k_ddl, self.config.top_k_sql),
                    database=database
                )
            else:
                # 仅使用向量检索
                retrieved = await self.vector_store.retrieve_multi(
                    query=question,
                    top_k_ddl=self.config.top_k_ddl,
                    top_k_sql=self.config.top_k_sql,
                    top_k_doc=self.config.top_k_doc,
                    database=database
                )
                # 转换为统一格式
                retrieved = self._convert_vector_results(retrieved)

            # 构建生成上下文
            context = self._build_generation_context(
                question, database, retrieved
            )

            # 生成 SQL
            if self.generator:
                generation_result = await self.generator.generate(
                    context,
                    use_cot=self.config.use_cot
                )

                generation_time = (datetime.now() - start_time).total_seconds()

                return RAGResult(
                    sql=generation_result.sql,
                    explanation=generation_result.explanation,
                    confidence=generation_result.confidence,
                    retrieved_context=retrieved,
                    generation_time=generation_time,
                    tokens_used=generation_result.tokens_used,
                    error=generation_result.error
                )
            else:
                return RAGResult(
                    sql="",
                    explanation="",
                    confidence=0.0,
                    retrieved_context=retrieved,
                    generation_time=(datetime.now() - start_time).total_seconds(),
                    error="生成器未初始化"
                )

        except Exception as e:
            logger.error(f"RAG 查询失败: {e}")
            return RAGResult(
                sql="",
                explanation="",
                confidence=0.0,
                retrieved_context={},
                generation_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )

    def _convert_vector_results(
        self,
        vector_results: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """转换向量检索结果为统一格式"""
        converted = []

        for doc_type, results in vector_results.items():
            for result in results:
                if hasattr(result, "content"):
                    converted.append({
                        "content": result.content,
                        "metadata": result.metadata,
                        "score": result.score,
                        "type": result.type
                    })
                elif isinstance(result, dict):
                    converted.append({
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0.0),
                        "type": result.get("type", doc_type)
                    })

        # 按分数排序
        converted.sort(key=lambda x: x.get("score", 0), reverse=True)
        return converted

    def _build_generation_context(
        self,
        question: str,
        database: str,
        retrieved: List[Dict[str, Any]]
    ) -> GenerationContext:
        """构建生成上下文"""
        ddl_docs = []
        sql_examples = []
        business_docs = []

        for item in retrieved:
            doc_type = item.get("type", "")

            if doc_type == "ddl" or "表名" in item.get("content", ""):
                ddl_docs.append(item)
            elif doc_type == "sql" or "问题:" in item.get("content", ""):
                sql_examples.append(item)
            elif doc_type == "doc" or "标题:" in item.get("content", ""):
                business_docs.append(item)

        return GenerationContext(
            query=question,
            database=database,
            ddl_docs=ddl_docs[:self.config.top_k_ddl],
            sql_examples=sql_examples[:self.config.top_k_sql],
            business_docs=business_docs[:self.config.top_k_doc]
        )

    # ========== 知识库管理 ==========

    async def add_ddl(
        self,
        table_name: str,
        ddl: str,
        description: str = "",
        database: str = ""
    ) -> bool:
        """添加 DDL 文档"""
        doc = DDLDocument(
            table_name=table_name,
            ddl=ddl,
            description=description,
            database=database
        )

        # 添加到知识库
        kb_result = await self.knowledge_base.add_ddl(doc)

        # 添加到向量存储
        if kb_result and self.vector_store:
            content = doc.to_document()
            await self.vector_store.add_ddl(
                table_name=table_name,
                content=content,
                metadata={"database": database, "description": description}
            )

        return kb_result

    async def add_sql_example(
        self,
        question: str,
        sql: str,
        database: str = "",
        tags: List[str] = None
    ) -> bool:
        """添加 SQL 示例"""
        example = SQLExample(
            question=question,
            sql=sql,
            database=database,
            tags=tags or []
        )

        # 添加到知识库
        kb_result = await self.knowledge_base.add_sql_example(example)

        # 添加到向量存储
        if kb_result and self.vector_store:
            content = example.to_document()
            await self.vector_store.add_sql_example(
                example_id=example.id,
                content=content,
                metadata={"database": database, "difficulty": example.difficulty}
            )

        return kb_result

    async def add_business_doc(
        self,
        title: str,
        content: str,
        database: str = "",
        related_tables: List[str] = None
    ) -> bool:
        """添加业务文档"""
        doc = BusinessDocumentation(
            title=title,
            content=content,
            database=database,
            related_tables=related_tables or []
        )

        # 添加到知识库
        kb_result = await self.knowledge_base.add_documentation(doc)

        # 添加到向量存储
        if kb_result and self.vector_store:
            content = doc.to_document()
            await self.vector_store.add_business_doc(
                doc_id=doc.id,
                content=content,
                metadata={"database": database, "title": title}
            )

        return kb_result

    async def import_from_file(
        self,
        file_path: str,
        file_type: str = "json"
    ) -> Dict[str, int]:
        """从文件导入数据"""
        stats = {"ddl": 0, "sql": 0, "doc": 0}

        try:
            if file_type == "json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        if "ddl" in item or "table_name" in item:
                            if await self.add_ddl(
                                item.get("table_name", ""),
                                item.get("ddl", ""),
                                item.get("description", ""),
                                item.get("database", "")
                            ):
                                stats["ddl"] += 1

                        elif "question" in item and "sql" in item:
                            if await self.add_sql_example(
                                item.get("question", ""),
                                item.get("sql", ""),
                                item.get("database", ""),
                                item.get("tags", [])
                            ):
                                stats["sql"] += 1

                        elif "title" in item and "content" in item:
                            if await self.add_business_doc(
                                item.get("title", ""),
                                item.get("content", ""),
                                item.get("database", ""),
                                item.get("related_tables", [])
                            ):
                                stats["doc"] += 1

            elif file_type == "csv":
                import csv
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "question" in row and "sql" in row:
                            if await self.add_sql_example(
                                row["question"],
                                row["sql"],
                                row.get("database", "")
                            ):
                                stats["sql"] += 1

            logger.info(f"导入完成: {stats}")
            return stats

        except Exception as e:
            logger.error(f"导入失败: {e}")
            return stats

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        kb_stats = await self.knowledge_base.get_stats()
        vector_stats = await self.vector_store.get_collection_stats() if self.vector_store else {}

        return {
            "knowledge_base": kb_stats,
            "vector_store": vector_stats,
            "initialized": self._initialized
        }


class RAGService:
    """RAG 服务 - 高层 API"""

    def __init__(self, config: RAGConfig):
        self.pipeline = RAGPipeline(config)
        self._loop = asyncio.get_event_loop()

    async def initialize(self):
        """初始化服务"""
        await self.pipeline.initialize()

    def query_sync(self, question: str, database: str = "") -> Dict[str, Any]:
        """同步查询接口"""
        result = asyncio.run(self.pipeline.query(question, database))
        return {
            "sql": result.sql,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "generation_time": result.generation_time,
            "error": result.error
        }


def create_rag_pipeline(config: RAGConfig) -> RAGPipeline:
    """创建 RAG 管道实例"""
    return RAGPipeline(config)


def create_rag_service(config: RAGConfig) -> RAGService:
    """创建 RAG 服务实例"""
    return RAGService(config)
