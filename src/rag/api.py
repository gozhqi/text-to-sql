"""
RAG 方案 - API 接口

提供 REST API 接口
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from loguru import logger

from .knowledge.base import (
    DDLDocument,
    SQLExample,
    BusinessDocumentation,
    create_knowledge_base
)
from .retrieval.vector_store import VectorStore, HybridRetriever, create_vector_store
from .generation.generator import RAGSQLGenerator, AdvancedRAGGenerator, GenerationResult


# ==================== Request/Response Models ====================

class DDLRequest(BaseModel):
    """DDL 请求"""
    table_name: str = Field(..., description="表名")
    ddl: str = Field(..., description="DDL 语句")
    description: str = Field("", description="表描述")
    business_rules: List[str] = Field(default_factory=list, description="业务规则")
    database: str = Field("", description="数据库名")


class SQLExampleRequest(BaseModel):
    """SQL 示例请求"""
    question: str = Field(..., description="问题")
    sql: str = Field(..., description="SQL 语句")
    database: str = Field("", description="数据库名")
    table_names: List[str] = Field(default_factory=list, description="涉及的表")
    difficulty: str = Field("easy", description="难度")
    tags: List[str] = Field(default_factory=list, description="标签")


class DocumentRequest(BaseModel):
    """业务文档请求"""
    title: str = Field(..., description="标题")
    content: str = Field(..., description="内容")
    database: str = Field("", description="数据库名")
    related_tables: List[str] = Field(default_factory=list, description="相关表")
    tags: List[str] = Field(default_factory=list, description="标签")


class GenerateRequest(BaseModel):
    """生成 SQL 请求"""
    question: str = Field(..., description="用户问题")
    db_name: str = Field("", description="数据库名")
    top_k: int = Field(3, description="检索数量")
    use_advanced: bool = Field(False, description="使用高级生成")


class GenerateResponse(BaseModel):
    """生成响应"""
    success: bool
    sql: str
    explanation: str
    confidence: float
    context_used: Dict[str, Any]
    error: str = ""


class StatsResponse(BaseModel):
    """统计响应"""
    ddl_count: int
    sql_count: int
    doc_count: int
    vector_stats: Dict[str, int] = {}


# ==================== API Router ====================

class RAGAPI:
    """RAG API 接口"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.knowledge_base = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self._initialized = False

    async def init(self):
        """初始化 RAG API"""
        if self._initialized:
            return

        # 初始化组件
        self.knowledge_base = create_knowledge_base()
        self.vector_store = create_vector_store()
        await self.vector_store.init()

        self.retriever = HybridRetriever(self.vector_store, self.knowledge_base)

        if self.llm_client:
            self.generator = RAGSQLGenerator(self.llm_client, self.retriever)

        self._initialized = True
        logger.info("RAG API 初始化完成")

    def create_router(self) -> APIRouter:
        """创建 API 路由"""
        router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

        @router.post("/train/ddl", response_model=Dict[str, Any])
        async def train_ddl(request: DDLRequest):
            """训练 DDL"""
            if not self._initialized:
                await self.init()

            doc = DDLDocument(
                table_name=request.table_name,
                ddl=request.ddl,
                description=request.description,
                business_rules=request.business_rules,
                database=request.database
            )

            # 添加到知识库
            success_kb = await self.knowledge_base.add_ddl(doc)

            # 添加到向量库
            content = doc.to_document()
            success_vs = await self.vector_store.add_ddl(
                request.table_name,
                content,
                {"database": request.database}
            )

            return {
                "success": success_kb and success_vs,
                "table_name": request.table_name
            }

        @router.post("/train/sql", response_model=Dict[str, Any])
        async def train_sql(request: SQLExampleRequest):
            """训练 SQL 示例"""
            if not self._initialized:
                await self.init()

            import hashlib
            example_id = hashlib.md5(
                f"{request.question}{request.sql}".encode()
            ).hexdigest()[:16]

            example = SQLExample(
                id=example_id,
                question=request.question,
                sql=request.sql,
                database=request.database,
                table_names=request.table_names,
                difficulty=request.difficulty,
                tags=request.tags
            )

            # 添加到知识库
            success_kb = await self.knowledge_base.add_sql_example(example)

            # 添加到向量库
            content = example.to_document()
            success_vs = await self.vector_store.add_sql_example(
                example_id,
                content,
                {"database": request.database, "difficulty": request.difficulty}
            )

            return {
                "success": success_kb and success_vs,
                "example_id": example_id
            }

        @router.post("/train/document", response_model=Dict[str, Any])
        async def train_document(request: DocumentRequest):
            """训练业务文档"""
            if not self._initialized:
                await self.init()

            import hashlib
            doc_id = hashlib.md5(
                f"{request.title}{request.content}".encode()
            ).hexdigest()[:16]

            doc = BusinessDocumentation(
                id=doc_id,
                title=request.title,
                content=request.content,
                database=request.database,
                related_tables=request.related_tables,
                tags=request.tags
            )

            # 添加到知识库
            success_kb = await self.knowledge_base.add_documentation(doc)

            # 添加到向量库
            content = doc.to_document()
            success_vs = await self.vector_store.add_business_doc(
                doc_id,
                content,
                {"database": request.database}
            )

            return {
                "success": success_kb and success_vs,
                "doc_id": doc_id
            }

        @router.post("/train/batch", response_model=Dict[str, int])
        async def batch_train(
            ddl_docs: List[DDLRequest] = [],
            sql_examples: List[SQLExampleRequest] = [],
            documents: List[DocumentRequest] = []
        ):
            """批量训练"""
            if not self._initialized:
                await self.init()

            stats = {"ddl": 0, "sql": 0, "doc": 0}

            for ddl_req in ddl_docs:
                result = await train_ddl(ddl_req)
                if result.get("success"):
                    stats["ddl"] += 1

            for sql_req in sql_examples:
                result = await train_sql(sql_req)
                if result.get("success"):
                    stats["sql"] += 1

            for doc_req in documents:
                result = await train_document(doc_req)
                if result.get("success"):
                    stats["doc"] += 1

            return stats

        @router.post("/train/import", response_model=Dict[str, int])
        async def import_file(
            file_type: str = "sql",  # ddl, sql, doc
            file: UploadFile = File(...)
        ):
            """从文件导入"""
            if not self._initialized:
                await self.init()

            content = await file.read()

            try:
                data = __import__("json").loads(content)

                if file_type == "sql":
                    count = 0
                    for item in data:
                        example = SQLExample.from_dict(item)
                        if await self.knowledge_base.add_sql_example(example):
                            count += 1
                    return {"imported": count}

                elif file_type == "ddl":
                    count = 0
                    for item in data:
                        doc = DDLDocument.from_dict(item)
                        if await self.knowledge_base.add_ddl(doc):
                            count += 1
                    return {"imported": count}

                else:
                    raise HTTPException(status_code=400, detail="不支持的文件类型")

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"导入失败: {str(e)}")

        @router.post("/generate", response_model=GenerateResponse)
        async def generate_sql(request: GenerateRequest):
            """生成 SQL"""
            if not self._initialized:
                await self.init()

            if not self.generator:
                raise HTTPException(status_code=400, detail="生成器未初始化")

            if request.use_advanced:
                gen = AdvancedRAGGenerator(self.llm_client, self.retriever)
                result = await gen.generate_complex(
                    request.question,
                    request.db_name
                )
            else:
                result = await self.generator.generate(
                    request.question,
                    request.db_name,
                    top_k=request.top_k
                )

            return GenerateResponse(
                success=result.success,
                sql=result.sql,
                explanation=result.explanation,
                confidence=result.confidence,
                context_used=result.context_used,
                error=result.error
            )

        @router.get("/search")
        async def search(query: str, limit: int = 10):
            """搜索知识库"""
            if not self._initialized:
                await self.init()

            results = await self.knowledge_base.search(query, limit)
            return {"results": results}

        @router.get("/stats", response_model=StatsResponse)
        async def get_stats():
            """获取统计信息"""
            if not self._initialized:
                await self.init()

            kb_stats = await self.knowledge_base.get_stats()
            vs_stats = await self.vector_store.get_collection_stats()

            return StatsResponse(
                ddl_count=kb_stats.get("ddl_count", 0),
                sql_count=kb_stats.get("sql_count", 0),
                doc_count=kb_stats.get("doc_count", 0),
                vector_stats=vs_stats
            )

        @router.delete("/clear")
        async def clear_knowledge():
            """清空知识库"""
            if not self._initialized:
                await self.init()

            await self.vector_store.clear_collection("ddl_docs")
            await self.vector_store.clear_collection("sql_examples")
            await self.vector_store.clear_collection("business_docs")

            return {"success": True, "message": "知识库已清空"}

        return router


# ==================== 便捷函数 ====================

def create_rag_api(llm_client=None) -> RAGAPI:
    """创建 RAG API 实例"""
    return RAGAPI(llm_client)
