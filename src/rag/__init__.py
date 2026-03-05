"""
RAG 方案 - 模块初始化

完整的 RAG 驱动 Text-to-SQL 方案
"""
from .knowledge.base import (
    DDLDocument,
    SQLExample,
    BusinessDocumentation,
    KnowledgeBase,
    FileKnowledgeBase,
    create_knowledge_base
)

from .retrieval.vector_store import (
    VectorStore,
    HybridRetriever,
    RetrievalResult,
    create_vector_store,
    create_hybrid_retriever
)

from .generation.generator import (
    RAGSQLGenerator,
    AdvancedRAGGenerator,
    GenerationResult
)

__all__ = [
    # Knowledge
    "DDLDocument",
    "SQLExample",
    "BusinessDocumentation",
    "KnowledgeBase",
    "FileKnowledgeBase",
    "create_knowledge_base",

    # Retrieval
    "VectorStore",
    "HybridRetriever",
    "RetrievalResult",
    "create_vector_store",
    "create_hybrid_retriever",

    # Generation
    "RAGSQLGenerator",
    "AdvancedRAGGenerator",
    "GenerationResult",
]
