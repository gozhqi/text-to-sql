"""
模型模块初始化
"""
from app.models.schemas import (
    QueryIntent, TableColumn, TableSchema,
    ConversationTurn, ConversationSession,
    QueryRequest, QueryResponse,
    ChatRequest, ChatResponse,
    SchemaResponse, SQLGenerationResult
)
from app.models.database import DatabaseManager, get_db_manager

__all__ = [
    "QueryIntent", "TableColumn", "TableSchema",
    "ConversationTurn", "ConversationSession",
    "QueryRequest", "QueryResponse",
    "ChatRequest", "ChatResponse",
    "SchemaResponse", "SQLGenerationResult",
    "DatabaseManager", "get_db_manager"
]