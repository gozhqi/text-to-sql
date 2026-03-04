"""
数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class QueryIntent(str, Enum):
    """查询意图类型"""
    NEW_QUERY = "new_query"
    REFINE = "refine"
    MODIFY = "modify"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    SORT = "sort"
    COMPARE = "compare"
    DRILL_DOWN = "drill_down"
    CLARIFICATION = "clarification"


class TableColumn(BaseModel):
    """表字段"""
    name: str
    type: str
    comment: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: Optional[str] = None  # 格式: table.column


class TableSchema(BaseModel):
    """表结构信息"""
    table_name: str
    table_comment: str = ""
    columns: List[TableColumn] = []
    row_count: int = 0
    
    def to_prompt_text(self) -> str:
        """转换为Prompt文本"""
        lines = [f"表名: {self.table_name}"]
        if self.table_comment:
            lines.append(f"描述: {self.table_comment}")
        lines.append("字段:")
        for col in self.columns:
            pk = " [主键]" if col.is_primary_key else ""
            fk = f" [外键→{col.foreign_key_ref}]" if col.is_foreign_key else ""
            lines.append(f"  - {col.name} ({col.type}): {col.comment}{pk}{fk}")
        return "\n".join(lines)


class ConversationTurn(BaseModel):
    """单轮对话记录"""
    turn_id: int
    timestamp: datetime
    user_question: str
    rewritten_question: Optional[str] = None
    generated_sql: str
    sql_result_summary: str = ""
    referenced_tables: List[str] = []
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    intent: QueryIntent = QueryIntent.NEW_QUERY


class ConversationSession(BaseModel):
    """对话会话"""
    session_id: str
    db_name: str
    created_at: datetime = Field(default_factory=datetime.now)
    turns: List[ConversationTurn] = []
    
    def get_last_turn(self) -> Optional[ConversationTurn]:
        """获取最后一轮对话"""
        return self.turns[-1] if self.turns else None
    
    def get_context_summary(self, max_turns: int = 3) -> str:
        """生成上下文摘要"""
        if not self.turns:
            return ""
        
        summary_parts = []
        for turn in self.turns[-max_turns:]:
            summary_parts.append(f"用户: {turn.user_question}")
            summary_parts.append(f"SQL: {turn.generated_sql}")
        
        return "\n".join(summary_parts)


# ========== API请求/响应模型 ==========

class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., description="用户问题")
    db_name: str = Field(..., description="数据库名称")
    session_id: Optional[str] = Field(None, description="会话ID(多轮对话需要)")


class QueryResponse(BaseModel):
    """查询响应"""
    success: bool = True
    sql: str = ""
    explanation: str = ""
    is_multi_turn: bool = False
    rewritten_question: Optional[str] = None
    relevant_tables: List[str] = []
    intent: str = "new_query"
    results: Optional[List[Dict[str, Any]]] = None
    result_count: int = 0
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: str
    db_name: str


class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool
    message: str
    sql: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    session_id: str


class SchemaResponse(BaseModel):
    """Schema响应"""
    tables: List[TableSchema]
    total_tables: int
    total_columns: int


class SQLGenerationResult(BaseModel):
    """SQL生成结果"""
    sql: str
    explanation: str = ""
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 0.0