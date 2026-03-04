"""
多轮对话上下文管理模块
"""
from typing import Dict, Optional, List
from datetime import datetime
import json
from loguru import logger

from app.models.schemas import (
    ConversationSession, ConversationTurn, QueryIntent
)


class ContextManager:
    """上下文管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_history_turns = 10  # 最大历史轮数
        self.session_timeout_minutes = 30  # 会话超时时间
    
    def create_session(self, session_id: str, db_name: str) -> ConversationSession:
        """创建新会话"""
        session = ConversationSession(
            session_id=session_id,
            db_name=db_name,
            created_at=datetime.now()
        )
        self.sessions[session_id] = session
        logger.info(f"创建会话: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def get_or_create_session(
        self, 
        session_id: str, 
        db_name: str
    ) -> ConversationSession:
        """获取或创建会话"""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id, db_name)
        return session
    
    def add_turn(
        self,
        session_id: str,
        question: str,
        rewritten_question: Optional[str],
        sql: str,
        result_summary: str,
        tables: List[str],
        intent: str = "new_query"
    ) -> ConversationTurn:
        """添加对话轮次"""
        
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"会话不存在: {session_id}")
            return None
        
        turn = ConversationTurn(
            turn_id=len(session.turns) + 1,
            timestamp=datetime.now(),
            user_question=question,
            rewritten_question=rewritten_question,
            generated_sql=sql,
            sql_result_summary=result_summary,
            referenced_tables=tables,
            intent=QueryIntent(intent)
        )
        
        # 限制历史长度
        if len(session.turns) >= self.max_history_turns:
            session.turns = session.turns[-(self.max_history_turns - 1):]
        
        session.turns.append(turn)
        logger.info(f"会话 {session_id} 添加第 {turn.turn_id} 轮对话")
        
        return turn
    
    def get_context_summary(
        self, 
        session_id: str,
        max_turns: int = 3
    ) -> str:
        """获取上下文摘要"""
        
        session = self.sessions.get(session_id)
        if not session or not session.turns:
            return ""
        
        return session.get_context_summary(max_turns)
    
    def get_last_turn(self, session_id: str) -> Optional[ConversationTurn]:
        """获取最后一轮对话"""
        
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return session.get_last_turn()
    
    def is_multi_turn(self, session_id: str) -> bool:
        """判断是否多轮对话"""
        
        session = self.sessions.get(session_id)
        return session is not None and len(session.turns) > 0
    
    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"清除会话: {session_id}")
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        
        now = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            if session.turns:
                last_turn = session.turns[-1]
                elapsed = (now - last_turn.timestamp).total_seconds() / 60
                if elapsed > self.session_timeout_minutes:
                    expired.append(session_id)
        
        for session_id in expired:
            self.clear_session(session_id)
        
        if expired:
            logger.info(f"清理了 {len(expired)} 个过期会话")
    
    def get_session_stats(self, session_id: str) -> Dict:
        """获取会话统计"""
        
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        return {
            "session_id": session_id,
            "db_name": session.db_name,
            "created_at": session.created_at.isoformat(),
            "total_turns": len(session.turns),
            "tables_used": list(set(
                table for turn in session.turns 
                for table in turn.referenced_tables
            ))
        }


# 全局实例
context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """获取上下文管理器实例"""
    global context_manager
    if context_manager is None:
        context_manager = ContextManager()
    return context_manager