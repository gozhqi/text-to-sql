"""
Text-to-SQL 处理流水线
"""
from typing import Dict, List, Optional, Any
from loguru import logger

from app.config import get_settings
from app.models.schemas import (
    TableSchema, QueryResponse, QueryIntent
)
from app.models.database import get_db_manager
from app.core.schema_retriever import SchemaRetriever
from app.core.sql_generator import get_sql_generator
from app.core.sql_validator import get_sql_validator
from app.core.context_manager import get_context_manager


class TextToSQLPipeline:
    """Text-to-SQL 完整处理流水线"""
    
    def __init__(self):
        self.settings = get_settings()
        self.schema_retriever = SchemaRetriever()
        self._initialized = False
    
    async def init(self):
        """初始化流水线"""
        if self._initialized:
            return
        
        # 初始化各组件
        await self.schema_retriever.init()
        await get_sql_generator()
        get_sql_validator()
        get_context_manager()
        
        self._initialized = True
        logger.info("Text-to-SQL Pipeline 初始化完成")
    
    async def process(
        self,
        question: str,
        db_name: str,
        session_id: Optional[str] = None
    ) -> QueryResponse:
        """
        处理用户问题，生成SQL并执行
        
        Args:
            question: 用户问题
            db_name: 数据库名称
            session_id: 会话ID（多轮对话需要）
        
        Returns:
            QueryResponse
        """
        
        if not self._initialized:
            await self.init()
        
        try:
            # 获取数据库管理器
            db_manager = await get_db_manager()
            
            # 获取Schema
            schemas = await db_manager.get_table_schemas(db_name)
            
            # 上下文管理
            rewritten_question = question
            is_multi_turn = False
            intent = QueryIntent.NEW_QUERY.value
            context_summary = ""
            
            context_mgr = get_context_manager()
            
            if session_id:
                session = context_mgr.get_or_create_session(session_id, db_name)
                
                if session.turns:
                    is_multi_turn = True
                    last_turn = session.get_last_turn()
                    
                    # 问题改写
                    sql_gen = await get_sql_generator()
                    rewritten_question = await sql_gen.rewrite_question(
                        current_question=question,
                        last_question=last_turn.user_question,
                        last_sql=last_turn.generated_sql,
                        referenced_tables=last_turn.referenced_tables
                    )
                    
                    # 意图识别
                    intent = await sql_gen.classify_intent(
                        current_question=question,
                        last_question=last_turn.user_question,
                        last_sql=last_turn.generated_sql
                    )
                    
                    context_summary = context_mgr.get_context_summary(session_id)
            
            # Schema检索
            relevant_table_names = await self.schema_retriever.hybrid_retrieve(
                question=rewritten_question,
                db_name=db_name,
                schemas=schemas,
                top_k=5
            )
            
            relevant_tables = [
                schemas[name] for name in relevant_table_names 
                if name in schemas
            ]
            
            # 如果没找到相关表，使用所有表
            if not relevant_tables:
                relevant_tables = list(schemas.values())[:5]
            
            # 生成SQL
            sql_gen = await get_sql_generator()
            result = await sql_gen.generate(
                question=rewritten_question,
                tables=relevant_tables,
                context_summary=context_summary
            )
            
            sql = result.sql
            
            # SQL校验与修复
            validator = get_sql_validator()
            sql, is_valid, error = validator.validate_and_fix(sql, relevant_tables)
            
            if not is_valid:
                return QueryResponse(
                    success=False,
                    sql=sql,
                    explanation=result.explanation,
                    error=f"SQL校验失败: {error}"
                )
            
            # 执行SQL
            try:
                results = await db_manager.execute_sql(sql)
                result_count = len(results)
                result_summary = f"返回 {result_count} 条记录"
            except Exception as e:
                logger.error(f"SQL执行失败: {e}")
                return QueryResponse(
                    success=False,
                    sql=sql,
                    explanation=result.explanation,
                    error=f"SQL执行失败: {str(e)}"
                )
            
            # 保存对话轮次
            if session_id:
                context_mgr.add_turn(
                    session_id=session_id,
                    question=question,
                    rewritten_question=rewritten_question if is_multi_turn else None,
                    sql=sql,
                    result_summary=result_summary,
                    tables=relevant_table_names,
                    intent=intent
                )
            
            return QueryResponse(
                success=True,
                sql=sql,
                explanation=result.explanation,
                is_multi_turn=is_multi_turn,
                rewritten_question=rewritten_question if is_multi_turn else None,
                relevant_tables=relevant_table_names,
                intent=intent,
                results=results[:100],  # 限制返回数量
                result_count=result_count
            )
            
        except Exception as e:
            logger.exception(f"处理失败: {e}")
            return QueryResponse(
                success=False,
                error=f"处理失败: {str(e)}"
            )
    
    async def get_schema_info(self, db_name: str) -> Dict[str, Any]:
        """获取数据库Schema信息"""
        
        db_manager = await get_db_manager()
        schemas = await db_manager.get_table_schemas(db_name)
        
        return {
            "db_name": db_name,
            "tables": [
                {
                    "name": schema.table_name,
                    "comment": schema.table_comment,
                    "columns": [
                        {
                            "name": col.name,
                            "type": col.type,
                            "comment": col.comment
                        }
                        for col in schema.columns
                    ]
                }
                for schema in schemas.values()
            ],
            "total_tables": len(schemas),
            "total_columns": sum(len(s.columns) for s in schemas.values())
        }
    
    async def build_schema_index(self, db_name: str):
        """构建Schema向量索引"""
        
        if not self._initialized:
            await self.init()
        
        db_manager = await get_db_manager()
        schemas = await db_manager.get_table_schemas(db_name)
        
        await self.schema_retriever.build_schema_index(db_name, schemas)
        
        return {"success": True, "tables_indexed": len(schemas)}


# 全局实例
pipeline: Optional[TextToSQLPipeline] = None


async def get_pipeline() -> TextToSQLPipeline:
    """获取Pipeline实例"""
    global pipeline
    if pipeline is None:
        pipeline = TextToSQLPipeline()
        await pipeline.init()
    return pipeline