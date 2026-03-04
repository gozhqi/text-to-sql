"""
数据库连接管理
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from loguru import logger

from app.config import get_settings
from app.models.schemas import TableSchema, TableColumn


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.async_engine = None
        self.sync_engine = None
        self.async_session_factory = None
        self._schema_cache: Dict[str, Dict[str, TableSchema]] = {}
    
    async def init(self):
        """初始化数据库连接"""
        # 异步引擎
        self.async_engine = create_async_engine(
            self.settings.database_url,
            echo=self.settings.debug,
            pool_size=5,
            max_overflow=10
        )
        self.async_session_factory = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # 同步引擎 (用于Schema获取等同步操作)
        self.sync_engine = create_engine(
            self.settings.sync_database_url,
            echo=self.settings.debug
        )
        
        logger.info("数据库连接初始化完成")
    
    async def close(self):
        """关闭数据库连接"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.sync_engine:
            self.sync_engine.dispose()
        logger.info("数据库连接已关闭")
    
    @asynccontextmanager
    async def get_session(self):
        """获取异步会话"""
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"数据库操作错误: {e}")
                raise
    
    async def get_table_schemas(self, db_name: str) -> Dict[str, TableSchema]:
        """获取数据库所有表结构"""
        if db_name in self._schema_cache:
            return self._schema_cache[db_name]
        
        schemas = {}
        
        async with self.get_session() as session:
            # 获取所有表名
            if self.settings.db_type == "mysql":
                tables_result = await session.execute(text("""
                    SELECT TABLE_NAME, TABLE_COMMENT 
                    FROM information_schema.TABLES 
                    WHERE TABLE_SCHEMA = :db_name
                """), {"db_name": db_name})
            else:  # PostgreSQL
                tables_result = await session.execute(text("""
                    SELECT table_name as TABLE_NAME, 
                           obj_description((table_schema || '.' || table_name)::regclass) as TABLE_COMMENT
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """))
            
            tables = tables_result.fetchall()
            
            for table_row in tables:
                table_name = table_row[0]
                table_comment = table_row[1] or ""
                
                # 获取列信息
                if self.settings.db_type == "mysql":
                    columns_result = await session.execute(text("""
                        SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT, COLUMN_KEY
                        FROM information_schema.COLUMNS
                        WHERE TABLE_SCHEMA = :db_name AND TABLE_NAME = :table_name
                        ORDER BY ORDINAL_POSITION
                    """), {"db_name": db_name, "table_name": table_name})
                else:
                    columns_result = await session.execute(text("""
                        SELECT column_name, data_type, 
                               col_description((table_schema || '.' || table_name)::regclass, ordinal_position),
                               ''
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = :table_name
                        ORDER BY ordinal_position
                    """), {"table_name": table_name})
                
                columns = []
                for col_row in columns_result.fetchall():
                    columns.append(TableColumn(
                        name=col_row[0],
                        type=col_row[1],
                        comment=col_row[2] or "",
                        is_primary_key=(col_row[3] == "PRI") if self.settings.db_type == "mysql" else False
                    ))
                
                schemas[table_name] = TableSchema(
                    table_name=table_name,
                    table_comment=table_comment,
                    columns=columns
                )
        
        self._schema_cache[db_name] = schemas
        return schemas
    
    async def execute_sql(self, sql: str, limit: int = None) -> List[Dict[str, Any]]:
        """执行SQL查询"""
        if limit is None:
            limit = self.settings.max_query_rows
        
        # 添加LIMIT
        sql_upper = sql.upper().strip()
        if "LIMIT" not in sql_upper and (sql_upper.startswith("SELECT") or sql_upper.startswith("SHOW")):
            sql = f"{sql.rstrip(';')} LIMIT {limit}"
        
        async with self.get_session() as session:
            result = await session.execute(text(sql))
            
            if result.returns_rows:
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return rows
            else:
                await session.commit()
                return []


# 全局数据库管理器实例
db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    """获取数据库管理器实例"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.init()
    return db_manager