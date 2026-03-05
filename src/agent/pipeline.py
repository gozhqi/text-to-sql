"""
Agent 方案 - 完整管道

整合 ReAct 循环、工具集和自我纠错，提供完整的 Agent 驱动 Text-to-SQL 服务
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .react_loop import (
    ReActLoop,
    SelfCorrectionReActLoop,
    ReActResult,
    Step,
    ThoughtType,
    create_react_loop,
    create_self_correcting_react_loop
)
from .tools import (
    create_database_tools,
    get_tool_schemas,
    Tool
)


@dataclass
class AgentConfig:
    """Agent 配置"""
    # LLM 配置
    llm_api_key: str = ""
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1

    # 数据库配置
    db_type: str = "mysql"
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""

    # Agent 配置
    max_iterations: int = 10
    enable_self_correction: bool = True
    verbose: bool = True

    @property
    def database_url(self) -> str:
        """构建数据库连接 URL"""
        if self.db_type == "mysql":
            return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "postgresql":
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "sqlite":
            return f"sqlite:///{self.db_name}"
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")


@dataclass
class AgentResult:
    """Agent 结果"""
    sql: str
    explanation: str
    steps: List[Dict[str, Any]]
    success: bool
    execution_time: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    query_result: Optional[List[Dict[str, Any]]] = None


class SQLValidator:
    """SQL 验证器"""

    def __init__(self, engine: Engine):
        self.engine = engine

    async def validate(self, sql: str, database: str = "") -> "ValidationResult":
        """验证 SQL"""
        try:
            # 检查危险关键词
            sql_upper = sql.upper()
            dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "EXEC"]
            if any(kw in sql_upper for kw in dangerous):
                return ValidationResult(
                    is_valid=False,
                    error=f"SQL 包含危险操作: {', '.join([kw for kw in dangerous if kw in sql_upper])}"
                )

            # 尝试解析 SQL
            with self.engine.connect() as conn:
                # 使用 EXPLAIN 验证语法
                if sql_upper.strip().startswith("SELECT"):
                    result = conn.execute(text(f"EXPLAIN {sql}"))
                else:
                    # 其他类型直接尝试执行
                    result = conn.execute(text(sql))

            return ValidationResult(is_valid=True, error="")

        except Exception as e:
            return ValidationResult(is_valid=False, error=str(e))


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    error: str = ""


class AgentPipeline:
    """Agent 完整管道"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self.tools: List[Tool] = []
        self.react_loop: Optional[ReActLoop] = None
        self.validator: Optional[SQLValidator] = None
        self.llm_client: Optional[AsyncOpenAI] = None

        self._initialized = False

    async def initialize(self):
        """初始化管道"""
        if self._initialized:
            return

        logger.info("初始化 Agent 管道...")

        try:
            # 初始化数据库连接
            if SQLALCHEMY_AVAILABLE:
                self.engine = create_engine(
                    self.config.database_url,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
                logger.info(f"数据库连接成功: {self.config.db_type}")

                # 初始化工具集
                self.tools = create_database_tools(self.engine)
                logger.info(f"加载 {len(self.tools)} 个工具")

                # 初始化验证器
                self.validator = SQLValidator(self.engine)
            else:
                logger.warning("SQLAlchemy 不可用，数据库工具将无法使用")

            # 初始化 LLM 客户端
            if OPENAI_AVAILABLE and self.config.llm_api_key:
                self.llm_client = AsyncOpenAI(
                    api_key=self.config.llm_api_key,
                    base_url=self.config.llm_base_url
                )
                logger.info(f"LLM 客户端初始化成功: {self.config.llm_model}")
            else:
                logger.warning("OpenAI 不可用，Agent 将无法运行")

            # 初始化 ReAct 循环
            if self.llm_client:
                if self.config.enable_self_correction and self.validator:
                    self.react_loop = create_self_correcting_react_loop(
                        self.llm_client,
                        self.tools,
                        validator=self.validator
                    )
                else:
                    self.react_loop = create_react_loop(
                        self.llm_client,
                        self.tools,
                        max_iterations=self.config.max_iterations
                    )

            self._initialized = True
            logger.info("Agent 管道初始化完成")

        except Exception as e:
            logger.error(f"Agent 管道初始化失败: {e}")
            raise

    async def query(
        self,
        question: str,
        execute_query: bool = False,
        return_steps: bool = True
    ) -> AgentResult:
        """执行 Agent 查询"""
        start_time = datetime.now()

        try:
            if not self._initialized:
                await self.initialize()

            if not self.react_loop:
                return AgentResult(
                    sql="",
                    explanation="",
                    steps=[],
                    success=False,
                    error="Agent 未初始化"
                )

            logger.info(f"处理查询: {question}")

            # 运行 ReAct 循环
            react_result = await self.react_loop.run(
                question,
                context={"database": self.config.db_name}
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # 转换步骤格式
            steps = [self._convert_step(step) for step in react_result.steps]

            # 执行查询（可选）
            query_result = None
            if execute_query and react_result.sql and self.engine:
                query_result = await self._execute_query(react_result.sql)

            return AgentResult(
                sql=react_result.sql,
                explanation=react_result.explanation,
                steps=steps if return_steps else [],
                success=react_result.success,
                execution_time=execution_time,
                tokens_used=react_result.total_tokens,
                error=react_result.error,
                query_result=query_result
            )

        except Exception as e:
            logger.error(f"Agent 查询失败: {e}")
            return AgentResult(
                sql="",
                explanation="",
                steps=[],
                success=False,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )

    def _convert_step(self, step: Step) -> Dict[str, Any]:
        """转换步骤为字典格式"""
        return {
            "type": step.step_type.value,
            "content": step.content,
            "tool_name": step.tool_name,
            "tool_input": step.tool_input,
            "tool_output": step.tool_output,
            "timestamp": step.timestamp
        }

    async def _execute_query(self, sql: str, limit: int = 100) -> List[Dict[str, Any]]:
        """执行查询"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()

                return [dict(zip(columns, row)) for row in rows[:limit]]

        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            return []

    async def validate_sql(self, sql: str) -> ValidationResult:
        """验证 SQL"""
        if not self.validator:
            return ValidationResult(is_valid=False, error="验证器未初始化")

        return await self.validator.validate(sql, self.config.db_name)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取工具模式"""
        return get_tool_schemas(self.tools)

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "initialized": self._initialized,
            "tools_count": len(self.tools),
            "enable_self_correction": self.config.enable_self_correction,
            "max_iterations": self.config.max_iterations
        }

        # 数据库统计
        if self.engine:
            try:
                from sqlalchemy import inspect
                inspector = inspect(self.engine)
                stats["tables"] = len(inspector.get_table_names())
            except:
                stats["tables"] = 0

        return stats


class AgentService:
    """Agent 服务 - 高层 API"""

    def __init__(self, config: AgentConfig):
        self.pipeline = AgentPipeline(config)

    async def initialize(self):
        """初始化服务"""
        await self.pipeline.initialize()

    def query_sync(self, question: str, execute_query: bool = False) -> Dict[str, Any]:
        """同步查询接口"""
        result = asyncio.run(self.pipeline.query(question, execute_query))
        return {
            "sql": result.sql,
            "explanation": result.explanation,
            "steps": result.steps,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error,
            "query_result": result.query_result
        }


def create_agent_pipeline(config: AgentConfig) -> AgentPipeline:
    """创建 Agent 管道实例"""
    return AgentPipeline(config)


def create_agent_service(config: AgentConfig) -> AgentService:
    """创建 Agent 服务实例"""
    return AgentService(config)


# 便捷函数
async def quick_query(
    question: str,
    db_url: str,
    api_key: str,
    model: str = "gpt-4o"
) -> str:
    """快速查询"""
    config = AgentConfig(
        llm_api_key=api_key,
        llm_model=model,
        database_url=db_url
    )

    pipeline = create_agent_pipeline(config)
    await pipeline.initialize()

    result = await pipeline.query(question)
    return result.sql
