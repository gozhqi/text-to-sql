"""
Agent 方案 - API 接口

提供 REST API 接口
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from .tools.base import create_tool_registry
from .react.loop import create_react_loop
from .orchestrator.base import create_orchestrator


# ==================== Request/Response Models ====================

class AgentGenerateRequest(BaseModel):
    """Agent 生成请求"""
    question: str = Field(..., description="用户问题")
    db_name: str = Field(..., description="数据库名称")
    mode: str = Field("react", description="运行模式: react, multi_agent")
    max_iterations: int = Field(10, description="最大迭代次数")


class AgentGenerateResponse(BaseModel):
    """Agent 生成响应"""
    success: bool
    sql: str
    question: str
    iterations: int
    reasoning_trace: Dict[str, Any]
    method: str
    error: str = ""


class ToolsListResponse(BaseModel):
    """工具列表响应"""
    tools: list


# ==================== API Router ====================

class AgentAPI:
    """Agent API 接口"""

    def __init__(self, llm_client=None, db_manager=None):
        self.llm_client = llm_client
        self.db_manager = db_manager
        self.tool_registry = None
        self.react_loop = None
        self.orchestrator = None
        self._initialized = False

    async def init(self):
        """初始化 Agent API"""
        if self._initialized:
            return

        self.tool_registry = create_tool_registry(self.db_manager, self.llm_client)
        self.react_loop = create_react_loop(self.llm_client, self.tool_registry)
        self.orchestrator = create_orchestrator(self.llm_client, self.db_manager)

        self._initialized = True
        logger.info("Agent API 初始化完成")

    def create_router(self) -> APIRouter:
        """创建 API 路由"""
        router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])

        @router.post("/generate", response_model=AgentGenerateResponse)
        async def generate_sql(request: AgentGenerateRequest):
            """使用 Agent 生成 SQL"""
            if not self._initialized:
                await self.init()

            try:
                if request.mode == "react":
                    result = await self.react_loop.run(
                        request.question,
                        request.db_name
                    )
                elif request.mode == "multi_agent":
                    result = await self.orchestrator.process(
                        request.question,
                        request.db_name
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"不支持的运行模式: {request.mode}"
                    )

                return AgentGenerateResponse(
                    success=result.get("success", False),
                    sql=result.get("sql", ""),
                    question=result.get("question", request.question),
                    iterations=result.get("iterations", 0),
                    reasoning_trace=result.get("reasoning_trace", {}),
                    method=result.get("method", "agent"),
                    error=result.get("error", "")
                )

            except Exception as e:
                logger.error(f"Agent SQL 生成失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @router.get("/tools", response_model=ToolsListResponse)
        async def list_tools():
            """列出可用工具"""
            if not self._initialized:
                await self.init()

            tools = []
            for tool in self.tool_registry.list_tools():
                tools.append(tool.get_signature())

            return ToolsListResponse(tools=tools)

        @router.get("/tools/{tool_name}")
        async def get_tool_info(tool_name: str):
            """获取工具详情"""
            if not self._initialized:
                await self.init()

            tool = self.tool_registry.get(tool_name)
            if not tool:
                raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")

            return tool.get_signature()

        return router


# ==================== 便捷函数 ====================

def create_agent_api(llm_client=None, db_manager=None) -> AgentAPI:
    """创建 Agent API 实例"""
    return AgentAPI(llm_client, db_manager)
