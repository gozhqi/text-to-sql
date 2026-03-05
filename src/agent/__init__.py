"""
Agent 方案 - 模块初始化

完整的 Agent 驱动 Text-to-SQL 方案
"""
from .tools.base import (
    BaseTool,
    ToolRegistry,
    ToolResult,
    ToolCategory,
    ListTablesTool,
    GetSchemaTool,
    SearchColumnsTool,
    FindJoinPathTool,
    GenerateSQLTool,
    ValidateSQLTool,
    TestExecuteTool,
    create_tool_registry
)

from .react.loop import (
    ActionType,
    Thought,
    Observation,
    AgentState,
    ReActLoop,
    create_react_loop
)

from .orchestrator.base import (
    MultiAgentOrchestrator,
    SchemaExplorerAgent,
    SQLGeneratorAgent,
    ValidatorAgent,
    create_orchestrator
)

__all__ = [
    # Tools
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "ToolCategory",
    "ListTablesTool",
    "GetSchemaTool",
    "SearchColumnsTool",
    "FindJoinPathTool",
    "GenerateSQLTool",
    "ValidateSQLTool",
    "TestExecuteTool",
    "create_tool_registry",

    # ReAct
    "ActionType",
    "Thought",
    "Observation",
    "AgentState",
    "ReActLoop",
    "create_react_loop",

    # Orchestrator
    "MultiAgentOrchestrator",
    "SchemaExplorerAgent",
    "SQLGeneratorAgent",
    "ValidatorAgent",
    "create_orchestrator",
]
