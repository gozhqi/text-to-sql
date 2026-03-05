"""
Agent 方案 - ReAct 循环模块

实现 ReAct (Reasoning + Acting) 循环
"""
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..tools.base import BaseTool, ToolRegistry, ToolResult


class ActionType(Enum):
    """动作类型"""
    LIST_TABLES = "list_tables"
    GET_SCHEMA = "get_schema"
    SEARCH_COLUMNS = "search_columns"
    FIND_JOIN_PATH = "find_join_path"
    GENERATE_SQL = "generate_sql"
    VALIDATE_SQL = "validate_sql"
    TEST_EXECUTE = "test_execute"
    FINAL_ANSWER = "final_answer"


@dataclass
class Thought:
    """思考记录"""
    content: str
    iteration: int
    action: Optional[ActionType] = None
    reasoning: str = ""


@dataclass
class Observation:
    """观察记录"""
    action: ActionType
    result: ToolResult
    iteration: int
    raw_output: str = ""


@dataclass
class AgentState:
    """Agent 状态"""
    question: str
    db_name: str
    iteration: int = 0
    thoughts: List[Thought] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    current_sql: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # 状态追踪
    known_tables: List[str] = field(default_factory=list)
    known_schemas: Dict[str, Any] = field(default_factory=dict)
    search_history: List[str] = field(default_factory=list)

    def add_thought(self, thought: Thought):
        """添加思考"""
        self.thoughts.append(thought)

    def add_observation(self, observation: Observation):
        """添加观察"""
        self.observations.append(observation)

    def update_context(self, key: str, value: Any):
        """更新上下文"""
        self.context[key] = value

    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        parts = []

        if self.known_tables:
            parts.append(f"已知表: {', '.join(self.known_tables)}")

        if self.known_schemas:
            parts.append(f"已获取结构的表: {list(self.known_schemas.keys())}")

        if self.current_sql:
            parts.append(f"当前 SQL: {self.current_sql[:100]}...")

        if self.search_history:
            parts.append(f"搜索历史: {self.search_history}")

        return "\n".join(parts) if parts else "尚未收集信息"


class ReActLoop:
    """ReAct 循环"""

    def __init__(
        self,
        llm_client,
        tool_registry: ToolRegistry,
        max_iterations: int = 10
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    async def run(
        self,
        question: str,
        db_name: str
    ) -> Dict[str, Any]:
        """
        运行 ReAct 循环

        Args:
            question: 用户问题
            db_name: 数据库名称

        Returns:
            包含 SQL 和推理过程的结果
        """
        state = AgentState(question=question, db_name=db_name)

        for iteration in range(self.max_iterations):
            state.iteration = iteration + 1
            logger.info(f"=== ReAct 迭代 {iteration + 1} ===")

            # 1. 思考：决定下一步动作
            thought = await self._think(state)
            state.add_thought(thought)

            logger.info(f"思考: {thought.content}")
            logger.info(f"动作: {thought.action.value if thought.action else 'None'}")

            # 2. 检查是否完成
            if thought.action == ActionType.FINAL_ANSWER:
                return self._build_result(state, success=True)

            # 3. 执行动作
            if thought.action is None:
                # 无法决定动作，尝试生成 SQL
                thought.action = ActionType.GENERATE_SQL

            observation = await self._act(thought, state)
            state.add_observation(observation)

            logger.info(f"观察: {observation.result.data if observation.result.success else observation.result.error}")

            # 4. 更新状态
            self._update_state(state, observation)

            # 5. 检查是否有 SQL 且已验证
            if state.current_sql and self._is_complete(state):
                return self._build_result(state, success=True)

        # 达到最大迭代次数
        return self._build_result(state, success=False, error="达到最大迭代次数")

    async def _think(self, state: AgentState) -> Thought:
        """思考：决定下一步动作"""
        prompt = self._build_decision_prompt(state)

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content
            return self._parse_thought(content, state.iteration)

        except Exception as e:
            logger.error(f"思考失败: {e}")
            # 回退到规则决策
            return self._rule_based_decision(state)

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        tool_descriptions = self.tool_registry.get_tool_descriptions()

        return f"""你是 SQL Agent 的决策模块。你的任务是根据当前状态决定下一步动作。

可用工具：
{tool_descriptions}

工作流程：
1. 首先列出数据库中的表 (list_tables)
2. 获取相关表的结构 (get_schema)
3. 如果需要，搜索特定字段 (search_columns)
4. 生成 SQL (generate_sql)
5. 验证 SQL (validate_sql)
6. 测试执行 (test_execute)
7. 完成后返回结果 (final_answer)

返回格式（JSON）：
{{
    "thought": "你的思考过程",
    "action": "动作名称",
    "reasoning": "选择该动作的原因",
    "params": {{"参数名": "参数值"}}
}}

注意：
- 一次只执行一个动作
- 参数值必须是字符串或简单的 JSON 对象
- 当 SQL 生成并验证成功后，使用 final_answer 动作
"""

    def _build_decision_prompt(self, state: AgentState) -> str:
        """构建决策 Prompt"""
        parts = [
            f"## 用户问题\n{state.question}",
            f"\n## 数据库\n{state.db_name}",
            f"\n## 当前迭代\n{state.iteration}/{self.max_iterations}",
        ]

        # 添加上下文摘要
        if state.get_context_summary():
            parts.append(f"\n## 当前状态\n{state.get_context_summary()}")

        # 添加最近的思考
        if state.thoughts:
            parts.append("\n## 最近的思考")
            for thought in state.thoughts[-3:]:
                parts.append(f"- {thought.content}")

        # 添加最近的观察
        if state.observations:
            parts.append("\n## 最近的观察")
            for obs in state.observations[-3:]:
                if obs.result.success:
                    parts.append(f"- {obs.action.value}: {obs.result.data}")
                else:
                    parts.append(f"- {obs.action.value}: 失败 - {obs.result.error}")

        parts.append("\n## 请决定下一步动作")

        return "\n".join(parts)

    def _parse_thought(self, content: str, iteration: int) -> Thought:
        """解析思考内容"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]*"thought"[^{}]*\}', content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)

            if json_match:
                data = json.loads(json_match.group(1))
                action_name = data.get("action", "")
                action = self._get_action(action_name)

                return Thought(
                    content=data.get("thought", content[:200]),
                    iteration=iteration,
                    action=action,
                    reasoning=data.get("reasoning", "")
                )
        except Exception as e:
            logger.warning(f"解析思考失败: {e}")

        # 回退：从内容中提取动作
        for action_type in ActionType:
            if action_type.value in content.lower():
                return Thought(
                    content=content[:200],
                    iteration=iteration,
                    action=action_type
                )

        # 默认：生成 SQL
        return Thought(
            content="决定生成 SQL",
            iteration=iteration,
            action=ActionType.GENERATE_SQL
        )

    def _get_action(self, action_name: str) -> Optional[ActionType]:
        """从名称获取动作类型"""
        for action_type in ActionType:
            if action_type.value == action_name.lower():
                return action_type
        return None

    def _rule_based_decision(self, state: AgentState) -> Thought:
        """基于规则的决策（回退方案）"""
        iteration = state.iteration

        if not state.known_tables:
            return Thought(
                content="需要先列出数据库中的表",
                iteration=iteration,
                action=ActionType.LIST_TABLES
            )

        if not state.known_schemas and state.known_tables:
            return Thought(
                content=f"需要获取表 {state.known_tables[0]} 的结构",
                iteration=iteration,
                action=ActionType.GET_SCHEMA
            )

        if not state.current_sql:
            return Thought(
                content="已收集足够信息，开始生成 SQL",
                iteration=iteration,
                action=ActionType.GENERATE_SQL
            )

        if iteration < 5:
            return Thought(
                content="需要验证和测试 SQL",
                iteration=iteration,
                action=ActionType.VALIDATE_SQL
            )

        return Thought(
            content="已完成所有步骤",
            iteration=iteration,
            action=ActionType.FINAL_ANSWER
        )

    async def _act(self, thought: Thought, state: AgentState) -> Observation:
        """执行动作"""
        action = thought.action

        # 准备参数
        params = self._prepare_params(action, state)

        # 获取工具
        tool = self.tool_registry.get(action.value)
        if not tool:
            return Observation(
                action=action,
                result=ToolResult(success=False, error=f"工具不存在: {action.value}"),
                iteration=state.iteration
            )

        # 执行工具
        try:
            result = await tool.execute(**params)
        except Exception as e:
            result = ToolResult(success=False, error=str(e))

        return Observation(
            action=action,
            result=result,
            iteration=state.iteration
        )

    def _prepare_params(self, action: ActionType, state: AgentState) -> Dict[str, Any]:
        """准备动作参数"""
        params = {}

        if action == ActionType.LIST_TABLES:
            params = {"db_name": state.db_name}

        elif action == ActionType.GET_SCHEMA:
            # 选择第一个未知结构的表
            table = state.known_tables[0] if state.known_tables else ""
            params = {"table_name": table, "db_name": state.db_name}

        elif action == ActionType.SEARCH_COLUMNS:
            # 从问题中提取关键词
            keywords = self._extract_keywords(state.question)
            keyword = keywords[0] if keywords else "id"
            params = {"keyword": keyword, "db_name": state.db_name}

        elif action == ActionType.FIND_JOIN_PATH:
            # 需要两个表
            tables = state.known_tables[:2]
            if len(tables) == 2:
                params = {
                    "table1": tables[0],
                    "table2": tables[1],
                    "db_name": state.db_name
                }

        elif action == ActionType.GENERATE_SQL:
            params = {
                "question": state.question,
                "context": {
                    "tables": state.known_tables,
                    "schemas": state.known_schemas,
                    "db_name": state.db_name
                }
            }

        elif action == ActionType.VALIDATE_SQL:
            params = {"sql": state.current_sql}

        elif action == ActionType.TEST_EXECUTE:
            params = {"sql": state.current_sql}

        return params

    def _extract_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        # 简化实现：移除常见词，返回剩余的词
        stop_words = {"的", "是", "在", "和", "或", "查询", "显示", "列出", "获取"}
        words = question.replace("?", "").replace("？", "").split()
        return [w for w in words if w not in stop_words and len(w) > 1]

    def _update_state(self, state: AgentState, observation: Observation):
        """根据观察更新状态"""
        action = observation.action
        result = observation.result

        if not result.success:
            return

        if action == ActionType.LIST_TABLES:
            tables = result.data.get("tables", [])
            state.known_tables = tables
            state.update_context("tables", tables)

        elif action == ActionType.GET_SCHEMA:
            table_name = result.data.get("table_name", "")
            state.known_schemas[table_name] = result.data
            state.update_context("schemas", state.known_schemas)

        elif action == ActionType.SEARCH_COLUMNS:
            keyword = result.data.get("keyword", "")
            if keyword not in state.search_history:
                state.search_history.append(keyword)

        elif action == ActionType.GENERATE_SQL:
            sql = result.data.get("sql", "")
            state.current_sql = sql
            state.update_context("generated_sql", sql)

    def _is_complete(self, state: AgentState) -> bool:
        """检查是否完成"""
        if not state.current_sql:
            return False

        # 检查是否有成功的验证
        for obs in state.observations:
            if (obs.action == ActionType.VALIDATE_SQL and
                obs.result.success and
                obs.result.data.get("valid")):
                return True

        # 检查是否有成功的测试执行
        for obs in state.observations:
            if (obs.action == ActionType.TEST_EXECUTE and
                obs.result.success and
                obs.result.data.get("executed")):
                return True

        return False

    def _build_result(
        self,
        state: AgentState,
        success: bool,
        error: str = ""
    ) -> Dict[str, Any]:
        """构建返回结果"""
        return {
            "success": success,
            "sql": state.current_sql,
            "question": state.question,
            "db_name": state.db_name,
            "iterations": state.iteration,
            "reasoning_trace": {
                "thoughts": [
                    {
                        "content": t.content,
                        "action": t.action.value if t.action else None,
                        "iteration": t.iteration
                    }
                    for t in state.thoughts
                ],
                "observations": [
                    {
                        "action": o.action.value,
                        "success": o.result.success,
                        "data": o.result.data if o.result.success else None,
                        "error": o.result.error if not o.result.success else None
                    }
                    for o in state.observations
                ],
                "context": state.context
            },
            "method": "agent_react",
            "error": error
        }


def create_react_loop(
    llm_client,
    tool_registry: ToolRegistry,
    max_iterations: int = 10
) -> ReActLoop:
    """创建 ReAct 循环"""
    return ReActLoop(llm_client, tool_registry, max_iterations)
