"""
方案二：基于 Agent 的 Text-to-SQL 方案（LangChain SQL Agent 风格）

核心理念：
1. LLM 作为推理引擎，动态调用工具
2. ReAct 循环：Thought → Action → Observation → Thought
3. 自我纠错能力强，适合复杂查询

适用场景：
- Schema 较大或未知
- 复杂、多步查询
- 需要动态探索数据库

论文参考：
- ReAct: Synergizing Reasoning and Acting in Language Models (ICLR 2023)
- DB-Surfer: A Decoupled Schema Text-to-SQL Framework with Multi-Agent Collaboration
- Agent-based approaches for Text-to-SQL (2024-2025)
"""

from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import json
import re
from loguru import logger
from dataclasses import dataclass, field

from app.config import get_settings


class AgentAction(Enum):
    """Agent 动作类型"""
    LIST_TABLES = "list_tables"
    GET_SCHEMA = "get_schema"
    SEARCH_COLUMNS = "search_columns"
    FIND_JOIN_PATH = "find_join_path"
    GENERATE_SQL = "generate_sql"
    VALIDATE_SQL = "validate_sql"
    TEST_EXECUTE = "test_execute"
    FINAL_ANSWER = "final_answer"


@dataclass
class AgentObservation:
    """Agent 观察结果"""
    action: AgentAction
    result: Any
    thought: str = ""
    error: str = ""


@dataclass
class AgentTool:
    """Agent 工具定义"""
    name: str
    description: str
    function: Callable
    action_type: AgentAction


@dataclass
class AgentState:
    """Agent 状态"""
    question: str
    thoughts: List[str] = field(default_factory=list)
    observations: List[AgentObservation] = field(default_factory=list)
    current_sql: str = ""
    relevant_tables: List[str] = field(default_factory=list)
    schema_context: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 0

    def add_thought(self, thought: str):
        """添加思考"""
        self.thoughts.append(thought)

    def add_observation(self, observation: AgentObservation):
        """添加观察"""
        self.observations.append(observation)


class SQLAgent:
    """
    SQL Agent - 使用 ReAct 模式动态探索 Schema 并生成 SQL

    工作流程：
    1. 分析问题
    2. 决定下一步动作
    3. 执行动作获得观察
    4. 基于观察更新思考
    5. 重复直到可以生成 SQL
    """

    def __init__(self, db_manager=None, llm_client=None):
        self.settings = get_settings()
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.tools: Dict[str, AgentTool] = {}
        self._register_tools()

    def _register_tools(self):
        """注册可用工具"""
        self.tools = {
            "list_tables": AgentTool(
                name="list_tables",
                description="列出数据库中的所有表",
                function=self._list_tables,
                action_type=AgentAction.LIST_TABLES
            ),
            "get_schema": AgentTool(
                name="get_schema",
                description="获取指定表的详细结构，包括字段名、类型、注释",
                function=self._get_schema,
                action_type=AgentAction.GET_SCHEMA
            ),
            "search_columns": AgentTool(
                name="search_columns",
                description="搜索包含特定关键词的列，用于查找相关字段",
                function=self._search_columns,
                action_type=AgentAction.SEARCH_COLUMNS
            ),
            "find_join_path": AgentTool(
                name="find_join_path",
                description="找到两个表之间的关联路径（外键关系）",
                function=self._find_join_path,
                action_type=AgentAction.FIND_JOIN_PATH
            ),
            "generate_sql": AgentTool(
                name="generate_sql",
                description="根据已收集的信息生成 SQL 查询",
                function=self._generate_sql,
                action_type=AgentAction.GENERATE_SQL
            ),
            "validate_sql": AgentTool(
                name="validate_sql",
                description="验证 SQL 的语法和正确性",
                function=self._validate_sql,
                action_type=AgentAction.VALIDATE_SQL
            ),
            "test_execute": AgentTool(
                name="test_execute",
                description="测试执行 SQL（使用 LIMIT 1），检查是否能正确执行",
                function=self._test_execute,
                action_type=AgentAction.TEST_EXECUTE
            ),
        }

    # ==================== 工具实现 ====================

    async def _list_tables(self, db_name: str) -> str:
        """列出所有表"""
        if self.db_manager:
            schemas = await self.db_manager.get_table_schemas(db_name)
            tables = list(schemas.keys())
            return f"数据库 {db_name} 中有以下表: {', '.join(tables)}"
        return "数据库管理器未初始化"

    async def _get_schema(self, table_name: str, db_name: str) -> str:
        """获取表结构"""
        if self.db_manager:
            schemas = await self.db_manager.get_table_schemas(db_name)
            if table_name in schemas:
                schema = schemas[table_name]
                cols = []
                for col in schema.columns:
                    col_desc = f"  - {col.name}: {col.type}"
                    if col.comment:
                        col_desc += f" ({col.comment})"
                    cols.append(col_desc)
                return f"表 {table_name} 的结构:\n" + "\n".join(cols)
        return f"未找到表 {table_name}"

    async def _search_columns(self, keyword: str, db_name: str) -> str:
        """搜索列"""
        if self.db_manager:
            schemas = await self.db_manager.get_table_schemas(db_name)
            results = []
            for table_name, schema in schemas.items():
                for col in schema.columns:
                    if keyword.lower() in col.name.lower() or \
                       (col.comment and keyword.lower() in col.comment.lower()):
                        results.append(f"{table_name}.{col.name}")
            if results:
                return f"包含 '{keyword}' 的列:\n" + "\n".join(f"  - {r}" for r in results)
        return f"未找到包含 '{keyword}' 的列"

    async def _find_join_path(
        self,
        table1: str,
        table2: str,
        db_name: str
    ) -> str:
        """查找关联路径"""
        # 简化实现：实际需要分析外键关系
        return f"表 {table1} 和 {table2} 可以通过相关字段关联（具体关系请查看表结构）"

    async def _generate_sql(
        self,
        question: str,
        state: AgentState
    ) -> str:
        """生成 SQL"""
        prompt = self._build_sql_prompt(question, state)

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": "你是 SQL 专家。根据收集的 Schema 信息生成准确的 SQL 查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            content = response.choices[0].message.content
            sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            if sql_match:
                return sql_match.group(1).strip()
            return content

        except Exception as e:
            return f"生成失败: {str(e)}"

    async def _validate_sql(self, sql: str) -> str:
        """验证 SQL"""
        # 简化验证
        if not sql.strip():
            return "SQL 为空"
        if "SELECT" not in sql.upper():
            return "SQL 不是 SELECT 语句"
        return "SQL 语法检查通过"

    async def _test_execute(self, sql: str) -> str:
        """测试执行"""
        if self.db_manager:
            try:
                test_sql = f"{sql} LIMIT 1" if "LIMIT" not in sql.upper() else sql
                results = await self.db_manager.execute_sql(test_sql)
                return f"测试执行成功，返回 {len(results)} 条记录（LIMIT 1）"
            except Exception as e:
                return f"测试执行失败: {str(e)}"
        return "数据库管理器未初始化"

    def _build_sql_prompt(self, question: str, state: AgentState) -> str:
        """构建 SQL 生成 Prompt"""
        parts = [
            f"用户问题: {question}",
            "",
            "已收集的 Schema 信息:"
        ]

        if state.schema_context:
            for table_name, schema_info in state.schema_context.items():
                parts.append(f"\n{schema_info}")

        if state.relevant_tables:
            parts.append(f"\n相关表: {', '.join(state.relevant_tables)}")

        parts.extend([
            "",
            "请根据上述信息生成 SQL 查询。",
            "只返回 SQL，不要解释。"
        ])

        return "\n".join(parts)

    # ==================== ReAct 循环 ====================

    async def run(
        self,
        question: str,
        db_name: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        运行 Agent

        Args:
            question: 用户问题
            db_name: 数据库名称
            max_iterations: 最大迭代次数

        Returns:
            包含 SQL 和推理过程的结果
        """
        state = AgentState(question=question)

        for iteration in range(max_iterations):
            state.iterations = iteration + 1

            # 1. 思考：决定下一步动作
            action = await self._decide_action(state, db_name)
            logger.info(f"Iteration {iteration + 1}: {action.action.value}")

            # 2. 检查是否完成
            if action.action == AgentAction.FINAL_ANSWER:
                return self._build_result(state, success=True)

            # 3. 执行动作
            observation = await self._execute_action(action, state, db_name)
            state.add_observation(observation)

            # 4. 更新状态
            self._update_state(state, action, observation)

            # 5. 检查是否出错
            if observation.error:
                logger.warning(f"执行出错: {observation.error}")

        # 达到最大迭代次数
        return self._build_result(state, success=False, error="达到最大迭代次数")

    async def _decide_action(
        self,
        state: AgentState,
        db_name: str
    ) -> AgentAction:
        """决定下一步动作"""
        # 构建决策 Prompt
        prompt = self._build_decision_prompt(state, db_name)

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": """你是 SQL Agent 的决策模块。根据当前状态和用户问题，决定下一步动作。

可用动作：
- list_tables: 列出所有表
- get_schema: 获取表结构
- search_columns: 搜索列
- find_join_path: 查找关联路径
- generate_sql: 生成 SQL（在收集足够信息后）
- validate_sql: 验证 SQL
- test_execute: 测试执行 SQL
- final_answer: 完成任务（已有有效 SQL）

返回 JSON 格式：{"action": "动作名", "thought": "思考过程", "params": {参数}}
"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content
            return self._parse_decision(content)

        except Exception as e:
            logger.error(f"决策失败: {e}")
            # 回退到基于规则的决策
            return self._rule_based_decision(state)

    def _build_decision_prompt(self, state: AgentState, db_name: str) -> str:
        """构建决策 Prompt"""
        parts = [
            f"用户问题: {state.question}",
            f"当前迭代: {state.iterations}",
            ""
        ]

        if state.thoughts:
            parts.append("之前的思考:")
            for i, thought in enumerate(state.thoughts[-3:], 1):
                parts.append(f"  {i}. {thought}")

        if state.relevant_tables:
            parts.append(f"\n已知相关表: {', '.join(state.relevant_tables)}")

        if state.current_sql:
            parts.append(f"\n当前 SQL: {state.current_sql}")

        if state.observations:
            parts.append("\n最近的观察:")
            for obs in state.observations[-3:]:
                parts.append(f"  - {obs.action.value}: {obs.result}")

        parts.append("\n请决定下一步动作：")

        return "\n".join(parts)

    def _parse_decision(self, content: str) -> AgentAction:
        """解析决策结果"""
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                action_name = data.get("action", "")
                for action_enum in AgentAction:
                    if action_enum.value == action_name:
                        return action_enum
            except json.JSONDecodeError:
                pass

        # 回退到 generate_sql
        return AgentAction.GENERATE_SQL

    def _rule_based_decision(self, state: AgentState) -> AgentAction:
        """基于规则的决策（回退方案）"""
        if not state.relevant_tables:
            return AgentAction.LIST_TABLES
        elif not state.schema_context:
            return AgentAction.GET_SCHEMA
        elif not state.current_sql:
            return AgentAction.GENERATE_SQL
        elif state.iterations < 3:
            return AgentAction.TEST_EXECUTE
        else:
            return AgentAction.FINAL_ANSWER

    async def _execute_action(
        self,
        decision: AgentAction,
        state: AgentState,
        db_name: str
    ) -> AgentObservation:
        """执行动作"""
        try:
            tool = self.tools.get(decision.value)
            if not tool:
                return AgentObservation(
                    action=decision,
                    result="",
                    error=f"未找到工具: {decision.value}"
                )

            # 执行工具函数
            if decision == AgentAction.LIST_TABLES:
                result = await tool.function(db_name)
            elif decision == AgentAction.GET_SCHEMA:
                # 需要从状态中获取表名
                table = state.relevant_tables[0] if state.relevant_tables else ""
                result = await tool.function(table, db_name)
            elif decision == AgentAction.GENERATE_SQL:
                result = await tool.function(state.question, state)
            elif decision == AgentAction.VALIDATE_SQL:
                result = await tool.function(state.current_sql)
            elif decision == AgentAction.TEST_EXECUTE:
                result = await tool.function(state.current_sql)
            else:
                result = await tool.function(db_name)

            return AgentObservation(
                action=decision,
                result=result
            )

        except Exception as e:
            return AgentObservation(
                action=decision,
                result="",
                error=str(e)
            )

    def _update_state(
        self,
        state: AgentState,
        action: AgentAction,
        observation: AgentObservation
    ):
        """根据观察更新状态"""
        if action == AgentAction.LIST_TABLES:
            # 从结果中提取表名
            if "有以下表:" in observation.result:
                tables_str = observation.result.split("有以下表:")[1].strip()
                state.relevant_tables = [t.strip() for t in tables_str.split(",")]
            state.add_thought(f"列出了数据库中的表")

        elif action == AgentAction.GET_SCHEMA:
            # 保存 Schema 信息
            if state.relevant_tables:
                table_name = state.relevant_tables[0]
                state.schema_context[table_name] = observation.result
            state.add_thought(f"获取了表结构信息")

        elif action == AgentAction.GENERATE_SQL:
            state.current_sql = observation.result
            state.add_thought(f"生成了 SQL: {observation.result[:100]}...")

        elif action == AgentAction.TEST_EXECUTE:
            if "成功" in observation.result:
                state.add_thought("SQL 测试执行成功，可以返回结果")
            else:
                state.add_thought(f"SQL 测试失败: {observation.result}")

        state.add_thought(observation.thought or f"执行了 {action.value}")

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
            "iterations": state.iterations,
            "reasoning_trace": {
                "thoughts": state.thoughts,
                "relevant_tables": state.relevant_tables,
                "schema_context": state.schema_context
            },
            "error": error,
            "method": "agent_based"
        }


# ==================== 多 Agent 协作系统 ====================

class MultiAgentOrchestrator:
    """
    多 Agent 协调器

    将复杂任务分解给专门的 Agent：
    - SchemaExplorerAgent: 探索数据库 Schema
    - SQLGeneratorAgent: 生成 SQL
    - ValidatorAgent: 验证和修复 SQL
    """

    def __init__(self, db_manager=None, llm_client=None):
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.agents = {}
        self._init_agents()

    def _init_agents(self):
        """初始化子 Agent"""
        self.agents = {
            "schema_explorer": SchemaExplorerAgent(self.db_manager, self.llm_client),
            "sql_generator": SQLGeneratorAgent(self.llm_client),
            "validator": ValidatorAgent(self.db_manager, self.llm_client)
        }

    async def process(
        self,
        question: str,
        db_name: str
    ) -> Dict[str, Any]:
        """
        处理查询

        流程：Schema Explorer → SQL Generator → Validator
        """
        result = {
            "question": question,
            "steps": []
        }

        # Step 1: Schema 探索
        logger.info("Step 1: Schema 探索")
        schema_result = await self.agents["schema_explorer"].explore(question, db_name)
        result["steps"].append({
            "agent": "schema_explorer",
            "result": schema_result
        })

        # Step 2: SQL 生成
        logger.info("Step 2: SQL 生成")
        sql_result = await self.agents["sql_generator"].generate(
            question,
            schema_result["context"]
        )
        result["steps"].append({
            "agent": "sql_generator",
            "result": sql_result
        })

        sql = sql_result.get("sql", "")

        # Step 3: 验证和修复
        logger.info("Step 3: 验证和修复")
        validation_result = await self.agents["validator"].validate_and_fix(
            sql,
            schema_result["context"]
        )
        result["steps"].append({
            "agent": "validator",
            "result": validation_result
        })

        result["success"] = validation_result.get("is_valid", False)
        result["sql"] = validation_result.get("sql", sql)

        return result


class SchemaExplorerAgent:
    """Schema 探索 Agent"""

    def __init__(self, db_manager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client

    async def explore(self, question: str, db_name: str) -> Dict[str, Any]:
        """探索 Schema，找出相关表和字段"""
        # 简化实现
        schemas = await self.db_manager.get_table_schemas(db_name)

        # 使用 LLM 分析问题找出相关表
        context = {
            "tables": list(schemas.keys()),
            "schemas": {}
        }

        # 简单关键词匹配
        question_lower = question.lower()
        for table_name, schema in schemas.items():
            if table_name.lower() in question_lower:
                context["schemas"][table_name] = str(schema)

        return {"context": context}


class SQLGeneratorAgent:
    """SQL 生成 Agent"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def generate(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成 SQL"""
        # 简化实现
        return {"sql": "SELECT * FROM table LIMIT 10"}


class ValidatorAgent:
    """验证和修复 Agent"""

    def __init__(self, db_manager, llm_client):
        self.db_manager = db_manager
        self.llm_client = llm_client

    async def validate_and_fix(
        self,
        sql: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证和修复 SQL"""
        # 简化实现
        return {
            "is_valid": bool(sql and "SELECT" in sql.upper()),
            "sql": sql,
            "errors": []
        }


# ==================== 示例用法 ====================

async def example_usage():
    """示例用法"""
    from openai import AsyncOpenAI

    settings = get_settings()
    llm_client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url
    )

    # 使用单 Agent
    agent = SQLAgent(llm_client=llm_client)

    result = await agent.run(
        question="查询销售额最高的前10个产品",
        db_name="sales_db",
        max_iterations=5
    )

    print(f"生成的 SQL: {result['sql']}")
    print(f"推理过程: {result['reasoning_trace']}")

    # 使用多 Agent
    orchestrator = MultiAgentOrchestrator(llm_client=llm_client)

    result = await orchestrator.process(
        question="查询每个地区的季度销售额趋势",
        db_name="sales_db"
    )

    print(f"多 Agent 结果: {result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
