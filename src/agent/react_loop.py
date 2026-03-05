"""
Agent 方案 - ReAct 循环实现

实现推理-行动循环，让 Agent 能够逐步思考并调用工具
"""
import json
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime
from enum import Enum

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ThoughtType(Enum):
    """思考类型"""
    OBSERVATION = "observation"  # 观察
    THOUGHT = "thought"  # 思考
    ACTION = "action"  # 行动
    ANSWER = "answer"  # 最终答案


@dataclass
class Step:
    """执行步骤"""
    step_type: ThoughtType
    content: str
    tool_name: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.now().timestamp()


@dataclass
class ReActResult:
    """ReAct 执行结果"""
    sql: str
    explanation: str
    steps: List[Step]
    success: bool
    error: Optional[str] = None
    total_tokens: int = 0


class Tool:
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, **kwargs) -> str:
        """执行工具"""
        raise NotImplementedError

    def get_schema(self) -> Dict[str, Any]:
        """获取工具模式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {}
        }


class ReActLoop:
    """ReAct 循环"""

    def __init__(
        self,
        llm_client: Any,
        tools: List[Tool],
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.steps: List[Step] = []

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        return f"""你是一个专业的 SQL 查询生成助手。使用 ReAct 方法（推理-行动）来回答用户的数据库查询问题。

# 可用工具
{tool_descriptions}

# 回答格式
你必须按照以下格式思考：

思考：[你的思考过程]
行动：[工具名称](工具参数)

或者当你有足够信息时：

思考：[你的最终思考]
答案：[生成的 SQL 查询]

# 注意事项
1. 每次"思考"后必须跟随"行动"或"答案"
2. 只使用上述列出的工具
3. 行动参数必须是有效的 JSON 格式
4. 在给出答案前，确保你已经收集了足够的信息
5. 最终答案只输出 SQL 语句，不需要其他解释

# 示例
用户：查询销售额前10的产品

思考：我需要先了解数据库中有哪些表，以及它们的结构
行动：list_tables[]

观察：数据库包含 products 表和 sales 表

思考：我需要查看这两个表的字段，以便正确关联
行动：describe_table[{{"table_name": "products"}}]

观察：products 表包含 id, name, price 等字段

思考：现在我需要查看 sales 表的结构
行动：describe_table[{{"table_name": "sales"}}]

观察：sales 表包含 product_id, quantity, amount 等字段

思考：我了解了表结构，可以通过 JOIN sales 和 products 表，按产品分组计算销售额
答案：SELECT p.name, SUM(s.amount) as total_sales FROM products p JOIN sales s ON p.id = s.product_id GROUP BY p.id ORDER BY total_sales DESC LIMIT 10
"""

    async def _parse_response(self, response: str) -> Tuple[ThoughtType, str, Optional[str], Dict[str, Any]]:
        """解析 LLM 响应"""
        response = response.strip()

        # 匹配思考
        thought_match = re.search(r'思考[：:]\s*(.+?)(?=\n(?:行动|答案|思考)[：:])', response, re.DOTALL)
        if not thought_match:
            thought_match = re.search(r'思考[：:]\s*(.+)', response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""

        # 匹配答案
        answer_match = re.search(r'答案[：:]\s*(.+)', response, re.DOTALL)
        if answer_match:
            return ThoughtType.ANSWER, thought, None, {}, answer_match.group(1).strip()

        # 匹配行动
        action_match = re.search(r'行动[：:]\s*(\w+)\[(.+?)\]', response, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            try:
                tool_input = json.loads(action_match.group(2))
                return ThoughtType.ACTION, thought, tool_name, tool_input, ""
            except json.JSONDecodeError:
                logger.warning(f"无法解析工具参数: {action_match.group(2)}")
                return ThoughtType.ACTION, thought, tool_name, {}, ""

        return ThoughtType.THOUGHT, thought, None, {}, ""

    async def run(self, question: str, context: Dict[str, Any] = None) -> ReActResult:
        """运行 ReAct 循环"""
        context = context or {}
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"用户问题：{question}"}
        ]

        total_tokens = 0
        last_observation = ""

        for iteration in range(self.max_iterations):
            # 调用 LLM
            try:
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )

                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                total_tokens += tokens_used

                if self.verbose:
                    logger.info(f"\n=== 迭代 {iteration + 1} ===")
                    logger.info(f"LLM 响应:\n{content}")

                # 解析响应
                thought_type, thought, tool_name, tool_input, answer = await self._parse_response(content)

                # 记录思考步骤
                self.steps.append(Step(
                    step_type=ThoughtType.THOUGHT,
                    content=thought
                ))

                # 处理答案
                if thought_type == ThoughtType.ANSWER:
                    self.steps.append(Step(
                        step_type=ThoughtType.ANSWER,
                        content=answer
                    ))

                    # 提取 SQL
                    sql = self._extract_sql(answer)
                    explanation = self._build_explanation()

                    return ReActResult(
                        sql=sql,
                        explanation=explanation,
                        steps=self.steps,
                        success=True,
                        total_tokens=total_tokens
                    )

                # 处理行动
                elif thought_type == ThoughtType.ACTION:
                    if tool_name not in self.tools:
                        observation = f"错误：工具 '{tool_name}' 不存在"
                    else:
                        # 记录行动步骤
                        self.steps.append(Step(
                            step_type=ThoughtType.ACTION,
                            content=f"调用工具: {tool_name}",
                            tool_name=tool_name,
                            tool_input=tool_input
                        ))

                        # 执行工具
                        try:
                            tool = self.tools[tool_name]
                            observation = await tool.execute(**tool_input)

                            self.steps[-1].tool_output = observation[:500]  # 限制输出长度

                        except Exception as e:
                            observation = f"工具执行错误: {str(e)}"
                            logger.error(f"工具 {tool_name} 执行失败: {e}")

                    # 记录观察
                    self.steps.append(Step(
                        step_type=ThoughtType.OBSERVATION,
                        content=observation
                    ))

                    if self.verbose:
                        logger.info(f"观察：{observation[:200]}")

                    # 添加到对话历史
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"观察：{observation}\n\n请继续思考。"
                    })

                else:
                    # 只有思考，没有行动
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "请继续思考并采取行动或给出答案。"
                    })

            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                return ReActResult(
                    sql="",
                    explanation="",
                    steps=self.steps,
                    success=False,
                    error=str(e),
                    total_tokens=total_tokens
                )

        # 达到最大迭代次数
        explanation = self._build_explanation()
        return ReActResult(
            sql="",
            explanation=explanation,
            steps=self.steps,
            success=False,
            error="达到最大迭代次数，未能生成有效的 SQL",
            total_tokens=total_tokens
        )

    def _extract_sql(self, text: str) -> str:
        """从文本中提取 SQL"""
        # 移除 markdown 代码块
        text = text.replace("```sql", "").replace("```", "").strip()

        # 检查 SQL 关键词
        if "SELECT" in text.upper():
            return text
        return ""

    def _build_explanation(self) -> str:
        """构建解释"""
        explanation_parts = []
        for step in self.steps:
            if step.step_type == ThoughtType.THOUGHT:
                explanation_parts.append(f"- {step.content}")

        return "\n".join(explanation_parts)


class SelfCorrectionReActLoop(ReActLoop):
    """支持自我纠错的 ReAct 循环"""

    def __init__(self, *args, validator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.validator = validator

    async def run(self, question: str, context: Dict[str, Any] = None) -> ReActResult:
        """运行带自我纠错的 ReAct 循环"""
        result = await super().run(question, context)

        # 如果有验证器且生成了 SQL
        if self.validator and result.sql:
            validation = await self.validator.validate(result.sql, context.get("database", ""))

            if not validation.is_valid:
                logger.info(f"SQL 验证失败: {validation.error}")

                # 添加验证反馈到步骤
                self.steps.append(Step(
                    step_type=ThoughtType.OBSERVATION,
                    content=f"SQL 验证失败: {validation.error}\n请根据错误信息修复 SQL。"
                ))

                # 尝试修复
                if self._attempt_correction(result, validation):
                    # 重新验证
                    validation = await self.validator.validate(result.sql, context.get("database", ""))
                    result.success = validation.is_valid
                    if not validation.is_valid:
                        result.error = validation.error

        return result

    def _attempt_correction(self, result: ReActResult, validation) -> bool:
        """尝试修正 SQL"""
        # 简单的修正策略：在步骤中添加错误信息
        # 更复杂的实现可以使用 LLM 重新生成
        error_msg = validation.error if hasattr(validation, 'error') else "验证失败"
        result.error = error_msg
        return False


def create_react_loop(llm_client, tools: List[Tool], max_iterations: int = 10) -> ReActLoop:
    """创建 ReAct 循环实例"""
    return ReActLoop(llm_client, tools, max_iterations)


def create_self_correcting_react_loop(llm_client, tools: List[Tool], validator=None) -> SelfCorrectionReActLoop:
    """创建支持自我纠错的 ReAct 循环"""
    return SelfCorrectionReActLoop(llm_client, tools, validator=validator)
