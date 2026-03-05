"""
RAG 方案 - SQL 生成模块

使用检索到的上下文生成 SQL 查询
"""
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 库未安装")


@dataclass
class GenerationContext:
    """生成上下文"""
    query: str
    database: str
    ddl_docs: List[Dict[str, Any]]
    sql_examples: List[Dict[str, Any]]
    business_docs: List[Dict[str, Any]]


@dataclass
class GenerationResult:
    """生成结果"""
    sql: str
    explanation: str
    confidence: float
    sources: List[Dict[str, Any]]
    tokens_used: int = 0
    error: Optional[str] = None


class SQLGenerator:
    """SQL 生成器"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        if OPENAI_AVAILABLE and api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"SQL生成器初始化成功: {model}")
        else:
            logger.warning("SQL生成器初始化失败：OpenAI不可用")

    def _build_prompt(
        self,
        context: GenerationContext,
        use_few_shot: bool = True
    ) -> str:
        """构建提示词"""
        prompt_parts = [
            "# 任务",
            "你是一个专业的 SQL 查询生成助手。根据用户的自然语言问题和提供的数据库结构，生成准确的 SQL 查询语句。",
            "",
            "# 用户问题",
            f"数据库: {context.database}",
            f"问题: {context.query}",
            "",
        ]

        # 添加 DDL 文档
        if context.ddl_docs:
            prompt_parts.append("# 数据库表结构")
            for i, doc in enumerate(context.ddl_docs, 1):
                if isinstance(doc, dict):
                    prompt_parts.append(f"## 表 {i}")
                    if "table_name" in doc:
                        prompt_parts.append(f"表名: {doc['table_name']}")
                    if "ddl" in doc:
                        prompt_parts.append(f"DDL: {doc['ddl']}")
                    if "description" in doc and doc["description"]:
                        prompt_parts.append(f"说明: {doc['description']}")
                    if "business_rules" in doc and doc["business_rules"]:
                        prompt_parts.append(f"业务规则: {'; '.join(doc['business_rules'])}")
                else:
                    prompt_parts.append(str(doc))
                prompt_parts.append("")

        # 添加业务文档
        if context.business_docs:
            prompt_parts.append("# 业务文档")
            for i, doc in enumerate(context.business_docs, 1):
                if isinstance(doc, dict):
                    prompt_parts.append(f"## 文档 {i}")
                    if "title" in doc:
                        prompt_parts.append(f"标题: {doc['title']}")
                    if "content" in doc:
                        prompt_parts.append(f"内容: {doc['content']}")
                else:
                    prompt_parts.append(str(doc))
                prompt_parts.append("")

        # 添加 SQL 示例（Few-shot Learning）
        if use_few_shot and context.sql_examples:
            prompt_parts.append("# 参考示例")
            for i, example in enumerate(context.sql_examples[:3], 1):
                if isinstance(example, dict):
                    if "question" in example and "sql" in example:
                        prompt_parts.append(f"## 示例 {i}")
                        prompt_parts.append(f"问题: {example['question']}")
                        prompt_parts.append(f"SQL: {example['sql']}")
                else:
                    prompt_parts.append(str(example))
                prompt_parts.append("")

        # 添加输出要求
        prompt_parts.extend([
            "# 输出要求",
            "1. 只输出 SQL 查询语句，不要有任何解释",
            "2. 使用正确的 SQL 语法",
            "3. 只执行 SELECT 查询，不要生成 UPDATE/DELETE/INSERT/DROP 等操作",
            "4. 如果问题无法用 SQL 回答，输出 -- 无法生成查询",
            "",
            "# 输出"
        ])

        return "\n".join(prompt_parts)

    def _build_cot_prompt(self, context: GenerationContext) -> str:
        """构建思维链提示词"""
        prompt_parts = [
            "# 任务",
            "你是一个专业的 SQL 查询生成助手。请使用思维链方法，逐步分析问题并生成 SQL。",
            "",
            "# 用户问题",
            f"数据库: {context.database}",
            f"问题: {context.query}",
            "",
        ]

        # 添加上下文
        if context.ddl_docs:
            prompt_parts.append("# 数据库表结构")
            for doc in context.ddl_docs[:3]:
                if isinstance(doc, dict):
                    prompt_parts.append(f"表: {doc.get('table_name', 'unknown')}")
                    prompt_parts.append(f"DDL: {doc.get('ddl', '')}")
                prompt_parts.append("")

        if context.sql_examples:
            prompt_parts.append("# 参考示例")
            for example in context.sql_examples[:2]:
                if isinstance(example, dict):
                    prompt_parts.append(f"Q: {example.get('question', '')}")
                    prompt_parts.append(f"A: {example.get('sql', '')}")
                prompt_parts.append("")

        prompt_parts.extend([
            "# 思维过程",
            "请按以下步骤思考：",
            "1. 理解用户问题的意图",
            "2. 识别涉及的数据表和字段",
            "3. 确定查询类型（简单查询、连接查询、聚合查询等）",
            "4. 构建 WHERE 条件",
            "5. 添加必要的排序和限制",
            "",
            "请先输出你的思考过程，然后输出最终的 SQL。",
            ""
        ])

        return "\n".join(prompt_parts)

    async def generate(
        self,
        context: GenerationContext,
        method: str = "direct",
        use_cot: bool = False
    ) -> GenerationResult:
        """生成 SQL"""
        if not self.client:
            return GenerationResult(
                sql="",
                explanation="",
                confidence=0.0,
                sources=[],
                error="生成器未初始化"
            )

        try:
            # 构建提示词
            if use_cot:
                prompt = self._build_cot_prompt(context)
            else:
                prompt = self._build_prompt(context, use_few_shot=True)

            # 调用 LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的 SQL 查询生成助手。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 解析结果
            content = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            # 提取 SQL 和解释
            sql, explanation = self._parse_response(content, use_cot)

            # 计算置信度（基于 tokens 和响应长度）
            confidence = self._calculate_confidence(content, tokens_used)

            # 收集来源
            sources = self._collect_sources(context)

            logger.info(f"SQL 生成完成，使用 tokens: {tokens_used}")

            return GenerationResult(
                sql=sql,
                explanation=explanation,
                confidence=confidence,
                sources=sources,
                tokens_used=tokens_used
            )

        except Exception as e:
            logger.error(f"SQL 生成失败: {e}")
            return GenerationResult(
                sql="",
                explanation="",
                confidence=0.0,
                sources=[],
                error=str(e)
            )

    def _parse_response(self, content: str, use_cot: bool) -> Tuple[str, str]:
        """解析响应内容"""
        if use_cot:
            # 思维链模式：先思考后输出
            parts = content.split("SQL:", 1)
            if len(parts) == 2:
                explanation = parts[0].strip()
                sql = parts[1].strip()
                # 移除 markdown 代码块
                sql = sql.replace("```sql", "").replace("```", "").strip()
                return sql, explanation
            else:
                # 尝试其他分隔符
                lines = content.split("\n")
                sql_lines = []
                explanation_lines = []
                in_sql = False

                for line in lines:
                    if "SELECT" in line.upper() or "select" in line:
                        in_sql = True
                    if in_sql:
                        sql_lines.append(line)
                    else:
                        explanation_lines.append(line)

                if sql_lines:
                    sql = "\n".join(sql_lines).replace("```sql", "").replace("```", "").strip()
                    explanation = "\n".join(explanation_lines).strip()
                    return sql, explanation

        # 直接模式
        content = content.replace("```sql", "").replace("```", "").strip()

        # 检查是否是无效响应
        if content.startswith("--") or "无法" in content:
            return "", "无法生成有效的查询"

        return content, ""

    def _calculate_confidence(self, content: str, tokens_used: int) -> float:
        """计算置信度"""
        # 基础置信度
        confidence = 0.5

        # 检查 SQL 关键词
        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "HAVING"]
        content_upper = content.upper()
        keyword_count = sum(1 for kw in sql_keywords if kw in content_upper)
        confidence += min(keyword_count * 0.1, 0.3)

        # 检查语法完整性
        if content_upper.count("SELECT") == content_upper.count("FROM"):
            confidence += 0.1

        # 检查长度
        if 50 < len(content) < 500:
            confidence += 0.1

        return min(confidence, 1.0)

    def _collect_sources(self, context: GenerationContext) -> List[Dict[str, Any]]:
        """收集来源信息"""
        sources = []

        for doc in context.ddl_docs:
            if isinstance(doc, dict):
                sources.append({
                    "type": "ddl",
                    "table": doc.get("table_name", ""),
                    "description": doc.get("description", "")[:100]
                })

        for example in context.sql_examples:
            if isinstance(example, dict):
                sources.append({
                    "type": "sql_example",
                    "question": example.get("question", "")[:100],
                    "similarity": "high"
                })

        return sources

    async def generate_with_validation(
        self,
        context: GenerationContext,
        validator=None
    ) -> GenerationResult:
        """生成并验证 SQL"""
        # 首次生成
        result = await self.generate(context)

        # 如果有验证器且 SQL 有效
        if validator and result.sql:
            validation = await validator.validate(result.sql, context.database)

            if not validation.is_valid:
                # 尝试修复
                logger.info(f"SQL 验证失败，尝试修复: {validation.error}")
                fixed_context = self._build_fix_prompt(context, result.sql, validation)
                result = await self.generate(fixed_context)

        return result

    def _build_fix_prompt(
        self,
        original_context: GenerationContext,
        failed_sql: str,
        validation
    ) -> GenerationContext:
        """构建修复提示"""
        # 创建新的上下文，包含错误信息
        fix_context = GenerationContext(
            query=f"{original_context.query}\n\n注意：之前的尝试失败，错误：{validation.error}",
            database=original_context.database,
            ddl_docs=original_context.ddl_docs,
            sql_examples=original_context.sql_examples,
            business_docs=original_context.business_docs
        )
        return fix_context


class StreamSQLGenerator(SQLGenerator):
    """流式 SQL 生成器"""

    async def generate_stream(
        self,
        context: GenerationContext,
        callback=None
    ):
        """流式生成 SQL"""
        if not self.client:
            yield {"error": "生成器未初始化"}
            return

        try:
            prompt = self._build_prompt(context, use_few_shot=True)

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的 SQL 查询生成助手。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if callback:
                        await callback(content)
                    yield {"content": content}

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield {"error": str(e)}


def create_sql_generator(
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.1
) -> SQLGenerator:
    """创建 SQL 生成器实例"""
    return SQLGenerator(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature
    )
