"""
RAG 方案 - SQL 生成模块

使用检索增强生成 SQL
"""
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..knowledge.base import DDLDocument, SQLExample
from ..retrieval.vector_store import HybridRetriever


@dataclass
class GenerationResult:
    """生成结果"""
    success: bool
    sql: str
    explanation: str
    confidence: float
    context_used: Dict[str, Any]
    error: str = ""


class RAGSQLGenerator:
    """RAG SQL 生成器"""

    def __init__(
        self,
        llm_client,
        retriever: HybridRetriever,
        system_prompt: str = None
    ):
        self.llm_client = llm_client
        self.retriever = retriever
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个专业的 SQL 生成助手。你的任务是将自然语言问题转换为准确的 SQL 查询。

请遵循以下规则：
1. 只生成 SELECT 查询，不要生成 INSERT/UPDATE/DELETE
2. 使用检索到的相关 DDL 和 SQL 示例作为参考
3. 如果不确定表名或字段名，优先参考检索到的 DDL
4. 如果有相似的 SQL 示例，参考其结构但根据当前问题调整
5. 返回 JSON 格式：{"sql": "你的SQL", "explanation": "解释", "confidence": 0.0-1.0}
"""

    async def generate(
        self,
        question: str,
        db_name: str = "",
        schemas: Optional[List[Any]] = None,
        top_k: int = 3
    ) -> GenerationResult:
        """
        使用 RAG 生成 SQL

        Args:
            question: 用户问题
            db_name: 数据库名称
            schemas: 可选的表结构信息
            top_k: 检索数量

        Returns:
            生成结果
        """
        try:
            # 1. 检索相关上下文
            context = await self.retriever.retrieve(
                question,
                top_k=top_k,
                database=db_name
            )

            # 2. 构建 Prompt
            prompt = self._build_prompt(question, context, schemas)

            # 3. 调用 LLM
            response = await self._call_llm(prompt)

            # 4. 解析响应
            return self._parse_response(response, context)

        except Exception as e:
            logger.error(f"SQL 生成失败: {e}")
            return GenerationResult(
                success=False,
                sql="",
                explanation="",
                confidence=0.0,
                context_used={},
                error=str(e)
            )

    async def generate_with_feedback(
        self,
        question: str,
        db_name: str,
        feedback_sql: str,
        feedback_score: float,
        top_k: int = 3
    ) -> GenerationResult:
        """
        使用反馈重新生成 SQL

        当用户对结果不满意时，根据反馈重新生成
        """
        try:
            context = await self.retriever.retrieve(
                question,
                top_k=top_k,
                database=db_name
            )

            # 添加反馈信息
            prompt = self._build_prompt_with_feedback(
                question,
                context,
                feedback_sql,
                feedback_score
            )

            response = await self._call_llm(prompt)
            return self._parse_response(response, context)

        except Exception as e:
            logger.error(f"带反馈的 SQL 生成失败: {e}")
            return GenerationResult(
                success=False,
                sql="",
                explanation="",
                confidence=0.0,
                context_used={},
                error=str(e)
            )

    def _build_prompt(
        self,
        question: str,
        context: List[Dict[str, Any]],
        schemas: Optional[List[Any]] = None
    ) -> str:
        """构建 RAG 增强的 Prompt"""
        parts = [f"用户问题: {question}\n"]

        # 分类添加检索结果
        ddl_results = [r for r in context if r.get("type") == "ddl"]
        sql_results = [r for r in context if r.get("type") == "sql"]
        doc_results = [r for r in context if r.get("type") == "doc"]

        # 添加 DDL
        if ddl_results:
            parts.append("\n## 相关表结构 (DDL):")
            for result in ddl_results[:3]:
                if isinstance(result.get("content"), str):
                    parts.append(f"\n{result['content']}")
                elif isinstance(result.get("content"), dict):
                    content = result["content"]
                    if "ddl" in content:
                        parts.append(f"\n表: {content.get('table_name', '')}")
                        parts.append(f"描述: {content.get('description', '')}")
                        parts.append(f"DDL: {content['ddl']}")

        # 添加 SQL 示例
        if sql_results:
            parts.append("\n## 相似 SQL 示例:")
            for i, result in enumerate(sql_results[:3], 1):
                if isinstance(result.get("content"), str):
                    # 从内容中提取问题和 SQL
                    lines = result["content"].split("\n")
                    q_line = next((l for l in lines if l.startswith("问题:")), "")
                    s_line = next((l for l in lines if l.startswith("SQL:")), "")
                    parts.append(f"\n示例 {i}:")
                    if q_line: parts.append(q_line)
                    if s_line: parts.append(s_line)
                elif isinstance(result.get("content"), dict):
                    content = result["content"]
                    parts.append(f"\n示例 {i}:")
                    parts.append(f"问题: {content.get('question', '')}")
                    parts.append(f"SQL: {content.get('sql', '')}")

        # 添加业务文档
        if doc_results:
            parts.append("\n## 相关业务文档:")
            for result in doc_results[:2]:
                if isinstance(result.get("content"), str):
                    parts.append(f"\n{result['content']}")
                elif isinstance(result.get("content"), dict):
                    content = result["content"]
                    parts.append(f"\n{content.get('title', '')}")
                    parts.append(f"{content.get('content', '')}")

        # 添加 Schema 信息（如果提供）
        if schemas:
            parts.append("\n## 可用表结构:")
            for schema in schemas[:5]:
                table_name = getattr(schema, "table_name", "unknown")
                parts.append(f"\n表: {table_name}")
                columns = getattr(schema, "columns", [])
                for col in columns:
                    col_name = getattr(col, "name", "")
                    col_type = getattr(col, "type", "")
                    parts.append(f"  - {col_name}: {col_type}")

        parts.append("\n## 请根据上述信息生成 SQL:")
        parts.append("\n返回格式：")
        parts.append('```json')
        parts.append('{"sql": "你的SQL", "explanation": "解释", "confidence": 0.0-1.0}')
        parts.append('```')

        return "\n".join(parts)

    def _build_prompt_with_feedback(
        self,
        question: str,
        context: List[Dict[str, Any]],
        feedback_sql: str,
        feedback_score: float
    ) -> str:
        """构建带反馈的 Prompt"""
        base_prompt = self._build_prompt(question, context, None)

        feedback_section = f"""

## 之前的尝试和反馈
之前的 SQL: {feedback_sql}
用户评分: {feedback_score}/1.0

请根据反馈改进 SQL。如果分数较低，请：
1. 重新分析问题
2. 参考更多的示例
3. 检查表名和字段名是否正确
"""

        return base_prompt + feedback_section

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def _parse_response(
        self,
        content: str,
        context: List[Dict[str, Any]]
    ) -> GenerationResult:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{[^{}]*"sql"[^{}]*\}', content, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return GenerationResult(
                    success=bool(data.get("sql")),
                    sql=data.get("sql", ""),
                    explanation=data.get("explanation", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    context_used={
                        "results_count": len(context),
                        "types": list(set(r.get("type", "unknown") for r in context))
                    }
                )
            except json.JSONDecodeError:
                pass

        # 尝试提取 SQL
        sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return GenerationResult(
                success=True,
                sql=sql_match.group(1).strip(),
                explanation=content,
                confidence=0.6,
                context_used={
                    "results_count": len(context),
                    "types": list(set(r.get("type", "unknown") for r in context))
                }
            )

        # 尝试直接提取 SQL
        if "SELECT" in content.upper():
            lines = content.split("\n")
            sql_lines = []
            in_sql = False
            for line in lines:
                if "SELECT" in line.upper():
                    in_sql = True
                if in_sql:
                    sql_lines.append(line)
                    if ";" in line:
                        break

            if sql_lines:
                sql = "\n".join(sql_lines).strip()
                return GenerationResult(
                    success=True,
                    sql=sql,
                    explanation=content,
                    confidence=0.5,
                    context_used={
                        "results_count": len(context),
                        "types": list(set(r.get("type", "unknown") for r in context))
                    }
                )

        return GenerationResult(
            success=False,
            sql="",
            explanation="",
            confidence=0.0,
            context_used={},
            error="无法解析响应"
        )

    async def batch_generate(
        self,
        questions: List[str],
        db_name: str = "",
        top_k: int = 3
    ) -> List[GenerationResult]:
        """批量生成 SQL"""
        results = []

        for question in questions:
            result = await self.generate(question, db_name, top_k=top_k)
            results.append(result)

        return results


class AdvancedRAGGenerator(RAGSQLGenerator):
    """高级 RAG 生成器：支持查询分解和多步生成"""

    async def generate_complex(
        self,
        question: str,
        db_name: str = "",
        schemas: Optional[List[Any]] = None
    ) -> GenerationResult:
        """
        处理复杂查询

        步骤：
        1. 分析问题复杂度
        2. 如果复杂，分解为子问题
        3. 分别生成子 SQL
        4. 合并或选择最佳结果
        """
        # 检测是否为复杂查询
        is_complex = await self._detect_complexity(question)

        if not is_complex:
            return await self.generate(question, db_name, schemas)

        # 分解问题
        sub_questions = await self._decompose_question(question)

        logger.info(f"复杂查询分解: {question} -> {sub_questions}")

        # 为每个子问题生成 SQL
        sub_results = []
        for sub_q in sub_questions:
            result = await self.generate(sub_q, db_name, schemas)
            sub_results.append(result)

        # 合并结果
        return await self._merge_results(question, sub_results)

    async def _detect_complexity(self, question: str) -> bool:
        """检测问题复杂度"""
        complex_keywords = [
            "总计", "平均", "每个", "分别", "趋势", "对比",
            "top", "bottom", "highest", "lowest", "排名"
        ]

        question_lower = question.lower()
        return any(kw in question_lower for kw in complex_keywords) or len(question) > 50

    async def _decompose_question(self, question: str) -> List[str]:
        """分解复杂问题"""
        # 简化实现：基于规则分解
        # 实际可使用 LLM 进行分解

        # 检测是否包含多个问题
        separators = ["，", "。", "，然后", "，接着", " and ", " then "]
        for sep in separators:
            if sep in question:
                return [q.strip() for q in question.split(sep) if q.strip()]

        # 如果无法分解，返回原问题
        return [question]

    async def _merge_results(
        self,
        original_question: str,
        sub_results: List[GenerationResult]
    ) -> GenerationResult:
        """合并子结果"""
        if len(sub_results) == 1:
            return sub_results[0]

        # 如果所有子查询都成功，尝试用 UNION 或子查询合并
        successful = [r for r in sub_results if r.success]

        if not successful:
            return GenerationResult(
                success=False,
                sql="",
                explanation="所有子查询都失败了",
                confidence=0.0,
                context_used={},
                error="子查询失败"
            )

        if len(successful) == 1:
            return successful[0]

        # 尝试合并 SQL
        merged_sql = self._merge_sqls(successful)

        return GenerationResult(
            success=True,
            sql=merged_sql,
            explanation=f"合并了 {len(successful)} 个子查询",
            confidence=sum(r.confidence for r in successful) / len(successful),
            context_used={"merged_from": len(successful)}
        )

    def _merge_sqls(self, results: List[GenerationResult]) -> str:
        """合并多个 SQL"""
        sqls = [r.sql for r in results if r.sql]

        if len(sqls) == 1:
            return sqls[0]

        # 尝试 UNION
        if all("UNION" not in sql.upper() for sql in sqls):
            try:
                # 简单 UNION
                return " UNION ALL ".join(f"({sql})" for sql in sqls)
            except:
                pass

        # 返回第一个
        return sqls[0]
