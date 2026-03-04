"""
SQL生成模块
"""
from typing import Dict, Optional, List
import json
import re
from loguru import logger

from app.config import get_settings
from app.models.schemas import SQLGenerationResult, TableSchema
from app.core.prompt_builder import SQLPromptBuilder


class SQLGenerator:
    """SQL生成器"""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    async def init(self):
        """初始化LLM客户端"""
        from openai import AsyncOpenAI
        
        if self.settings.llm_provider == "openai":
            self._client = AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url
            )
        logger.info(f"SQLGenerator 初始化完成，使用模型: {self.settings.llm_model}")
    
    async def generate(
        self,
        question: str,
        tables: List[TableSchema],
        context_summary: str = ""
    ) -> SQLGenerationResult:
        """生成SQL"""
        
        # 构建Prompt
        prompt = SQLPromptBuilder.build_complete_prompt(
            question=question,
            tables=tables,
            context_summary=context_summary
        )
        
        # 调用LLM
        try:
            response = await self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": SQLPromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return SQLGenerationResult(
                sql="",
                explanation=f"生成失败: {str(e)}",
                confidence=0.0
            )
    
    def _parse_response(self, content: str) -> SQLGenerationResult:
        """解析LLM响应"""
        
        # 尝试提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return SQLGenerationResult(
                    sql=data.get("sql", ""),
                    explanation=data.get("explanation", ""),
                    assumptions=data.get("assumptions", []),
                    confidence=0.8
                )
            except json.JSONDecodeError:
                pass
        
        # 尝试提取SQL
        sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return SQLGenerationResult(
                sql=sql_match.group(1).strip(),
                explanation="",
                confidence=0.6
            )
        
        # 直接返回内容
        return SQLGenerationResult(
            sql="",
            explanation=content,
            confidence=0.0
        )
    
    async def rewrite_question(
        self,
        current_question: str,
        last_question: str,
        last_sql: str,
        referenced_tables: List[str]
    ) -> str:
        """改写问题（多轮对话）"""
        
        from app.core.prompt_builder import RewritePromptBuilder
        
        prompt = RewritePromptBuilder.build_rewrite_prompt(
            current_question=current_question,
            last_question=last_question,
            last_sql=last_sql,
            referenced_tables=referenced_tables
        )
        
        try:
            response = await self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("rewritten_question", current_question)
            
        except Exception as e:
            logger.warning(f"问题改写失败: {e}")
        
        return current_question
    
    async def classify_intent(
        self,
        current_question: str,
        last_question: str,
        last_sql: str
    ) -> str:
        """识别意图"""
        
        from app.core.prompt_builder import IntentPromptBuilder
        
        prompt = IntentPromptBuilder.build_classify_prompt(
            current_question=current_question,
            last_question=last_question,
            last_sql=last_sql
        )
        
        try:
            response = await self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("intent", "new_query")
            
        except Exception as e:
            logger.warning(f"意图识别失败: {e}")
        
        return "new_query"


# 全局实例
sql_generator: Optional[SQLGenerator] = None


async def get_sql_generator() -> SQLGenerator:
    """获取SQL生成器实例"""
    global sql_generator
    if sql_generator is None:
        sql_generator = SQLGenerator()
        await sql_generator.init()
    return sql_generator