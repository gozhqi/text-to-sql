"""
Text-to-SQL Inference Service

高性能推理服务，支持:
1. 多种模型后端 (OpenAI API, 本地模型, 微调模型)
2. 多级缓存策略
3. 复杂度自适应路由
4. 流式输出
"""
import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Literal, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import aiohttp
import redis.asyncio as redis


# ============================================================================
# 类型定义
# ============================================================================

class ModelBackend(Enum):
    """模型后端类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    VLLM = "vllm"


class ComplexityLevel(Enum):
    """查询复杂度"""
    SIMPLE = "simple"  # < 0.3
    MEDIUM = "medium"  # 0.3 - 0.7
    COMPLEX = "complex"  # > 0.7


@dataclass
class QueryRequest:
    """查询请求"""
    question: str
    db_name: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    stream: bool = False
    include_reasoning: bool = False


@dataclass
class SchemaInfo:
    """Schema 信息"""
    tables: Dict[str, Dict[str, Any]]
    relationships: List[Dict[str, str]]


@dataclass
class QueryResponse:
    """查询响应"""
    success: bool
    sql: str = ""
    explanation: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    execution_time: float = 0.0
    complexity: str = ""
    tables_used: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ============================================================================
# 复杂度评估器
# ============================================================================

class ComplexityEvaluator:
    """查询复杂度评估器"""

    # 复杂度权重
    TABLES_WEIGHT = 0.2
    JOIN_WEIGHT = 0.25
    AGGREGATION_WEIGHT = 0.2
    SUBQUERY_WEIGHT = 0.2
    KEYWORD_WEIGHT = 0.15

    def __init__(self):
        self.complexity_keywords = {
            "simple": ["显示", "列出", "查询", "获取", "有几个", "有多少"],
            "medium": ["统计", "平均", "总和", "最大", "最小", "分组", "排序"],
            "complex": [
                "每年的", "每月的", "每天的",
                "占比", "同比增长", "环比增长",
                "累计", "滚动", "排名",
                "条件是", "其中", "并且", "或者"
            ]
        }

    def evaluate(self, question: str, retrieved_schemas: List[str]) -> tuple[float, ComplexityLevel]:
        """评估查询复杂度"""
        score = 0.0

        # 1. 表数量
        table_count = len(retrieved_schemas)
        if table_count == 1:
            score += 0.1
        elif table_count == 2:
            score += 0.3
        else:
            score += 0.5

        # 2. 关键词复杂度
        question_lower = question.lower()

        # 复杂关键词
        for keyword in self.complexity_keywords["complex"]:
            if keyword in question_lower:
                score += 0.2
                break

        # 中等关键词
        for keyword in self.complexity_keywords["medium"]:
            if keyword in question_lower:
                score += 0.15
                break

        # 3. 问题长度（复杂问题通常更长）
        question_length = len(question)
        if question_length > 50:
            score += 0.1
        elif question_length > 30:
            score += 0.05

        # 4. 问句结构
        if "，" in question or "。" in question:
            score += 0.1

        if "并且" in question or "或者" in question or "然后" in question:
            score += 0.15

        # 归一化
        score = min(score, 1.0)

        # 确定复杂度级别
        if score < 0.3:
            level = ComplexityLevel.SIMPLE
        elif score < 0.7:
            level = ComplexityLevel.MEDIUM
        else:
            level = ComplexityLevel.COMPLEX

        return score, level


# ============================================================================
# 缓存管理器
# ============================================================================

class CacheManager:
    """多级缓存管理器"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.l1_cache: Dict[str, tuple[Any, float]] = {}  # {key: (value, expiry)}
        self.redis_client: Optional[redis.Redis] = None

    async def init(self):
        """初始化"""
        if self.redis_url:
            self.redis_client = await redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis: {self.redis_url}")

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # L1 缓存检查
        if key in self.l1_cache:
            value, expiry = self.l1_cache[key]
            if expiry > time.time():
                return value
            else:
                del self.l1_cache[key]

        # L2 缓存检查 (Redis)
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    # 回填 L1
                    deserialized = json.loads(value)
                    self.l1_cache[key] = (deserialized, time.time() + 300)
                    return deserialized
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        expiry = time.time() + ttl

        # 设置 L1 缓存
        self.l1_cache[key] = (value, expiry)

        # 设置 L2 缓存
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value, ensure_ascii=False)
                )
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

        # 清理过期缓存
        self._cleanup_l1()

    def _cleanup_l1(self):
        """清理 L1 缓存中的过期项"""
        now = time.time()
        expired_keys = [k for k, (_, expiry) in self.l1_cache.items() if expiry <= now]
        for key in expired_keys:
            del self.l1_cache[key]

    def generate_key(self, question: str, db_name: str, model: str) -> str:
        """生成缓存键"""
        content = f"{question}:{db_name}:{model}"
        return f"text2sql:{hashlib.md5(content.encode()).hexdigest()}"

    async def invalidate(self, pattern: str):
        """使缓存失效"""
        # 清理 L1
        keys_to_delete = [k for k in self.l1_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self.l1_cache[key]

        # 清理 Redis
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis invalidate error: {e}")


# ============================================================================
# 模型后端
# ============================================================================

class ModelBackend:
    """模型后端基类"""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """生成文本"""
        raise NotImplementedError

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式生成"""
        raise NotImplementedError


class OpenAIBackend(ModelBackend):
    """OpenAI API 后端"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """生成文本"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式生成"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    line_str = line.decode().strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass


class LocalModelBackend(ModelBackend):
    """本地模型后端 (支持 PEFT/LoRA)"""

    def __init__(
        self,
        base_model: str,
        lora_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device_map: str = "auto"
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self._loaded = False

    async def load(self):
        """加载模型"""
        if self._loaded:
            return

        logger.info(f"Loading local model: {self.base_model}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型配置
        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.load_in_4bit:
            model_kwargs.update({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            })

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs
        )

        # 加载 LoRA adapter
        if self.lora_path:
            logger.info(f"Loading LoRA adapter from: {self.lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_path
            )

        self._loaded = True
        logger.info("Model loaded successfully")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """生成文本"""
        await self.load()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只返回生成的部分
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式生成"""
        await self.load()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        streamer = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=None,  # 需要实现自定义 streamer
        )

        # 简化实现：直接返回完整结果
        # 实际实现应该使用 TextIteratorStreamer
        async for token in self._tokenize_stream(streamer):
            yield token

    async def _tokenize_stream(self, output_ids: torch.Tensor) -> AsyncGenerator[str, None]:
        """将输出 token 流式化"""
        # 简化实现
        pass


# ============================================================================
# Text-to-SQL 服务
# ============================================================================

class Text2SQLService:
    """Text-to-SQL 推理服务"""

    def __init__(
        self,
        model_backend: ModelBackend,
        cache_manager: Optional[CacheManager] = None,
        schemas: Optional[Dict[str, SchemaInfo]] = None,
    ):
        self.model_backend = model_backend
        self.cache_manager = cache_manager
        self.schemas = schemas or {}
        self.complexity_evaluator = ComplexityEvaluator()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """处理查询请求"""
        start_time = time.time()

        try:
            # 1. 检查缓存
            cache_key = None
            if self.cache_manager:
                cache_key = self.cache_manager.generate_key(
                    request.question,
                    request.db_name,
                    getattr(self.model_backend, "model", "unknown")
                )
                cached = await self.cache_manager.get(cache_key)
                if cached:
                    logger.info(f"Cache hit for key: {cache_key[:16]}...")
                    return QueryResponse(**cached)

            # 2. 获取 Schema
            schema_info = self.schemas.get(request.db_name)
            if not schema_info:
                return QueryResponse(
                    success=False,
                    error=f"Database not found: {request.db_name}"
                )

            # 3. 评估复杂度
            relevant_tables = list(schema_info.tables.keys())
            complexity_score, complexity_level = self.complexity_evaluator.evaluate(
                request.question,
                relevant_tables
            )

            # 4. 构建 Prompt
            prompt = self._build_prompt(request, schema_info, complexity_level)

            # 5. 生成 SQL
            if request.include_reasoning:
                response = await self._generate_with_reasoning(prompt)
            else:
                response = await self.model_backend.generate(prompt)

            # 6. 解析响应
            sql, explanation, confidence = self._parse_response(response)

            # 7. 计算执行时间
            execution_time = time.time() - start_time

            # 8. 构建响应
            query_response = QueryResponse(
                success=True,
                sql=sql,
                explanation=explanation,
                confidence=confidence,
                execution_time=execution_time,
                complexity=complexity_level.value,
                tables_used=relevant_tables,
            )

            # 9. 缓存结果
            if self.cache_manager and cache_key and query_response.success:
                await self.cache_manager.set(
                    cache_key,
                    query_response.__dict__,
                    ttl=3600
                )

            return query_response

        except Exception as e:
            logger.exception(f"Query failed: {e}")
            return QueryResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def query_stream(
        self,
        request: QueryRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式查询"""
        start_time = time.time()

        try:
            # 获取 Schema
            schema_info = self.schemas.get(request.db_name)
            if not schema_info:
                yield {
                    "type": "error",
                    "error": f"Database not found: {request.db_name}"
                }
                return

            # 构建 Prompt
            _, complexity_level = self.complexity_evaluator.evaluate(
                request.question,
                list(schema_info.tables.keys())
            )
            prompt = self._build_prompt(request, schema_info, complexity_level)

            # 发送状态
            yield {
                "type": "status",
                "message": "Generating SQL...",
                "complexity": complexity_level.value
            }

            # 流式生成
            full_response = ""
            async for chunk in self.model_backend.generate_stream(prompt):
                full_response += chunk
                yield {
                    "type": "token",
                    "content": chunk
                }

            # 解析最终结果
            sql, explanation, confidence = self._parse_response(full_response)

            yield {
                "type": "result",
                "sql": sql,
                "explanation": explanation,
                "confidence": confidence,
                "execution_time": time.time() - start_time
            }

        except Exception as e:
            logger.exception(f"Stream query failed: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }

    def _build_prompt(
        self,
        request: QueryRequest,
        schema_info: SchemaInfo,
        complexity_level: ComplexityLevel
    ) -> str:
        """构建 Prompt"""
        # Schema 部分
        schema_text = self._format_schema(schema_info)

        # 系统提示
        system_prompt = """你是一个专业的 SQL 生成助手。请根据用户的问题和数据库结构，生成准确、高效的 SQL 查询语句。

## 规则要求：
1. 只生成 SELECT 查询语句，禁止 INSERT/UPDATE/DELETE/DROP 等操作
2. 使用标准 SQL 语法
3. 合理使用 JOIN 连接多个表
4. 对于聚合查询，正确使用 GROUP BY 和 HAVING

## 输出格式：
```json
{
  "sql": "生成的SQL语句",
  "explanation": "查询逻辑说明",
  "confidence": 0.95
}
```"""

        # 完整 Prompt
        prompt = f"""{system_prompt}

## 数据库结构

{schema_text}

## 用户问题

{request.question}

请生成对应的 SQL 查询语句："""

        return prompt

    def _format_schema(self, schema_info: SchemaInfo) -> str:
        """格式化 Schema"""
        text = ""
        for table_name, table_schema in schema_info.tables.items():
            text += f"### 表名: {table_name}\n"

            if table_schema.get("table_comment"):
                text += f"说明: {table_schema['table_comment']}\n"

            text += "\n#### 列信息:\n"
            for col in table_schema.get("columns", []):
                pk = " [PK]" if col.get("is_primary_key") else ""
                fk = " [FK]" if col.get("is_foreign_key") else ""
                text += f"- {col['name']}{pk}{fk}: {col['type']}"
                if col.get("comment"):
                    text += f" # {col['comment']}"
                text += "\n"

            text += "\n"

        return text.strip()

    def _parse_response(self, response: str) -> tuple[str, str, float]:
        """解析模型响应"""
        import re

        # 尝试提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return (
                    data.get("sql", ""),
                    data.get("explanation", ""),
                    data.get("confidence", 0.8)
                )
            except json.JSONDecodeError:
                pass

        # 尝试提取 SQL
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip(), response, 0.7

        # 没有找到 SQL
        return "", response, 0.0

    async def _generate_with_reasoning(self, prompt: str) -> str:
        """带思维链的生成"""
        reasoning_prompt = f"""{prompt}

在生成 SQL 之前，请先分析问题：

1. 确定需要查询哪些表
2. 确定需要哪些列
3. 确定查询条件
4. 确定是否需要聚合或排序

然后逐步构建 SQL 查询。"""

        return await self.model_backend.generate(reasoning_prompt)


# ============================================================================
# 服务工厂
# ============================================================================

class ServiceFactory:
    """Text-to-SQL 服务工厂"""

    @staticmethod
    async def create_openai_service(
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        schemas: Optional[Dict[str, SchemaInfo]] = None,
    ) -> Text2SQLService:
        """创建 OpenAI 后端服务"""
        backend = OpenAIBackend(api_key=api_key, base_url=base_url, model=model)
        cache = CacheManager(redis_url) if redis_url else None
        if cache:
            await cache.init()

        return Text2SQLService(
            model_backend=backend,
            cache_manager=cache,
            schemas=schemas,
        )

    @staticmethod
    async def create_local_service(
        base_model: str,
        lora_path: Optional[str] = None,
        load_in_4bit: bool = False,
        redis_url: Optional[str] = None,
        schemas: Optional[Dict[str, SchemaInfo]] = None,
    ) -> Text2SQLService:
        """创建本地模型服务"""
        backend = LocalModelBackend(
            base_model=base_model,
            lora_path=lora_path,
            load_in_4bit=load_in_4bit
        )
        cache = CacheManager(redis_url) if redis_url else None
        if cache:
            await cache.init()

        return Text2SQLService(
            model_backend=backend,
            cache_manager=cache,
            schemas=schemas,
        )
