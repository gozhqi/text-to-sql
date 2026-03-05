"""
微调方案 - 推理模块

使用微调后的模型生成 SQL
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..data.preparation import TrainingExample


@dataclass
class GenerationResult:
    """生成结果"""
    success: bool
    sql: str
    explanation: str
    confidence: float
    full_output: str
    error: str = ""


class FinetunedPredictor:
    """微调模型预测器"""

    def __init__(
        self,
        model_path: str,
        base_model: str = None,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _get_device(self, device: str) -> str:
        """获取设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """加载微调后的模型"""
        logger.info(f"加载微调模型: {self.model_path}")
        logger.info(f"使用设备: {self.device}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        if self.base_model:
            # 加载 base model + adapter
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            self.model = PeftModel.from_pretrained(base, self.model_path)
            self.model = self.model.merge_and_unload()  # 合并权重
        else:
            # 直接加载完整模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device
            )

        self.model.eval()
        self._loaded = True
        logger.info("模型加载完成")

    def generate_sql(
        self,
        question: str,
        schema: str = "",
        max_length: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        num_beams: int = 1
    ) -> GenerationResult:
        """
        生成 SQL

        Args:
            question: 用户问题
            schema: 数据库 Schema
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling 参数
            num_beams: 束搜索数量

        Returns:
            生成结果
        """
        if not self._loaded:
            self.load_model()

        try:
            # 构建 Prompt
            prompt = self._build_prompt(question, schema)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - max_length
            ).to(self.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # 解码
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # 提取 SQL
            sql = self._extract_sql(generated_text, prompt)

            return GenerationResult(
                success=bool(sql),
                sql=sql,
                explanation=f"基于微调模型生成",
                confidence=0.8,
                full_output=generated_text
            )

        except Exception as e:
            logger.error(f"SQL 生成失败: {e}")
            return GenerationResult(
                success=False,
                sql="",
                explanation="",
                confidence=0.0,
                full_output="",
                error=str(e)
            )

    def _build_prompt(self, question: str, schema: str) -> str:
        """构建推理 Prompt"""
        if schema:
            # 使用与训练相同的格式
            return f"""### Instruction:
Convert the following question to SQL query.

Schema:
{schema}

Question:
{question}

### Response:
"""
        return f"""### Instruction:
{question}

### Response:
"""

    def _extract_sql(self, output: str, prompt: str) -> str:
        """从输出中提取 SQL"""
        # 移除 prompt 部分
        sql_part = output[len(prompt):].strip()

        # 提取 SQL（可能在代码块中）
        import re

        # 尝试提取 ```sql 代码块
        sql_match = re.search(r'```sql\s*(.*?)\s*```', sql_part, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # 尝试提取 ``` 代码块
        code_match = re.search(r'```\s*(.*?)\s*```', sql_part, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()
            if "SELECT" in content.upper():
                return content

        # 直接提取，找到第一个 SELECT
        if "SELECT" in sql_part.upper():
            lines = sql_part.split("\n")
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
                return "\n".join(sql_lines).strip()

        return sql_part.split(";")[0].strip() + ";" if sql_part else ""

    async def batch_generate(
        self,
        questions: list,
        schemas: list = None,
        **kwargs
    ) -> list:
        """批量生成"""
        if schemas is None:
            schemas = [""] * len(questions)

        results = []
        for question, schema in zip(questions, schemas):
            result = self.generate_sql(question, schema, **kwargs)
            results.append(result)

        return results


class AdaptivePredictor(FinetunedPredictor):
    """自适应预测器：支持回退到 API"""

    def __init__(
        self,
        model_path: str,
        base_model: str = None,
        fallback_llm=None,
        confidence_threshold: float = 0.5
    ):
        super().__init__(model_path, base_model)
        self.fallback_llm = fallback_llm
        self.confidence_threshold = confidence_threshold

    def generate_sql_with_fallback(
        self,
        question: str,
        schema: str = "",
        **kwargs
    ) -> GenerationResult:
        """生成 SQL，支持回退"""
        # 首先尝试使用微调模型
        result = self.generate_sql(question, schema, **kwargs)

        # 如果置信度低且有回退 LLM
        if result.confidence < self.confidence_threshold and self.fallback_llm:
            logger.info(f"微调模型置信度低 ({result.confidence})，使用回退 LLM")

            # 使用回退 LLM 生成
            try:
                import asyncio
                fallback_result = asyncio.run(self._fallback_generate(question, schema))

                if fallback_result.get("sql"):
                    return GenerationResult(
                        success=True,
                        sql=fallback_result["sql"],
                        explanation=f"使用回退 LLM 生成（微调模型置信度不足）",
                        confidence=fallback_result.get("confidence", 0.7),
                        full_output=""
                    )
            except Exception as e:
                logger.warning(f"回退 LLM 失败: {e}")

        return result

    async def _fallback_generate(self, question: str, schema: str) -> Dict[str, Any]:
        """使用回退 LLM 生成"""
        prompt = f"Convert to SQL:\n\n{question}"
        if schema:
            prompt = f"Schema:\n{schema}\n\nQuestion: {question}"

        response = await self.fallback_llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content

        # 提取 SQL
        import re
        sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        sql = sql_match.group(1).strip() if sql_match else content.strip()

        return {
            "sql": sql,
            "confidence": 0.7
        }


def create_predictor(
    model_path: str,
    base_model: str = None
) -> FinetunedPredictor:
    """创建预测器"""
    return FinetunedPredictor(model_path, base_model)
