"""
方案三：基于 Fine-tuning 的 Text-to-SQL 方案

核心理念：
1. 在特定数据集上微调 LLM，使其更好地理解 SQL 生成任务
2. 使用 PEFT（参数高效微调）方法如 LoRA 降低成本
3. 结合 Instruction Tuning 提升模型遵循指令的能力

适用场景：
- 有大量标注数据
- 需要高准确率
- 对特定领域/方言有要求

论文参考：
- RESDSQL: Instruction Learning for Text-to-SQL (2024)
- SQL-Crafter: " Crafting Text-to-SQL Models with Pre-training and Fine-tuning"
- Alpha-SQL: Monte Carlo Tree Search for Text-to-SQL (ICML 2025)
- Fine-tuning methods for Text-to-SQL (2024-2025)

注意：
- 此方案主要用于模型训练，实际推理时可使用微调后的模型
- 完整的微调需要 GPU 资源和大量时间
- 这里提供训练流程和代码框架
"""

from typing import List, Dict, Optional, Any, Tuple
import json
import os
from dataclasses import dataclass, field
from loguru import logger
from enum import Enum

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

from app.config import get_settings


class FineTuningMethod(Enum):
    """微调方法"""
    FULL = "full"  # 全参数微调
    LORA = "lora"  # LoRA 微调
    QLORA = "qlora"  # QLoRA 微调（量化+LoRA）


@dataclass
class TrainingExample:
    """训练样本"""
    question: str
    sql: str
    database_schema: str = ""
    db_id: str = ""
    difficulty: str = "easy"  # easy, medium, hard, extra_hard

    def to_prompt(self, template: str = "default") -> str:
        """转换为训练 Prompt"""
        if template == "default":
            return f"""-- Task: Generate SQL based on the natural language question
-- Database Schema:
{self.database_schema}

-- Question:
{self.question}

-- SQL:
{self.sql}"""
        elif template == "instruction":
            return f"""### Instruction:
Convert the following question to SQL query.

Schema:
{self.database_schema}

Question:
{self.question}

### Response:
{self.sql}"""
        else:
            return f"Question: {self.question}\nSQL: {self.sql}"


@dataclass
class FineTuningConfig:
    """微调配置"""
    # 模型配置
    base_model: str = "codellama/CodeLlama-7b-Instruct-hf"
    output_dir: str = "./models/finetuned_sql"

    # 训练配置
    method: FineTuningMethod = FineTuningMethod.LORA
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100

    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # 其他配置
    max_length: int = 2048
    use_flash_attention: bool = False


class SQLFineTuner:
    """
    SQL 模型微调器

    支持的微调方法：
    1. Full Fine-tuning: 全参数微调（需要大显存）
    2. LoRA: 低秩适应，只训练少量参数
    3. QLoRA: 量化版 LoRA，进一步降低显存需求
    """

    def __init__(self, config: FineTuningConfig = None):
        self.config = config or FineTuningConfig()
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.trainer = None

    def load_base_model(self):
        """加载基础模型和分词器"""
        logger.info(f"加载基础模型: {self.config.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        logger.info("基础模型加载完成")

    def setup_peft(self):
        """设置 PEFT 模型"""
        if self.config.method == FineTuningMethod.LORA:
            self._setup_lora()
        elif self.config.method == FineTuningMethod.QLORA:
            self._setup_qlora()
        else:
            logger.info("使用全参数微调")

    def _setup_lora(self):
        """设置 LoRA"""
        logger.info("设置 LoRA 微调")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            inference_mode=False
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    def _setup_qlora(self):
        """设置 QLoRA"""
        logger.info("设置 QLoRA 微调")

        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # 重新加载模型为 4-bit
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            inference_mode=False
        )

        self.peft_model = get_peft_model(self.model, lora_config)

    def prepare_dataset(
        self,
        examples: List[TrainingExample],
        template: str = "default"
    ) -> Dataset:
        """准备训练数据集"""
        logger.info(f"准备数据集，样本数: {len(examples)}")

        texts = []
        for ex in examples:
            prompt = ex.to_prompt(template)
            texts.append(prompt)

        # 创建 Dataset
        dataset = Dataset.from_dict({"text": texts})

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

    def train(
        self,
        train_examples: List[TrainingExample],
        eval_examples: List[TrainingExample] = None
    ):
        """训练模型"""
        logger.info("开始训练流程")

        # 加载模型
        if self.model is None:
            self.load_base_model()

        # 设置 PEFT
        if self.config.method in [FineTuningMethod.LORA, FineTuningMethod.QLORA]:
            self.setup_peft()

        # 准备数据集
        train_dataset = self.prepare_dataset(train_examples)
        eval_dataset = None
        if eval_examples:
            eval_dataset = self.prepare_dataset(eval_examples)

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            fp16=True,
            optim="adamw_torch",
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            load_best_model_at_end=bool(eval_dataset),
            report_to=["tensorboard"],
        )

        # 创建 Trainer
        model_to_train = self.peft_model if self.peft_model else self.model

        self.trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )

        # 开始训练
        logger.info("开始训练...")
        self.trainer.train()

        # 保存模型
        logger.info(f"保存模型到: {self.config.output_dir}")
        self.trainer.save_model()

        # 保存 tokenizer
        self.tokenizer.save_pretrained(self.config.output_dir)

        logger.info("训练完成")

    def save_for_inference(self, path: str = None):
        """保存用于推理的模型"""
        save_path = path or self.config.output_dir

        if self.peft_model:
            # 保存 LoRA adapter
            self.peft_model.save_pretrained(save_path)
            logger.info(f"LoRA adapter 已保存到: {save_path}")

        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer 已保存到: {save_path}")


class FineTunedSQLGenerator:
    """
    使用微调后模型生成 SQL 的推理器
    """

    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = model_path
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.device = None

    def load_model(self):
        """加载微调后的模型"""
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # 如果有 base_model，加载为 PEFT 模型
        if self.base_model:
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model_obj, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.model.eval()
        logger.info("模型加载完成")

    def generate_sql(
        self,
        question: str,
        schema: str = "",
        max_length: int = 512,
        temperature: float = 0.1,
        num_beams: int = 4
    ) -> Dict[str, Any]:
        """
        生成 SQL

        Args:
            question: 用户问题
            schema: 数据库 Schema
            max_length: 最大生成长度
            temperature: 采样温度
            num_beams: 束搜索数量

        Returns:
            包含 SQL 和元数据的结果
        """
        if self.model is None:
            self.load_model()

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
                num_beams=num_beams,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 提取 SQL
        sql = self._extract_sql(generated_text, prompt)

        return {
            "success": bool(sql),
            "sql": sql,
            "full_output": generated_text,
            "method": "finetuned"
        }

    def _build_prompt(self, question: str, schema: str) -> str:
        """构建推理 Prompt"""
        if schema:
            return f"""-- Task: Generate SQL based on the natural language question
-- Database Schema:
{schema}

-- Question:
{self.question}

-- SQL:"""
        return f"Question: {question}\nSQL:"

    def _extract_sql(self, output: str, prompt: str) -> str:
        """从输出中提取 SQL"""
        # 移除 prompt 部分
        sql_part = output[len(prompt):].strip()

        # 提取 SQL（可能在代码块中）
        import re
        sql_match = re.search(r'```sql\s*(.*?)\s*```', sql_part, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        return sql_part.split(";")[0].strip() + ";" if sql_part else ""


# ==================== 数据准备工具 ====================

class DataPreparator:
    """训练数据准备工具"""

    @staticmethod
    def load_spider_dataset(data_path: str) -> List[TrainingExample]:
        """
        加载 Spider 数据集

        Spider 是最常用的 Text-to-SQL 数据集
        """
        examples = []

        # 加载数据
        train_path = os.path.join(data_path, "train.json")
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            # Spider 数据格式
            example = TrainingExample(
                question=item["question"],
                sql=item["query"],
                db_id=item["db_id"],
                difficulty=item.get("difficulty", "easy")
            )
            examples.append(example)

        logger.info(f"从 Spider 加载了 {len(examples)} 个样本")
        return examples

    @staticmethod
    def load_custom_dataset(
        questions: List[str],
        sqls: List[str],
        schemas: List[str] = None
    ) -> List[TrainingExample]:
        """加载自定义数据集"""
        examples = []

        for i, (question, sql) in enumerate(zip(questions, sqls)):
            schema = schemas[i] if schemas and i < len(schemas) else ""
            examples.append(TrainingExample(
                question=question,
                sql=sql,
                database_schema=schema
            ))

        logger.info(f"创建了 {len(examples)} 个自定义样本")
        return examples

    @staticmethod
    def augment_data(
        examples: List[TrainingExample],
        augment_factor: int = 3
    ) -> List[TrainingExample]:
        """
        数据增强

        方法：
        1. SQL 等价改写
        2. 问题同义改写
        3. Schema 变体
        """
        # 简化实现，实际可以使用 LLM 进行增强
        augmented = examples.copy()

        for _ in range(augment_factor - 1):
            for ex in examples:
                # 创建变体（这里只是复制，实际需要改写）
                augmented.append(TrainingExample(
                    question=ex.question,
                    sql=ex.sql,
                    database_schema=ex.database_schema,
                    db_id=ex.db_id,
                    difficulty=ex.difficulty
                ))

        logger.info(f"数据增强: {len(examples)} -> {len(augmented)}")
        return augmented


# ==================== 评估工具 ====================

class ModelEvaluator:
    """模型评估工具"""

    @staticmethod
    def evaluate_exact_match(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """精确匹配评估"""
        from sqlparse import parse, format

        exact_matches = 0
        for pred, ref in zip(predictions, references):
            # 规范化 SQL
            pred_formatted = format(pred[0] if parse(pred) else pred, reindent=True)
            ref_formatted = format(ref[0] if parse(ref) else ref, reindent=True)

            if pred_formatted.strip() == ref_formatted.strip():
                exact_matches += 1

        return {
            "exact_match": exact_matches / len(predictions),
            "total": len(predictions),
            "matches": exact_matches
        }

    @staticmethod
    def evaluate_execution_accuracy(
        sqls: List[str],
        db_manager,
        db_name: str
    ) -> Dict[str, float]:
        """执行准确率评估"""
        successful = 0

        for sql in sqls:
            try:
                db_manager.execute_sql(sql)
                successful += 1
            except:
                pass

        return {
            "execution_accuracy": successful / len(sqls),
            "total": len(sqls),
            "successful": successful
        }


# ==================== 示例用法 ====================

async def example_training():
    """示例：训练流程"""

    # 1. 准备数据
    examples = DataPreparator.load_custom_dataset(
        questions=[
            "查询所有用户",
            "查询销售额最高的产品",
            "统计每个订单的总金额"
        ],
        sqls=[
            "SELECT * FROM users",
            "SELECT * FROM products ORDER BY sales DESC LIMIT 1",
            "SELECT order_id, SUM(amount) FROM order_items GROUP BY order_id"
        ]
    )

    # 2. 配置微调
    config = FineTuningConfig(
        base_model="codellama/CodeLlama-7b-Instruct-hf",
        method=FineTuningMethod.LORA,
        output_dir="./models/finetuned_sql",
        num_train_epochs=3,
        per_device_train_batch_size=2
    )

    # 3. 创建微调器
    tuner = SQLFineTuner(config)

    # 4. 开始训练
    tuner.train(examples)

    # 5. 保存模型
    tuner.save_for_inference()


async def example_inference():
    """示例：推理流程"""

    # 1. 加载微调后的模型
    generator = FineTunedSQLGenerator(
        model_path="./models/finetuned_sql",
        base_model="codellama/CodeLlama-7b-Instruct-hf"
    )
    generator.load_model()

    # 2. 生成 SQL
    result = generator.generate_sql(
        question="查询销售额最高的前10个产品",
        schema="CREATE TABLE products (id INT, name VARCHAR, sales DECIMAL)"
    )

    print(f"生成的 SQL: {result['sql']}")


class HybridSQLGenerator:
    """
    混合 SQL 生成器

    结合多种方法：
    1. 优先使用微调模型生成
    2. 如果置信度低，使用 RAG 检索示例
    3. 最后使用 Agent 进行验证
    """

    def __init__(
        self,
        finetuned_generator: FineTunedSQLGenerator = None,
        rag_generator = None,
        agent = None
    ):
        self.finetuned = finetuned_generator
        self.rag = rag_generator
        self.agent = agent

    async def generate(
        self,
        question: str,
        schema: str = "",
        db_name: str = ""
    ) -> Dict[str, Any]:
        """
        混合生成 SQL

        策略：
        - 简单问题：直接使用微调模型
        - 中等问题：微调 + RAG
        - 复杂问题：微调 + RAG + Agent 验证
        """
        # 1. 使用微调模型生成
        if self.finetuned:
            result = self.finetuned.generate_sql(question, schema)
            confidence = self._estimate_confidence(result)
        else:
            confidence = 0

        # 2. 如果置信度低，使用 RAG
        if confidence < 0.7 and self.rag:
            rag_result = await self.rag.generate_sql(question, db_name)
            # 可以结合两个结果
            if rag_result.get("success"):
                result = rag_result

        # 3. 复杂问题使用 Agent 验证
        if confidence < 0.5 and self.agent:
            agent_result = await self.agent.run(question, db_name)
            if agent_result.get("success"):
                result = agent_result

        return result

    def _estimate_confidence(self, result: Dict) -> float:
        """估算置信度"""
        # 简化实现
        if result.get("sql"):
            return 0.8
        return 0.0


if __name__ == "__main__":
    import asyncio
    # asyncio.run(example_training())
    # asyncio.run(example_inference())
    print("请根据需要调用 example_training() 或 example_inference()")
