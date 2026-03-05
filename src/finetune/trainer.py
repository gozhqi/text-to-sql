"""
微调方案 - LoRA 训练模块

提供 QLoRA 和 LoRA 训练功能，用于微调大语言模型进行 Text-to-SQL 任务
"""
import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime
from pathlib import Path
import torch

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/peft 未安装")

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes 未安装，QLoRA 不可用")


@dataclass
class LoraConfig:
    """LoRA 配置"""
    # 模型配置
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    trust_remote_code: bool = True

    # LoRA 参数
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 训练参数
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # 量化配置 (QLoRA)
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # 输出配置
    output_dir: str = "./output/finetune"
    cache_dir: str = "./cache"

    # 其他
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4


@dataclass
class TrainingResult:
    """训练结果"""
    output_dir: str
    checkpoint: str
    train_loss: float = 0.0
    eval_loss: float = 0.0
    train_samples: int = 0
    eval_samples: int = 0
    training_time: float = 0.0


class PromptFormatter:
    """提示词格式化器"""

    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            padding_side="right"
        )

        # 设置特殊 token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def format_prompt(self, example: Dict[str, Any], format_type: str = "alpaca") -> str:
        """格式化提示词"""
        if format_type == "alpaca":
            return self._format_alpaca(example)
        elif format_type == "instruction":
            return self._format_instruction(example)
        elif format_type == "chat":
            return self._format_chat(example)
        else:
            return self._format_simple(example)

    def _format_alpaca(self, example: Dict[str, Any]) -> str:
        """Alpaca 格式"""
        instruction = example.get("instruction", "将以下问题转换为 SQL 查询。")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

        return prompt

    def _format_instruction(self, example: Dict[str, Any]) -> str:
        """指令格式"""
        prompt = example.get("prompt", example.get("question", ""))
        completion = example.get("completion", example.get("sql", ""))

        return f"{prompt}\n{completion}"

    def _format_chat(self, example: Dict[str, Any]) -> str:
        """聊天格式"""
        conversations = example.get("conversations", [])

        messages = []
        for conv in conversations:
            role = conv.get("from", "user")
            if role == "human":
                messages.append({"role": "user", "content": conv.get("value", "")})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": conv.get("value", "")})

        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def _format_simple(self, example: Dict[str, Any]) -> str:
        """简单格式"""
        question = example.get("question", "")
        sql = example.get("sql", "")

        return f"Question: {question}\nSQL: {sql}"


class LoRATrainer:
    """LoRA 训练器"""

    def __init__(self, config: LoraConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.prompt_formatter = None

    def initialize(self):
        """初始化模型和分词器"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers 库未安装")

        logger.info("初始化 LoRA 训练器...")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        load_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.cache_dir,
            "device_map": "auto"
        }

        # 量化配置
        if self.config.use_4bit and BITSANDBYTES_AVAILABLE:
            load_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": getattr(torch, self.config.bnb_4bit_compute_dtype),
                "bnb_4bit_quant_type": self.config.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": True
            })
            logger.info("使用 4-bit 量化 (QLoRA)")
        elif self.config.use_8bit:
            load_kwargs["load_in_8bit"] = True
            logger.info("使用 8-bit 量化")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs
        )

        # 准备模型以进行 k-bit 训练
        if self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # 配置 LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )

        # 应用 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 初始化提示词格式化器
        self.prompt_formatter = PromptFormatter(self.config.model_name)

        logger.info("LoRA 训练器初始化完成")

    def prepare_dataset(
        self,
        data: List[Dict[str, Any]],
        format_type: str = "alpaca"
    ) -> Dataset:
        """准备训练数据集"""
        logger.info(f"准备数据集，共 {len(data)} 条样本")

        # 格式化提示词
        formatted_data = []
        for example in data:
            prompt = self.prompt_formatter.format_prompt(example, format_type)

            # 处理不同的输入格式
            if format_type == "alpaca":
                formatted_data.append({"text": prompt})
            elif format_type == "instruction":
                formatted_data.append({"text": prompt})
            else:
                formatted_data.append({"text": prompt})

        # 创建 Dataset
        dataset = Dataset.from_list(formatted_data)

        # 分词
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=["text"]
        )

        return tokenized_dataset

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
        format_type: str = "alpaca"
    ) -> TrainingResult:
        """训练模型"""
        start_time = datetime.now()

        if not self.model:
            self.initialize()

        # 准备数据集
        train_dataset = self.prepare_dataset(train_data, format_type)
        eval_dataset = None
        if eval_data:
            eval_dataset = self.prepare_dataset(eval_data, format_type)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            fp16=torch.cuda.is_available() and not (self.config.use_4bit or self.config.use_8bit),
            bf16=torch.cuda.is_bf16_supported() and not (self.config.use_4bit or self.config.use_8bit),
            gradient_checkpointing=True,
            optim="paged_adamw_32bit" if self.config.use_4bit else "adamw_torch",
            report_to=["tensorboard"],
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            ddp_find_unused_parameters=False,
            save_strategy="steps",
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # 创建 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # 开始训练
        logger.info("开始训练...")
        train_result = self.trainer.train()

        # 保存最终模型
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        # 计算训练时间
        training_time = (datetime.now() - start_time).total_seconds()

        # 获取指标
        metrics = train_result.metrics
        train_loss = metrics.get("train_loss", 0.0)
        eval_loss = metrics.get("eval_loss", 0.0)

        logger.info(f"训练完成！用时: {training_time:.2f}秒")
        logger.info(f"最终训练损失: {train_loss:.4f}")
        if eval_loss:
            logger.info(f"验证损失: {eval_loss:.4f}")

        return TrainingResult(
            output_dir=self.config.output_dir,
            checkpoint=self.config.output_dir,
            train_loss=train_loss,
            eval_loss=eval_loss,
            train_samples=len(train_data),
            eval_samples=len(eval_data) if eval_data else 0,
            training_time=training_time
        )

    def resume_from_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练"""
        if self.trainer:
            self.trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            logger.warning("Trainer 未初始化，无法恢复训练")


def create_lora_trainer(config: LoraConfig) -> LoRATrainer:
    """创建 LoRA 训练器实例"""
    return LoRATrainer(config)


# 预定义配置
def get_codellama_config() -> LoraConfig:
    """获取 CodeLlama 配置"""
    return LoraConfig(
        model_name="codellama/CodeLlama-7b-Instruct-hf",
        lora_r=16,
        lora_alpha=32,
        learning_rate=2e-4,
        num_train_epochs=3
    )


def get_qwen_config() -> LoraConfig:
    """获取 Qwen 配置"""
    return LoraConfig(
        model_name="Qwen/Qwen-7B-Chat",
        lora_r=16,
        lora_alpha=32,
        learning_rate=2e-4,
        num_train_epochs=3
    )


def get_deepseek_config() -> LoraConfig:
    """获取 DeepSeek 配置"""
    return LoraConfig(
        model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
        lora_r=16,
        lora_alpha=32,
        learning_rate=2e-4,
        num_train_epochs=3
    )


async def quick_train(
    data_path: str,
    output_dir: str = "./output/finetune",
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
    use_4bit: bool = True
) -> TrainingResult:
    """快速训练函数"""
    from .data import load_dataset

    # 加载数据
    train_data, _ = await load_dataset(data_path, format_type="alpaca")

    # 创建训练器
    config = LoraConfig(
        model_name=model_name,
        output_dir=output_dir,
        use_4bit=use_4bit
    )

    trainer = LoRATrainer(config)
    result = trainer.train(train_data)

    return result
