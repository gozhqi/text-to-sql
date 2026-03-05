"""
微调方案 - 训练模块

支持 LoRA/QLoRA 微调
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

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
    PeftModel,
    prepare_model_for_kbit_training
)

from ..data.preparation import TrainingExample


class FineTuningMethod(Enum):
    """微调方法"""
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    base_model: str = "codellama/CodeLlama-7b-Instruct-hf"
    output_dir: str = "./data/models/finetuned_sql"

    # 训练配置
    method: FineTuningMethod = FineTuningMethod.LORA
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 其他配置
    max_length: int = 2048
    use_flash_attention: bool = False
    bf16: bool = True
    fp16: bool = False

    # 量化配置 (QLoRA)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


class SQLFineTuner:
    """SQL 模型微调器"""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.trainer = None

    def load_base_model(self):
        """加载基础模型和分词器"""
        logger.info(f"加载基础模型: {self.config.base_model}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            use_fast=True
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto"
        }

        if self.config.method == FineTuningMethod.QLORA:
            # QLoRA 需要量化配置
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs
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
            inference_mode=False,
            bias="none"
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    def _setup_qlora(self):
        """设置 QLoRA"""
        logger.info("设置 QLoRA 微调")

        # 准备模型 for k-bit 训练
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            inference_mode=False,
            bias="none"
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    def prepare_dataset(
        self,
        examples: List[TrainingExample],
        template: str = "alpaca"
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
                padding="max_length",
                return_tensors=None
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
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
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_examples else None,
            save_total_limit=3,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            optim="adamw_torch",
            evaluation_strategy="steps" if eval_examples else "no",
            load_best_model_at_end=bool(eval_examples),
            report_to=["tensorboard"],
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            ddp_find_unused_parameters=False,
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

        # 保存训练配置
        import json
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self._serialize_config(), f, indent=2)

        logger.info("训练完成")

    def _serialize_config(self) -> Dict[str, Any]:
        """序列化配置"""
        return {
            "base_model": self.config.base_model,
            "method": self.config.method.value,
            "num_train_epochs": self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "max_length": self.config.max_length
        }

    def save_for_inference(self, path: str = None):
        """保存用于推理的模型"""
        save_path = path or self.config.output_dir

        if self.peft_model:
            # 保存 LoRA adapter
            self.peft_model.save_pretrained(save_path)
            logger.info(f"LoRA adapter 已保存到: {save_path}")

        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer 已保存到: {save_path}")

    def resume_training(self, checkpoint_path: str):
        """从检查点恢复训练"""
        logger.info(f"从检查点恢复训练: {checkpoint_path}")

        if self.trainer is None:
            raise ValueError("Trainer 未初始化")

        self.trainer.train(resume_from_checkpoint=checkpoint_path)


def create_finetuner(config: TrainingConfig = None) -> SQLFineTuner:
    """创建微调器"""
    return SQLFineTuner(config)
