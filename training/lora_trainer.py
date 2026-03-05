"""
Text-to-SQL Training Pipeline - LoRA/QLoRA Trainer Module

LoRA/QLoRA 微调训练器，支持:
1. 多种基座模型 (CodeLlama, DeepSeek, Qwen)
2. LoRA 和 QLoRA 训练
3. 自动化训练流程
4. Checkpoint 管理和评估
"""
import os
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from datetime import datetime
from loguru import logger
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class ModelConfig:
    """模型配置"""
    # 基座模型
    base_model: str = "codellama/CodeLlama-34b-hf"
    # 模型类型
    model_type: Literal["codellama", "deepseek", "qwen", "sqlcoder"] = "codellama"

    # 最大序列长度
    max_length: int = 2048

    # 特殊 token
    pad_token: Optional[str] = None
    eos_token: Optional[str] = None


@dataclass
class LoRAConfig:
    """LoRA 配置"""
    # 是否使用 QLoRA (量化)
    use_qlora: bool = False

    # LoRA 参数
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # 目标模块
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # QLoRA 量化配置
    quantization_bits: int = 4
    quantization_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True


@dataclass
class TrainingConfig:
    """训练配置"""
    # 输出目录
    output_dir: str = "outputs/text2sql_lora"

    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # 学习率
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # 优化器
    optim: str = "adamw_torch"

    # 调度器
    lr_scheduler_type: str = "cosine"

    # 保存和评估
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3

    # 日志
    logging_steps: int = 10
    logging_dir: Optional[str] = None

    # 精度
    bf16: bool = True  # 需要 GPU 支持
    fp16: bool = False

    # 其他
    dataloader_num_workers: int = 4
    group_by_length: bool = True  # 减少填充
    gradient_checkpointing: bool = True


# ============================================================================
# 数据处理
# ============================================================================

class Text2SQLDataset:
    """Text-to-SQL 数据集处理器"""

    def __init__(
        self,
        model_config: ModelConfig,
        max_length: int = 2048
    ):
        self.model_config = model_config
        self.max_length = max_length
        self.tokenizer = None

    def load_tokenizer(self):
        """加载 tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=True,
            use_fast=True
        )

        # 设置特殊 token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model_config.pad_token:
                self.tokenizer.add_special_tokens({"pad_token": self.model_config.pad_token})

        logger.info(f"Loaded tokenizer: {self.model_config.base_model}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """格式化为训练 prompt"""
        # 构建格式
        prompt = f"""<|user|>
{self._build_user_prompt(sample)}
<|end|>
<|assistant|>
{sample['sql']}<|end|>"""

        return prompt

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        """构建用户 prompt"""
        # Schema 描述
        schema_text = self._format_schemas(sample.get("schemas", {}))

        # 问题
        question = sample.get("question", "")

        return f"""## 数据库结构

{schema_text}

## 用户问题

{question}

请生成对应的 SQL 查询语句。"""

    def _format_schemas(self, schemas: Dict[str, Any]) -> str:
        """格式化 Schema"""
        text = ""
        for table_name, table_schema in schemas.items():
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

            if table_schema.get("foreign_keys"):
                text += "\n#### 外键关系:\n"
                for col, ref in table_schema["foreign_keys"].items():
                    text += f"- {col} → {ref}\n"

            text += "\n"

        return text.strip()

    def load_data(self, data_path: str) -> Dataset:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                # 跳过负样本（用于对比学习时可以保留）
                if sample.get("is_negative"):
                    continue
                data.append(sample)

        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return Dataset.from_list(data)

    def tokenize(self, samples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Tokenize 批量数据"""
        prompts = [self.format_prompt(sample) for sample in samples["text"]]

        # Tokenize
        encodings = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # 对于因果语言模型，labels 就是 input_ids
        labels = encodings["input_ids"].clone()

        # 对于填充位置，设置为 -100 (忽略 loss)
        attention_mask = encodings["attention_mask"]
        labels[attention_mask == 0] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    def prepare_datasets(
        self,
        train_path: str,
        val_path: str
    ) -> tuple[Dataset, Dataset]:
        """准备训练数据集"""
        if self.tokenizer is None:
            self.load_tokenizer()

        # 加载原始数据
        train_data = self.load_data(train_path)
        val_data = self.load_data(val_path)

        # 格式化为 prompt
        train_prompts = [self.format_prompt(sample) for sample in train_data]
        val_prompts = [self.format_prompt(sample) for sample in val_data]

        # 创建数据集
        train_dataset = Dataset.from_dict({"text": train_prompts})
        val_dataset = Dataset.from_dict({"text": val_prompts})

        # 应用 tokenization
        train_dataset = train_dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing training data"
        )

        val_dataset = val_dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing validation data"
        )

        logger.info(f"Prepared datasets: train={len(train_dataset)}, val={len(val_dataset)}")

        return train_dataset, val_dataset


# ============================================================================
# 模型准备
# ============================================================================

class ModelLoader:
    """模型加载器"""

    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config

    def load_base_model(self) -> AutoModelForCausalLM:
        """加载基座模型"""
        logger.info(f"Loading base model: {self.model_config.base_model}")

        # QLoRA 需要特殊处理
        if self.lora_config.use_qlora:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.lora_config.quantization_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=self.lora_config.double_quant,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            # 准备 kbit 训练
            model = prepare_model_for_kbit_training(model)

            logger.info("Loaded model with QLoRA quantization")

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            logger.info("Loaded base model")

        # 启用梯度检查点
        model.gradient_checkpointing_enable()

        return model

    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """应用 LoRA"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.lora_rank,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        logger.info(f"Applied LoRA:")
        logger.info(f"  - Rank: {self.lora_config.lora_rank}")
        logger.info(f"  - Alpha: {self.lora_config.lora_alpha}")
        logger.info(f"  - Target modules: {self.lora_config.target_modules}")
        logger.info(f"  - Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"  - Total params: {total_params:,}")

        return model


# ============================================================================
# 训练器
# ============================================================================

class Text2SQLTrainer:
    """Text-to-SQL 训练器"""

    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self, train_path: str, val_path: str):
        """设置训练环境"""
        logger.info("=" * 60)
        logger.info("Setting up Text-to-SQL Training Pipeline")
        logger.info("=" * 60)

        # 1. 准备数据集
        logger.info("\n[1/4] Preparing datasets...")
        dataset_processor = Text2SQLDataset(self.model_config, self.model_config.max_length)
        train_dataset, val_dataset = dataset_processor.prepare_datasets(train_path, val_path)
        self.tokenizer = dataset_processor.tokenizer

        # 2. 加载模型
        logger.info("\n[2/4] Loading model...")
        model_loader = ModelLoader(self.model_config, self.lora_config)
        model = model_loader.load_base_model()
        model = model_loader.apply_lora(model)
        self.model = model

        # 3. 创建训练器
        logger.info("\n[3/4] Creating trainer...")
        self._create_trainer(train_dataset, val_dataset)

        # 4. 设置完成
        logger.info("\n[4/4] Setup complete!")
        logger.info("=" * 60)

    def _create_trainer(self, train_dataset: Dataset, val_dataset: Dataset):
        """创建 Trainer"""
        # 训练参数
        training_args = TrainingArguments(
            # 输出
            output_dir=self.training_config.output_dir,

            # 训练
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,

            # 优化器和学习率
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            optim=self.training_config.optim,
            lr_scheduler_type=self.training_config.lr_scheduler_type,

            # 保存和评估
            save_strategy=self.training_config.save_strategy,
            evaluation_strategy=self.training_config.evaluation_strategy,
            save_total_limit=self.training_config.save_total_limit,

            # 日志
            logging_steps=self.training_config.logging_steps,
            logging_dir=self.training_config.logging_dir,

            # 精度
            bf16=self.training_config.bf16,
            fp16=self.training_config.fp16,

            # 数据加载
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            group_by_length=self.training_config.group_by_length,

            # 梯度检查点
            gradient_checkpointing=self.training_config.gradient_checkpointing,

            # 报告
            report_to=["tensorboard"],
            save_safetensors=True,
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        # 创建 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # 打印训练信息
        total_steps = self.trainer.args.max_steps * self.training_config.gradient_accumulation_steps
        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {self.training_config.num_train_epochs}")
        logger.info(f"  - Batch size: {self.training_config.per_device_train_batch_size}")
        logger.info(f"  - Gradient accumulation: {self.training_config.gradient_accumulation_steps}")
        logger.info(f"  - Effective batch size: {self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps}")
        logger.info(f"  - Learning rate: {self.training_config.learning_rate}")
        logger.info(f"  - Warmup ratio: {self.training_config.warmup_ratio}")

    def train(self):
        """开始训练"""
        logger.info("\n" + "=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60 + "\n")

        # 训练前评估
        logger.info("Running pre-training evaluation...")
        pre_eval = self.trainer.evaluate()
        logger.info(f"Pre-training eval loss: {pre_eval['eval_loss']:.4f}")

        # 开始训练
        train_result = self.trainer.train()

        # 训练后评估
        logger.info("\nRunning post-training evaluation...")
        post_eval = self.trainer.evaluate()
        logger.info(f"Post-training eval loss: {post_eval['eval_loss']:.4f}")

        # 保存最终模型
        logger.info("\nSaving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)

        # 保存训练指标
        metrics = {
            "train_samples": len(self.trainer.train_dataset),
            "eval_samples": len(self.trainer.eval_dataset),
            "pre_train_loss": pre_eval['eval_loss'],
            "post_train_loss": post_eval['eval_loss'],
            "train_loss": train_result.training_loss,
            "train_steps": len(train_result.log_history),
        }

        metrics_path = os.path.join(self.training_config.output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"\nTraining complete! Metrics saved to: {metrics_path}")
        logger.info(f"  - Pre-train loss: {metrics['pre_train_loss']:.4f}")
        logger.info(f"  - Post-train loss: {metrics['post_train_loss']:.4f}")
        logger.info(f"  - Improvement: {metrics['pre_train_loss'] - metrics['post_train_loss']:.4f}")

        return train_result

    def resume(self, checkpoint_path: str):
        """从 checkpoint 恢复训练"""
        logger.info(f"Resuming training from: {checkpoint_path}")
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
        return train_result


# ============================================================================
# 快捷配置
# ============================================================================

def get_qlora_config() -> tuple[ModelConfig, LoRAConfig, TrainingConfig]:
    """获取 QLoRA 推荐配置（低显存）"""
    model_config = ModelConfig(
        base_model="codellama/CodeLlama-34b-hf",
        max_length=2048,
    )

    lora_config = LoRAConfig(
        use_qlora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        quantization_bits=4,
    )

    training_config = TrainingConfig(
        output_dir="outputs/text2sql_qlora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        bf16=True,
    )

    return model_config, lora_config, training_config


def get_lora_config() -> tuple[ModelConfig, LoRAConfig, TrainingConfig]:
    """获取 LoRA 推荐配置（高性能）"""
    model_config = ModelConfig(
        base_model="codellama/CodeLlama-34b-hf",
        max_length=2048,
    )

    lora_config = LoRAConfig(
        use_qlora=False,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.05,
    )

    training_config = TrainingConfig(
        output_dir="outputs/text2sql_lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        bf16=True,
    )

    return model_config, lora_config, training_config


# ============================================================================
# CLI
# ============================================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL LoRA/QLoRA 训练器")
    parser.add_argument("--train-path", required=True, help="训练数据路径 (.jsonl)")
    parser.add_argument("--val-path", required=True, help="验证数据路径 (.jsonl)")
    parser.add_argument("--output-dir", default="outputs/text2sql_lora", help="输出目录")
    parser.add_argument("--base-model", default="codellama/CodeLlama-34b-hf", help="基座模型")
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora", help="训练模式")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--resume", help="从 checkpoint 恢复")

    args = parser.parse_args()

    # 创建配置
    model_config = ModelConfig(base_model=args.base_model)

    if args.mode == "qlora":
        lora_config = LoRAConfig(
            use_qlora=True,
            lora_rank=args.lora_rank // 4 if args.lora_rank > 16 else 8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    else:
        lora_config = LoRAConfig(
            use_qlora=False,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
        )

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # 创建训练器
    trainer = Text2SQLTrainer(model_config, lora_config, training_config)

    # 设置
    trainer.setup(args.train_path, args.val_path)

    # 训练或恢复
    if args.resume:
        trainer.resume(args.resume)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
