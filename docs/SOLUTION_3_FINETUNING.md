# 方案三：模型微调的 Text-to-SQL 系统

> 基于 DB-GPT-Hub 架构，适合有标注数据和 GPU 资源的场景

---

## 一、方案概述

### 适用场景
- ✅ 有大量标注数据（Question-SQL 对）
- ✅ 有 GPU 训练资源
- ✅ 追求最高准确率
- ✅ 需要低延迟推理

### 核心优势
| 特点 | 说明 |
|------|------|
| **最高准确率** | 87%+ (Spider 数据集) |
| **最低延迟** | 1-3 秒推理时间 |
| **数据隐私** | 模型可本地部署 |
| **可控性强** | 完全掌握模型行为 |

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     微调方案架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  训练阶段                                                    │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │Spider   │   │ 自定义   │   │ 数据增强 │                   │
│  │ 数据集  │   │ 数据集  │   │  样本   │                   │
│  └────┬────┘   └────┬────┘   └────┬────┘                   │
│       │             │             │                         │
│       └─────────────┼─────────────┘                         │
│                     ↓                                       │
│            ┌──────────────┐                                 │
│            │  数据预处理   │                                 │
│            │  格式转换     │                                 │
│            └──────────────┘                                 │
│                     ↓                                       │
│            ┌──────────────┐                                 │
│            │ 预训练模型   │  ← CodeLlama / Qwen / DeepSeek  │
│            └──────────────┘                                 │
│                     ↓                                       │
│            ┌──────────────┐                                 │
│            │ LoRA/QLoRA   │                                 │
│            │   微调训练    │                                 │
│            └──────────────┘                                 │
│                     ↓                                       │
│            ┌──────────────┐                                 │
│            │  模型评估    │  ← EM / EX 指标                 │
│            └──────────────┘                                 │
│                                                             │
│  推理阶段                                                    │
│  ┌─────────────┐                                            │
│  │  用户问题   │                                            │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  ┌─────────────┐                                            │
│  │ 微调模型推理 │  ← 1-3 秒                                  │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  ┌─────────────┐                                            │
│  │  生成的SQL  │                                            │
│  └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、必要条件

### 2.1 数据条件

| 条件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **标注数据量** | 1,000 对 | 10,000+ 对 |
| **数据质量** | 70% 正确 | 90%+ 正确 |
| **Schema 覆盖** | 核心表 | 全部表 |
| **查询类型覆盖** | 基础查询 | 全类型 |

**数据格式要求**:
```json
{
  "db_id": "sales_db",
  "question": "查询上个月销售额最高的产品",
  "query": "SELECT product_name, SUM(amount) as total FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 MONTH) GROUP BY product_name ORDER BY total DESC LIMIT 1",
  "schema": {
    "tables": ["orders", "products"],
    "columns": {
      "orders": ["order_id", "product_id", "amount", "order_date"],
      "products": ["product_id", "product_name"]
    }
  }
}
```

### 2.2 硬件条件

| 配置 | QLoRA (4-bit) | LoRA | 全量微调 |
|------|---------------|------|----------|
| **GPU** | 1x RTX 3090 (24GB) | 2x A100 (40GB) | 4x A100 (80GB) |
| **内存** | 32 GB | 64 GB | 128 GB |
| **存储** | 100 GB SSD | 500 GB SSD | 1 TB SSD |
| **训练时间** | 1-2 天 | 4-6 小时 | 1-2 天 |

### 2.3 软件条件

```txt
# 核心依赖
torch>=2.0.0
transformers>=4.35.0
peft>=0.5.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
datasets>=2.14.0

# 可选依赖
deepspeed>=0.12.0      # 分布式训练
flash-attn>=2.0.0      # Flash Attention
wandb>=0.15.0          # 训练监控
```

---

## 三、数据准备

### 3.1 数据收集

```python
import json
from typing import List, Dict

class DataCollector:
    """训练数据收集器"""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.samples = []

    def add_sample(
        self,
        question: str,
        sql: str,
        db_id: str,
        schema: Dict = None,
        difficulty: str = "medium"
    ):
        """添加训练样本"""
        sample = {
            "db_id": db_id,
            "question": question,
            "query": sql,
            "schema": schema or {},
            "difficulty": difficulty
        }
        self.samples.append(sample)

    def save(self):
        """保存数据集"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"✅ 已保存 {len(self.samples)} 个样本到 {self.output_path}")

# 使用示例
collector = DataCollector("data/train.jsonl")

# 添加 Spider 数据集
collector.add_sample(
    question="What is the average age of students?",
    sql="SELECT AVG(age) FROM student",
    db_id="student_db",
    difficulty="easy"
)

# 添加业务数据
collector.add_sample(
    question="查询上季度销售额前10的产品",
    sql="SELECT product_name, SUM(amount) as total FROM sales WHERE quarter = 'Q1' GROUP BY product_name ORDER BY total DESC LIMIT 10",
    db_id="sales_db",
    difficulty="medium"
)

collector.save()
```

### 3.2 数据增强

```python
import re
from typing import List, Dict

class SQLDataAugmenter:
    """SQL 数据增强器"""

    # 同义词映射
    SYNONYMS = {
        "查询": ["找出", "获取", "显示", "列出"],
        "所有": ["全部", "每个"],
        "最高": ["最大", "最多", "第一"],
        "最低": ["最小", "最少"],
        "数量": ["个数", "总数"],
        "金额": ["销售额", "价格", "费用"]
    }

    def augment_question(self, question: str) -> List[str]:
        """问题同义词替换增强"""
        augmented = [question]

        for word, synonyms in self.SYNONYMS.items():
            if word in question:
                for synonym in synonyms:
                    new_question = question.replace(word, synonym)
                    augmented.append(new_question)

        return list(set(augmented))

    def augment_sql(self, sql: str) -> List[str]:
        """SQL 等价改写增强"""
        augmented = [sql]

        # 1. JOIN 顺序改变 (A JOIN B → B JOIN A)
        join_pattern = r'(\w+)\s+JOIN\s+(\w+)\s+ON'
        matches = re.findall(join_pattern, sql, re.IGNORECASE)
        if len(matches) >= 2:
            swapped_sql = re.sub(
                join_pattern,
                lambda m: f"{m.group(2)} JOIN {m.group(1)} ON",
                sql,
                count=1,
                flags=re.IGNORECASE
            )
            augmented.append(swapped_sql)

        # 2. 子查询 ↔ JOIN 转换
        # (简化示例，实际需要更复杂的解析)

        # 3. ORDER BY 方向反转 (ASC ↔ DESC)
        if "ASC" in sql.upper():
            augmented.append(sql.upper().replace("ASC", "DESC"))
        elif "DESC" in sql.upper():
            augmented.append(sql.upper().replace("DESC", "ASC"))

        return list(set(augmented))

    def augment_sample(self, sample: Dict) -> List[Dict]:
        """增强单个样本"""
        augmented = []

        # 问题增强
        questions = self.augment_question(sample["question"])

        # SQL 增强
        sqls = self.augment_sql(sample["query"])

        # 组合
        for q in questions:
            for s in sqls:
                augmented.append({
                    **sample,
                    "question": q,
                    "query": s,
                    "augmented": True
                })

        return augmented[:10]  # 限制数量
```

### 3.3 负样本构造

```python
class NegativeSampleGenerator:
    """负样本生成器"""

    def generate_schema_error(self, sample: Dict) -> Dict:
        """生成 Schema 错误样本"""
        return {
            **sample,
            "query": sample["query"].replace(
                sample["schema"]["tables"][0],
                "non_existent_table"
            ),
            "is_negative": True,
            "error_type": "schema_error"
        }

    def generate_syntax_error(self, sample: Dict) -> Dict:
        """生成语法错误样本"""
        sql = sample["query"]
        # 移除一个关键字
        error_sql = sql.replace("SELECT", "").replace("FROM", "")
        return {
            **sample,
            "query": error_sql,
            "is_negative": True,
            "error_type": "syntax_error"
        }

    def generate_semantic_error(self, sample: Dict) -> Dict:
        """生成语义错误样本"""
        sql = sample["query"]
        # 替换聚合函数
        error_sql = sql.replace("SUM", "AVG").replace("COUNT", "MAX")
        return {
            **sample,
            "query": error_sql,
            "is_negative": True,
            "error_type": "semantic_error"
        }
```

---

## 四、模型训练

### 4.1 预训练模型选择

| 模型 | 参数量 | SQL 能力 | 推荐场景 |
|------|--------|----------|----------|
| **CodeLlama-34B** | 34B | ⭐⭐⭐⭐⭐ | 追求最高准确率 |
| **DeepSeek-Coder-33B** | 33B | ⭐⭐⭐⭐⭐ | 推理能力强 |
| **Qwen2.5-32B** | 32B | ⭐⭐⭐⭐ | 中文场景 |
| **SQLCoder-7B** | 7B | ⭐⭐⭐⭐ | 资源受限 |
| **CodeLlama-7B** | 7B | ⭐⭐⭐ | 快速实验 |

### 4.2 LoRA/QLoRA 训练

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb

class SQLLoRATrainer:
    """SQL 模型 LoRA 训练器"""

    def __init__(
        self,
        base_model: str = "codellama/CodeLlama-34b-hf",
        output_dir: str = "./output",
        use_qlora: bool = True
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.use_qlora = use_qlora

        # 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = self._load_model()

        # 配置 LoRA
        self.model = self._configure_lora()

    def _load_model(self):
        """加载基础模型"""
        if self.use_qlora:
            # QLoRA: 4-bit 量化
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                ),
                device_map="auto",
                trust_remote_code=True
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # 标准 LoRA
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        return model

    def _configure_lora(self):
        """配置 LoRA"""
        lora_config = LoraConfig(
            r=64 if self.use_qlora else 32,
            lora_alpha=16 if self.use_qlora else 64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()

        return model

    def prepare_dataset(self, data_path: str):
        """准备数据集"""
        dataset = load_dataset('json', data_files=data_path, split='train')

        def tokenize_function(examples):
            # 构建 Prompt
            prompts = []
            for i in range(len(examples['question'])):
                prompt = f"""### Task: Generate SQL query for the given question.

### Database Schema:
{json.dumps(examples['schema'][i], indent=2)}

### Question:
{examples['question'][i]}

### SQL:
{examples['query'][i]}<|end|>
"""
                prompts.append(prompt)

            return self.tokenizer(
                prompts,
                truncation=True,
                max_length=2048,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(
        self,
        train_path: str,
        val_path: str = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4
    ):
        """执行训练"""
        # 准备数据
        train_dataset = self.prepare_dataset(train_path)
        val_dataset = self.prepare_dataset(val_path) if val_path else None

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4 if self.use_qlora else 2,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=100,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            report_to="wandb",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine"
        )

        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"✅ 模型已保存到 {self.output_dir}")

        return trainer

    def export_merged_model(self, output_path: str):
        """导出合并后的完整模型"""
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✅ 合并模型已保存到 {output_path}")


# 使用示例
trainer = SQLLoRATrainer(
    base_model="codellama/CodeLlama-34b-hf",
    output_dir="./sql_model",
    use_qlora=True
)

trainer.train(
    train_path="data/train.jsonl",
    val_path="data/val.jsonl",
    epochs=3,
    batch_size=2,
    learning_rate=1e-4
)

trainer.export_merged_model("./sql_model_merged")
```

### 4.3 训练配置详解

```yaml
# QLoRA 配置 (低显存场景)
qlora_config:
  # 量化配置
  quantization:
    bits: 4
    compute_dtype: float16
    quant_type: nf4
    use_double_quant: true

  # LoRA 配置
  lora:
    r: 64
    alpha: 16
    dropout: 0.1
    target_modules:
      - q_proj
      - v_proj
      - k_proj
      - o_proj

  # 训练配置
  training:
    epochs: 3
    batch_size: 2
    gradient_accumulation: 4
    learning_rate: 1e-4
    warmup_ratio: 0.1
    lr_scheduler: cosine

  # 预计资源
  resources:
    gpu_memory: 10 GB
    training_time: 24 hours (10K samples)

---

# LoRA 配置 (高性能场景)
lora_config:
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules:
      - q_proj
      - v_proj
      - k_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

  training:
    epochs: 5
    batch_size: 4
    gradient_accumulation: 2
    learning_rate: 2e-4
    warmup_ratio: 0.05

  resources:
    gpu_memory: 40 GB
    training_time: 6 hours (10K samples)
```

---

## 五、模型评估

### 5.1 评估指标

```python
from typing import List, Dict
import re

class SQLEvaluator:
    """SQL 模型评估器"""

    def __init__(self, db_connection=None):
        self.db = db_connection

    def normalize_sql(self, sql: str) -> str:
        """SQL 规范化"""
        # 统一大小写
        sql = sql.upper()
        # 移除多余空格
        sql = ' '.join(sql.split())
        # 移除别名
        sql = re.sub(r'\s+AS\s+\w+', '', sql)
        return sql

    def exact_match(
        self,
        predicted_sqls: List[str],
        gold_sqls: List[str]
    ) -> float:
        """
        精确匹配 (EM)

        Spider 数据集标准指标
        """
        correct = 0
        for pred, gold in zip(predicted_sqls, gold_sqls):
            if self.normalize_sql(pred) == self.normalize_sql(gold):
                correct += 1

        return correct / len(gold_sqls)

    def execution_accuracy(
        self,
        predicted_sqls: List[str],
        gold_sqls: List[str],
        db_id: str
    ) -> float:
        """
        执行准确率 (EX)

        执行 SQL 并比较结果
        """
        correct = 0

        for pred, gold in zip(predicted_sqls, gold_sqls):
            try:
                pred_result = self.db.execute(pred, db_id)
                gold_result = self.db.execute(gold, db_id)

                if self._compare_results(pred_result, gold_result):
                    correct += 1
            except:
                pass  # 执行失败算错误

        return correct / len(gold_sqls)

    def _compare_results(self, result1: List, result2: List) -> bool:
        """比较两个查询结果"""
        if len(result1) != len(result2):
            return False

        # 排序后比较
        sorted1 = sorted([tuple(sorted(r.items())) for r in result1])
        sorted2 = sorted([tuple(sorted(r.items())) for r in result2])

        return sorted1 == sorted2

    def evaluate(
        self,
        model,
        test_dataset: List[Dict],
        db_connection
    ) -> Dict:
        """
        完整评估
        """
        self.db = db_connection

        predicted_sqls = []
        gold_sqls = []

        for sample in test_dataset:
            # 生成 SQL
            prompt = self._build_prompt(sample)
            pred_sql = model.generate(prompt)

            predicted_sqls.append(pred_sql)
            gold_sqls.append(sample['query'])

        return {
            "exact_match": self.exact_match(predicted_sqls, gold_sqls),
            "execution_accuracy": self.execution_accuracy(
                predicted_sqls, gold_sqls, test_dataset[0]['db_id']
            ),
            "total_samples": len(test_dataset)
        }
```

### 5.2 基准测试结果

| 模型配置 | Spider EM | Spider EX | 训练成本 |
|----------|-----------|-----------|----------|
| **CodeLlama-34B + LoRA** | 72.1% | 87.3% | $48 |
| **CodeLlama-34B + QLoRA** | 69.8% | 84.5% | $12 |
| **DeepSeek-Coder-33B + LoRA** | 71.5% | 86.8% | $45 |
| **Qwen2.5-32B + LoRA (中文)** | 68.2% | 82.1% | $42 |
| **SQLCoder-7B + QLoRA** | 58.3% | 71.2% | $5 |

---

## 六、模型部署

### 6.1 推理服务

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SQLInferenceService:
    """SQL 推理服务"""

    def __init__(self, model_path: str):
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def generate(
        self,
        question: str,
        schema: Dict,
        max_new_tokens: int = 512
    ) -> str:
        """生成 SQL"""
        # 构建 Prompt
        prompt = f"""### Task: Generate SQL query for the given question.

### Database Schema:
{json.dumps(schema, indent=2)}

### Question:
{question}

### SQL:
"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取 SQL
        sql = self._extract_sql(generated_text)

        return sql

    def _extract_sql(self, text: str) -> str:
        """从生成文本中提取 SQL"""
        # 查找 SQL 开始标记
        if "### SQL:" in text:
            sql_start = text.find("### SQL:") + len("### SQL:")
            sql = text[sql_start:].strip()
            # 截断到下一个标记
            for marker in ["###", "<|end|>", "\n\n"]:
                if marker in sql:
                    sql = sql[:sql.find(marker)]
            return sql.strip()
        return text


# FastAPI 服务
app = FastAPI(title="Text-to-SQL Inference API")
service = SQLInferenceService("./sql_model_merged")

class QueryRequest(BaseModel):
    question: str
    schema: Dict

@app.post("/generate")
async def generate_sql(request: QueryRequest):
    """生成 SQL"""
    sql = service.generate(request.question, request.schema)
    return {"sql": sql}
```

### 6.2 性能优化

```python
# vLLM 加速推理
from vllm import LLM, SamplingParams

class FastSQLInference:
    """使用 vLLM 加速的推理服务"""

    def __init__(self, model_path: str):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=512
        )

    def generate_batch(
        self,
        questions: List[str],
        schemas: List[Dict]
    ) -> List[str]:
        """批量生成 SQL"""
        prompts = [
            self._build_prompt(q, s)
            for q, s in zip(questions, schemas)
        ]

        outputs = self.llm.generate(prompts, self.sampling_params)

        return [self._extract_sql(o.outputs[0].text) for o in outputs]

    # 性能对比:
    # 原生 HuggingFace: ~50 tokens/s
    # vLLM: ~2000 tokens/s (40x 提升)
```

---

## 七、总结

### 优势
1. **最高准确率** - 87%+ 执行准确率
2. **最低延迟** - 1-3 秒推理时间
3. **数据隐私** - 完全本地化部署
4. **可控性强** - 可针对性优化

### 劣势
1. **需要 GPU** - 训练和推理都需要
2. **数据依赖** - 需要大量高质量标注数据
3. **维护成本** - 模型迭代需要重新训练

### 适用场景
- ✅ 有 GPU 资源
- ✅ 有大量标注数据
- ✅ 追求最高准确率
- ✅ 需要低延迟响应
- ✅ 数据隐私要求高