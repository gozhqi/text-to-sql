# Google Colab 免费训练 Text-to-SQL 模型教程

## 一、准备工作

### 1.1 打开 Colab
访问：https://colab.research.google.com/

### 1.2 启用 GPU
```
菜单栏 → 运行时 → 更改运行时类型 → 硬件加速器选择 GPU → 保存
```

### 1.3 验证 GPU
```python
!nvidia-smi
# 应该看到 Tesla T4 (16GB)
```

---

## 二、环境安装

### 2.1 安装依赖
```python
!pip install -q torch transformers peft bitsandbytes accelerate datasets
```

### 2.2 验证安装
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 三、准备数据

### 3.1 上传训练数据
```python
from google.colab import files
uploaded = files.upload()  # 上传 train.jsonl
```

### 3.2 或使用示例数据
```python
# 创建示例训练数据
import json

train_data = [
    {
        "db_id": "sales",
        "question": "查询所有客户",
        "query": "SELECT * FROM customers",
        "schema": {"tables": ["customers"]}
    },
    {
        "db_id": "sales",
        "question": "查询销售额最高的产品",
        "query": "SELECT product_name, SUM(amount) as total FROM sales GROUP BY product_name ORDER BY total DESC LIMIT 1",
        "schema": {"tables": ["sales"]}
    },
    {
        "db_id": "sales",
        "question": "统计每个地区的订单数量",
        "query": "SELECT region, COUNT(*) as order_count FROM orders GROUP BY region",
        "schema": {"tables": ["orders"]}
    }
]

with open('train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Created {len(train_data)} training samples")
```

---

## 四、加载模型 (QLoRA)

### 4.1 配置量化
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "codellama/CodeLlama-7b-hf"  # 小模型适合 T4

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 加载模型
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {model_name}")
print(f"Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

### 4.2 配置 LoRA
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 准备模型
model = prepare_model_for_kbit_training(model)

# LoRA 配置
lora_config = LoraConfig(
    r=16,  # 小一点适合 T4
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## 五、训练

### 5.1 准备数据集
```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='train.jsonl', split='train')

def tokenize(examples):
    prompts = [
        f"### Question: {q}\n### SQL: {s}<|end|>"
        for q, s in zip(examples['question'], examples['query'])
    ]
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)
print(f"Dataset size: {len(tokenized_dataset)}")
```

### 5.2 开始训练
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./sql_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # T4 显存小，batch=1
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Starting training...")
trainer.train()
print("Training completed!")
```

---

## 六、测试模型

### 6.1 生成 SQL
```python
def generate_sql(question):
    prompt = f"### Question: {question}\n### SQL:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            top_p=0.9
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试
test_questions = [
    "查询所有订单",
    "找出销售额最高的客户",
    "统计每个产品的销售数量"
]

for q in test_questions:
    print(f"\n问题: {q}")
    print(f"SQL: {generate_sql(q)}")
```

---

## 七、保存模型

### 7.1 保存到 Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# 保存模型
model.save_pretrained("/content/drive/MyDrive/sql_model")
tokenizer.save_pretrained("/content/drive/MyDrive/sql_model")
print("Model saved to Google Drive!")
```

### 7.2 或下载到本地
```python
model.save_pretrained("./sql_model_final")
tokenizer.save_pretrained("./sql_model_final")

# 打包下载
!zip -r sql_model.zip ./sql_model_final
files.download("sql_model.zip")
```

---

## 八、注意事项

### 8.1 Colab 限制
- 免费版每次最长 **12 小时**
- 空闲超过 90 分钟会断开
- GPU 分配不保证（高峰期可能没有）

### 8.2 显存优化
- 使用小模型 (7B)
- batch_size = 1
- 使用 gradient_accumulation
- 使用 4-bit 量化

### 8.3 建议
- 先用小数据测试流程
- 确认可行后再用大数据训练
- 训练完成立即保存模型

---

## 九、完整脚本

将以上代码合并成一个完整脚本，复制到 Colab 运行即可。

完整代码见项目：`training/colab_train.ipynb`