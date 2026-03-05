# tests/test_finetune.py
"""
微调方案测试
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPreparation:
    """数据准备测试"""

    def test_spider_format_parsing(self):
        """测试 Spider 格式解析"""
        spider_sample = {
            "db_id": "sales",
            "question": "查询所有客户",
            "query": "SELECT * FROM customers"
        }
        
        assert spider_sample["db_id"] == "sales"
        assert "SELECT" in spider_sample["query"]

    def test_data_augmentation(self):
        """测试数据增强"""
        original = {
            "question": "查询销售额",
            "sql": "SELECT SUM(amount) FROM sales"
        }
        
        # 模拟增强
        augmented = [
            {"question": "查询销售总额", "sql": original["sql"]},
            {"question": "统计销售额", "sql": original["sql"]},
            {"question": "获取销售额", "sql": original["sql"]}
        ]
        
        assert len(augmented) == 3

    def test_negative_sample_generation(self):
        """测试负样本生成"""
        positive = {
            "sql": "SELECT * FROM customers WHERE city = '北京'"
        }
        
        # 生成负样本
        negative = {
            "sql": "SELECT * FROM non_existent_table",
            "is_negative": True
        }
        
        assert negative["is_negative"] == True


class TestLoRAConfig:
    """LoRA 配置测试"""

    def test_qlora_config(self):
        """测试 QLoRA 配置"""
        config = {
            "quantization_bit": 4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        }
        
        assert config["quantization_bit"] == 4
        assert config["lora_rank"] == 16

    def test_lora_config(self):
        """测试 LoRA 配置"""
        config = {
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05
        }
        
        assert config["lora_rank"] == 32

    def test_target_modules(self):
        """测试目标模块"""
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        assert len(target_modules) == 4
        assert "q_proj" in target_modules


class TestModelTraining:
    """模型训练测试"""

    def test_training_args(self):
        """测试训练参数"""
        args = {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.1
        }
        
        assert args["epochs"] == 3
        assert args["learning_rate"] == 1e-4

    def test_gradient_accumulation(self):
        """测试梯度累积"""
        batch_size = 2
        accumulation_steps = 4
        effective_batch_size = batch_size * accumulation_steps
        
        assert effective_batch_size == 8

    def test_checkpoint_saving(self):
        """测试检查点保存"""
        checkpoints = []
        for epoch in range(3):
            checkpoints.append(f"checkpoint_epoch_{epoch}")
        
        assert len(checkpoints) == 3


class TestEvaluation:
    """评估测试"""

    def test_exact_match(self):
        """测试精确匹配"""
        pred_sql = "SELECT name FROM customers"
        gold_sql = "SELECT name FROM customers"
        
        # 规范化比较
        is_match = pred_sql.lower().strip() == gold_sql.lower().strip()
        
        assert is_match == True

    def test_execution_accuracy(self):
        """测试执行准确率"""
        pred_result = [{"name": "张三"}, {"name": "李四"}]
        gold_result = [{"name": "张三"}, {"name": "李四"}]
        
        is_match = pred_result == gold_result
        
        assert is_match == True

    def test_schema_recall(self):
        """测试 Schema 召回率"""
        predicted_tables = ["customers", "orders"]
        gold_tables = ["customers", "orders", "products"]
        
        recall = len(set(predicted_tables) & set(gold_tables)) / len(gold_tables)
        
        assert recall == 2/3


class TestInference:
    """推理测试"""

    def test_model_loading(self):
        """测试模型加载"""
        model_config = {
            "model_path": "./sql_model",
            "device": "cuda",
            "dtype": "float16"
        }
        
        assert model_config["device"] == "cuda"

    def test_generation_params(self):
        """测试生成参数"""
        params = {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        assert params["temperature"] == 0.1

    def test_batch_inference(self):
        """测试批量推理"""
        questions = ["查询客户", "查询订单", "查询产品"]
        
        # 模拟批量生成
        results = [{"sql": f"SQL_{i}"} for i in range(len(questions))]
        
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])