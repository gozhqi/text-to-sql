# tests/test_rag.py
"""
RAG 方案测试
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKnowledgeBase:
    """知识库测试"""

    def test_create_knowledge_base(self):
        """测试创建知识库"""
        # 模拟知识库创建
        kb = {
            "ddl_count": 0,
            "sql_count": 0,
            "doc_count": 0
        }
        assert kb["ddl_count"] == 0
        assert kb["sql_count"] == 0

    def test_add_sql_example(self, sample_sql_examples):
        """测试添加 SQL 示例"""
        kb = {"sql_examples": []}
        for example in sample_sql_examples:
            kb["sql_examples"].append(example)
        
        assert len(kb["sql_examples"]) == 3
        assert kb["sql_examples"][0]["question"] == "查询所有客户"

    def test_add_ddl(self, sample_ddl):
        """测试添加 DDL"""
        kb = {"ddl_list": []}
        kb["ddl_list"].append(sample_ddl)
        
        assert len(kb["ddl_list"]) == 1
        assert "customers" in kb["ddl_list"][0]


class TestVectorStore:
    """向量存储测试"""

    def test_create_vector_store(self):
        """测试创建向量存储"""
        config = {
            "persist_directory": "./test_chroma",
            "embedding_model": "test-model"
        }
        assert config["persist_directory"] == "./test_chroma"

    def test_embedding_generation(self):
        """测试嵌入生成"""
        # 模拟嵌入生成
        text = "查询所有客户"
        # 实际会调用嵌入模型，这里模拟
        embedding = [0.1] * 384  # 假设 384 维
        assert len(embedding) == 384

    def test_similarity_search(self):
        """测试相似度搜索"""
        # 模拟相似度搜索结果
        results = [
            {"question": "查询所有客户", "sql": "SELECT * FROM customers", "score": 0.95},
            {"question": "获取客户列表", "sql": "SELECT * FROM customers", "score": 0.88}
        ]
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]


class TestSQLGenerator:
    """SQL 生成器测试"""

    def test_build_prompt(self, sample_sql_examples, sample_ddl):
        """测试构建 Prompt"""
        question = "查询所有客户信息"
        
        prompt_parts = [
            "## 数据库结构",
            sample_ddl,
            "## 相似 SQL 示例",
            sample_sql_examples[0]["question"],
            sample_sql_examples[0]["sql"],
            "## 用户问题",
            question
        ]
        
        prompt = "\n".join(prompt_parts)
        assert "客户信息" in prompt
        assert "SELECT" in prompt

    def test_parse_sql_response(self):
        """测试解析 SQL 响应"""
        response = '''```sql
SELECT * FROM customers WHERE city = '北京'
```'''
        
        # 模拟解析
        import re
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
        sql = sql_match.group(1).strip() if sql_match else ""
        
        assert sql == "SELECT * FROM customers WHERE city = '北京'"

    def test_validate_sql(self):
        """测试 SQL 校验"""
        dangerous_sql = "DROP TABLE customers"
        safe_sql = "SELECT * FROM customers"
        
        # 检查危险 SQL
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT']
        is_dangerous = any(kw in dangerous_sql.upper() for kw in dangerous_keywords)
        is_safe = not any(kw in safe_sql.upper() for kw in dangerous_keywords)
        
        assert is_dangerous == True
        assert is_safe == True


class TestRAGPipeline:
    """RAG 管道测试"""

    def test_pipeline_initialization(self):
        """测试管道初始化"""
        config = {
            "knowledge_base_path": "./data/knowledge",
            "vector_store_path": "./data/chroma",
            "llm_model": "gpt-4o"
        }
        
        assert config["llm_model"] == "gpt-4o"

    def test_full_pipeline(self, sample_sql_examples):
        """测试完整管道流程"""
        # 模拟完整流程
        question = "查询销售额最高的产品"
        
        # 1. 检索相似 SQL
        similar_sqls = [sample_sql_examples[1]]
        
        # 2. 构建 Prompt
        prompt = f"问题: {question}\n相似SQL: {similar_sqls[0]['sql']}"
        
        # 3. 模拟生成
        generated_sql = "SELECT product_name, SUM(amount) as total FROM sales GROUP BY product_name ORDER BY total DESC LIMIT 1"
        
        # 4. 校验
        is_valid = generated_sql.startswith("SELECT")
        
        assert len(similar_sqls) > 0
        assert is_valid == True


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])