# tests/test_rag_real.py
"""
RAG 方案真实 API 测试
测试会实际调用 LLM API 和向量数据库
"""
import pytest
import sys
import os
import json
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查环境变量
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")


@pytest.mark.skipif(not API_KEY, reason="需要设置 OPENAI_API_KEY 或 LLM_API_KEY 环境变量")
class TestRAGRealAPI:
    """RAG 方案真实 API 测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """测试前准备"""
        from src.rag.knowledge.base import FileKnowledgeBase
        from src.rag.retrieval.vector_store import VectorStore
        from src.rag.generator.sql_generator import SQLGenerator

        # 创建临时知识库
        self.kb_path = "/tmp/test_kb"
        self.vector_path = "/tmp/test_vector"

        os.makedirs(self.kb_path, exist_ok=True)
        os.makedirs(self.vector_path, exist_ok=True)

        # 初始化知识库
        self.kb = FileKnowledgeBase(self.kb_path)
        self.vector_store = VectorStore(self.vector_path)
        self.generator = SQLGenerator(api_key=API_KEY)

    def test_01_add_ddl_to_knowledge_base(self):
        """测试添加 DDL 到知识库"""
        ddl = """
        CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            customer_name VARCHAR(100),
            city VARCHAR(50),
            created_at DATE
        );
        """

        result = self.kb.add_ddl(ddl, db_name="test_db")
        assert result["success"] == True
        assert result["tables_count"] >= 1

    def test_02_add_sql_example_to_knowledge_base(self):
        """测试添加 SQL 示例到知识库"""
        examples = [
            {
                "question": "查询所有客户",
                "sql": "SELECT * FROM customers",
                "db_name": "test_db"
            },
            {
                "question": "查询北京地区的客户",
                "sql": "SELECT * FROM customers WHERE city = '北京'",
                "db_name": "test_db"
            }
        ]

        for ex in examples:
            result = self.kb.add_sql_example(
                question=ex["question"],
                sql=ex["sql"],
                db_name=ex["db_name"]
            )
            assert result["success"] == True

    def test_03_build_vector_index(self):
        """测试构建向量索引"""
        result = self.vector_store.build_index(self.kb)
        assert result["success"] == True
        assert result["vectors_count"] > 0

    def test_04_retrieve_similar_sql(self):
        """测试检索相似 SQL"""
        query = "查询北京客户"

        results = self.vector_store.search(query, top_k=3)
        assert len(results) > 0
        assert "sql" in results[0]
        assert "score" in results[0]

    def test_05_generate_sql_with_llm(self):
        """测试使用 LLM 生成 SQL"""
        question = "查询所有客户的姓名"
        schema = """
        CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            customer_name VARCHAR(100)
        );
        """

        start_time = time.time()
        result = self.generator.generate(
            question=question,
            schema=schema,
            similar_sqls=[]
        )
        elapsed = time.time() - start_time

        assert result["sql"] != ""
        assert "SELECT" in result["sql"].upper()
        assert elapsed < 30  # 应该在 30 秒内完成

        print(f"\n生成的 SQL: {result['sql']}")
        print(f"耗时: {elapsed:.2f}s")

    def test_06_full_rag_pipeline(self):
        """测试完整 RAG 流程"""
        from src.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            knowledge_base_path=self.kb_path,
            vector_store_path=self.vector_path,
            api_key=API_KEY
        )

        question = "查询北京地区的客户信息"

        start_time = time.time()
        result = pipeline.generate_sql(question, db_name="test_db")
        elapsed = time.time() - start_time

        assert result["success"] == True
        assert result["sql"] != ""
        assert "SELECT" in result["sql"].upper()
        assert result["confidence"] > 0

        print(f"\n问题: {question}")
        print(f"生成的 SQL: {result['sql']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"耗时: {elapsed:.2f}s")

    def test_07_sql_validation(self):
        """测试 SQL 校验"""
        valid_sql = "SELECT * FROM customers WHERE city = '北京'"
        invalid_sql = "DROP TABLE customers"

        # 校验有效 SQL
        result1 = self.generator.validate_sql(valid_sql)
        assert result1["is_valid"] == True

        # 校验危险 SQL
        result2 = self.generator.validate_sql(invalid_sql)
        assert result2["is_valid"] == False
        assert "危险" in result2["error"] or "DROP" in result2["error"]

    def test_08_multi_turn_context(self):
        """测试多轮对话上下文"""
        from src.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            knowledge_base_path=self.kb_path,
            vector_store_path=self.vector_path,
            api_key=API_KEY
        )

        session_id = "test_session_001"

        # 第一轮
        result1 = pipeline.generate_sql(
            "查询客户数量",
            db_name="test_db",
            session_id=session_id
        )
        assert result1["success"] == True

        # 第二轮（引用上一轮）
        result2 = pipeline.generate_sql(
            "按城市分组",
            db_name="test_db",
            session_id=session_id
        )
        assert result2["success"] == True
        assert result2["is_multi_turn"] == True


@pytest.mark.skipif(not API_KEY, reason="需要设置 OPENAI_API_KEY 或 LLM_API_KEY 环境变量")
class TestRAGPerformance:
    """RAG 性能测试"""

    def test_response_time(self):
        """测试响应时间"""
        from src.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(api_key=API_KEY)

        questions = [
            "查询所有客户",
            "统计客户数量",
            "查询北京地区的客户"
        ]

        times = []
        for q in questions:
            start = time.time()
            result = pipeline.generate_sql(q, db_name="test_db")
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"\n平均响应时间: {avg_time:.2f}s")
        print(f"最大响应时间: {max_time:.2f}s")

        assert avg_time < 5  # 平均响应时间应小于 5 秒
        assert max_time < 10  # 最大响应时间应小于 10 秒


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])