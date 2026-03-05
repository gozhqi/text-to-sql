# tests/test_agent_real.py
"""
Agent 方案真实 API 测试
测试会实际调用 LLM API 和数据库
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
DB_URL = os.environ.get("TEST_DB_URL")  # 如: mysql://user:pass@host/db


@pytest.mark.skipif(not API_KEY, reason="需要设置 API_KEY 环境变量")
class TestAgentRealAPI:
    """Agent 方案真实 API 测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """测试前准备"""
        from src.agent.pipeline import AgentPipeline

        self.pipeline = AgentPipeline(
            api_key=API_KEY,
            db_url=DB_URL,
            max_iterations=10
        )

    def test_01_simple_query(self):
        """测试简单查询"""
        question = "查询所有客户信息"

        start_time = time.time()
        result = self.pipeline.run(question)
        elapsed = time.time() - start_time

        assert result["success"] == True
        assert result["sql"] != ""
        assert "SELECT" in result["sql"].upper()

        print(f"\n问题: {question}")
        print(f"生成的 SQL: {result['sql']}")
        print(f"迭代次数: {result['iterations']}")
        print(f"耗时: {elapsed:.2f}s")

    def test_02_aggregation_query(self):
        """测试聚合查询"""
        question = "统计每个城市的客户数量"

        result = self.pipeline.run(question)

        assert result["success"] == True
        assert "GROUP BY" in result["sql"].upper() or "COUNT" in result["sql"].upper()

        print(f"\n问题: {question}")
        print(f"生成的 SQL: {result['sql']}")

    def test_03_join_query(self):
        """测试 JOIN 查询"""
        question = "查询每个客户的订单总额"

        result = self.pipeline.run(question)

        assert result["success"] == True
        assert "JOIN" in result["sql"].upper()

        print(f"\n问题: {question}")
        print(f"生成的 SQL: {result['sql']}")

    def test_04_error_correction(self):
        """测试错误纠正"""
        # 故意问一个可能导致错误的问题
        question = "查询不存在表的XXX字段"

        result = self.pipeline.run(question)

        # Agent 应该能够识别并处理错误
        assert "error" in result or result["success"] == False or "不存在" in result.get("explanation", "")

        print(f"\n问题: {question}")
        print(f"结果: {result}")

    def test_05_reasoning_steps(self):
        """测试推理步骤"""
        question = "查询销售额最高的前5个产品，并显示产品名称和销售总额"

        result = self.pipeline.run(question)

        assert result["success"] == True
        assert len(result.get("reasoning_steps", [])) > 0

        print(f"\n推理步骤:")
        for i, step in enumerate(result.get("reasoning_steps", []), 1):
            print(f"  {i}. {step}")

    def test_06_tool_calls(self):
        """测试工具调用"""
        question = "查询客户表的结构"

        result = self.pipeline.run(question)

        assert result["success"] == True
        assert "tool_calls" in result
        assert len(result["tool_calls"]) > 0

        print(f"\n工具调用:")
        for call in result["tool_calls"]:
            print(f"  - {call['tool']}: {call.get('result', '')[:50]}...")


@pytest.mark.skipif(not API_KEY, reason="需要设置 API_KEY 环境变量")
class TestAgentPerformance:
    """Agent 性能测试"""

    def test_iteration_count(self):
        """测试迭代次数"""
        from src.agent.pipeline import AgentPipeline

        pipeline = AgentPipeline(api_key=API_KEY, max_iterations=5)

        result = pipeline.run("查询所有客户")

        assert result["iterations"] <= 5
        print(f"\n迭代次数: {result['iterations']}")

    def test_timeout(self):
        """测试超时处理"""
        from src.agent.pipeline import AgentPipeline

        pipeline = AgentPipeline(
            api_key=API_KEY,
            timeout_seconds=30
        )

        start = time.time()
        result = pipeline.run("查询客户信息")
        elapsed = time.time() - start

        assert elapsed < 60  # 应该在超时时间内完成
        print(f"\n耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])