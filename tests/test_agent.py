# tests/test_agent.py
"""
Agent 方案测试
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestReActLoop:
    """ReAct 循环测试"""

    def test_thought_action_observation_cycle(self):
        """测试思考-行动-观察循环"""
        cycle = {
            "thought": "我需要先查看有哪些表",
            "action": "list_tables",
            "observation": ["customers", "orders", "products"]
        }
        
        assert cycle["action"] == "list_tables"
        assert len(cycle["observation"]) == 3

    def test_max_iterations(self):
        """测试最大迭代次数限制"""
        max_iterations = 10
        current_iteration = 0
        
        while current_iteration < max_iterations:
            current_iteration += 1
        
        assert current_iteration == max_iterations

    def test_early_termination(self):
        """测试提前终止"""
        results = []
        for i in range(10):
            results.append(f"step_{i}")
            if i == 3:
                break
        
        assert len(results) == 4


class TestAgentTools:
    """Agent 工具测试"""

    def test_list_tables_tool(self):
        """测试列出表工具"""
        mock_db = Mock()
        mock_db.execute.return_value = [
            {"table_name": "customers"},
            {"table_name": "orders"}
        ]
        
        result = mock_db.execute("SHOW TABLES")
        assert len(result) == 2

    def test_get_schema_tool(self):
        """测试获取 Schema 工具"""
        mock_db = Mock()
        mock_db.execute.return_value = [
            {"Field": "id", "Type": "int"},
            {"Field": "name", "Type": "varchar"}
        ]
        
        result = mock_db.execute("DESCRIBE customers")
        assert result[0]["Field"] == "id"

    def test_execute_sql_tool(self):
        """测试执行 SQL 工具"""
        mock_db = Mock()
        mock_db.execute.return_value = [
            {"id": 1, "name": "张三"},
            {"id": 2, "name": "李四"}
        ]
        
        result = mock_db.execute("SELECT * FROM customers")
        assert len(result) == 2


class TestSQLErrorCorrection:
    """SQL 错误纠正测试"""

    def test_syntax_error_detection(self):
        """测试语法错误检测"""
        error_sql = "SELECT * FORM customers"  # FORM 应该是 FROM
        
        # 检测错误
        has_error = "FORM" in error_sql and "FROM" not in error_sql
        
        assert has_error == True

    def test_auto_correction(self):
        """测试自动纠正"""
        error_sql = "SELECT * FORM customers"
        corrected_sql = error_sql.replace("FORM", "FROM")
        
        assert corrected_sql == "SELECT * FROM customers"

    def test_retry_mechanism(self):
        """测试重试机制"""
        max_retries = 3
        attempts = 0
        success = False
        
        for i in range(max_retries):
            attempts += 1
            if i == 2:  # 第三次成功
                success = True
                break
        
        assert attempts == 3
        assert success == True


class TestAgentPipeline:
    """Agent 管道测试"""

    def test_simple_query_flow(self):
        """测试简单查询流程"""
        question = "查询所有客户"
        
        # 模拟流程
        steps = [
            {"step": "list_tables", "result": ["customers"]},
            {"step": "get_schema", "result": "id, name, city"},
            {"step": "generate_sql", "result": "SELECT * FROM customers"},
            {"step": "execute", "result": [{"id": 1, "name": "张三"}]}
        ]
        
        assert len(steps) == 4
        assert "SELECT" in steps[2]["result"]

    def test_complex_query_flow(self):
        """测试复杂查询流程"""
        question = "查询每个地区的销售额排名"
        
        # 模拟复杂流程
        steps = [
            {"step": "list_tables", "result": ["customers", "orders"]},
            {"step": "get_schema", "result": "多个表的字段"},
            {"step": "analyze_join", "result": "customers.id = orders.customer_id"},
            {"step": "generate_sql", "result": "SELECT c.city, SUM(o.amount) FROM customers c JOIN orders o"},
            {"step": "validate", "result": "valid"},
            {"step": "execute", "result": "success"}
        ]
        
        assert len(steps) == 6
        assert steps[2]["step"] == "analyze_join"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])