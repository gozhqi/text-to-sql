# tests/test_e2e.py
"""
端到端集成测试
真实调用完整流程
"""
import pytest
import sys
import os
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


@pytest.mark.skipif(not API_KEY, reason="需要设置 API_KEY 环境变量")
class TestEndToEnd:
    """端到端测试"""

    def test_01_rag_api_endpoint(self):
        """测试 RAG API 端点"""
        url = f"{BASE_URL}/api/rag/generate"

        payload = {
            "question": "查询所有客户",
            "db_name": "test_db"
        }

        response = requests.post(url, json=payload, timeout=60)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "sql" in data
        assert "SELECT" in data["sql"].upper()

        print(f"\n生成的 SQL: {data['sql']}")

    def test_02_agent_api_endpoint(self):
        """测试 Agent API 端点"""
        url = f"{BASE_URL}/api/agent/generate"

        payload = {
            "question": "统计每个城市的订单数量",
            "db_name": "test_db"
        }

        response = requests.post(url, json=payload, timeout=120)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "sql" in data

        print(f"\n生成的 SQL: {data['sql']}")

    def test_03_knowledge_import(self):
        """测试知识导入"""
        url = f"{BASE_URL}/api/knowledge/import"

        # 导入 SQL 示例
        payload = {
            "type": "sql_example",
            "data": {
                "question": "查询活跃客户",
                "sql": "SELECT * FROM customers WHERE status = 'active'",
                "db_name": "test_db"
            }
        }

        response = requests.post(url, json=payload, timeout=30)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

        print(f"\n导入结果: {data}")

    def test_04_multi_turn_conversation(self):
        """测试多轮对话"""
        session_url = f"{BASE_URL}/api/chat"

        session_id = f"test_session_{int(time.time())}"

        # 第一轮
        response1 = requests.post(session_url, json={
            "message": "查询客户数量",
            "session_id": session_id,
            "db_name": "test_db"
        }, timeout=60)

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["success"] == True

        # 第二轮
        response2 = requests.post(session_url, json={
            "message": "按城市分组",
            "session_id": session_id,
            "db_name": "test_db"
        }, timeout=60)

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["success"] == True
        assert data2.get("is_multi_turn") == True

        print(f"\n第一轮 SQL: {data1['sql']}")
        print(f"第二轮 SQL: {data2['sql']}")

    def test_05_schema_endpoint(self):
        """测试 Schema 端点"""
        url = f"{BASE_URL}/api/schema/test_db"

        response = requests.get(url, timeout=30)

        assert response.status_code == 200
        data = response.json()
        assert "tables" in data

        print(f"\n表数量: {len(data['tables'])}")


class TestHealthCheck:
    """健康检查测试"""

    def test_api_health(self):
        """测试 API 健康状态"""
        url = f"{BASE_URL}/health"

        try:
            response = requests.get(url, timeout=5)
            assert response.status_code == 200
            print("\n✅ API 服务正常")
        except requests.exceptions.ConnectionError:
            pytest.skip("API 服务未启动")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])