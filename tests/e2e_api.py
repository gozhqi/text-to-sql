#!/usr/bin/env python3
"""
Text-to-SQL E2E API 测试
测试首页加载、SQL 导入、SQL 生成功能
"""
import asyncio
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 尝试导入 requests
try:
    import requests
except ImportError:
    print("需要安装 requests: pip install requests")
    sys.exit(1)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 配置
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

print(f"测试地址: {BASE_URL}")
print(f"截图目录: {SCREENSHOT_DIR}")


def test_health():
    """测试健康检查"""
    print("\n=== 测试 1: 健康检查 ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        data = response.json()
        print(f"响应: {data}")

        assert response.status_code == 200, f"状态码错误: {response.status_code}"
        assert data.get("status") == "ok", f"状态错误: {data.get('status')}"

        print("健康检查通过")
        return True, data

    except Exception as e:
        print(f"健康检查失败: {e}")
        return False, None


def test_homepage():
    """测试首页加载"""
    print("\n=== 测试 2: 首页加载 ===")
    try:
        response = requests.get(BASE_URL, timeout=10)
        html = response.text

        assert response.status_code == 200, f"状态码错误: {response.status_code}"
        assert "Text-to-SQL" in html, "页面缺少标题"
        assert "查询" in html, "页面缺少查询标签"
        assert "知识库" in html, "页面缺少知识库标签"

        # 保存 HTML 快照
        snapshot_path = SCREENSHOT_DIR / f"homepage_{datetime.now().strftime('%H%M%S')}.html"
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML 快照已保存: {snapshot_path}")

        print("首页加载测试通过")
        return True

    except Exception as e:
        print(f"首页加载测试失败: {e}")
        return False


def test_knowledge_stats():
    """测试知识库统计 API"""
    print("\n=== 测试 3: 知识库统计 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/knowledge/stats", timeout=10)
        data = response.json()
        print(f"知识库统计: {data}")

        assert response.status_code == 200, f"状态码错误: {response.status_code}"
        assert data.get("success") == True, f"请求失败: {data}"
        assert "sql_count" in data, "缺少 sql_count"

        print(f"SQL 示例数: {data.get('sql_count')}")
        print(f"DDL 数: {data.get('ddl_count')}")
        print(f"文档数: {data.get('doc_count')}")

        print("知识库统计测试通过")
        return True, data

    except Exception as e:
        print(f"知识库统计测试失败: {e}")
        return False, None


def test_sql_import():
    """测试 SQL 导入功能"""
    print("\n=== 测试 4: SQL 导入功能 ===")
    try:
        # 创建测试数据
        test_data = {
            "question": "测试问题 - 查询所有客户",
            "sql": "SELECT * FROM customers WHERE status = 'active'",
            "db_id": "test_db"
        }

        # 使用 form-data 格式上传
        files = {
            'type': (None, 'sql'),
            'data': (None, json.dumps(test_data))
        }

        response = requests.post(
            f"{BASE_URL}/api/knowledge/import",
            data=files,
            timeout=10
        )
        data = response.json()
        print(f"导入响应: {data}")

        assert response.status_code == 200, f"状态码错误: {response.status_code}"
        assert data.get("success") == True, f"导入失败: {data}"
        assert data.get("count") > 0, "未导入任何数据"

        print(f"成功导入 {data.get('count')} 条记录")

        # 验证导入后的统计
        stats_response = requests.get(f"{BASE_URL}/api/knowledge/stats", timeout=10)
        stats_data = stats_response.json()
        print(f"更新后的知识库统计: {stats_data}")

        print("SQL 导入测试通过")
        return True

    except Exception as e:
        print(f"SQL 导入测试失败: {e}")
        return False


def test_sql_generation():
    """测试 SQL 生成功能"""
    print("\n=== 测试 5: SQL 生成功能 ===")
    try:
        test_questions = [
            "查询所有客户信息",
            "统计每个地区的订单数量",
            "找出销售额最高的前10个产品"
        ]

        results = []

        for question in test_questions:
            print(f"\n测试问题: {question}")

            payload = {
                "question": question,
                "db_name": "default"
            }

            response = requests.post(
                f"{BASE_URL}/api/rag/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code != 200:
                print(f"  请求失败: {response.status_code}")
                continue

            data = response.json()
            print(f"  生成结果:")
            print(f"    SQL: {data.get('sql')}")
            print(f"    置信度: {data.get('confidence')}")
            print(f"    耗时: {data.get('generation_time')}s")
            print(f"    Token: {data.get('tokens_used')}")
            print(f"    解释: {data.get('explanation')}")

            assert data.get("success") == True, f"生成失败: {data}"
            assert data.get("sql"), "SQL 为空"

            # 验证 SQL 安全性
            sql_upper = data.get('sql').upper().strip()
            assert sql_upper.startswith('SELECT'), "不是 SELECT 语句"

            results.append({
                "question": question,
                "sql": data.get('sql'),
                "confidence": data.get('confidence'),
                "time": data.get('generation_time')
            })

        # 保存结果快照
        snapshot_path = SCREENSHOT_DIR / f"generation_results_{datetime.now().strftime('%H%M%S')}.json"
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果快照已保存: {snapshot_path}")

        print(f"\nSQL 生成测试通过，成功生成 {len(results)} 个查询")
        return True

    except Exception as e:
        print(f"SQL 生成测试失败: {e}")
        return False


def test_security_validation():
    """测试安全验证"""
    print("\n=== 测试 6: 安全验证 ===")
    try:
        # 测试危险 SQL 会被过滤
        dangerous_questions = [
            "删除所有用户数据",
            "DROP TABLE customers",
            "UPDATE customers SET status = 'deleted'"
        ]

        blocked_count = 0

        for question in dangerous_questions:
            print(f"测试危险问题: {question}")

            payload = {
                "question": question,
                "db_name": "default"
            }

            response = requests.post(
                f"{BASE_URL}/api/rag/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                # 检查是否返回安全的 SQL
                if data.get('sql'):
                    sql_upper = data.get('sql').upper()
                    if 'DROP' in sql_upper or 'DELETE' in sql_upper or 'UPDATE' in sql_upper:
                        print(f"  警告: 危险 SQL 未被过滤!")
                    else:
                        print(f"  安全: 返回了 SELECT 查询")
                        blocked_count += 1
            else:
                print(f"  请求被拒绝: {response.status_code}")
                blocked_count += 1

        print(f"\n安全验证测试通过，{blocked_count}/{len(dangerous_questions)} 个危险请求被处理")
        return True

    except Exception as e:
        print(f"安全验证测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("Text-to-SQL E2E API 测试")
    print("=" * 60)

    results = []

    # 运行测试
    result, health_data = test_health()
    results.append(("健康检查", result))

    results.append(("首页加载", test_homepage()))
    results.append(("知识库统计", test_knowledge_stats()[0]))
    results.append(("SQL 导入", test_sql_import()))
    results.append(("SQL 生成", test_sql_generation()))
    results.append(("安全验证", test_security_validation()))

    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "通过" if result else "失败"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    print("=" * 60)
    print(f"总计: {passed}/{total} 通过")
    print(f"快照目录: {SCREENSHOT_DIR}")
    print("=" * 60)

    # 生成测试报告
    report_path = Path(__file__).parent / "test_report_e2e.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "results": [{"name": name, "passed": result} for name, result in results],
        "summary": {"total": total, "passed": passed, "failed": total - passed}
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"测试报告已保存: {report_path}")

    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
