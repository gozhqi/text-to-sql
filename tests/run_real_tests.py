# tests/run_real_tests.py
"""
运行真实 API 测试脚本
"""
import os
import sys
import subprocess

def check_environment():
    """检查环境变量"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")

    if api_key:
        print(f"✅ API_KEY 已设置 ({api_key[:10]}...)")
    else:
        print("❌ API_KEY 未设置")
        print("   请设置环境变量: export OPENAI_API_KEY=your_key")
        return False

    return True


def run_tests():
    """运行测试"""
    print("\n" + "=" * 50)
    print("运行真实 API 测试")
    print("=" * 50 + "\n")

    # 测试命令
    test_commands = [
        # RAG 真实测试
        ["python3", "-m", "pytest", "tests/test_rag_real.py", "-v", "-s", "--tb=short"],

        # Agent 真实测试
        ["python3", "-m", "pytest", "tests/test_agent_real.py", "-v", "-s", "--tb=short"],

        # 端到端测试
        ["python3", "-m", "pytest", "tests/test_e2e.py", "-v", "-s", "--tb=short"],
    ]

    results = []

    for cmd in test_commands:
        test_name = cmd[1]
        print(f"\n运行: {test_name}")
        print("-" * 40)

        result = subprocess.run(cmd, cwd="/home/admin/.openclaw/workspace/text-to-sql")
        results.append({
            "test": test_name,
            "returncode": result.returncode
        })

    # 打印结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要")
    print("=" * 50)

    for r in results:
        status = "✅ 通过" if r["returncode"] == 0 else "❌ 失败"
        print(f"{r['test']}: {status}")


if __name__ == "__main__":
    if check_environment():
        run_tests()
    else:
        print("\n请先设置环境变量后再运行测试")
        sys.exit(1)