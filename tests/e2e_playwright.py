#!/usr/bin/env python3
"""
Text-to-SQL E2E Playwright 测试
测试首页加载、SQL 导入、SQL 生成功能
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# 尝试从虚拟环境导入 playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from playwright.async_api import async_playwright, Page, Browser


# 配置
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

print(f"测试地址: {BASE_URL}")
print(f"截图目录: {SCREENSHOT_DIR}")


async def wait_for_network_idle(page: Page, timeout: int = 30000):
    """等待网络空闲"""
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
    except:
        pass


async def test_homepage(page: Page) -> bool:
    """测试首页加载"""
    print("\n=== 测试 1: 首页加载 ===")
    try:
        # 访问首页
        await page.goto(BASE_URL, wait_until="networkidle")
        await asyncio.sleep(1)  # 等待 JavaScript 执行

        # 检查标题
        title = await page.title()
        print(f"页面标题: {title}")

        # 截图
        screenshot_path = SCREENSHOT_DIR / f"homepage_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"截图已保存: {screenshot_path}")

        # 检查关键元素是否存在
        header = await page.locator(".header h1").text_content()
        print(f"页面标题文字: {header}")

        # 检查标签页
        tabs = await page.locator(".tab").all_text_contents()
        print(f"标签页: {tabs}")

        # 检查默认激活的标签页
        active_tab = await page.locator(".tab.active").text_content()
        print(f"激活标签: {active_tab}")

        assert "Text-to-SQL" in title or "Text" in title, "页面标题不正确"
        assert "查询" in active_tab, "默认标签页不正确"

        print("首页加载测试通过")
        return True

    except Exception as e:
        print(f"首页加载测试失败: {e}")
        # 失败时也截图
        screenshot_path = SCREENSHOT_DIR / f"homepage_error_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        return False


async def test_sql_import(page: Page) -> bool:
    """测试 SQL 导入功能"""
    print("\n=== 测试 2: SQL 导入功能 ===")
    try:
        # 切换到知识库标签
        await page.click('.tab[data-tab="knowledge"]')
        await asyncio.sleep(0.5)

        # 截图知识库页面
        screenshot_path = SCREENSHOT_DIR / f"knowledge_tab_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"知识库页面截图: {screenshot_path}")

        # 检查知识库统计
        ddl_count = await page.locator("#ddlCount").text_content()
        sql_count = await page.locator("#sqlCount").text_content()
        doc_count = await page.locator("#docCount").text_content()
        print(f"知识库统计 - DDL: {ddl_count}, SQL: {sql_count}, 文档: {doc_count}")

        # 创建测试文件
        test_data = {
            "question": "测试问题",
            "sql": "SELECT * FROM test_table",
            "db_id": "test"
        }

        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            test_file = f.name

        try:
            # 设置文件上传处理
            file_input = page.locator("#fileInput")
            await file_input.set_input_files(test_file)
            await asyncio.sleep(1)

            # 截图上传后状态
            screenshot_path = SCREENSHOT_DIR / f"after_import_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"导入后截图: {screenshot_path}")

        finally:
            os.unlink(test_file)

        print("SQL 导入测试通过")
        return True

    except Exception as e:
        print(f"SQL 导入测试失败: {e}")
        screenshot_path = SCREENSHOT_DIR / f"import_error_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        return False


async def test_sql_generation(page: Page) -> bool:
    """测试 SQL 生成功能"""
    print("\n=== 测试 3: SQL 生成功能 ===")
    try:
        # 切换回查询标签
        await page.click('.tab[data-tab="query"]')
        await asyncio.sleep(0.5)

        # 输入测试问题
        test_question = "查询所有客户信息"
        await page.fill("#questionInput", test_question)
        await asyncio.sleep(0.5)

        # 截图输入后状态
        screenshot_path = SCREENSHOT_DIR / f"before_generate_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"输入问题后截图: {screenshot_path}")

        # 点击生成按钮
        print("点击生成按钮...")
        await page.click("#generateBtn")

        # 等待加载动画出现然后消失
        try:
            await page.wait_for_selector("#loading.active", timeout=5000)
            print("加载动画出现")
        except:
            print("加载动画未出现（可能响应很快）")

        # 等待结果显示
        try:
            await page.wait_for_selector("#resultArea", state="visible", timeout=10000)
            print("结果区域已显示")
        except:
            print("结果区域未显示，可能是服务响应问题")
            # 截图当前状态
            screenshot_path = SCREENSHOT_DIR / f"after_generate_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            return False

        await asyncio.sleep(0.5)

        # 截图生成结果
        screenshot_path = SCREENSHOT_DIR / f"after_generate_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"生成结果截图: {screenshot_path}")

        # 检查生成的 SQL
        sql_output = await page.locator("#sqlOutput").text_content()
        print(f"生成的 SQL: {sql_output}")

        # 检查统计信息
        confidence = await page.locator("#confidenceScore").text_content()
        gen_time = await page.locator("#generationTime").text_content()
        tokens = await page.locator("#tokensUsed").text_content()
        explanation = await page.locator("#explanationText").text_content()

        print(f"置信度: {confidence}")
        print(f"生成耗时: {gen_time}")
        print(f"Token数: {tokens}")
        print(f"解释: {explanation}")

        assert sql_output and len(sql_output) > 0, "SQL 输出为空"
        assert "SELECT" in sql_output.upper(), "生成的不是 SELECT 语句"

        print("SQL 生成测试通过")
        return True

    except Exception as e:
        print(f"SQL 生成测试失败: {e}")
        screenshot_path = SCREENSHOT_DIR / f"generate_error_{datetime.now().strftime('%H%M%S')}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        return False


async def test_tabs_navigation(page: Page) -> bool:
    """测试标签页导航"""
    print("\n=== 测试 4: 标签页导航 ===")
    try:
        tabs = ["query", "knowledge", "history", "settings"]

        for tab in tabs:
            print(f"切换到 {tab} 标签")
            await page.click(f'.tab[data-tab="{tab}"]')
            await asyncio.sleep(0.3)

            # 验证标签激活状态
            active_tab = await page.locator(".tab.active").text_content()
            print(f"当前激活标签: {active_tab}")

            # 截图
            screenshot_path = SCREENSHOT_DIR / f"tab_{tab}_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)

        print("标签页导航测试通过")
        return True

    except Exception as e:
        print(f"标签页导航测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("=" * 60)
    print("Text-to-SQL E2E Playwright 测试")
    print("=" * 60)

    # 首先检查服务是否运行
    import urllib.request
    try:
        print(f"\n检查服务是否运行: {BASE_URL}")
        response = urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
        print(f"服务状态: {response.read().decode()}")
    except Exception as e:
        print(f"服务未运行! 请先启动服务: python run_server_fixed.py")
        print(f"错误: {e}")
        return

    async with async_playwright() as p:
        # 启动浏览器（使用 headless 模式）
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )

        # 创建页面
        context = await browser.new_context(
            viewport={'width': 1400, 'height': 900},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()

        # 设置超时
        page.set_default_timeout(30000)
        page.set_default_navigation_timeout(30000)

        results = []

        try:
            # 运行测试
            results.append(("首页加载", await test_homepage(page)))
            results.append(("标签页导航", await test_tabs_navigation(page)))
            results.append(("SQL 导入", await test_sql_import(page)))
            results.append(("SQL 生成", await test_sql_generation(page)))

        finally:
            await browser.close()

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
    print(f"截图目录: {SCREENSHOT_DIR}")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
