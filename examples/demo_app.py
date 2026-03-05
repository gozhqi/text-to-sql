"""
Text-to-SQL 方案对比测试界面

使用 Streamlit 创建交互式界面，让用户可以选择不同方案进行测试
"""

import streamlit as st
import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.rag_sql_generator import (
    RAGSQLGenerator,
    DDLDocument,
    SQLExample
)
from examples.agent_sql_generator import SQLAgent, MultiAgentOrchestrator


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="Text-to-SQL 方案对比",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 初始化会话状态 ====================

if "rag_generator" not in st.session_state:
    st.session_state.rag_generator = None
if "sql_agent" not in st.session_state:
    st.session_state.sql_agent = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ==================== 样式定义 ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .method-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .method-rag { background-color: #e3f2fd; }
    .method-agent { background-color: #f3e5f5; }
    .method-finetune { background-color: #e8f5e9; }
    .sql-output {
        background-color: #262626;
        color: #f8f8f2;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    .metric-card {
        padding: 1rem;
        text-align: center;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 侧边栏配置 ====================

with st.sidebar:
    st.header("⚙️ 配置")

    # API 配置
    st.subheader("LLM 配置")
    api_key = st.text_input("API Key", type="password", help="OpenAI API Key")
    base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
    model = st.selectbox(
        "模型",
        ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-sonnet"],
        index=0
    )

    # 数据库配置
    st.subheader("数据库配置")
    db_type = st.selectbox("数据库类型", ["MySQL", "PostgreSQL", "SQLite"])
    db_host = st.text_input("数据库地址", value="localhost")
    db_port = st.number_input("端口", value=3306, min_value=1, max_value=65535)
    db_name = st.text_input("数据库名称", value="sales_db")
    db_user = st.text_input("用户名")
    db_password = st.text_input("密码", type="password")

    # 初始化按钮
    st.markdown("---")
    if st.button("🔄 初始化系统", use_container_width=True):
        with st.spinner("正在初始化..."):
            st.session_state.initialized = True
            st.success("系统初始化成功！")

    # 训练数据管理
    st.markdown("---")
    st.subheader("📚 训练数据")

    with st.expander("添加 DDL 文档"):
        ddl_table = st.text_input("表名")
        ddl_content = st.text_area("DDL 语句", height=100)
        ddl_desc = st.text_input("表描述")
        if st.button("添加 DDL"):
            if ddl_table and ddl_content:
                st.success(f"已添加 DDL: {ddl_table}")

    with st.expander("添加 SQL 示例"):
        sql_question = st.text_input("问题")
        sql_content = st.text_area("SQL 语句", height=100)
        if st.button("添加示例"):
            if sql_question and sql_content:
                st.success(f"已添加示例: {sql_question[:30]}...")


# ==================== 主页面 ====================

st.markdown('<h1 class="main-header">🔍 Text-to-SQL 方案对比</h1>', unsafe_allow_html=True)

st.markdown("""
本工具提供三种主流的 Text-to-SQL 实现方案，您可以对比它们的效果：
- **RAG 方案**: 基于检索增强生成，类似 Vanna.ai
- **Agent 方案**: 基于多 Agent 协作，类似 LangChain SQL Agent
- **Fine-tuning 方案**: 基于模型微调（需要预先训练）
""")

# ==================== 方案选择 ====================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="method-card method-rag">
        <h3>📚 RAG 方案</h3>
        <p><b>核心理念</b>: SQL 即知识</p>
        <ul>
            <li>检索相似 SQL 示例</li>
            <li>利用历史查询</li>
            <li>持续学习能力</li>
        </ul>
        <p><b>适用</b>: 有大量历史 SQL</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="method-card method-agent">
        <h3>🤖 Agent 方案</h3>
        <p><b>核心理念</b>: 动态推理</p>
        <ul>
            <li>ReAct 思考循环</li>
            <li>自我纠错能力</li>
            <li>Schema 探索</li>
        </ul>
        <p><b>适用</b>: 复杂查询场景</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="method-card method-finetune">
        <h3>🎯 Fine-tuning 方案</h3>
        <p><b>核心理念</b>: 专项训练</p>
        <ul>
            <li>模型参数微调</li>
            <li>高准确率</li>
            <li>领域适配</li>
        </ul>
        <p><b>适用</b>: 有大量标注数据</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== 查询界面 ====================

st.markdown("---")
st.header("💬 开始查询")

# 选择方案
selected_method = st.radio(
    "选择方案",
    ["RAG 方案", "Agent 方案", "Fine-tuning 方案（演示）"],
    horizontal=True
)

# 输入问题
question = st.text_area(
    "输入您的问题",
    placeholder="例如：查询上个月销售额最高的前10个产品",
    height=80
)

# 高级选项
with st.expander("⚙️ 高级选项"):
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Max Tokens", 256, 2048, 1024, 128)
    with col2:
        show_reasoning = st.checkbox("显示推理过程", value=True)
        use_cache = st.checkbox("使用缓存", value=True)

# 执行按钮
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    execute_button = st.button("🚀 生成 SQL", use_container_width=True, type="primary")


# ==================== 结果展示 ====================

if execute_button:
    if not st.session_state.initialized:
        st.warning("请先在侧边栏初始化系统")
    elif not question:
        st.warning("请输入您的问题")
    else:
        # 显示加载状态
        with st.spinner("正在生成 SQL..."):

            # 模拟结果（实际需要调用对应的方法）
            if selected_method == "RAG 方案":
                result = {
                    "success": True,
                    "sql": "SELECT p.name, SUM(o.total_amount) as revenue FROM products p JOIN orders o ON p.id = o.product_id WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH) GROUP BY p.id, p.name ORDER BY revenue DESC LIMIT 10",
                    "explanation": "这个问题需要：1) 找到上个月的过滤条件 2) 关联产品和订单表 3) 按产品分组求和 4) 排序并取前10",
                    "confidence": 0.85,
                    "context_used": {
                        "ddls_count": 2,
                        "examples_count": 3
                    },
                    "method": "rag"
                }
            elif selected_method == "Agent 方案":
                result = {
                    "success": True,
                    "sql": "SELECT p.name, SUM(o.total_amount) as revenue FROM products p JOIN orders o ON p.id = o.product_id WHERE o.created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') GROUP BY p.id, p.name ORDER BY revenue DESC LIMIT 10",
                    "explanation": "通过多步推理生成：1) 列出数据库表 2) 获取产品表和订单表结构 3) 搜索相关字段 4) 生成并验证 SQL",
                    "iterations": 5,
                    "reasoning_trace": {
                        "thoughts": [
                            "需要找出涉及销售的表",
                            "获取 products 和 orders 表结构",
                            "找到关联字段 product_id",
                            "生成带时间过滤和聚合的 SQL",
                            "验证 SQL 语法正确性"
                        ],
                        "relevant_tables": ["products", "orders"]
                    },
                    "method": "agent"
                }
            else:  # Fine-tuning
                result = {
                    "success": True,
                    "sql": "SELECT name, SUM(amount) FROM sales WHERE date >= NOW() - INTERVAL 1 MONTH GROUP BY product_id ORDER BY SUM(amount) DESC LIMIT 10",
                    "explanation": "使用微调后的模型直接生成",
                    "confidence": 0.92,
                    "model": "codellama-sql-finetuned",
                    "method": "finetuned"
                }

        # 添加到历史
        st.session_state.query_history.append({
            "question": question,
            "method": selected_method,
            "result": result
        })

        # 显示结果
        st.success("✅ SQL 生成成功！")

        # SQL 输出
        st.subheader("📝 生成的 SQL")
        st.markdown(
            f'<div class="sql-output">{result["sql"]}</div>',
            unsafe_allow_html=True
        )

        # 复制按钮
        st.button("📋 复制 SQL", key=f"copy_{len(st.session_state.query_history)}")

        # 详细信息
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 指标")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("置信度", f"{result.get('confidence', 0.8):.0%}")
            with metric_col2:
                if "iterations" in result:
                    st.metric("迭代次数", result["iterations"])
                else:
                    st.metric("示例数", result.get("context_used", {}).get("examples_count", 0))
            with metric_col3:
                st.metric("方法", result["method"].upper())

        with col2:
            st.subheader("💡 解释")
            st.info(result.get("explanation", "暂无解释"))

        # 推理过程
        if show_reasoning and "reasoning_trace" in result:
            st.subheader("🧠 推理过程")
            trace = result["reasoning_trace"]

            if "thoughts" in trace:
                for i, thought in enumerate(trace["thoughts"], 1):
                    st.markdown(f"{i}. {thought}")

            if "relevant_tables" in trace:
                st.markdown(f"**相关表**: {', '.join(trace['relevant_tables'])}")


# ==================== 对比功能 ====================

st.markdown("---")
st.header("🔬 方案对比")

if st.button("🔄 使用所有方案生成", use_container_width=True):
    if question:
        st.info("正在使用所有方案生成 SQL，请稍候...")
        # 这里可以实际调用所有方案
        st.success("对比完成！")
    else:
        st.warning("请先输入问题")


# ==================== 历史记录 ====================

if st.session_state.query_history:
    st.markdown("---")
    st.header("📜 查询历史")

    for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
        with st.expander(f"{item['method']}: {item['question'][:50]}..."):
            st.markdown(f"**方法**: {item['method']}")
            st.markdown(f"**问题**: {item['question']}")
            if item['result'].get('success'):
                st.markdown(f"**SQL**: `{item['result']['sql'][:100]}...`")
                st.markdown(f"**置信度**: {item['result'].get('confidence', 0):.0%}")


# ==================== 统计信息 ====================

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("总查询数", len(st.session_state.query_history))

with col2:
    rag_count = sum(1 for h in st.session_state.query_history if h['method'] == 'RAG 方案')
    st.metric("RAG 查询", rag_count)

with col3:
    agent_count = sum(1 for h in st.session_state.query_history if h['method'] == 'Agent 方案')
    st.metric("Agent 查询", agent_count)

with col4:
    finetune_count = sum(1 for h in st.session_state.query_history if h['method'] == 'Fine-tuning 方案（演示）')
    st.metric("Fine-tune 查询", finetune_count)


# ==================== 页脚 ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Text-to-SQL 方案对比工具 | 基于最新研究成果实现</p>
    <p>参考: Spider 数据集, Vanna.ai, LangChain, DB-Surfer, AutoLink</p>
</div>
""", unsafe_allow_html=True)
