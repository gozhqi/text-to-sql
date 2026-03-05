# tests/conftest.py
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_sql_examples():
    """SQL 示例数据"""
    return [
        {
            "question": "查询所有客户",
            "sql": "SELECT * FROM customers",
            "db_id": "sales"
        },
        {
            "question": "查询销售额最高的产品",
            "sql": "SELECT product_name, SUM(amount) as total FROM sales GROUP BY product_name ORDER BY total DESC LIMIT 1",
            "db_id": "sales"
        },
        {
            "question": "统计每个地区的订单数量",
            "sql": "SELECT region, COUNT(*) as order_count FROM orders GROUP BY region",
            "db_id": "sales"
        }
    ]


@pytest.fixture
def sample_ddl():
    """DDL 示例"""
    return """
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        customer_name VARCHAR(100),
        city VARCHAR(50)
    );
    
    CREATE TABLE orders (
        order_id INT PRIMARY KEY,
        customer_id INT,
        amount DECIMAL(10,2),
        order_date DATE
    );
    """