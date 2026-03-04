"""
Prompt构建模块
"""
from typing import List, Dict, Optional
from app.models.schemas import TableSchema


class SQLPromptBuilder:
    """SQL生成的Prompt构建器"""
    
    SYSTEM_PROMPT = """你是一个专业的SQL生成助手。你的任务是根据用户的问题和数据库结构，生成准确、高效的SQL查询语句。

## 规则要求：
1. 只生成SELECT查询语句，禁止INSERT/UPDATE/DELETE/DROP等操作
2. 使用标准SQL语法，兼容MySQL/PostgreSQL
3. 合理使用JOIN连接多个表
4. 对于聚合查询，正确使用GROUP BY和HAVING
5. 使用中文注释说明关键逻辑
6. 如果问题模糊，选择最合理的解释

## 输出格式：
```json
{
  "sql": "生成的SQL语句",
  "explanation": "查询逻辑说明",
  "assumptions": ["假设条件列表"]
}
```"""

    @staticmethod
    def build_schema_prompt(tables: List[TableSchema]) -> str:
        """构建Schema描述"""
        schema_text = "## 数据库结构：\n\n"
        
        for table in tables:
            schema_text += table.to_prompt_text()
            schema_text += "\n"
        
        return schema_text

    @staticmethod
    def build_few_shot_examples() -> str:
        """构建Few-shot示例"""
        return """
## 示例问答：

### 示例1：
用户问题: 查询最近一个月销售额最高的前10个产品
SQL:
```sql
SELECT 
    p.product_id,
    p.product_name,
    SUM(o.amount) as total_sales
FROM products p
JOIN order_items o ON p.product_id = o.product_id
JOIN orders ord ON o.order_id = ord.order_id
WHERE ord.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)
GROUP BY p.product_id, p.product_name
ORDER BY total_sales DESC
LIMIT 10;
```

### 示例2：
用户问题: 统计每个部门的员工数量和平均薪资
SQL:
```sql
SELECT 
    d.department_name,
    COUNT(e.employee_id) as employee_count,
    AVG(e.salary) as avg_salary
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id, d.department_name;
```

### 示例3：
用户问题: 找出从未下单的客户
SQL:
```sql
SELECT c.customer_id, c.customer_name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```
"""

    @classmethod
    def build_complete_prompt(
        cls,
        question: str,
        tables: List[TableSchema],
        context_summary: str = ""
    ) -> str:
        """构建完整Prompt"""
        
        parts = [cls.SYSTEM_PROMPT]
        
        if tables:
            parts.append(cls.build_schema_prompt(tables))
        
        parts.append(cls.build_few_shot_examples())
        
        if context_summary:
            parts.append(f"## 对话上下文：\n{context_summary}\n")
        
        parts.append(f"## 用户问题：\n{question}\n")
        parts.append("请生成对应的SQL查询语句：")
        
        return "\n".join(parts)


class RewritePromptBuilder:
    """问题改写Prompt构建器"""
    
    @staticmethod
    def build_rewrite_prompt(
        current_question: str,
        last_question: str,
        last_sql: str,
        referenced_tables: List[str]
    ) -> str:
        """构建问题改写Prompt"""
        
        return f"""
你是一个问题改写助手。用户正在与数据库进行多轮对话，请根据上下文改写当前问题，使其成为独立完整的查询问题。

## 上下文信息：
上一轮问题: {last_question}
上一轮SQL: {last_sql}
涉及的表: {', '.join(referenced_tables)}

## 当前问题：
{current_question}

## 改写规则：
1. 补充省略的主语、宾语、时间等关键信息
2. 继承上一轮的查询范围和过滤条件（除非明确要修改）
3. 处理代词指代（"它"、"那个"、"他们"等）
4. 如果是追加条件，融合到完整查询中
5. 如果是新查询，保持原样

## 输出格式：
```json
{
  "rewritten_question": "改写后的完整问题",
  "reasoning": "改写理由"
}
```
"""


class IntentPromptBuilder:
    """意图识别Prompt构建器"""
    
    @staticmethod
    def build_classify_prompt(
        current_question: str,
        last_question: str,
        last_sql: str
    ) -> str:
        """构建意图分类Prompt"""
        
        return f"""
分析当前问题与上一轮对话的关系，判断用户意图。

## 上一轮：
问题: {last_question}
SQL: {last_sql}

## 当前问题：
{current_question}

## 意图类型说明：
- new_query: 全新的独立查询，与之前无关
- refine: 细化查询，追加更具体的条件
- modify: 修改之前的某些条件（如更换时间范围、地区等）
- aggregate: 在之前基础上进行聚合统计
- filter: 添加过滤条件筛选数据
- sort: 对之前结果进行排序
- compare: 进行对比分析
- drill_down: 下钻查看更详细的数据

## 输出：
```json
{
  "intent": "意图类型",
  "confidence": 0.95,
  "reasoning": "判断理由"
}
```
"""