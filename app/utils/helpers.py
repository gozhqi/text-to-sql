"""
工具函数模块
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List
import json
import re


def generate_session_id() -> str:
    """生成会话ID"""
    return str(uuid.uuid4())


def format_datetime(dt: datetime) -> str:
    """格式化时间"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """从文本中提取JSON"""
    
    # 尝试提取JSON块
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return {}


def truncate_results(results: List[Dict], limit: int = 100) -> List[Dict]:
    """截断结果集"""
    return results[:limit]


def format_sql(sql: str) -> str:
    """格式化SQL"""
    
    # 移除多余空格
    sql = ' '.join(sql.split())
    
    # 关键字大写
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 
        'INNER JOIN', 'ON', 'AND', 'OR', 'ORDER BY', 'GROUP BY', 
        'HAVING', 'LIMIT', 'OFFSET', 'AS', 'DISTINCT', 'COUNT', 
        'SUM', 'AVG', 'MAX', 'MIN', 'IN', 'NOT IN', 'IS NULL', 
        'IS NOT NULL', 'LIKE', 'BETWEEN', 'ASC', 'DESC'
    ]
    
    for keyword in keywords:
        sql = re.sub(
            rf'\b{keyword}\b', 
            keyword, 
            sql, 
            flags=re.IGNORECASE
        )
    
    return sql


def extract_table_names(sql: str) -> List[str]:
    """提取SQL中的表名"""
    tables = []
    
    patterns = [
        r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        tables.extend(matches)
    
    return list(set(tables))


def sanitize_input(text: str) -> str:
    """清理输入文本"""
    
    # 移除特殊字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 限制长度
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def build_error_response(error: str, details: str = "") -> Dict[str, Any]:
    """构建错误响应"""
    return {
        "success": False,
        "error": error,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }


def build_success_response(data: Any, message: str = "") -> Dict[str, Any]:
    """构建成功响应"""
    return {
        "success": True,
        "data": data,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }