"""
SQL校验模块
"""
import re
from typing import List, Tuple, Optional
from loguru import logger

from app.config import get_settings
from app.models.schemas import TableSchema


class SQLValidator:
    """SQL安全校验器"""
    
    # 危险SQL关键词
    DANGEROUS_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
        'TRUNCATE', 'REPLACE', 'MERGE', 'CALL', 'EXEC', 'EXECUTE',
        'GRANT', 'REVOKE', 'INTO OUTFILE', 'INTO DUMPFILE'
    ]
    
    # SQL注释模式
    COMMENT_PATTERNS = [
        r'--.*$',           # 单行注释
        r'/\*.*?\*/',       # 多行注释
    ]
    
    def __init__(self):
        self.settings = get_settings()
        self.allowed_operations = [
            op.strip().upper() 
            for op in self.settings.allowed_operations.split(',')
        ]
    
    def validate(self, sql: str, tables: List[TableSchema] = None) -> Tuple[bool, str]:
        """
        校验SQL安全性
        
        Returns:
            (is_valid, error_message)
        """
        
        # 空检查
        if not sql or not sql.strip():
            return False, "SQL语句为空"
        
        sql_upper = sql.upper().strip()
        
        # 移除注释
        clean_sql = sql
        for pattern in self.COMMENT_PATTERNS:
            clean_sql = re.sub(pattern, '', clean_sql, flags=re.MULTILINE | re.DOTALL)
        clean_sql_upper = clean_sql.upper().strip()
        
        # 检查危险关键词
        for keyword in self.DANGEROUS_KEYWORDS:
            # 使用单词边界匹配
            if re.search(rf'\b{keyword}\b', clean_sql_upper):
                return False, f"SQL包含危险操作: {keyword}"
        
        # 检查是否以允许的操作开头
        first_word = clean_sql_upper.split()[0] if clean_sql_upper.split() else ""
        if first_word not in self.allowed_operations:
            return False, f"不允许的操作类型: {first_word}，仅支持: {', '.join(self.allowed_operations)}"
        
        # 检查表名是否在允许列表中
        if tables:
            allowed_tables = {t.table_name.lower() for t in tables}
            extracted_tables = self._extract_table_names(clean_sql)
            
            for table in extracted_tables:
                if table.lower() not in allowed_tables:
                    logger.warning(f"SQL引用了未授权的表: {table}")
        
        # 检查SQL注入模式
        injection_patterns = [
            r';\s*\w',           # 分号后跟语句
            r'UNION\s+SELECT',   # UNION注入
            r'OR\s+1\s*=\s*1',   # 常见注入
            r"'\s*OR\s*'",       # 字符串注入
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, clean_sql_upper):
                return False, f"检测到潜在的SQL注入模式"
        
        return True, ""
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """提取SQL中的表名"""
        tables = []
        
        # FROM 子句
        from_matches = re.findall(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
        tables.extend(from_matches)
        
        # JOIN 子句
        join_matches = re.findall(r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
        tables.extend(join_matches)
        
        # INTO 子句
        into_matches = re.findall(r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
        tables.extend(into_matches)
        
        # UPDATE 子句
        update_matches = re.findall(r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
        tables.extend(update_matches)
        
        return list(set(tables))
    
    def fix_sql(self, sql: str) -> str:
        """修复常见SQL问题"""
        
        # 移除末尾分号
        sql = sql.rstrip().rstrip(';')
        
        # 标准化空格
        sql = ' '.join(sql.split())
        
        # 添加LIMIT（如果没有）
        sql_upper = sql.upper()
        if sql_upper.startswith('SELECT') and 'LIMIT' not in sql_upper:
            sql = f"{sql} LIMIT {self.settings.max_query_rows}"
        
        return sql
    
    def validate_and_fix(
        self, 
        sql: str, 
        tables: List[TableSchema] = None
    ) -> Tuple[str, bool, str]:
        """
        校验并修复SQL
        
        Returns:
            (fixed_sql, is_valid, error_message)
        """
        
        # 先修复
        fixed_sql = self.fix_sql(sql)
        
        # 再校验
        is_valid, error = self.validate(fixed_sql, tables)
        
        return fixed_sql, is_valid, error


# 全局实例
sql_validator: Optional[SQLValidator] = None


def get_sql_validator() -> SQLValidator:
    """获取SQL校验器实例"""
    global sql_validator
    if sql_validator is None:
        sql_validator = SQLValidator()
    return sql_validator