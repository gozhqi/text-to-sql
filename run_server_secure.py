# run_server_secure.py
"""
安全加固版 Text-to-SQL 服务
"""
import os
import sys
import re
import time
from typing import Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json

app = FastAPI(title="Text-to-SQL RAG 服务 (安全版)", version="1.0.0")

# ========== 安全配置 ==========

# API Keys（生产环境应该用环境变量）
API_KEYS = os.environ.get("API_KEYS", "demo-key-12345").split(",")

# IP 白名单（为空则允许所有）
ALLOWED_IPS = os.environ.get("ALLOWED_IPS", "").split(",") if os.environ.get("ALLOWED_IPS") else []

# 请求频率限制
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "10"))  # 每分钟请求数
rate_limit_store = defaultdict(list)

# 危险 SQL 关键字
DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
    'EXEC', 'EXECUTE', 'INTO OUTFILE', 'INTO DUMPFILE',
    'UNION', 'INFORMATION_SCHEMA'
]

# 允许的表
ALLOWED_TABLES = []


# ========== 安全中间件 ==========

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """安全中间件"""
    
    # 1. IP 白名单检查
    if ALLOWED_IPS:
        client_ip = request.client.host
        if client_ip not in ALLOWED_IPS and not client_ip.startswith("127."):
            raise HTTPException(status_code=403, detail="访问被拒绝")
    
    # 2. 请求频率限制
    client_ip = request.client.host
    current_time = time.time()
    
    # 清理过期记录
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip]
        if current_time - t < 60
    ]
    
    # 检查频率
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    
    rate_limit_store[client_ip].append(current_time)
    
    # 3. 记录访问日志
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {client_ip} - {request.method} {request.url}")
    
    return await call_next(request)


# ========== SQL 安全校验 ==========

def validate_sql(sql: str) -> tuple:
    """
    校验 SQL 安全性
    返回: (is_valid, error_message)
    """
    
    if not sql or not sql.strip():
        return False, "SQL 为空"
    
    sql_upper = sql.upper().strip()
    
    # 1. 检查危险关键字
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"禁止的操作: {keyword}"
    
    # 2. 只允许 SELECT
    if not sql_upper.startswith('SELECT'):
        return False, "只允许 SELECT 查询"
    
    # 3. 检查 SQL 注入模式
    injection_patterns = [
        r';\s*\w',           # 分号后跟语句
        r'--',               # SQL 注释
        r'/\*',              # 多行注释
        r"'\s*OR\s*'",       # 字符串注入
        r'"\s*OR\s*"',
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, sql_upper):
            return False, "检测到潜在的注入攻击"
    
    # 4. 表权限检查（如果配置了白名单）
    if ALLOWED_TABLES:
        tables = extract_tables(sql)
        for table in tables:
            if table.lower() not in [t.lower() for t in ALLOWED_TABLES]:
                return False, f"无权访问表: {table}"
    
    return True, ""


def extract_tables(sql: str) -> list:
    """提取 SQL 中的表名"""
    tables = []
    
    # FROM 子句
    from_matches = re.findall(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
    tables.extend(from_matches)
    
    # JOIN 子句
    join_matches = re.findall(r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
    tables.extend(join_matches)
    
    return list(set(tables))


# ========== API 端点 ==========

class GenerateRequest(BaseModel):
    question: str
    db_name: str = "default"
    session_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "version": "1.0.0-secure"}


@app.post("/api/rag/generate")
async def generate_sql(
    request: GenerateRequest,
    x_api_key: Optional[str] = Header(None)
):
    """
    生成 SQL（需要 API Key）
    """
    
    # API Key 验证
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    
    # 问题长度限制
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="问题过长，最多 500 字符")
    
    # TODO: 调用 RAG 管道生成 SQL
    sql = f"-- 问题: {request.question}\nSELECT * FROM customers LIMIT 10;"
    
    # 安全校验
    is_valid, error = validate_sql(sql)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "sql": ""
        }
    
    return {
        "success": True,
        "sql": sql,
        "explanation": "示例 SQL",
        "confidence": 0.85
    }


@app.post("/api/knowledge/import")
async def import_knowledge(
    type: str = Form(...),
    file: UploadFile = File(None),
    data: str = Form(None),
    x_api_key: Optional[str] = Header(None)
):
    """
    导入知识（需要 API Key）
    """
    
    # API Key 验证
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    
    # 文件大小限制
    if file and file.size and file.size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="文件过大，最大 10MB")
    
    count = 0
    # ... 导入逻辑
    
    return {
        "success": True,
        "count": count,
        "message": f"成功导入 {count} 条记录"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🔒 启动安全加固版 Text-to-SQL 服务")
    print("=" * 50)
    print(f"📍 访问地址: http://localhost:8000")
    print(f"🔑 API Key: {API_KEYS[0]}")
    print(f"📊 频率限制: {RATE_LIMIT} 次/分钟")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)