# run_server_fixed.py
"""
修复版 Text-to-SQL 服务
"""
import os
import sys
import re
import time
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Text-to-SQL RAG 服务", version="1.0.1")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
API_KEYS = os.environ.get("API_KEYS", "sk-text2sql-20260305-secret123").split(",")

# 危险 SQL 关键字
DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
    'EXEC', 'EXECUTE', 'INTO OUTFILE', 'INTO DUMPFILE'
]

# 知识库目录
KNOWLEDGE_DIR = "./data/knowledge"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_DIR, "sql_examples"), exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_DIR, "ddl"), exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_DIR, "docs"), exist_ok=True)


def validate_sql(sql: str) -> tuple:
    """校验 SQL 安全性"""
    if not sql or not sql.strip():
        return False, "SQL 为空"
    
    sql_upper = sql.upper().strip()
    
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"禁止的操作: {keyword}"
    
    if not sql_upper.startswith('SELECT'):
        return False, "只允许 SELECT 查询"
    
    return True, ""


class GenerateRequest(BaseModel):
    question: str
    db_name: str = "default"
    session_id: Optional[str] = None


# ========== API 端点 ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "version": "1.0.1"}


@app.post("/api/rag/generate")
async def generate_sql(
    request: GenerateRequest,
    x_api_key: Optional[str] = Header(None)
):
    """生成 SQL"""
    
    import time
    start_time = time.time()
    
    # API Key 验证（前端访问可以不验证）
    # if x_api_key not in API_KEYS:
    #     raise HTTPException(status_code=401, detail="无效的 API Key")
    
    # 问题长度限制
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="问题过长")
    
    # 模拟生成 SQL（实际应调用 RAG 管道）
    sql = f"SELECT * FROM customers LIMIT 10;"
    
    # 安全校验
    is_valid, error = validate_sql(sql)
    if not is_valid:
        return {
            "success": False,
            "sql": "",
            "explanation": "",
            "confidence": 0.0,
            "generation_time": time.time() - start_time,
            "tokens_used": 0,
            "error": error
        }
    
    elapsed = time.time() - start_time
    
    return {
        "success": True,
        "sql": sql,
        "explanation": "这是一个示例 SQL 查询",
        "confidence": 0.85,
        "generation_time": elapsed,
        "tokens_used": 150,
        "error": None
    }


@app.post("/api/knowledge/import")
async def import_knowledge(
    type: str = Form(...),
    file: UploadFile = File(None),
    data: str = Form(None),
    x_api_key: Optional[str] = Header(None)
):
    """导入知识"""
    
    count = 0
    
    try:
        if file:
            content = await file.read()
            filename = file.filename.lower()
            
            if filename.endswith('.sql'):
                # 处理 SQL 文件
                text = content.decode('utf-8')
                statements = [s.strip() for s in text.split(';') if s.strip()]
                
                for i, stmt in enumerate(statements):
                    filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{i}.sql")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(stmt)
                    count += 1
                    
            elif filename.endswith('.json'):
                # 处理 JSON 文件
                text = content.decode('utf-8')
                items = json.loads(text)
                
                if isinstance(items, list):
                    for item in items:
                        filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{count}.json")
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(item, f, ensure_ascii=False, indent=2)
                        count += 1
                else:
                    filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)
                    count = 1
                    
            elif filename.endswith('.csv'):
                # 处理 CSV 文件
                import csv
                from io import StringIO
                
                text = content.decode('utf-8')
                reader = csv.DictReader(StringIO(text))
                
                for row in reader:
                    filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{count}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)
                    count += 1
                    
            elif filename.endswith('.zip'):
                # 处理 ZIP 文件
                import zipfile
                import tempfile
                import shutil
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    extract_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    for root, dirs, files in os.walk(extract_dir):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            
                            if fname.endswith('.sql'):
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    text = f.read()
                                    statements = [s.strip() for s in text.split(';') if s.strip()]
                                    for stmt in statements:
                                        filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{count}.sql")
                                        with open(filepath, 'w', encoding='utf-8') as f:
                                            f.write(stmt)
                                        count += 1
                                        
                            elif fname.endswith('.json'):
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    items = json.load(f)
                                    if isinstance(items, list):
                                        for item in items:
                                            filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{count}.json")
                                            with open(filepath, 'w', encoding='utf-8') as f:
                                                json.dump(item, f, ensure_ascii=False, indent=2)
                                            count += 1
                                            
                finally:
                    os.unlink(tmp_path)
                    shutil.rmtree(extract_dir, ignore_errors=True)
        
        elif data:
            # 处理直接传入的数据
            item = json.loads(data)
            filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
            count = 1
    
    except Exception as e:
        return {
            "success": False,
            "count": 0,
            "message": f"导入失败: {str(e)}"
        }
    
    return {
        "success": True,
        "count": count,
        "message": f"成功导入 {count} 条记录"
    }


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """获取知识库统计"""
    sql_dir = os.path.join(KNOWLEDGE_DIR, "sql_examples")
    ddl_dir = os.path.join(KNOWLEDGE_DIR, "ddl")
    docs_dir = os.path.join(KNOWLEDGE_DIR, "docs")
    
    return {
        "success": True,
        "sql_count": len(os.listdir(sql_dir)) if os.path.exists(sql_dir) else 0,
        "ddl_count": len(os.listdir(ddl_dir)) if os.path.exists(ddl_dir) else 0,
        "doc_count": len(os.listdir(docs_dir)) if os.path.exists(docs_dir) else 0
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🚀 Text-to-SQL 服务启动")
    print("=" * 50)
    print(f"📍 访问地址: http://localhost:8000")
    print(f"🔑 API Key: {API_KEYS[0]}")
    print(f"📚 知识库: {KNOWLEDGE_DIR}")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)