# run_server.py
"""
启动 Text-to-SQL 服务
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import zipfile
import tempfile
import shutil

app = FastAPI(title="Text-to-SQL RAG 服务", version="1.0.0")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ========== 数据模型 ==========

class GenerateRequest(BaseModel):
    question: str
    db_name: str = "default"
    session_id: Optional[str] = None


class GenerateResponse(BaseModel):
    success: bool
    sql: str = ""
    explanation: str = ""
    confidence: float = 0.0
    generation_time: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None


class ImportResponse(BaseModel):
    success: bool
    count: int = 0
    message: str = ""


# ========== 知识库存储 ==========

KNOWLEDGE_DIR = "./data/knowledge"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)


# ========== API 端点 ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/api/rag/generate", response_model=GenerateResponse)
async def generate_sql(request: GenerateRequest):
    """
    生成 SQL
    """
    import time

    start_time = time.time()

    # TODO: 实际调用 RAG 管道
    # 这里先返回模拟结果
    sql = f"-- 问题: {request.question}\nSELECT * FROM customers LIMIT 10;"

    return GenerateResponse(
        success=True,
        sql=sql,
        explanation="这是一个示例 SQL，实际部署后会调用 LLM 生成。",
        confidence=0.85,
        generation_time=time.time() - start_time,
        tokens_used=150
    )


@app.post("/api/knowledge/import", response_model=ImportResponse)
async def import_knowledge(
    type: str = Form(...),
    file: UploadFile = File(None),
    data: str = Form(None)
):
    """
    导入知识
    支持:
    - 单条 SQL (data 参数)
    - 文件上传 (file 参数)
    - 压缩包上传 (.zip)
    """
    count = 0

    if file:
        # 处理文件上传
        filename = file.filename.lower()

        if filename.endswith('.zip'):
            # 处理压缩包
            count = await process_zip_file(file, type)
        elif filename.endswith('.sql'):
            # 处理单个 SQL 文件
            content = await file.read()
            count = await save_sql_file(content.decode('utf-8'), type)
        elif filename.endswith('.json'):
            # 处理 JSON 文件
            content = await file.read()
            count = await save_json_file(content.decode('utf-8'), type)
        elif filename.endswith('.csv'):
            # 处理 CSV 文件
            content = await file.read()
            count = await save_csv_file(content.decode('utf-8'), type)

    elif data:
        # 处理直接传入的数据
        count = await save_single_item(data, type)

    return ImportResponse(
        success=True,
        count=count,
        message=f"成功导入 {count} 条记录"
    )


async def process_zip_file(file: UploadFile, type: str) -> int:
    """处理 ZIP 压缩包"""
    count = 0

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 解压
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 遍历文件
        for root, dirs, files in os.walk(extract_dir):
            for filename in files:
                filepath = os.path.join(root, filename)

                if filename.endswith('.sql'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        count += await save_sql_file(f.read(), type)
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        count += await save_json_file(f.read(), type)
                elif filename.endswith('.csv'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        count += await save_csv_file(f.read(), type)

    finally:
        # 清理
        os.unlink(tmp_path)
        shutil.rmtree(extract_dir, ignore_errors=True)

    return count


async def save_sql_file(content: str, type: str) -> int:
    """保存 SQL 文件内容"""
    # 解析 SQL 语句
    statements = []
    current = []

    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('--'):
            continue

        current.append(line)

        if line.endswith(';'):
            statements.append(' '.join(current))
            current = []

    # 保存
    sql_dir = os.path.join(KNOWLEDGE_DIR, "sql_examples")
    os.makedirs(sql_dir, exist_ok=True)

    for i, sql in enumerate(statements):
        filepath = os.path.join(sql_dir, f"imported_{len(os.listdir(sql_dir))}_{i}.sql")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sql)

    return len(statements)


async def save_json_file(content: str, type: str) -> int:
    """保存 JSON 文件内容"""
    count = 0

    try:
        data = json.loads(content)

        if isinstance(data, list):
            for item in data:
                await save_single_item(json.dumps(item), type)
                count += 1
        elif isinstance(data, dict):
            await save_single_item(json.dumps(data), type)
            count += 1
    except json.JSONDecodeError:
        pass

    return count


async def save_csv_file(content: str, type: str) -> int:
    """保存 CSV 文件内容"""
    import csv
    from io import StringIO

    count = 0
    reader = csv.DictReader(StringIO(content))

    sql_dir = os.path.join(KNOWLEDGE_DIR, "sql_examples")
    os.makedirs(sql_dir, exist_ok=True)

    for row in reader:
        if 'question' in row and 'sql' in row:
            filepath = os.path.join(sql_dir, f"imported_{len(os.listdir(sql_dir))}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(row, f, ensure_ascii=False)
            count += 1

    return count


async def save_single_item(data: str, type: str) -> int:
    """保存单条数据"""
    try:
        item = json.loads(data)
    except:
        return 0

    save_dir = os.path.join(KNOWLEDGE_DIR, f"{type}_items")
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, f"item_{len(os.listdir(save_dir))}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(item, f, ensure_ascii=False, indent=2)

    return 1


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """获取知识库统计"""
    stats = {
        "ddl_count": 0,
        "sql_count": 0,
        "doc_count": 0
    }

    # 统计 SQL 示例
    sql_dir = os.path.join(KNOWLEDGE_DIR, "sql_examples")
    if os.path.exists(sql_dir):
        stats["sql_count"] = len(os.listdir(sql_dir))

    return stats


# ========== 启动 ==========

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("🚀 启动 Text-to-SQL 服务")
    print("=" * 50)
    print(f"📍 访问地址: http://localhost:8000")
    print(f"📚 知识库路径: {KNOWLEDGE_DIR}")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)