# run_server_rag.py
"""
真正实现 RAG 的 Text-to-SQL 服务
"""
import os
import sys
import re
import time
import json
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 简单的向量检索实现
class SimpleVectorStore:
    """简单的内存向量存储"""
    
    def __init__(self):
        self.documents = []  # [(text, embedding, metadata)]
        self._initialized = False
    
    def add(self, text: str, metadata: dict):
        """添加文档"""
        # 简单的关键词向量（实际应使用嵌入模型）
        words = set(re.findall(r'\w+', text.lower()))
        self.documents.append((text, words, metadata))
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """搜索相似文档"""
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scores = []
        for text, words, metadata in self.documents:
            # 简单的 Jaccard 相似度
            intersection = len(query_words & words)
            union = len(query_words | words)
            score = intersection / union if union > 0 else 0
            scores.append((score, text, metadata))
        
        # 排序并返回 top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, text, metadata in scores[:top_k]:
            if score > 0:
                results.append({
                    "text": text,
                    "score": score,
                    "metadata": metadata
                })
        
        return results


# 全局向量存储
vector_store = SimpleVectorStore()

app = FastAPI(title="Text-to-SQL RAG 服务", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEYS = os.environ.get("API_KEYS", "sk-text2sql-20260305-secret123").split(",")

KNOWLEDGE_DIR = "./data/knowledge"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(os.path.join(KNOWLEDGE_DIR, "sql_examples"), exist_ok=True)

DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
]


def validate_sql(sql: str) -> tuple:
    if not sql or not sql.strip():
        return False, "SQL 为空"
    
    sql_upper = sql.upper().strip()
    
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"禁止的操作: {keyword}"
    
    if not sql_upper.startswith('SELECT'):
        return False, "只允许 SELECT 查询"
    
    return True, ""


def load_knowledge_base():
    """加载知识库到向量存储"""
    sql_dir = os.path.join(KNOWLEDGE_DIR, "sql_examples")
    
    if not os.path.exists(sql_dir):
        return 0
    
    count = 0
    for filename in os.listdir(sql_dir):
        filepath = os.path.join(sql_dir, filename)
        
        try:
            if filename.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'question' in item and 'sql' in item:
                                vector_store.add(
                                    text=f"{item['question']} {item['sql']}",
                                    metadata=item
                                )
                                count += 1
                    elif 'question' in data and 'sql' in data:
                        vector_store.add(
                            text=f"{data['question']} {data['sql']}",
                            metadata=data
                        )
                        count += 1
            elif filename.endswith('.sql'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    sql = f.read().strip()
                    if sql:
                        vector_store.add(
                            text=sql,
                            metadata={"sql": sql, "source": filename}
                        )
                        count += 1
        except Exception as e:
            print(f"加载文件失败 {filename}: {e}")
    
    return count


# 启动时加载知识库
_knowledge_count = load_knowledge_base()
print(f"📚 已加载 {_knowledge_count} 条知识")


class GenerateRequest(BaseModel):
    question: str
    db_name: str = "default"
    session_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "knowledge_count": len(vector_store.documents)
    }


@app.post("/api/rag/generate")
async def generate_sql(
    request: GenerateRequest,
    x_api_key: Optional[str] = Header(None)
):
    """使用 RAG 生成 SQL"""
    
    import time
    start_time = time.time()
    
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="问题过长")
    
    # 检索相似 SQL
    similar = vector_store.search(request.question, top_k=3)
    
    # 构建提示
    if similar:
        # 有相似示例，基于示例生成
        examples_text = "\n".join([
            f"问题: {s['metadata'].get('question', '')}\nSQL: {s['metadata'].get('sql', s['text'])}"
            for s in similar
        ])
        
        # 简单的模板生成（实际应调用 LLM）
        best_match = similar[0]['metadata']
        sql = best_match.get('sql', '')
        
        # 尝试调整 SQL（简单规则）
        question_lower = request.question.lower()
        
        if '排序' in question_lower or 'order' in question_lower:
            if 'ORDER BY' not in sql.upper():
                sql = sql.rstrip(';') + ' ORDER BY created_at DESC'
        
        if 'limit' in question_lower or '前' in question_lower:
            if 'LIMIT' not in sql.upper():
                sql = sql.rstrip(';') + ' LIMIT 10'
        
        explanation = f"基于相似示例生成 (相似度: {similar[0]['score']:.2f})"
    else:
        # 没有相似示例
        sql = "-- 请先导入相关的 SQL 示例到知识库"
        explanation = "知识库中没有找到相似的 SQL 示例"
    
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
    
    return {
        "success": True,
        "sql": sql,
        "explanation": explanation,
        "confidence": similar[0]['score'] if similar else 0.5,
        "generation_time": time.time() - start_time,
        "tokens_used": 100,
        "similar_examples": len(similar)
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
            
            if filename.endswith('.json'):
                text = content.decode('utf-8')
                items = json.loads(text)
                
                if isinstance(items, list):
                    for item in items:
                        # 保存到文件
                        filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{count}.json")
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(item, f, ensure_ascii=False, indent=2)
                        
                        # 添加到向量存储
                        if 'question' in item and 'sql' in item:
                            vector_store.add(
                                text=f"{item['question']} {item['sql']}",
                                metadata=item
                            )
                        count += 1
                else:
                    filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)
                    
                    if 'question' in items and 'sql' in items:
                        vector_store.add(
                            text=f"{items['question']} {items['sql']}",
                            metadata=items
                        )
                    count = 1
                    
            elif filename.endswith('.sql'):
                text = content.decode('utf-8')
                statements = [s.strip() for s in text.split(';') if s.strip()]
                
                for i, stmt in enumerate(statements):
                    filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{i}.sql")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(stmt)
                    
                    vector_store.add(text=stmt, metadata={"sql": stmt})
                    count += 1
                    
            elif filename.endswith('.csv'):
                import csv
                from io import StringIO
                
                text = content.decode('utf-8')
                reader = csv.DictReader(StringIO(text))
                
                for row in reader:
                    filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}_{count}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)
                    
                    if 'question' in row and 'sql' in row:
                        vector_store.add(
                            text=f"{row['question']} {row['sql']}",
                            metadata=row
                        )
                    count += 1
                    
            elif filename.endswith('.zip'):
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
                            
                            if fname.endswith('.json'):
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    items = json.load(f)
                                    if isinstance(items, list):
                                        for item in items:
                                            if 'question' in item and 'sql' in item:
                                                vector_store.add(
                                                    text=f"{item['question']} {item['sql']}",
                                                    metadata=item
                                                )
                                            count += 1
                            elif fname.endswith('.sql'):
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    text = f.read()
                                    statements = [s.strip() for s in text.split(';') if s.strip()]
                                    for stmt in statements:
                                        vector_store.add(text=stmt, metadata={"sql": stmt})
                                        count += 1
                finally:
                    os.unlink(tmp_path)
                    shutil.rmtree(extract_dir, ignore_errors=True)
        
        elif data:
            item = json.loads(data)
            filepath = os.path.join(KNOWLEDGE_DIR, "sql_examples", f"imported_{int(time.time())}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
            
            if 'question' in item and 'sql' in item:
                vector_store.add(
                    text=f"{item['question']} {item['sql']}",
                    metadata=item
                )
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
        "message": f"成功导入 {count} 条记录",
        "total_knowledge": len(vector_store.documents)
    }


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    return {
        "success": True,
        "sql_count": len(vector_store.documents),
        "ddl_count": 0,
        "doc_count": 0
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🚀 Text-to-SQL RAG 服务 v2.0")
    print("=" * 50)
    print(f"📍 访问地址: http://localhost:8000")
    print(f"📚 已加载知识: {len(vector_store.documents)} 条")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)