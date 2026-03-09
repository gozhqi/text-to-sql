#!/usr/bin/env python3
"""
Text-to-SQL 完整服务
- 智谱 GLM SQL 生成
- 真正的向量检索
- 多格式 SQL 导入
- 交互式问答
"""
import os
import sys
from pathlib import Path

# 加载 .env 文件
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value
import re
import time
import json
import hashlib
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ============ 配置 ============
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4")

KNOWLEDGE_DIR = Path("./data/knowledge")
SQL_EXAMPLES_DIR = KNOWLEDGE_DIR / "sql_examples"
DDL_DIR = KNOWLEDGE_DIR / "ddl"
DOCS_DIR = KNOWLEDGE_DIR / "docs"

# 创建目录
for d in [SQL_EXAMPLES_DIR, DDL_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============ 向量存储（使用简单的 TF-IDF + 余弦相似度）============
class VectorStore:
    """向量存储 - 支持 TF-IDF 和可选的嵌入模型"""
    
    def __init__(self):
        self.documents = []  # [(id, text, metadata, tfidf_vector)]
        self.idf = {}  # 逆文档频率
        self.doc_count = 0
        self.use_embedding = False
        self.embedding_model = None
        print("✅ 使用 TF-IDF 向量检索")
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 中英文分词
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_]+', text.lower())
        return words
    
    def _compute_tfidf(self, text: str) -> Dict[str, float]:
        """计算 TF-IDF 向量"""
        words = self._tokenize(text)
        if not words:
            return {}
        
        # TF
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        for w in tf:
            tf[w] = tf[w] / len(words)
        
        # TF-IDF
        tfidf = {}
        for w, t in tf.items():
            idf = self.idf.get(w, 1.0)
            tfidf[w] = t * idf
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        # 所有词
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in all_words)
        norm1 = sum(v ** 2 for v in vec1.values()) ** 0.5
        norm2 = sum(v ** 2 for v in vec2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def add(self, doc_id: str, text: str, metadata: dict):
        """添加文档"""
        # 更新 IDF
        words = set(self._tokenize(text))
        for w in words:
            self.idf[w] = self.idf.get(w, 0) + 1
        self.doc_count += 1
        
        # 更新现有文档的 IDF
        for i, (existing_id, existing_text, existing_meta, existing_vec) in enumerate(self.documents):
            new_tfidf = {}
            for w, v in existing_vec.items():
                new_idf = self.doc_count / self.idf.get(w, 1)
                new_tfidf[w] = v * new_idf
            self.documents[i] = (existing_id, existing_text, existing_meta, new_tfidf)
        
        # 计算当前文档的 TF-IDF
        tfidf = self._compute_tfidf(text)
        
        # 如果有嵌入模型，也计算嵌入
        if self.use_embedding and self.embedding_model:
            metadata['_embedding'] = self.embedding_model.encode(text).tolist()
        
        self.documents.append((doc_id, text, metadata, tfidf))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相似文档"""
        if self.use_embedding and self.embedding_model:
            return self._search_with_embedding(query, top_k)
        else:
            return self._search_with_tfidf(query, top_k)
    
    def _search_with_tfidf(self, query: str, top_k: int) -> List[Dict]:
        """使用 TF-IDF 搜索"""
        query_tfidf = self._compute_tfidf(query)
        
        scores = []
        for doc_id, text, metadata, doc_tfidf in self.documents:
            sim = self._cosine_similarity(query_tfidf, doc_tfidf)
            scores.append((sim, doc_id, text, metadata))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for sim, doc_id, text, metadata in scores[:top_k]:
            if sim > 0.01:  # 过滤太低的分数
                results.append({
                    "id": doc_id,
                    "text": text,
                    "score": sim,
                    "metadata": {k: v for k, v in metadata.items() if not k.startswith('_')}
                })
        
        return results
    
    def _search_with_embedding(self, query: str, top_k: int) -> List[Dict]:
        """使用嵌入模型搜索"""
        query_embedding = self.embedding_model.encode(query)
        
        scores = []
        for doc_id, text, metadata, doc_tfidf in self.documents:
            if '_embedding' in metadata:
                import numpy as np
                doc_embedding = np.array(metadata['_embedding'])
                sim = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                scores.append((float(sim), doc_id, text, metadata))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for sim, doc_id, text, metadata in scores[:top_k]:
            if sim > 0.1:
                results.append({
                    "id": doc_id,
                    "text": text,
                    "score": sim,
                    "metadata": {k: v for k, v in metadata.items() if not k.startswith('_')}
                })
        
        return results
    
    def count(self) -> int:
        return len(self.documents)


# ============ 智谱 GLM API ============
class ZhipuLLM:
    """智谱 GLM API 客户端"""
    
    def __init__(self, api_key: str = None, model: str = "glm-4"):
        self.api_key = api_key or ZHIPU_API_KEY
        self.model = model
        self.api_url = ZHIPU_API_URL
    
    def chat(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """调用智谱 GLM API"""
        if not self.api_key:
            raise ValueError("未配置 ZHIPU_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"智谱 API 调用失败: {e}")


# ============ SQL 解析器 ============
class SQLParser:
    """SQL 解析器 - 支持多种格式"""
    
    @staticmethod
    def parse_oracle_ddl(sql: str) -> List[Dict]:
        """解析 Oracle DDL 语句"""
        results = []
        
        # 提取表名
        create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)'
        for match in re.finditer(create_table_pattern, sql, re.IGNORECASE):
            table_name = match.group(1).strip('"').strip("'")
            results.append({
                "type": "table",
                "name": table_name,
                "sql": match.group(0)
            })
        
        # 提取 SELECT 语句
        select_pattern = r'SELECT\s+.+?\s+FROM\s+[^\s;]+(?:\s+(?:WHERE|JOIN|GROUP|ORDER|HAVING).+?)?(?:;)'
        for i, match in enumerate(re.finditer(select_pattern, sql, re.IGNORECASE | re.DOTALL)):
            results.append({
                "type": "select",
                "name": f"query_{i+1}",
                "sql": match.group(0).strip().rstrip(';')
            })
        
        # 提取 CTE (WITH ... AS)
        cte_pattern = r'WITH\s+\w+\s+AS\s*\([^)]+\)'
        if re.search(cte_pattern, sql, re.IGNORECASE):
            # 整个 WITH 语句
            results.append({
                "type": "cte",
                "name": "cte_query",
                "sql": sql.strip()
            })
        
        return results
    
    @staticmethod
    def extract_tables_from_sql(sql: str) -> List[str]:
        """从 SQL 中提取表名"""
        tables = []
        
        # FROM table
        from_pattern = r'\bFROM\s+([^\s,()]+)'
        tables.extend(re.findall(from_pattern, sql, re.IGNORECASE))
        
        # JOIN table
        join_pattern = r'\bJOIN\s+([^\s,()]+)'
        tables.extend(re.findall(join_pattern, sql, re.IGNORECASE))
        
        # 清理
        tables = [t.strip('"').strip("'").strip('`') for t in tables]
        tables = [t for t in tables if not t.upper().startswith(('SELECT', '(', 'WITH'))]
        
        return list(set(tables))


# ============ 全局实例 ============
vector_store = VectorStore()
llm = ZhipuLLM()

# ============ FastAPI 应用 ============
app = FastAPI(title="Text-to-SQL Pro", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 危险关键词
DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE', 'EXEC'
]


def validate_sql(sql: str) -> tuple:
    """校验 SQL 安全性"""
    if not sql or not sql.strip():
        return False, "SQL 为空"
    
    sql_upper = sql.upper().strip()
    
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"禁止的操作: {keyword}"
    
    # 允许 WITH (CTE) 开头
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        return False, "只允许 SELECT 查询"
    
    return True, ""


def load_knowledge_base():
    """启动时加载知识库"""
    count = 0
    
    # 加载 SQL 示例
    for filepath in SQL_EXAMPLES_DIR.glob("**/*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if 'sql' in item:
                        text = f"{item.get('question', '')} {item['sql']}"
                        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
                        vector_store.add(doc_id, text, item)
                        count += 1
        except Exception as e:
            print(f"加载失败 {filepath}: {e}")
    
    # 加载 DDL
    for filepath in DDL_DIR.glob("**/*.sql"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sql = f.read()
                # 解析 DDL
                parsed = SQLParser.parse_oracle_ddl(sql)
                for p in parsed:
                    text = f"{p['name']} {p['sql']}"
                    doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
                    vector_store.add(doc_id, text, {"type": "ddl", **p})
                    count += 1
        except Exception as e:
            print(f"加载失败 {filepath}: {e}")
    
    return count


# 启动加载
_knowledge_count = load_knowledge_base()
print(f"📚 已加载 {_knowledge_count} 条知识")


# ============ 数据模型 ============
class GenerateRequest(BaseModel):
    question: str
    db_name: str = "default"
    context: Optional[str] = None
    conversation_id: Optional[str] = None


class ClarifyRequest(BaseModel):
    question: str
    db_name: str = "default"


# ============ API 路由 ============
@app.get("/", response_class=HTMLResponse)
async def index():
    """首页"""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Text-to-SQL Pro</h1><p>请配置前端页面</p>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "knowledge_count": vector_store.count(),
        "embedding_enabled": vector_store.use_embedding,
        "llm_configured": bool(ZHIPU_API_KEY)
    }


@app.post("/api/clarify")
async def clarify_question(request: ClarifyRequest):
    """澄清问题 - 分析用户需求并提问"""
    
    if not ZHIPU_API_KEY:
        # 没有配置 API Key，返回默认问题
        return {
            "success": True,
            "need_clarification": True,
            "questions": [
                {"id": "tables", "question": "您要查询哪些表的数据？", "options": None},
                {"id": "filters", "question": "需要什么筛选条件？", "options": None},
                {"id": "output", "question": "您希望输出哪些字段？", "options": None}
            ],
            "analysis": {
                "intent": "数据查询",
                "confidence": 0.5
            }
        }
    
    # 使用 LLM 分析问题
    prompt = f"""你是一个数据库查询专家。分析用户的问题，判断是否需要澄清，如果需要则提出具体问题。

用户问题: {request.question}

请分析:
1. 用户想查询什么数据？
2. 需要哪些表？
3. 有什么筛选条件？
4. 需要排序或限制吗？

如果问题足够清晰，返回 {{"need_clarification": false, "analysis": {{...}}}}
如果需要澄清，返回具体的问题列表。

返回 JSON 格式：
{{
    "need_clarification": true/false,
    "questions": [
        {{"id": "xxx", "question": "具体问题？", "options": ["选项1", "选项2"] 或 null}}
    ],
    "analysis": {{
        "intent": "查询意图",
        "tables": ["推测的表名"],
        "filters": ["推测的筛选条件"],
        "confidence": 0.0-1.0
    }}
}}
"""
    
    try:
        response = llm.chat([{"role": "user", "content": prompt}])
        # 解析 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return {"success": True, **json.loads(json_match.group())}
    except Exception as e:
        print(f"澄清失败: {e}")
    
    return {
        "success": True,
        "need_clarification": True,
        "questions": [
            {"id": "tables", "question": "您要查询哪些表的数据？", "options": None}
        ]
    }


@app.post("/api/generate")
async def generate_sql(request: GenerateRequest):
    """生成 SQL"""
    
    start_time = time.time()
    
    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="问题过长")
    
    # 检索相似示例
    similar = vector_store.search(request.question, top_k=5)
    
    # 提取相关的 DDL/表结构
    ddl_context = []
    for s in similar:
        if s['metadata'].get('type') == 'ddl':
            ddl_context.append(s['text'])
    
    # 构建 prompt
    if ZHIPU_API_KEY:
        # 使用智谱 GLM
        examples_text = "\n\n".join([
            f"-- 示例 {i+1}\n-- 问题: {s['metadata'].get('question', 'N/A')}\n{s['metadata'].get('sql', s['text'])}"
            for i, s in enumerate(similar[:3])
        ])
        
        ddl_text = "\n\n".join(ddl_context[:2]) if ddl_context else "无"
        
        prompt = f"""你是一个 SQL 专家。根据用户问题生成标准 SQL 查询语句。

## 相关表结构
{ddl_text}

## 相似示例
{examples_text if examples_text else "无"}

## 用户问题
{request.question}

## 要求
1. 只输出 SQL 语句，不要解释
2. 使用标准 SQL 语法
3. 如果使用了 CTE (WITH 子句)，确保语法正确
4. 只生成 SELECT 查询，不要包含 INSERT/UPDATE/DELETE 等操作

## SQL
"""
        
        try:
            sql = llm.chat([{"role": "user", "content": prompt}])
            # 清理 SQL
            sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'^```\s*', '', sql)
            sql = re.sub(r'\s*```$', '', sql)
            sql = sql.strip()
            
            explanation = f"由智谱 GLM 生成，参考了 {len(similar)} 个相似示例"
        except Exception as e:
            return {
                "success": False,
                "sql": "",
                "explanation": f"生成失败: {str(e)}",
                "confidence": 0,
                "generation_time": time.time() - start_time
            }
    else:
        # 没有 API Key，使用模板匹配
        if similar:
            best = similar[0]
            sql = best['metadata'].get('sql', best['text'])
            explanation = f"基于知识库匹配 (相似度: {best['score']:.2f})，请配置 ZHIPU_API_KEY 以启用智能生成"
        else:
            sql = "-- 请先导入 SQL 示例或配置 ZHIPU_API_KEY"
            explanation = "知识库为空，无法生成"
    
    # 安全校验
    is_valid, error = validate_sql(sql)
    if not is_valid:
        return {
            "success": False,
            "sql": "",
            "explanation": error,
            "confidence": 0,
            "generation_time": time.time() - start_time
        }
    
    return {
        "success": True,
        "sql": sql,
        "explanation": explanation,
        "confidence": similar[0]['score'] if similar else 0.7,
        "generation_time": time.time() - start_time,
        "similar_count": len(similar),
        "tables_used": SQLParser.extract_tables_from_sql(sql)
    }


@app.post("/api/knowledge/import")
async def import_knowledge(
    type: str = Form(...),
    file: UploadFile = File(None),
    data: str = Form(None),
    x_api_key: Optional[str] = Header(None)
):
    """导入知识 - 支持多种格式"""
    
    count = 0
    errors = []
    
    try:
        if file:
            content = await file.read()
            filename = file.filename.lower()
            
            if filename.endswith('.json'):
                # JSON 格式
                text = content.decode('utf-8')
                items = json.loads(text)
                if not isinstance(items, list):
                    items = [items]
                
                for item in items:
                    if 'sql' in item:
                        text = f"{item.get('question', '')} {item['sql']}"
                        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
                        
                        # 保存文件
                        save_path = SQL_EXAMPLES_DIR / f"{int(time.time())}_{doc_id}.json"
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(item, f, ensure_ascii=False, indent=2)
                        
                        # 添加到向量存储
                        vector_store.add(doc_id, text, item)
                        count += 1
                        
            elif filename.endswith('.sql'):
                # SQL 文件 - 支持 Oracle DDL
                text = content.decode('utf-8')
                
                # 保存原始文件
                save_path = DDL_DIR / f"{int(time.time())}_{filename}"
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # 解析并添加
                parsed = SQLParser.parse_oracle_ddl(text)
                for p in parsed:
                    doc_text = f"{p['name']} {p['sql']}"
                    doc_id = hashlib.md5(doc_text.encode()).hexdigest()[:8]
                    vector_store.add(doc_id, doc_text, {"type": "ddl", **p})
                    count += 1
                
                # 如果没有解析出结构，直接添加整个文件
                if not parsed:
                    doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
                    vector_store.add(doc_id, text, {"type": "sql", "source": filename})
                    count += 1
                    
            elif filename.endswith('.csv'):
                # CSV 格式
                import csv
                from io import StringIO
                
                text = content.decode('utf-8')
                reader = csv.DictReader(StringIO(text))
                
                for row in reader:
                    if 'sql' in row:
                        doc_text = f"{row.get('question', '')} {row['sql']}"
                        doc_id = hashlib.md5(doc_text.encode()).hexdigest()[:8]
                        
                        save_path = SQL_EXAMPLES_DIR / f"{int(time.time())}_{doc_id}.json"
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(row, f, ensure_ascii=False, indent=2)
                        
                        vector_store.add(doc_id, doc_text, row)
                        count += 1
                        
            elif filename.endswith('.zip'):
                # ZIP 压缩包
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
                                    if not isinstance(items, list):
                                        items = [items]
                                    for item in items:
                                        if 'sql' in item:
                                            doc_text = f"{item.get('question', '')} {item['sql']}"
                                            doc_id = hashlib.md5(doc_text.encode()).hexdigest()[:8]
                                            vector_store.add(doc_id, doc_text, item)
                                            count += 1
                                            
                            elif fname.endswith('.sql'):
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    sql = f.read()
                                    parsed = SQLParser.parse_oracle_ddl(sql)
                                    for p in parsed:
                                        doc_text = f"{p['name']} {p['sql']}"
                                        doc_id = hashlib.md5(doc_text.encode()).hexdigest()[:8]
                                        vector_store.add(doc_id, doc_text, {"type": "ddl", **p})
                                        count += 1
                finally:
                    os.unlink(tmp_path)
                    shutil.rmtree(extract_dir, ignore_errors=True)
        
        elif data:
            # 直接传 JSON 数据
            item = json.loads(data)
            if 'sql' in item:
                doc_text = f"{item.get('question', '')} {item['sql']}"
                doc_id = hashlib.md5(doc_text.encode()).hexdigest()[:8]
                
                save_path = SQL_EXAMPLES_DIR / f"{int(time.time())}_{doc_id}.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
                
                vector_store.add(doc_id, doc_text, item)
                count = 1
    
    except Exception as e:
        errors.append(str(e))
    
    return {
        "success": count > 0,
        "count": count,
        "errors": errors,
        "message": f"成功导入 {count} 条记录" if count > 0 else "导入失败",
        "total_knowledge": vector_store.count()
    }


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """获取知识库统计"""
    sql_count = sum(1 for _, _, m, _ in vector_store.documents if m.get('type') != 'ddl')
    ddl_count = sum(1 for _, _, m, _ in vector_store.documents if m.get('type') == 'ddl')
    
    return {
        "success": True,
        "sql_count": sql_count,
        "ddl_count": ddl_count,
        "total_count": vector_store.count(),
        "embedding_enabled": vector_store.use_embedding
    }


@app.post("/api/knowledge/clear")
async def clear_knowledge(x_api_key: Optional[str] = Header(None)):
    """清空知识库"""
    global vector_store
    vector_store = VectorStore()
    
    # 清空文件
    import shutil
    for d in [SQL_EXAMPLES_DIR, DDL_DIR]:
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()
    
    return {
        "success": True,
        "message": "知识库已清空",
        "total_count": 0
    }


# ============ 启动 ============
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Text-to-SQL Pro v3.0")
    print("=" * 60)
    print(f"📍 访问地址: http://localhost:8000")
    print(f"📚 已加载知识: {vector_store.count()} 条")
    print(f"🧠 嵌入模型: {'已启用' if vector_store.use_embedding else '未启用 (使用 TF-IDF)'}")
    print(f"🔑 智谱 API: {'已配置' if ZHIPU_API_KEY else '未配置'}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)