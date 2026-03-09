# 安全配置建议

## 一、网络层安全

### 1.1 IP 白名单（推荐）
只允许特定 IP 访问服务：

```python
# 在 run_server.py 中添加
from fastapi import Request, HTTPException

ALLOWED_IPS = [
    "127.0.0.1",
    "172.17.48.0/24",  # 内网段
    # "你的办公网IP"
]

@app.middleware("http")
async def ip_filter(request: Request, call_next):
    client_ip = request.client.host
    if not any(client_ip.startswith(ip.split('/')[0]) for ip in ALLOWED_IPS):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    return await call_next(request)
```

### 1.2 防火墙配置
```bash
# 只允许特定 IP 访问 8000 端口
iptables -A INPUT -p tcp --dport 8000 -s 允许的IP -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

---

## 二、应用层安全

### 2.1 认证机制
```python
# 添加 API Key 认证
from fastapi import Header

API_KEYS = ["your-secret-key-1", "your-secret-key-2"]

@app.post("/api/rag/generate")
async def generate_sql(
    request: GenerateRequest,
    x_api_key: str = Header(None)
):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    # ... 业务逻辑
```

### 2.2 请求频率限制
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/rag/generate")
@limiter.limit("10/minute")  # 每分钟最多 10 次
async def generate_sql(request: GenerateRequest):
    # ... 业务逻辑
```

---

## 三、SQL 安全

### 3.1 危险操作拦截
```python
DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
    'EXEC', 'EXECUTE', 'INTO OUTFILE', 'INTO DUMPFILE'
]

def validate_sql(sql: str) -> tuple:
    sql_upper = sql.upper()
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"禁止的操作: {keyword}"
    
    if not sql_upper.strip().startswith('SELECT'):
        return False, "只允许 SELECT 查询"
    
    return True, ""
```

### 3.2 表权限控制
```python
# 只允许访问特定表
ALLOWED_TABLES = ["customers", "orders", "products"]

def check_table_access(sql: str):
    tables = extract_tables(sql)
    for table in tables:
        if table.lower() not in ALLOWED_TABLES:
            return False, f"无权访问表: {table}"
    return True, ""
```

---

## 四、日志与监控

### 4.1 访问日志
```python
import logging

logging.basicConfig(
    filename='/var/log/text2sql/access.log',
    level=logging.INFO,
    format='%(asctime)s - %(ip)s - %(request)s'
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"IP: {request.client.host} - {request.url}")
    return await call_next(request)
```

### 4.2 异常告警
```python
# 检测异常行为
def detect_attack(request: Request):
    # 短时间内大量请求
    # 尝试注入攻击
    # 访问不存在的端点
    pass
```

---

## 五、数据安全

### 5.1 敏感数据脱敏
```python
# SQL 结果脱敏
def mask_sensitive_data(results: list, sensitive_columns: list):
    for row in results:
        for col in sensitive_columns:
            if col in row:
                row[col] = "***"
    return results
```

### 5.2 查询结果限制
```python
MAX_ROWS = 1000  # 最多返回 1000 行

def limit_results(results: list):
    return results[:MAX_ROWS]
```

---

## 六、部署安全

### 6.1 HTTPS
```bash
# 使用 Nginx 反向代理 + Let's Encrypt
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

### 6.2 服务隔离
```bash
# 使用 Docker 容器隔离
docker run -d \
  --name text2sql \
  -p 127.0.0.1:8000:8000 \
  --memory="1g" \
  --cpus="1" \
  text2sql:latest
```

---

## 七、快速安全加固

如果现在就要开放公网访问，建议：

1. **添加 API Key 认证**
2. **限制请求频率**
3. **添加 IP 白名单**（如果知道访问者 IP）
4. **使用 HTTPS**

需要我帮你实现哪个安全措施？