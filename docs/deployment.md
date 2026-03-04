# 部署指南

## 环境要求

- Python 3.10+
- MySQL 5.7+ 或 PostgreSQL 12+
- Redis (可选，用于会话存储)
- 至少 4GB 内存

## 快速部署

### 1. 克隆项目

```bash
git clone https://github.com/your-username/text-to-sql.git
cd text-to-sql
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# LLM配置
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4

# 数据库配置
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3306
DB_USER=your-user
DB_PASSWORD=your-password
DB_NAME=your-database
```

### 5. 启动服务

```bash
python -m app.main
```

访问 http://localhost:8000

## Docker 部署

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "app.main"]
```

### 构建并运行

```bash
docker build -t text-to-sql .
docker run -p 8000:8000 --env-file .env text-to-sql
```

## 生产环境部署

### 使用 Gunicorn

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 常见问题

### Q: Schema检索不准确

A: 确保先构建Schema索引：
```bash
curl -X POST "http://localhost:8000/api/schema/your_db/build-index"
```

### Q: LLM调用超时

A: 检查网络连接，或使用代理：
```ini
LLM_BASE_URL=https://your-proxy.com/v1
```

### Q: 中文问题理解不准确

A: 建议使用GPT-4或支持中文的模型，并在Prompt中添加业务术语说明。