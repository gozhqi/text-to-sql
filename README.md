# Text-to-SQL 智能查询系统

基于大语言模型的多轮对话式SQL生成系统。

## 功能特性

- ✅ 单轮自然语言转SQL
- ✅ 多轮对话上下文理解
- ✅ Schema智能检索
- ✅ SQL安全校验
- ✅ Web API接口
- ✅ Web UI界面

## 项目结构

```
text-to-sql/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI主入口
│   ├── config.py               # 配置管理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py          # 数据模型
│   │   └── database.py         # 数据库连接
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schema_retriever.py # Schema检索
│   │   ├── prompt_builder.py   # Prompt构建
│   │   ├── sql_generator.py    # SQL生成
│   │   ├── sql_validator.py    # SQL校验
│   │   └── context_manager.py  # 上下文管理
│   ├── services/
│   │   ├── __init__.py
│   │   └── pipeline.py         # 处理流水线
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # 工具函数
├── frontend/
│   └── index.html              # Web界面
├── tests/
│   └── test_pipeline.py
├── .env.example                # 环境变量模板
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的配置
```

### 3. 启动服务

```bash
python -m app.main
```

### 4. 访问界面

打开浏览器访问 http://localhost:8000

## API接口

### POST /api/query

发送查询请求

```json
{
  "question": "查询上个月销售额最高的产品",
  "db_name": "sales_db",
  "session_id": "optional-session-id"
}
```

### POST /api/chat

多轮对话接口

```json
{
  "message": "按地区分组",
  "session_id": "your-session-id",
  "db_name": "sales_db"
}
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| LLM_PROVIDER | LLM提供商 | openai |
| LLM_API_KEY | API密钥 | - |
| LLM_MODEL | 模型名称 | gpt-4 |
| DB_HOST | 数据库地址 | localhost |
| DB_PORT | 数据库端口 | 3306 |
| DB_USER | 数据库用户 | - |
| DB_PASSWORD | 数据库密码 | - |
| DB_NAME | 数据库名称 | - |