# 开发指南

## 本地开发

### 1. 安装开发依赖

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio
```

### 2. 运行测试

```bash
pytest tests/
```

### 3. 代码格式化

```bash
pip install black isort
black app/
isort app/
```

## 项目结构说明

```
app/
├── main.py              # FastAPI应用入口
├── config.py            # 配置管理（使用pydantic-settings）
├── models/              # 数据模型
│   ├── schemas.py       # Pydantic模型定义
│   └── database.py      # 数据库连接管理
├── core/                # 核心业务逻辑
│   ├── schema_retriever.py  # Schema向量检索
│   ├── prompt_builder.py    # Prompt模板构建
│   ├── sql_generator.py     # LLM SQL生成
│   ├── sql_validator.py     # SQL安全校验
│   └── context_manager.py   # 多轮对话管理
├── services/
│   └── pipeline.py      # 完整处理流水线
└── utils/
    └── helpers.py       # 工具函数
```

## 扩展开发

### 添加新的LLM提供商

```python
# app/core/sql_generator.py

class SQLGenerator:
    async def init(self):
        if self.settings.llm_provider == "your_provider":
            # 初始化你的LLM客户端
            self._client = YourLLMClient(...)
```

### 添加新的数据库支持

```python
# app/models/database.py

class DatabaseManager:
    async def get_table_schemas(self, db_name: str):
        if self.settings.db_type == "your_db":
            # 实现你的Schema获取逻辑
            ...
```

### 自定义Prompt模板

```python
# app/core/prompt_builder.py

class CustomPromptBuilder(SQLPromptBuilder):
    SYSTEM_PROMPT = "你的自定义系统提示..."
    
    @staticmethod
    def build_few_shot_examples():
        return "你的自定义示例..."
```

## 调试技巧

### 启用调试日志

```ini
# .env
DEBUG=true
```

### 查看生成的Prompt

```python
# 在 sql_generator.py 中添加
logger.debug(f"Prompt: {prompt}")
```

### 测试单个组件

```python
# 测试Schema检索
from app.core.schema_retriever import SchemaRetriever

retriever = SchemaRetriever()
await retriever.init()
tables = await retriever.retrieve_relevant_tables("查询用户", "mydb")
print(tables)
```

## 性能优化

### 1. Schema缓存

Schema信息会自动缓存，避免重复查询数据库。

### 2. 向量索引预构建

启动后调用一次构建索引：
```bash
curl -X POST "http://localhost:8000/api/schema/mydb/build-index"
```

### 3. 连接池配置

```python
# app/config.py
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
```