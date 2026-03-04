# API 接口文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`

## 接口列表

### 1. 单次查询

**POST** `/api/query`

将自然语言问题转换为SQL并执行。

**请求体**:
```json
{
  "question": "查询最近一个月的订单数量",
  "db_name": "mydb",
  "session_id": "optional-session-id"
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| question | string | 是 | 用户问题 |
| db_name | string | 是 | 数据库名称 |
| session_id | string | 否 | 会话ID，多轮对话需要 |

**响应**:
```json
{
  "success": true,
  "sql": "SELECT COUNT(*) as order_count FROM orders WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)",
  "explanation": "查询最近一个月内所有订单的数量",
  "is_multi_turn": false,
  "rewritten_question": null,
  "relevant_tables": ["orders"],
  "intent": "new_query",
  "results": [
    {"order_count": 1234}
  ],
  "result_count": 1,
  "error": null
}
```

### 2. 多轮对话

**POST** `/api/chat`

支持上下文理解的连续对话。

**请求体**:
```json
{
  "message": "按地区分组",
  "session_id": "your-session-id",
  "db_name": "mydb"
}
```

**响应**:
```json
{
  "success": true,
  "message": "查询完成",
  "sql": "SELECT region, COUNT(*) as order_count FROM orders WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH) GROUP BY region",
  "results": [...],
  "session_id": "your-session-id"
}
```

### 3. 获取Schema

**GET** `/api/schema/{db_name}`

获取指定数据库的表结构信息。

**响应**:
```json
{
  "success": true,
  "data": {
    "db_name": "mydb",
    "tables": [
      {
        "name": "orders",
        "comment": "订单表",
        "columns": [
          {"name": "id", "type": "int", "comment": "订单ID"},
          {"name": "customer_id", "type": "int", "comment": "客户ID"},
          {"name": "order_date", "type": "datetime", "comment": "订单日期"}
        ]
      }
    ],
    "total_tables": 1,
    "total_columns": 3
  }
}
```

### 4. 构建Schema索引

**POST** `/api/schema/{db_name}/build-index`

为数据库构建向量索引，提高Schema检索精度。

**响应**:
```json
{
  "success": true,
  "tables_indexed": 10
}
```

### 5. 新建会话

**GET** `/api/session/new`

创建新的对话会话。

**响应**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "新会话已创建"
}
```

### 6. 健康检查

**GET** `/health`

检查服务状态。

**响应**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## 错误响应

当请求失败时，响应格式如下：

```json
{
  "success": false,
  "sql": "",
  "explanation": "",
  "error": "错误信息"
}
```

## 使用示例

### cURL

```bash
# 单次查询
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "查询用户数量", "db_name": "mydb"}'

# 多轮对话
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "只看活跃用户", "session_id": "your-session-id", "db_name": "mydb"}'
```

### Python

```python
import requests

# 单次查询
response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "question": "查询最近一周的销售额",
        "db_name": "mydb"
    }
)
result = response.json()
print(result["sql"])
```

### JavaScript

```javascript
// 单次查询
const response = await fetch('http://localhost:8000/api/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: '查询用户数量',
    db_name: 'mydb'
  })
});
const result = await response.json();
console.log(result.sql);
```