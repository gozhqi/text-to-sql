"""
Text-to-SQL FastAPI Server

高性能 Web API 服务器，提供:
1. RESTful API 接口
2. WebSocket 实时交互
3. 流式响应支持
4. 完整的错误处理
"""
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import redis.asyncio as redis

# 导入核心模块
import sys
sys.path.append(".")
from inference.text2sql_service import (
    Text2SQLService,
    QueryRequest as ServiceQueryRequest,
    QueryResponse,
    ServiceFactory,
    SchemaInfo
)


# ============================================================================
# 配置
# ============================================================================

class Settings:
    """服务配置"""
    # API
    app_name: str = "Text-to-SQL Service"
    version: str = "2.0.0"
    debug: bool = False

    # CORS
    allowed_origins: List[str] = ["*"]

    # Redis
    redis_url: Optional[str] = None

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds


settings = Settings()


# ============================================================================
# 数据模型
# ============================================================================

class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=1)
    db_name: str = Field(..., description="数据库名称")
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    user_id: Optional[str] = Field(None, description="用户ID")
    stream: bool = Field(False, description="是否使用流式响应")
    include_reasoning: bool = Field(False, description="是否包含推理过程")


class QueryResponseModel(BaseModel):
    """查询响应模型"""
    success: bool
    sql: str = ""
    explanation: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    execution_time: float = 0.0
    complexity: str = ""
    tables_used: List[str] = []
    error: Optional[str] = None


class ChatMessage(BaseModel):
    """聊天消息模型"""
    message: str
    session_id: str
    db_name: str


class FeedbackRequest(BaseModel):
    """用户反馈模型"""
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    sql_correct: bool
    result_correct: bool
    edited_sql: Optional[str] = None
    comment: Optional[str] = None


class TableInfo(BaseModel):
    """表信息模型"""
    name: str
    comment: str = ""
    columns: List[Dict[str, Any]]


class DatabaseInfo(BaseModel):
    """数据库信息模型"""
    db_name: str
    tables: List[TableInfo]
    total_tables: int
    total_columns: int


# ============================================================================
# 会话管理
# ============================================================================

class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Optional[redis.Redis] = None

    async def init(self, redis_url: Optional[str] = None):
        """初始化"""
        if redis_url:
            self.redis_client = await redis.from_url(redis_url)

    def get_or_create(self, session_id: str, db_name: str) -> Dict[str, Any]:
        """获取或创建会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "id": session_id,
                "db_name": db_name,
                "created_at": asyncio.get_event_loop().time(),
                "messages": [],
                "context": {}
            }
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": asyncio.get_event_loop().time()
            })

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取历史消息"""
        if session_id in self.sessions:
            return self.sessions[session_id]["messages"][-limit:]
        return []

    async def cleanup_expired(self, max_age: float = 3600):
        """清理过期会话"""
        current_time = asyncio.get_event_loop().time()
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if current_time - session.get("created_at", 0) > max_age
        ]
        for sid in expired_ids:
            del self.sessions[sid]
            logger.debug(f"Cleaned up expired session: {sid}")


# ============================================================================
# 限流器
# ============================================================================

class RateLimiter:
    """限流器"""

    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self.clients: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        now = asyncio.get_event_loop().time()

        if client_id not in self.clients:
            self.clients[client_id] = []

        # 清理过期记录
        self.clients[client_id] = [
            t for t in self.clients[client_id]
            if now - t < self.window
        ]

        # 检查是否超过限制
        if len(self.clients[client_id]) >= self.requests:
            return False

        # 记录本次请求
        self.clients[client_id].append(now)
        return True


# ============================================================================
# 全局状态
# =============================================================================

# 服务实例
text2sql_service: Optional[Text2SQLService] = None
session_manager = SessionManager()
rate_limiter = RateLimiter() if settings.rate_limit_enabled else None


# ============================================================================
# 示例 Schema 数据
# =============================================================================

# 示例：销售数据库
SALES_SCHEMA = SchemaInfo(
    tables={
        "products": {
            "table_name": "products",
            "table_comment": "产品信息表",
            "columns": [
                {"name": "product_id", "type": "INT", "is_primary_key": True, "comment": "产品ID"},
                {"name": "product_name", "type": "VARCHAR(100)", "comment": "产品名称"},
                {"name": "category", "type": "VARCHAR(50)", "comment": "产品类别"},
                {"name": "price", "type": "DECIMAL(10,2)", "comment": "单价"},
                {"name": "stock", "type": "INT", "comment": "库存数量"},
            ]
        },
        "customers": {
            "table_name": "customers",
            "table_comment": "客户信息表",
            "columns": [
                {"name": "customer_id", "type": "INT", "is_primary_key": True, "comment": "客户ID"},
                {"name": "customer_name", "type": "VARCHAR(100)", "comment": "客户名称"},
                {"name": "city", "type": "VARCHAR(50)", "comment": "城市"},
                {"name": "country", "type": "VARCHAR(50)", "comment": "国家"},
                {"name": "phone", "type": "VARCHAR(20)", "comment": "电话"},
            ]
        },
        "orders": {
            "table_name": "orders",
            "table_comment": "订单表",
            "columns": [
                {"name": "order_id", "type": "INT", "is_primary_key": True, "comment": "订单ID"},
                {"name": "customer_id", "type": "INT", "is_foreign_key": True, "foreign_key_refs": "customers(customer_id)", "comment": "客户ID"},
                {"name": "order_date", "type": "DATE", "comment": "订单日期"},
                {"name": "status", "type": "VARCHAR(20)", "comment": "订单状态"},
                {"name": "total_amount", "type": "DECIMAL(10,2)", "comment": "订单总额"},
            ]
        },
        "order_items": {
            "table_name": "order_items",
            "table_comment": "订单明细表",
            "columns": [
                {"name": "item_id", "type": "INT", "is_primary_key": True, "comment": "明细ID"},
                {"name": "order_id", "type": "INT", "is_foreign_key": True, "foreign_key_refs": "orders(order_id)", "comment": "订单ID"},
                {"name": "product_id", "type": "INT", "is_foreign_key": True, "foreign_key_refs": "products(product_id)", "comment": "产品ID"},
                {"name": "quantity", "type": "INT", "comment": "数量"},
                {"name": "unit_price", "type": "DECIMAL(10,2)", "comment": "单价"},
                {"name": "subtotal", "type": "DECIMAL(10,2)", "comment": "小计"},
            ]
        }
    },
    relationships=[
        {"from": "orders.customer_id", "to": "customers.customer_id"},
        {"from": "order_items.order_id", "to": "orders.order_id"},
        {"from": "order_items.product_id", "to": "products.product_id"},
    ]
)

# 可用数据库
AVAILABLE_SCHEMAS = {
    "sales_db": SALES_SCHEMA,
}


# ============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("Starting Text-to-SQL Service...")

    global text2sql_service

    # 初始化服务
    # 这里使用示例配置，实际部署时从环境变量读取
    # text2sql_service = await ServiceFactory.create_openai_service(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    #     redis_url=os.getenv("REDIS_URL"),
    #     schemas=AVAILABLE_SCHEMAS,
    # )

    # 初始化会话管理器
    await session_manager.init(settings.redis_url)

    logger.info("Service started successfully")

    yield

    # 关闭
    logger.info("Shutting down Text-to-SQL Service...")


# ============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 依赖项
# =============================================================================

def get_client_id(request: QueryRequest) -> str:
    """获取客户端ID用于限流"""
    return request.user_id or "anonymous"


# ============================================================================
# API 路由
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """根路径"""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "running"
    }


@app.get("/health", tags=["Health"])
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/api/v1/query", response_model=QueryResponseModel, tags=["Query"])
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(get_client_id)
):
    """执行 Text-to-SQL 查询"""
    # 限流检查
    if rate_limiter and not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    try:
        # 转换请求
        service_request = ServiceQueryRequest(
            question=request.question,
            db_name=request.db_name,
            session_id=request.session_id,
            user_id=request.user_id,
            stream=request.stream,
            include_reasoning=request.include_reasoning
        )

        # 如果服务未初始化，返回模拟响应
        if text2sql_service is None:
            # 模拟响应（用于开发测试）
            return QueryResponseModel(
                success=True,
                sql="-- 模拟 SQL (服务未配置)\nSELECT * FROM products LIMIT 10;",
                explanation="这是一个模拟响应，因为服务尚未配置真实的模型后端。请在生产环境中配置 OpenAI API 或本地模型。",
                confidence=0.9,
                execution_time=0.1,
                complexity="medium",
                tables_used=["products"]
            )

        # 执行查询
        response = await text2sql_service.query(service_request)

        # 添加到会话历史
        if request.session_id:
            session_manager.add_message(
                request.session_id,
                "user",
                request.question
            )
            session_manager.add_message(
                request.session_id,
                "assistant",
                response.sql
            )

        # 返回响应
        return QueryResponseModel(**response.__dict__)

    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """流式查询响应"""
    async def generate():
        try:
            service_request = ServiceQueryRequest(
                question=request.question,
                db_name=request.db_name,
                session_id=request.session_id,
                user_id=request.user_id,
                stream=True,
                include_reasoning=request.include_reasoning
            )

            if text2sql_service is None:
                # 模拟流式响应
                import json
                yield f"data: {json.dumps({'type': 'status', 'message': 'Processing...'})}\n\n"
                await asyncio.sleep(0.1)
                yield f"data: {json.dumps({'type': 'result', 'sql': '-- 模拟 SQL', 'explanation': '服务未配置'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for chunk in text2sql_service.query_stream(service_request):
                import json
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception(f"Stream query failed: {e}")
            import json
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/v1/schemas", tags=["Schema"])
async def list_schemas():
    """列出所有可用的数据库 Schema"""
    return {
        "databases": list(AVAILABLE_SCHEMAS.keys()),
        "count": len(AVAILABLE_SCHEMAS)
    }


@app.get("/api/v1/schemas/{db_name}", response_model=DatabaseInfo, tags=["Schema"])
async def get_schema(db_name: str):
    """获取指定数据库的 Schema"""
    if db_name not in AVAILABLE_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Database not found: {db_name}")

    schema = AVAILABLE_SCHEMAS[db_name]

    tables = []
    total_columns = 0
    for table_name, table_schema in schema.tables.items():
        columns = table_schema.get("columns", [])
        tables.append(TableInfo(
            name=table_name,
            comment=table_schema.get("table_comment", ""),
            columns=columns
        ))
        total_columns += len(columns)

    return DatabaseInfo(
        db_name=db_name,
        tables=tables,
        total_tables=len(tables),
        total_columns=total_columns
    )


@app.post("/api/v1/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """提交用户反馈"""
    # 在实际应用中，这里应该将反馈存储到数据库或文件中
    logger.info(f"Received feedback: {feedback}")
    return {"status": "success", "message": "Feedback received"}


# ============================================================================
# WebSocket 路由
# =============================================================================

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket 聊天接口"""
    await websocket.accept()

    session_id = str(uuid.uuid4())

    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message = data.get("message", "")
            db_name = data.get("db_name", "sales_db")

            if not message:
                await websocket.send_json({
                    "type": "error",
                    "error": "Message cannot be empty"
                })
                continue

            # 发送处理状态
            await websocket.send_json({
                "type": "status",
                "message": "Processing your query...",
                "session_id": session_id
            })

            # 处理查询
            try:
                service_request = ServiceQueryRequest(
                    question=message,
                    db_name=db_name,
                    session_id=session_id
                )

                if text2sql_service is None:
                    response = QueryResponse(
                        success=True,
                        sql="-- 模拟 SQL",
                        explanation="服务未配置",
                        confidence=0.9,
                        execution_time=0.1,
                        complexity="medium",
                        tables_used=[]
                    )
                else:
                    response = await text2sql_service.query(service_request)

                # 发送结果
                await websocket.send_json({
                    "type": "result",
                    "session_id": session_id,
                    **response.__dict__
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")


# ============================================================================
# 启动脚本
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
