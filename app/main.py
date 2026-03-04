"""
FastAPI 主应用
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os

from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, ChatRequest, ChatResponse
from app.models.database import get_db_manager
from app.services.pipeline import get_pipeline
from app.utils.helpers import generate_session_id, sanitize_input


# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("正在初始化应用...")
    
    # 初始化数据库连接
    await get_db_manager()
    
    # 初始化Pipeline
    await get_pipeline()
    
    logger.info("应用初始化完成")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭应用...")
    db_manager = await get_db_manager()
    await db_manager.close()


# 创建应用
settings = get_settings()
app = FastAPI(
    title="Text-to-SQL 智能查询系统",
    description="基于大语言模型的多轮对话式SQL生成系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== API路由 ==========

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    index_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Text-to-SQL API</h1><p>请访问 /docs 查看API文档</p>"


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    单次查询接口
    
    将自然语言问题转换为SQL并执行
    """
    try:
        # 清理输入
        question = sanitize_input(request.question)
        
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        # 获取Pipeline
        pipeline = await get_pipeline()
        
        # 处理查询
        result = await pipeline.process(
            question=question,
            db_name=request.db_name,
            session_id=request.session_id
        )
        
        return result
        
    except Exception as e:
        logger.exception(f"查询处理失败: {e}")
        return QueryResponse(
            success=False,
            error=f"处理失败: {str(e)}"
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    多轮对话接口
    
    支持上下文理解的连续对话
    """
    try:
        # 清理输入
        message = sanitize_input(request.message)
        
        if not message:
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        # 获取Pipeline
        pipeline = await get_pipeline()
        
        # 处理查询
        result = await pipeline.process(
            question=message,
            db_name=request.db_name,
            session_id=request.session_id
        )
        
        return ChatResponse(
            success=result.success,
            message=result.explanation or result.error or "查询完成",
            sql=result.sql,
            results=result.results,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.exception(f"对话处理失败: {e}")
        return ChatResponse(
            success=False,
            message=f"处理失败: {str(e)}",
            session_id=request.session_id
        )


@app.get("/api/schema/{db_name}")
async def get_schema(db_name: str):
    """
    获取数据库Schema信息
    """
    try:
        pipeline = await get_pipeline()
        result = await pipeline.get_schema_info(db_name)
        return {"success": True, "data": result}
    except Exception as e:
        logger.exception(f"获取Schema失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schema/{db_name}/build-index")
async def build_schema_index(db_name: str):
    """
    构建Schema向量索引
    """
    try:
        pipeline = await get_pipeline()
        result = await pipeline.build_schema_index(db_name)
        return result
    except Exception as e:
        logger.exception(f"构建索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/new")
async def new_session():
    """
    生成新的会话ID
    """
    return {
        "session_id": generate_session_id(),
        "message": "新会话已创建"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "version": "1.0.0"}


# 启动入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )