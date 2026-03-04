"""
服务模块初始化
"""
from app.services.pipeline import TextToSQLPipeline, get_pipeline

__all__ = ["TextToSQLPipeline", "get_pipeline"]