"""
配置管理模块
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置"""
    
    # LLM配置
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = "gpt-4"
    llm_base_url: Optional[str] = None
    
    # 数据库配置
    db_type: str = "mysql"
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    
    # 向量数据库
    chroma_persist_dir: str = "./data/chroma"
    
    # 安全配置
    allowed_operations: str = "SELECT,SHOW,DESCRIBE,EXPLAIN"
    max_query_rows: int = 10000
    query_timeout: int = 30
    
    # 服务配置
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        if self.db_type == "mysql":
            return f"mysql+aiomysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "postgresql":
            return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    @property
    def sync_database_url(self) -> str:
        """同步数据库连接URL"""
        if self.db_type == "mysql":
            return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "postgresql":
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()