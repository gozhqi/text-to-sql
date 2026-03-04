"""
Schema检索模块 - 基于向量检索的表结构匹配
"""
from typing import List, Dict, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import os

from app.config import get_settings
from app.models.schemas import TableSchema


class SchemaRetriever:
    """Schema检索器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialized = False
    
    async def init(self):
        """初始化向量模型和向量库"""
        if self._initialized:
            return
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 初始化ChromaDB
        persist_dir = self.settings.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self._initialized = True
        logger.info("SchemaRetriever 初始化完成")
    
    def _get_collection_name(self, db_name: str) -> str:
        """获取集合名称"""
        return f"schema_{db_name}".replace("-", "_").replace(".", "_")
    
    def _build_table_document(self, schema: TableSchema) -> str:
        """构建表的语义文档"""
        parts = [
            f"表名: {schema.table_name}",
        ]
        
        if schema.table_comment:
            parts.append(f"描述: {schema.table_comment}")
        
        # 字段信息
        column_info = []
        for col in schema.columns:
            col_desc = f"{col.name}"
            if col.comment:
                col_desc += f" ({col.comment})"
            column_info.append(col_desc)
        
        parts.append(f"字段: {', '.join(column_info)}")
        
        return "\n".join(parts)
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        return self.embedding_model.encode(text, convert_to_numpy=True).tolist()
    
    async def build_schema_index(self, db_name: str, schemas: Dict[str, TableSchema]):
        """构建Schema向量索引"""
        if not self._initialized:
            await self.init()
        
        collection_name = self._get_collection_name(db_name)
        
        # 删除已存在的集合
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        # 创建新集合
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"db_name": db_name}
        )
        
        # 批量添加文档
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for table_name, schema in schemas.items():
            doc = self._build_table_document(schema)
            ids.append(table_name)
            documents.append(doc)
            metadatas.append({
                "table_name": table_name,
                "table_comment": schema.table_comment or ""
            })
            embeddings.append(self._get_embedding(doc))
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        logger.info(f"已为数据库 {db_name} 构建了 {len(ids)} 个表的向量索引")
    
    async def retrieve_relevant_tables(
        self,
        question: str,
        db_name: str,
        top_k: int = 5
    ) -> List[str]:
        """检索与问题相关的表名"""
        if not self._initialized:
            await self.init()
        
        collection_name = self._get_collection_name(db_name)
        
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except Exception as e:
            logger.warning(f"未找到数据库 {db_name} 的Schema索引: {e}")
            return []
        
        # 查询
        query_embedding = self._get_embedding(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )
        
        if not results or not results.get("metadatas"):
            return []
        
        table_names = [
            meta["table_name"] 
            for meta in results["metadatas"][0]
        ]
        
        logger.info(f"问题 '{question}' 检索到相关表: {table_names}")
        return table_names
    
    async def hybrid_retrieve(
        self,
        question: str,
        db_name: str,
        schemas: Dict[str, TableSchema],
        top_k: int = 5
    ) -> List[str]:
        """混合检索：向量检索 + 关键词匹配"""
        
        # 1. 向量检索
        vector_results = await self.retrieve_relevant_tables(question, db_name, top_k)
        
        # 2. 关键词匹配
        question_lower = question.lower()
        keyword_results = []
        
        for table_name, schema in schemas.items():
            # 检查表名
            if table_name.lower() in question_lower:
                keyword_results.append(table_name)
                continue
            
            # 检查表注释
            if schema.table_comment and schema.table_comment.lower() in question_lower:
                keyword_results.append(table_name)
                continue
            
            # 检查字段名
            for col in schema.columns:
                if col.name.lower() in question_lower:
                    keyword_results.append(table_name)
                    break
        
        # 3. 合并去重
        all_results = list(dict.fromkeys(keyword_results + vector_results))
        
        return all_results[:top_k]