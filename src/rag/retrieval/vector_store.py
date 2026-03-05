"""
RAG 方案 - 向量检索模块

使用向量数据库进行语义检索
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    metadata: Dict[str, Any]
    score: float
    type: str  # "ddl", "sql", "doc"


class VectorStore:
    """向量存储和检索"""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.client = None
        self.collections = {}
        self._initialized = False

    async def init(self):
        """初始化向量存储"""
        if self._initialized:
            return

        # 初始化嵌入模型
        logger.info(f"加载嵌入模型: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # 初始化 ChromaDB
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        self._init_collections()

        self._initialized = True
        logger.info("向量存储初始化完成")

    def _init_collections(self):
        """初始化集合"""
        collection_names = ["ddl_docs", "sql_examples", "business_docs"]

        for name in collection_names:
            try:
                self.collections[name] = self.client.get_collection(name)
                logger.info(f"加载已有集合: {name}")
            except:
                self.collections[name] = self.client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"创建新集合: {name}")

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        return self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

    async def add_ddl(
        self,
        table_name: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """添加 DDL 文档"""
        if not self._initialized:
            await self.init()

        try:
            collection = self.collections["ddl_docs"]
            doc_id = f"ddl_{table_name}"

            # 检查是否已存在
            try:
                collection.get(ids=[doc_id])
                # 已存在，删除旧记录
                collection.delete(ids=[doc_id])
            except:
                pass

            embedding = self._get_embedding(content)

            collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    "table_name": table_name,
                    "type": "ddl",
                    **(metadata or {})
                }]
            )

            logger.debug(f"添加 DDL 向量: {table_name}")
            return True
        except Exception as e:
            logger.error(f"添加 DDL 向量失败: {e}")
            return False

    async def add_sql_example(
        self,
        example_id: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """添加 SQL 示例"""
        if not self._initialized:
            await self.init()

        try:
            collection = self.collections["sql_examples"]
            doc_id = f"sql_{example_id}"

            # 检查是否已存在
            try:
                collection.get(ids=[doc_id])
                collection.delete(ids=[doc_id])
            except:
                pass

            embedding = self._get_embedding(content)

            collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    "example_id": example_id,
                    "type": "sql",
                    **(metadata or {})
                }]
            )

            logger.debug(f"添加 SQL 示例向量: {example_id}")
            return True
        except Exception as e:
            logger.error(f"添加 SQL 示例向量失败: {e}")
            return False

    async def add_business_doc(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """添加业务文档"""
        if not self._initialized:
            await self.init()

        try:
            collection = self.collections["business_docs"]
            doc_id = f"doc_{doc_id}"

            # 检查是否已存在
            try:
                collection.get(ids=[doc_id])
                collection.delete(ids=[doc_id])
            except:
                pass

            embedding = self._get_embedding(content)

            collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    "doc_id": doc_id,
                    "type": "doc",
                    **(metadata or {})
                }]
            )

            logger.debug(f"添加业务文档向量: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"添加业务文档向量失败: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        collection_name: str = "sql_examples",
        top_k: int = 5,
        filter: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        """检索相关文档"""
        if not self._initialized:
            await self.init()

        try:
            collection = self.collections.get(collection_name)
            if not collection:
                logger.warning(f"集合不存在: {collection_name}")
                return []

            query_embedding = self._get_embedding(query)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter
            )

            retrieval_results = []
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distances = results.get("distances", [[]])
                    score = 1 - distances[0][i] if distances and i < len(distances[0]) else 0.0

                    retrieval_results.append(RetrievalResult(
                        content=doc,
                        metadata=metadata,
                        score=score,
                        type=metadata.get("type", collection_name)
                    ))

            return retrieval_results

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    async def retrieve_multi(
        self,
        query: str,
        top_k_ddl: int = 3,
        top_k_sql: int = 3,
        top_k_doc: int = 2,
        database: str = ""
    ) -> Dict[str, List[RetrievalResult]]:
        """从多个集合中检索"""
        results = {
            "ddl": [],
            "sql": [],
            "doc": []
        }

        # 检索 DDL
        ddl_results = await self.retrieve(
            query,
            "ddl_docs",
            top_k_ddl
        )
        results["ddl"] = ddl_results

        # 检索 SQL 示例
        sql_filter = {"database": database} if database else None
        sql_results = await self.retrieve(
            query,
            "sql_examples",
            top_k_sql,
            sql_filter
        )
        results["sql"] = sql_results

        # 检索业务文档
        doc_filter = {"database": database} if database else None
        doc_results = await self.retrieve(
            query,
            "business_docs",
            top_k_doc,
            doc_filter
        )
        results["doc"] = doc_results

        return results

    async def delete(
        self,
        doc_id: str,
        collection_name: str
    ) -> bool:
        """删除文档"""
        if not self._initialized:
            await self.init()

        try:
            collection = self.collections.get(collection_name)
            if collection:
                collection.delete(ids=[doc_id])
                logger.debug(f"删除向量: {doc_id} from {collection_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False

    async def clear_collection(self, collection_name: str) -> bool:
        """清空集合"""
        if not self._initialized:
            await self.init()

        try:
            collection = self.collections.get(collection_name)
            if collection:
                # 删除并重新创建集合
                self.client.delete_collection(name=collection_name)
                self.collections[collection_name] = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"清空集合: {collection_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, int]:
        """获取集合统计"""
        if not self._initialized:
            await self.init()

        stats = {}
        for name, collection in self.collections.items():
            try:
                stats[name] = collection.count()
            except:
                stats[name] = 0
        return stats


class HybridRetriever:
    """混合检索器：结合向量检索和关键词检索"""

    def __init__(self, vector_store: VectorStore, knowledge_base):
        self.vector_store = vector_store
        self.knowledge_base = knowledge_base
        self.alpha = 0.7  # 向量检索权重

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        database: str = ""
    ) -> List[Dict[str, Any]]:
        """混合检索"""
        # 向量检索
        vector_results = await self.vector_store.retrieve_multi(
            query,
            top_k_ddl=top_k,
            top_k_sql=top_k,
            top_k_doc=top_k // 2,
            database=database
        )

        # 关键词检索
        keyword_results = await self.knowledge_base.search(query, limit=top_k)

        # 合并结果
        combined = self._combine_results(
            vector_results,
            keyword_results,
            self.alpha
        )

        return combined[:top_k]

    def _combine_results(
        self,
        vector_results: Dict[str, List[RetrievalResult]],
        keyword_results: List[Dict[str, Any]],
        alpha: float
    ) -> List[Dict[str, Any]]:
        """合并检索结果"""
        combined = []

        # 处理向量检索结果
        for type_name, results in vector_results.items():
            for result in results:
                combined.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score * alpha,
                    "type": result.type,
                    "source": "vector"
                })

        # 处理关键词检索结果
        for result in keyword_results:
            # 归一化关键词分数
            normalized_score = min(result["score"] / 10.0, 1.0) * (1 - alpha)

            combined.append({
                "content": result.get("data", {}),
                "metadata": result.get("data", {}),
                "score": normalized_score,
                "type": result["type"],
                "source": "keyword"
            })

        # 按分数排序
        combined.sort(key=lambda x: x["score"], reverse=True)

        return combined


def create_vector_store(
    persist_dir: str = "./data/chroma",
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> VectorStore:
    """创建向量存储实例"""
    return VectorStore(persist_dir, embedding_model)


def create_hybrid_retriever(
    vector_store: VectorStore,
    knowledge_base
) -> HybridRetriever:
    """创建混合检索器"""
    return HybridRetriever(vector_store, knowledge_base)
