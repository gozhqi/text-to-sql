"""
方案一：基于 RAG 的 Text-to-SQL 方案（Vanna.ai 风格）

核心理念：
1. SQL 即知识 - 将历史 SQL 查询作为知识存储
2. 检索增强生成 - 检索相似 SQL 示例来辅助生成
3. 持续学习 - 可以从用户反馈中学习

适用场景：
- 有大量历史 SQL 查询
- 查询模式相对固定
- 业务分析师使用

论文参考：
- Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Text-to-SQL (EMNLP 2018)
- RAG-based approaches for Text-to-SQL (2024-2025)
"""

from typing import List, Dict, Optional, Any
import json
import re
from loguru import logger
from dataclasses import dataclass, field
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import os

from app.config import get_settings


@dataclass
class SQLExample:
    """SQL 示例数据类"""
    question: str
    sql: str
    database: str = ""
    table_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> str:
        """转换为文档格式用于向量化"""
        return f"问题: {self.question}\nSQL: {self.sql}"


@dataclass
class DDLDocument:
    """DDL 文档数据类"""
    table_name: str
    ddl: str
    description: str = ""
    business_rules: List[str] = field(default_factory=list)

    def to_document(self) -> str:
        """转换为文档格式用于向量化"""
        parts = [f"表名: {self.table_name}"]
        if self.description:
            parts.append(f"描述: {self.description}")
        if self.business_rules:
            parts.append(f"业务规则: {'; '.join(self.business_rules)}")
        parts.append(f"DDL:\n{self.ddl}")
        return "\n".join(parts)


class RAGSQLGenerator:
    """
    基于 RAG 的 SQL 生成器

    工作流程：
    1. 训练阶段：将 DDL、业务文档、历史 SQL 存入向量库
    2. 检索阶段：根据问题检索相关的 DDL 和历史 SQL
    3. 生成阶段：使用检索到的上下文构建 Prompt 生成 SQL
    """

    def __init__(self, llm_client=None):
        self.settings = get_settings()
        self.llm_client = llm_client
        self.embedding_model = None
        self.chroma_client = None
        self.ddl_collection = None
        self.sql_collection = None
        self._initialized = False

    async def init(self):
        """初始化 RAG 系统"""
        if self._initialized:
            return

        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )

        # 初始化 ChromaDB
        persist_dir = os.path.join(
            self.settings.chroma_persist_dir,
            "rag_sql"
        )
        os.makedirs(persist_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        self.ddl_collection = self._get_or_create_collection("ddl_docs")
        self.sql_collection = self._get_or_create_collection("sql_examples")

        if self.llm_client is None:
            from openai import AsyncOpenAI
            self.llm_client = AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url
            )

        self._initialized = True
        logger.info("RAGSQLGenerator 初始化完成")

    def _get_or_create_collection(self, name: str):
        """获取或创建集合"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(name=name)

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        return self.embedding_model.encode(
            text,
            convert_to_numpy=True
        ).tolist()

    # ==================== 训练方法 ====================

    async def train_ddl(self, ddl_doc: DDLDocument) -> bool:
        """
        训练 DDL 文档

        Args:
            ddl_doc: DDL 文档对象

        Returns:
            是否训练成功
        """
        if not self._initialized:
            await self.init()

        try:
            doc = ddl_doc.to_document()
            embedding = self._get_embedding(doc)

            self.ddl_collection.add(
                ids=[f"ddl_{ddl_doc.table_name}"],
                documents=[doc],
                embeddings=[embedding],
                metadatas=[{
                    "table_name": ddl_doc.table_name,
                    "type": "ddl",
                    "description": ddl_doc.description
                }]
            )
            logger.info(f"已训练 DDL: {ddl_doc.table_name}")
            return True
        except Exception as e:
            logger.error(f"DDL 训练失败: {e}")
            return False

    async def train_sql(self, example: SQLExample) -> bool:
        """
        训练 SQL 示例

        Args:
            example: SQL 示例对象

        Returns:
            是否训练成功
        """
        if not self._initialized:
            await self.init()

        try:
            doc = example.to_document()
            embedding = self._get_embedding(doc)

            self.sql_collection.add(
                ids=[f"sql_{hash(example.question + example.sql)}"],
                documents=[doc],
                embeddings=[embedding],
                metadatas=[{
                    "question": example.question,
                    "database": example.database,
                    "table_names": json.dumps(example.table_names),
                    "type": "sql_example"
                }]
            )
            logger.info(f"已训练 SQL 示例: {example.question[:50]}...")
            return True
        except Exception as e:
            logger.error(f"SQL 示例训练失败: {e}")
            return False

    async def train_documentation(self, documentation: str, metadata: Dict = None) -> bool:
        """
        训练业务文档

        Args:
            documentation: 业务文档内容
            metadata: 元数据

        Returns:
            是否训练成功
        """
        if not self._initialized:
            await self.init()

        try:
            embedding = self._get_embedding(documentation)

            self.ddl_collection.add(
                ids=[f"doc_{hash(documentation)}"],
                documents=[documentation],
                embeddings=[embedding],
                metadatas=[{"type": "documentation", **(metadata or {})}]
            )
            logger.info(f"已训练业务文档")
            return True
        except Exception as e:
            logger.error(f"业务文档训练失败: {e}")
            return False

    # ==================== 检索方法 ====================

    async def retrieve_relevant_context(
        self,
        question: str,
        db_name: str,
        top_k_ddl: int = 3,
        top_k_sql: int = 3
    ) -> Dict[str, Any]:
        """
        检索相关上下文

        Args:
            question: 用户问题
            db_name: 数据库名称
            top_k_ddl: 检索的 DDL 数量
            top_k_sql: 检索的 SQL 示例数量

        Returns:
            包含相关 DDL 和 SQL 示例的上下文
        """
        if not self._initialized:
            await self.init()

        query_embedding = self._get_embedding(question)

        # 检索相关 DDL
        ddl_results = self.ddl_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k_ddl
        )

        # 检索相关 SQL 示例
        sql_results = self.sql_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k_sql,
            where={"database": db_name} if db_name else None
        )

        # 格式化结果
        context = {
            "relevant_ddls": [],
            "similar_sqls": []
        }

        if ddl_results.get("documents"):
            for i, doc in enumerate(ddl_results["documents"][0]):
                context["relevant_ddls"].append({
                    "content": doc,
                    "metadata": ddl_results["metadatas"][0][i]
                })

        if sql_results.get("documents"):
            for i, doc in enumerate(sql_results["documents"][0]):
                meta = sql_results["metadatas"][0][i]
                context["similar_sqls"].append({
                    "question": meta.get("question", ""),
                    "sql": doc.split("SQL: ")[-1] if "SQL: " in doc else "",
                    "content": doc
                })

        return context

    # ==================== 生成方法 ====================

    async def generate_sql(
        self,
        question: str,
        db_name: str,
        schemas: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        使用 RAG 生成 SQL

        Args:
            question: 用户问题
            db_name: 数据库名称
            schemas: 可选的表结构信息

        Returns:
            包含 SQL、解释等的结果
        """
        if not self._initialized:
            await self.init()

        # 检索相关上下文
        context = await self.retrieve_relevant_context(question, db_name)

        # 构建 Prompt
        prompt = self._build_rag_prompt(question, context, schemas)

        # 调用 LLM
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            content = response.choices[0].message.content
            return self._parse_response(content, context)

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": "",
                "explanation": ""
            }

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的 SQL 生成助手。你的任务是将自然语言问题转换为准确的 SQL 查询。

请遵循以下规则：
1. 只生成 SELECT 查询，不要生成 INSERT/UPDATE/DELETE
2. 使用检索到的相关 DDL 和 SQL 示例作为参考
3. 如果不确定表名或字段名，优先参考检索到的 DDL
4. 如果有相似的 SQL 示例，参考其结构但根据当前问题调整
5. 返回 JSON 格式：{"sql": "你的SQL", "explanation": "解释", "confidence": 0.0-1.0}
"""

    def _build_rag_prompt(
        self,
        question: str,
        context: Dict[str, Any],
        schemas: Optional[List[Any]] = None
    ) -> str:
        """构建 RAG 增强的 Prompt"""
        parts = [f"用户问题: {question}\n"]

        # 添加相关 DDL
        if context["relevant_ddls"]:
            parts.append("\n## 相关表结构 (DDL):")
            for ddl in context["relevant_ddls"]:
                parts.append(f"\n{ddl['content']}")

        # 添加相似 SQL 示例
        if context["similar_sqls"]:
            parts.append("\n## 相似 SQL 示例:")
            for i, sql_ex in enumerate(context["similar_sqls"], 1):
                parts.append(f"\n示例 {i}:")
                parts.append(f"问题: {sql_ex['question']}")
                parts.append(f"SQL: {sql_ex['sql']}")

        # 添加 Schema 信息（如果提供）
        if schemas:
            parts.append("\n## 可用表结构:")
            for schema in schemas:
                parts.append(f"\n表: {schema.table_name}")
                for col in schema.columns:
                    parts.append(f"  - {col.name}: {col.type}")

        parts.append("\n## 请根据上述信息生成 SQL:")

        return "\n".join(parts)

    def _parse_response(self, content: str, context: Dict) -> Dict[str, Any]:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return {
                    "success": bool(data.get("sql")),
                    "sql": data.get("sql", ""),
                    "explanation": data.get("explanation", ""),
                    "confidence": data.get("confidence", 0.5),
                    "context_used": {
                        "ddls_count": len(context.get("relevant_ddls", [])),
                        "examples_count": len(context.get("similar_sqls", []))
                    }
                }
            except json.JSONDecodeError:
                pass

        # 尝试提取 SQL
        sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return {
                "success": True,
                "sql": sql_match.group(1).strip(),
                "explanation": content,
                "context_used": {
                    "ddls_count": len(context.get("relevant_ddls", [])),
                    "examples_count": len(context.get("similar_sqls", []))
                }
            }

        return {
            "success": False,
            "error": "无法解析响应",
            "raw_response": content
        }

    # ==================== 批量训练 ====================

    async def bulk_train(
        self,
        ddl_docs: List[DDLDocument] = None,
        sql_examples: List[SQLExample] = None,
        documentations: List[str] = None
    ) -> Dict[str, int]:
        """
        批量训练

        Returns:
            训练统计信息
        """
        stats = {"ddl": 0, "sql": 0, "doc": 0}

        if ddl_docs:
            for doc in ddl_docs:
                if await self.train_ddl(doc):
                    stats["ddl"] += 1

        if sql_examples:
            for example in sql_examples:
                if await self.train_sql(example):
                    stats["sql"] += 1

        if documentations:
            for doc in documentations:
                if await self.train_documentation(doc):
                    stats["doc"] += 1

        logger.info(f"批量训练完成: {stats}")
        return stats

    async def get_training_stats(self) -> Dict[str, int]:
        """获取训练统计"""
        stats = {
            "ddl_count": self.ddl_collection.count() if self.ddl_collection else 0,
            "sql_count": self.sql_collection.count() if self.sql_collection else 0
        }
        return stats


# ==================== 便捷函数 ====================

async def create_rag_generator(llm_client=None) -> RAGSQLGenerator:
    """创建 RAG SQL 生成器"""
    generator = RAGSQLGenerator(llm_client)
    await generator.init()
    return generator


# ==================== 示例用法 ====================

async def example_usage():
    """示例用法"""

    # 1. 创建生成器
    generator = await create_rag_generator()

    # 2. 训练 DDL
    await generator.train_ddl(DDLDocument(
        table_name="orders",
        ddl="CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, total_amount DECIMAL(10,2), status VARCHAR(20), created_at TIMESTAMP)",
        description="订单表，存储客户订单信息",
        business_rules=["status 只能是 pending/completed/cancelled"]
    ))

    await generator.train_ddl(DDLDocument(
        table_name="products",
        ddl="CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(100), category VARCHAR(50), price DECIMAL(10,2))",
        description="产品表"
    ))

    # 3. 训练 SQL 示例
    await generator.train_sql(SQLExample(
        question="查询销售额最高的前10个产品",
        sql="SELECT p.name, SUM(o.total_amount) as revenue FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY revenue DESC LIMIT 10",
        database="sales_db",
        table_names=["products", "order_items"]
    ))

    await generator.train_sql(SQLExample(
        question="查询本月的订单总数",
        sql="SELECT COUNT(*) as total_orders FROM orders WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)",
        database="sales_db",
        table_names=["orders"]
    ))

    # 4. 训练业务文档
    await generator.train_documentation(
        "销售额定义为订单的 total_amount 字段之和，只计算 status='completed' 的订单"
    )

    # 5. 生成 SQL
    result = await generator.generate_sql(
        question="查询上个月销售额最高的5个产品类别",
        db_name="sales_db"
    )

    print(f"生成的 SQL: {result['sql']}")
    print(f"解释: {result['explanation']}")

    # 6. 查看训练统计
    stats = await generator.get_training_stats()
    print(f"训练统计: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
