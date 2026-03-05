"""
Text-to-SQL Training Pipeline - Data Preparation Module

数据准备模块，负责:
1. Schema 提取和存储
2. 训练数据构造
3. 数据增强
4. 负样本构造
"""
import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
import aiomysql
from loguru import logger


class DatabaseType(Enum):
    """支持的数据库类型"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


@dataclass
class ColumnSchema:
    """列结构定义"""
    name: str
    type: str
    nullable: bool = True
    default: Optional[str] = None
    comment: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_refs: Optional[str] = None  # "table_name(column_name)"


@dataclass
class TableSchema:
    """表结构定义"""
    table_name: str
    columns: List[ColumnSchema] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, str] = field(default_factory=dict)  # {column: "ref_table(ref_column)"}
    table_comment: str = ""
    database_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "table_name": self.table_name,
            "table_comment": self.table_comment,
            "columns": [
                {
                    "name": col.name,
                    "type": col.type,
                    "nullable": col.nullable,
                    "default": col.default,
                    "comment": col.comment,
                    "is_primary_key": col.is_primary_key,
                    "is_foreign_key": col.is_foreign_key,
                    "foreign_key_refs": col.foreign_key_refs
                }
                for col in self.columns
            ],
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys
        }

    def to_prompt_text(self) -> str:
        """转换为 Prompt 文本格式"""
        text = f"## 表名: {self.table_name}\n"
        if self.table_comment:
            text += f"说明: {self.table_comment}\n"

        text += "\n### 列信息:\n"
        text += "| 列名 | 类型 | 可空 | 默认值 | 说明 |\n"
        text += "|------|------|------|--------|------|\n"

        for col in self.columns:
            pk_flag = " [PK]" if col.is_primary_key else ""
            fk_flag = " [FK]" if col.is_foreign_key else ""
            text += f"| {col.name}{pk_flag}{fk_flag} | {col.type} | {'是' if col.nullable else '否'} | {col.default or '-'} | {col.comment} |\n"

        if self.foreign_keys:
            text += "\n### 外键关系:\n"
            for col, ref in self.foreign_keys.items():
                text += f"- {col} → {ref}\n"

        return text + "\n"


class SchemaExtractor:
    """Schema 提取器 - 从数据库自动提取表结构"""

    def __init__(self, db_type: DatabaseType, connection_params: Dict[str, Any]):
        self.db_type = db_type
        self.connection_params = connection_params
        self._connection = None

    async def connect(self):
        """建立数据库连接"""
        if self.db_type == DatabaseType.POSTGRESQL:
            self._connection = await asyncpg.connect(**self.connection_params)
        elif self.db_type == DatabaseType.MYSQL:
            self._connection = await aiomysql.connect(**self.connection_params)
        # SQLite 不需要连接池

    async def close(self):
        """关闭连接"""
        if self._connection:
            if self.db_type == DatabaseType.POSTGRESQL:
                await self._connection.close()
            elif self.db_type == DatabaseType.MYSQL:
                self._connection.close()

    async def extract_all_schemas(self, database_name: str) -> Dict[str, TableSchema]:
        """提取数据库中所有表的 Schema"""
        await self.connect()

        # 获取所有表名
        tables = await self._get_all_tables(database_name)

        schemas = {}
        for table_name in tables:
            schema = await self._extract_table_schema(database_name, table_name)
            schemas[table_name] = schema
            logger.info(f"Extracted schema for table: {table_name}")

        await self.close()
        return schemas

    async def _get_all_tables(self, database_name: str) -> List[str]:
        """获取所有表名"""
        if self.db_type == DatabaseType.POSTGRESQL:
            query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            rows = await self._connection.fetch(query)
            return [row['table_name'] for row in rows]

        elif self.db_type == DatabaseType.MYSQL:
            async with self._connection.cursor() as cursor:
                await cursor.execute(f"SHOW TABLES FROM `{database_name}`")
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

        return []

    async def _extract_table_schema(self, database_name: str, table_name: str) -> TableSchema:
        """提取单个表的 Schema"""
        if self.db_type == DatabaseType.POSTGRESQL:
            return await self._extract_postgres_schema(database_name, table_name)
        elif self.db_type == DatabaseType.MYSQL:
            return await self._extract_mysql_schema(database_name, table_name)
        return TableSchema(table_name=table_name)

    async def _extract_postgres_schema(self, database_name: str, table_name: str) -> TableSchema:
        """提取 PostgreSQL 表结构"""
        # 获取列信息
        columns_query = """
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                pgd.description as comment,
                pk.primary_key is not null as is_primary_key
            FROM information_schema.columns c
            LEFT JOIN pg_catalog.pg_description pgd
                ON pgd.objoid = (
                    SELECT oid FROM pg_class WHERE relname = c.table_name
                )
                AND pgd.objsubid = c.ordinal_position
            LEFT JOIN (
                SELECT ku.column_name, TRUE as primary_key
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.table_name = $1
                AND tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.column_name = pk.column_name
            WHERE c.table_name = $1
            ORDER BY c.ordinal_position
        """

        rows = await self._connection.fetch(columns_query, table_name)

        columns = []
        for row in rows:
            columns.append(ColumnSchema(
                name=row['column_name'],
                type=row['data_type'],
                nullable=row['is_nullable'] == 'YES',
                default=row['column_default'],
                comment=row['comment'] or '',
                is_primary_key=row['is_primary_key'] or False
            ))

        # 获取表注释
        table_comment_query = """
            SELECT obj_description((SELECT oid FROM pg_class WHERE relname = $1)::regclass, 'pg_class') as comment
        """
        table_comment_row = await self._connection.fetchrow(table_comment_query, table_name)
        table_comment = table_comment_row['comment'] or ""

        # 获取外键
        foreign_keys_query = """
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = $1
        """
        fk_rows = await self._connection.fetch(foreign_keys_query, table_name)

        foreign_keys = {}
        for row in fk_rows:
            column = row['column_name']
            foreign_keys[column] = f"{row['foreign_table_name']}({row['foreign_column_name']})"

            # 标记列为外键
            for col in columns:
                if col.name == column:
                    col.is_foreign_key = True
                    col.foreign_key_refs = foreign_keys[column]

        # 主键列表
        primary_keys = [col.name for col in columns if col.is_primary_key]

        return TableSchema(
            table_name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            table_comment=table_comment,
            database_name=database_name
        )

    async def _extract_mysql_schema(self, database_name: str, table_name: str) -> TableSchema:
        """提取 MySQL 表结构"""
        async with self._connection.cursor() as cursor:
            # 获取列信息
            await cursor.execute(f"""
                SELECT
                    COLUMN_NAME as column_name,
                    DATA_TYPE as data_type,
                    IS_NULLABLE as is_nullable,
                    COLUMN_DEFAULT as column_default,
                    COLUMN_COMMENT as column_comment,
                    COLUMN_KEY as column_key
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION
            """, (database_name, table_name))

            rows = await cursor.fetchall()

            columns = []
            primary_keys = []

            for row in rows:
                col = ColumnSchema(
                    name=row[0],
                    type=row[1],
                    nullable=row[2] == 'YES',
                    default=row[3],
                    comment=row[4],
                    is_primary_key=row[5] == 'PRI'
                )
                columns.append(col)
                if col.is_primary_key:
                    primary_keys.append(col.name)

            # 获取表注释
            await cursor.execute(f"""
                SELECT TABLE_COMMENT
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (database_name, table_name))
            table_comment_row = await cursor.fetchone()
            table_comment = table_comment_row[0] if table_comment_row else ""

            # 获取外键
            await cursor.execute(f"""
                SELECT
                    COLUMN_NAME as column_name,
                    REFERENCED_TABLE_NAME as ref_table,
                    REFERENCED_COLUMN_NAME as ref_column
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                AND REFERENCED_TABLE_NAME IS NOT NULL
            """, (database_name, table_name))

            fk_rows = await cursor.fetchall()
            foreign_keys = {}

            for row in fk_rows:
                column, ref_table, ref_column = row
                foreign_keys[column] = f"{ref_table}({ref_column})"

                # 标记列为外键
                for col in columns:
                    if col.name == column:
                        col.is_foreign_key = True
                        col.foreign_key_refs = foreign_keys[column]

            return TableSchema(
                table_name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                table_comment=table_comment,
                database_name=database_name
            )


class TrainingDataBuilder:
    """训练数据构建器"""

    def __init__(self, schemas: Dict[str, TableSchema]):
        self.schemas = schemas

    def build_from_spider(self, spider_data_path: str) -> List[Dict[str, Any]]:
        """从 Spider 数据集构建训练数据"""
        training_data = []

        # Spider 数据集格式
        # data/spider/train.json
        # data/spider/tables.json

        with open(os.path.join(spider_data_path, "train.json"), 'r', encoding='utf-8') as f:
            spider_data = json.load(f)

        for item in spider_data:
            # 获取相关表结构
            db_id = item['db_id']
            table_names = item.get('table_used', [])

            relevant_schemas = {
                name: self.schemas.get(name, TableSchema(table_name=name))
                for name in table_names
            }

            training_data.append({
                "id": f"spider_{item['question_id']}",
                "question": item['question'],
                "sql": item['query'],
                "schemas": {name: schema.to_dict() for name, schema in relevant_schemas.items()},
                "difficulty": item.get('difficulty', 'unknown'),
                "tags": self._infer_tags(item['query'])
            })

        logger.info(f"Built {len(training_data)} training samples from Spider dataset")
        return training_data

    def build_from_pairs(self, question_sql_pairs: List[Tuple[str, str, List[str]]]) -> List[Dict[str, Any]]:
        """从问题-SQL对构建训练数据"""
        training_data = []

        for question, sql, table_names in question_sql_pairs:
            relevant_schemas = {
                name: self.schemas.get(name, TableSchema(table_name=name))
                for name in table_names
            }

            training_data.append({
                "id": f"custom_{len(training_data)}",
                "question": question,
                "sql": sql,
                "schemas": {name: schema.to_dict() for name, schema in relevant_schemas.items()},
                "difficulty": self._estimate_difficulty(sql),
                "tags": self._infer_tags(sql)
            })

        return training_data

    def _infer_tags(self, sql: str) -> List[str]:
        """推断 SQL 标签"""
        tags = []
        sql_upper = sql.upper()

        if " JOIN " in sql_upper:
            tags.append("join")
        if " GROUP BY " in sql_upper:
            tags.append("aggregate")
        if " ORDER BY " in sql_upper:
            tags.append("order_by")
        if " HAVING " in sql_upper:
            tags.append("having")
        if " UNION " in sql_upper:
            tags.append("union")
        if " INTERSECT " in sql_upper:
            tags.append("intersect")
        if " EXCEPT " in sql_upper:
            tags.append("except")
        if any(kw in sql_upper for kw in ["(SELECT", "SELECT * FROM"]):
            tags.append("subquery")
        if " CASE " in sql_upper:
            tags.append("case_when")
        if " DISTINCT " in sql_upper:
            tags.append("distinct")

        if not tags:
            tags.append("simple_select")

        return tags

    def _estimate_difficulty(self, sql: str) -> str:
        """估算 SQL 难度"""
        tags = self._infer_tags(sql)
        score = len(tags)

        if score <= 1:
            return "easy"
        elif score <= 3:
            return "medium"
        elif score <= 5:
            return "hard"
        else:
            return "extra_hard"


class DataAugmenter:
    """数据增强器"""

    def __init__(self, schemas: Dict[str, TableSchema]):
        self.schemas = schemas

    def augment(self, sample: Dict[str, Any], augment_count: int = 3) -> List[Dict[str, Any]]:
        """对单个样本进行增强"""
        augmented = [sample]

        # 问题改写
        for _ in range(augment_count):
            rewritten_question = self._rewrite_question(sample["question"])
            if rewritten_question and rewritten_question != sample["question"]:
                new_sample = sample.copy()
                new_sample["id"] = f"{sample['id']}_aug_{len(augmented)}"
                new_sample["question"] = rewritten_question
                augmented.append(new_sample)

        return augmented

    def _rewrite_question(self, question: str) -> Optional[str]:
        """问题改写"""
        # 同义词替换
        synonyms = {
            "查询": ["找出", "获取", "显示", "列出", "搜索"],
            "所有": ["全部", "每个"],
            "数量": ["个数", "总数", "计数"],
            "平均值": ["平均数", "均值"],
            "最大值": ["最多", "最大"],
            "最小值": ["最少", "最小"],
        }

        import random
        for original, replacements in synonyms.items():
            if original in question:
                replacement = random.choice(replacements)
                return question.replace(original, replacement, 1)

        return None


class NegativeSampleGenerator:
    """负样本生成器"""

    def __init__(self, schemas: Dict[str, TableSchema]):
        self.schemas = schemas

    def generate(self, positive_samples: List[Dict[str, Any]], ratio: float = 0.2) -> List[Dict[str, Any]]:
        """生成负样本"""
        negative_count = int(len(positive_samples) * ratio)
        negative_samples = []

        for i in range(negative_count):
            sample = positive_samples[i % len(positive_samples)]

            # 随机选择一种负样本类型
            neg_type = i % 3

            if neg_type == 0:
                neg_sample = self._introduce_syntax_error(sample)
            elif neg_type == 1:
                neg_sample = self._wrong_schema_reference(sample)
            else:
                neg_sample = self._semantic_drift(sample)

            neg_sample["id"] = f"{sample['id']}_neg_{i}"
            neg_sample["is_negative"] = True
            neg_sample["error_type"] = ["syntax", "schema", "semantic"][neg_type]
            negative_samples.append(neg_sample)

        return negative_samples

    def _introduce_syntax_error(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """引入语法错误"""
        sql = sample["sql"]

        # 随机引入错误
        error_types = [
            ("missing_parenthesis", lambda s: s.replace("(", "", 1)),
            ("missing_comma", lambda s: re.sub(r',\s*', ' ', s, count=1)),
            ("wrong_keyword", lambda s: s.replace("SELECT", "SELCT", 1)),
        ]

        error_type, error_func = random.choice(error_types)
        broken_sql = error_func(sql)

        return {**sample, "sql": broken_sql}

    def _wrong_schema_reference(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """错误的 Schema 引用"""
        sql = sample["sql"]

        # 随机替换一个表名
        tables = list(self.schemas.keys())
        if not tables:
            return sample

        for table_name in sample.get("schemas", {}).keys():
            wrong_table = random.choice([t for t in tables if t != table_name])
            sql = sql.replace(table_name, wrong_table, 1)
            break

        return {**sample, "sql": sql}

    def _semantic_drift(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """语义偏离"""
        question = sample["question"]

        # 随机修改问题中的关键词
        drifts = [
            ("最大", "最小"),
            ("前", "后"),
            ("升序", "降序"),
            ("大于", "小于"),
        ]

        for original, replacement in drifts:
            if original in question:
                question = question.replace(original, replacement, 1)
                break

        return {**sample, "question": question}


class DatasetManager:
    """数据集管理器"""

    def __init__(self, output_dir: str = "data/training"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_dataset(
        self,
        training_data: List[Dict[str, Any]],
        name: str,
        train_ratio: float = 0.9
    ):
        """保存数据集"""
        # 打乱数据
        import random
        random.shuffle(training_data)

        # 划分训练集和验证集
        split_idx = int(len(training_data) * train_ratio)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]

        # 保存
        train_path = os.path.join(self.output_dir, f"{name}_train.jsonl")
        val_path = os.path.join(self.output_dir, f"{name}_val.jsonl")

        with open(train_path, 'w', encoding='utf-8') as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        with open(val_path, 'w', encoding='utf-8') as f:
            for sample in val_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Saved dataset: {len(train_data)} train, {len(val_data)} val")
        logger.info(f"Train: {train_path}")
        logger.info(f"Val: {val_path}")

        return {
            "train_path": train_path,
            "val_path": val_path,
            "train_count": len(train_data),
            "val_count": len(val_data)
        }

    def load_dataset(self, train_path: str, val_path: str) -> Tuple[List[Dict], List[Dict]]:
        """加载数据集"""
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]

        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = [json.loads(line) for line in f]

        logger.info(f"Loaded dataset: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data


# CLI 工具
async def main():
    """命令行工具主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL 数据准备工具")
    parser.add_argument("--action", choices=["extract", "build", "augment"], required=True)
    parser.add_argument("--db-type", choices=["mysql", "postgresql"], default="postgresql")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--db-user", required=True)
    parser.add_argument("--db-password", required=True)
    parser.add_argument("--spider-path", help="Spider 数据集路径")
    parser.add_argument("--output-dir", default="data/training")

    args = parser.parse_args()

    # 数据库连接参数
    connection_params = {
        "host": args.db_host,
        "port": args.db_port,
        "database": args.db_name,
        "user": args.db_user,
        "password": args.db_password
    }

    if args.action == "extract":
        # 提取 Schema
        extractor = SchemaExtractor(DatabaseType(args.db_type), connection_params)
        schemas = await extractor.extract_all_schemas(args.db_name)

        # 保存 Schema
        output_path = os.path.join(args.output_dir, f"{args.db_name}_schemas.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({name: schema.to_dict() for name, schema in schemas.items()}, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved schemas to: {output_path}")

    elif args.action == "build":
        # 构建训练数据
        # 加载 Schema
        schema_path = os.path.join(args.output_dir, f"{args.db_name}_schemas.json")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)
            schemas = {name: TableSchema(**data) for name, data in schemas_data.items()}

        if args.spider_path:
            # 从 Spider 构建
            builder = TrainingDataBuilder(schemas)
            training_data = builder.build_from_spider(args.spider_path)
        else:
            # 从自定义数据构建
            builder = TrainingDataBuilder(schemas)
            training_data = builder.build_from_pairs([])

        # 保存数据集
        manager = DatasetManager(args.output_dir)
        manager.save_dataset(training_data, f"{args.db_name}_text2sql")

    elif args.action == "augment":
        # 数据增强
        schema_path = os.path.join(args.output_dir, f"{args.db_name}_schemas.json")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schemas_data = json.load(f)
            schemas = {name: TableSchema(**data) for name, data in schemas_data.items()}

        # 加载原始数据
        train_path = os.path.join(args.output_dir, f"{args.db_name}_text2sql_train.jsonl")
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]

        # 增强数据
        augmenter = DataAugmenter(schemas)
        augmented_data = []
        for sample in train_data:
            augmented_data.extend(augmenter.augment(sample, augment_count=2))

        # 生成负样本
        neg_generator = NegativeSampleGenerator(schemas)
        negative_samples = neg_generator.generate(augmented_data, ratio=0.15)
        augmented_data.extend(negative_samples)

        # 保存增强后的数据
        manager = DatasetManager(args.output_dir)
        manager.save_dataset(augmented_data, f"{args.db_name}_text2sql_augmented")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
