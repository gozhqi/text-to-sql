"""
微调方案 - 数据准备模块

提供数据加载、处理和格式化功能，用于 Text-to-SQL 模型微调
"""
import json
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from collections import defaultdict


@dataclass
class SQLExample:
    """SQL 示例数据类"""
    question: str
    sql: str
    database_id: str = ""
    db_id: str = ""
    difficulty: str = ""
    schema: Optional[Dict[str, Any]] = None
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "sql": self.sql,
            "database_id": self.database_id,
            "db_id": self.db_id,
            "difficulty": self.difficulty,
            "schema": self.schema,
            "evidence": self.evidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLExample":
        return cls(
            question=data.get("question", ""),
            sql=data.get("sql", ""),
            database_id=data.get("database_id", data.get("db_id", "")),
            db_id=data.get("db_id", data.get("database_id", "")),
            difficulty=data.get("difficulty", ""),
            schema=data.get("schema"),
            evidence=data.get("evidence")
        )


@dataclass
class DatasetStats:
    """数据集统计"""
    total_examples: int = 0
    databases: int = 0
    avg_question_length: float = 0.0
    avg_sql_length: float = 0.0
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    database_distribution: Dict[str, int] = field(default_factory=dict)


class DataLoader(ABC):
    """数据加载器抽象基类"""

    @abstractmethod
    async def load(self, path: str) -> List[SQLExample]:
        """加载数据"""
        pass


class SpiderDataLoader(DataLoader):
    """Spider 数据集加载器"""

    async def load(self, path: str) -> List[SQLExample]:
        """加载 Spider 数据集"""
        examples = []

        # 加载训练数据
        train_path = os.path.join(path, "train.json")
        if os.path.exists(train_path):
            with open(train_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    examples.append(SQLExample(
                        question=item.get("question", ""),
                        sql=item.get("sql", ""),
                        db_id=item.get("db_id", ""),
                        difficulty=item.get("difficulty", "")
                    ))

        logger.info(f"从 Spider 加载 {len(examples)} 个示例")
        return examples


class JsonDataLoader(DataLoader):
    """JSON 文件加载器"""

    async def load(self, path: str) -> List[SQLExample]:
        """加载 JSON 数据"""
        examples = []

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        examples.append(SQLExample.from_dict(item))

        elif os.path.isdir(path):
            # 加载目录中的所有 JSON 文件
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    filepath = os.path.join(path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                examples.append(SQLExample.from_dict(item))

        logger.info(f"从 JSON 加载 {len(examples)} 个示例")
        return examples


class CSVDataLoader(DataLoader):
    """CSV 文件加载器"""

    async def load(self, path: str) -> List[SQLExample]:
        """加载 CSV 数据"""
        examples = []

        try:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    examples.append(SQLExample(
                        question=row.get("question", ""),
                        sql=row.get("sql", ""),
                        database_id=row.get("database_id", row.get("db_id", "")),
                        difficulty=row.get("difficulty", "")
                    ))
        except Exception as e:
            logger.error(f"加载 CSV 失败: {e}")

        logger.info(f"从 CSV 加载 {len(examples)} 个示例")
        return examples


class DataProcessor:
    """数据处理器"""

    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_schemas()

    def _load_schemas(self):
        """加载数据库模式"""
        if not self.schema_path or not os.path.exists(self.schema_path):
            return

        # 尝试加载 schemas.json
        schema_file = os.path.join(self.schema_path, "schemas.json")
        if os.path.exists(schema_file):
            with open(schema_file, "r", encoding="utf-8") as f:
                self.schemas = json.load(f)

        # 或加载目录中的数据库模式
        elif os.path.isdir(self.schema_path):
            for filename in os.listdir(self.schema_path):
                if filename.endswith(".json"):
                    db_id = filename.replace(".json", "")
                    filepath = os.path.join(self.schema_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        self.schemas[db_id] = json.load(f)

        logger.info(f"加载 {len(self.schemas)} 个数据库模式")

    def format_with_schema(
        self,
        example: SQLExample,
        schema_format: str = "spider"
    ) -> str:
        """格式化示例（包含模式）"""
        schema = self.schemas.get(example.db_id or example.database_id, {})

        if schema_format == "spider":
            return self._format_spider(example, schema)
        elif schema_format == "natural":
            return self._format_natural(example, schema)
        elif schema_format == "instruction":
            return self._format_instruction(example, schema)
        else:
            return self._format_simple(example)

    def _format_spider(self, example: SQLExample, schema: Dict) -> str:
        """Spider 格式"""
        schema_text = self._build_schema_text(schema)

        return f"""{schema_text}

Question: {example.question}
SQL: {example.sql}"""

    def _format_natural(self, example: SQLExample, schema: Dict) -> str:
        """自然语言格式"""
        schema_text = self._build_schema_text_natural(schema)

        return f"""数据库结构：
{schema_text}

用户问题：{example.question}

请生成 SQL 查询语句：
{example.sql}"""

    def _format_instruction(self, example: SQLExample, schema: Dict) -> str:
        """指令格式（适用于指令微调）"""
        schema_text = self._build_schema_text(schema)

        prompt = f"""<指令>
根据以下数据库结构，将用户问题转换为 SQL 查询。

数据库结构：
{schema_text}

<用户问题>
{example.question}
</用户问题>

<SQL 查询>
{example.sql}
</SQL 查询>"""

        return prompt

    def _format_simple(self, example: SQLExample) -> str:
        """简单格式"""
        return f"Question: {example.question}\nSQL: {example.sql}"

    def _build_schema_text(self, schema: Dict) -> str:
        """构建模式文本（Spider 风格）"""
        if not schema:
            return ""

        tables = schema.get("table_names", [])
        columns = schema.get("column_names", [])

        text_parts = []

        # 表列表
        if tables:
            text_parts.append("Tables:")
            for i, table in enumerate(tables):
                text_parts.append(f"  {i}: {table}")

        # 列列表
        if columns:
            text_parts.append("\nColumns:")
            for i, (table_id, column) in enumerate(columns):
                table_name = tables[table_id] if table_id >= 0 and table_id < len(tables) else "*"
                text_parts.append(f"  {i}: {table_name}.{column}")

        return "\n".join(text_parts)

    def _build_schema_text_natural(self, schema: Dict) -> str:
        """构建自然语言模式文本"""
        if not schema:
            return ""

        tables = schema.get("table_names", [])
        columns = schema.get("column_names", [])
        column_types = schema.get("column_types", [])

        # 按表分组列
        table_columns = defaultdict(list)
        for i, (table_id, column) in enumerate(columns):
            if table_id >= 0 and table_id < len(tables):
                col_type = column_types[i] if i < len(column_types) else ""
                table_columns[tables[table_id]].append((column, col_type))

        # 构建文本
        text_parts = []
        for table in tables:
            text_parts.append(f"表 {table}:")
            for column, col_type in table_columns.get(table, []):
                type_str = f" ({col_type})" if col_type else ""
                text_parts.append(f"  - {column}{type_str}")

        return "\n".join(text_parts)


class DatasetFormatter:
    """数据集格式化器"""

    def __init__(self, processor: DataProcessor):
        self.processor = processor

    def format_for_training(
        self,
        examples: List[SQLExample],
        format_type: str = "alpaca",
        schema_format: str = "spider"
    ) -> List[Dict[str, str]]:
        """格式化为训练数据"""
        formatted = []

        for example in examples:
            if format_type == "alpaca":
                formatted.append(self._format_alpaca(example, schema_format))
            elif format_type == "sharegpt":
                formatted.append(self._format_sharegpt(example, schema_format))
            elif format_type == "instruction":
                formatted.append(self._format_instruction(example, schema_format))
            elif format_type == "openai":
                formatted.append(self._format_openai(example, schema_format))
            else:
                formatted.append(self._format_simple(example))

        return formatted

    def _format_alpaca(self, example: SQLExample, schema_format: str) -> Dict[str, str]:
        """Alpaca 格式"""
        schema = self.processor.schemas.get(example.db_id or example.database_id, {})
        schema_text = self.processor._build_schema_text(schema)

        instruction = f"""将以下自然语言问题转换为 SQL 查询。

数据库结构：
{schema_text}"""

        return {
            "instruction": instruction,
            "input": example.question,
            "output": example.sql
        }

    def _format_sharegpt(self, example: SQLExample, schema_format: str) -> Dict[str, Any]:
        """ShareGPT 格式"""
        schema = self.processor.schemas.get(example.db_id or example.database_id, {})
        schema_text = self.processor._build_schema_text(schema)

        return {
            "conversations": [
                {
                    "from": "human",
                    "value": f"""数据库结构：
{schema_text}

问题：{example.question}"""
                },
                {
                    "from": "gpt",
                    "value": example.sql
                }
            ]
        }

    def _format_instruction(self, example: SQLExample, schema_format: str) -> Dict[str, str]:
        """指令格式"""
        prompt = self.processor.format_with_schema(example, schema_format)

        return {
            "prompt": prompt,
            "completion": example.sql
        }

    def _format_openai(self, example: SQLExample, schema_format: str) -> Dict[str, Any]:
        """OpenAI 格式"""
        schema = self.processor.schemas.get(example.db_id or example.database_id, {})
        schema_text = self.processor._build_schema_text(schema)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个 SQL 查询生成助手。根据数据库结构和用户问题，生成正确的 SQL 查询语句。"
                },
                {
                    "role": "user",
                    "content": f"""数据库结构：
{schema_text}

问题：{example.question}"""
                },
                {
                    "role": "assistant",
                    "content": example.sql
                }
            ]
        }

    def _format_simple(self, example: SQLExample) -> Dict[str, str]:
        """简单格式"""
        return {
            "question": example.question,
            "sql": example.sql
        }

    def save_formatted(
        self,
        examples: List[SQLExample],
        output_path: str,
        format_type: str = "alpaca",
        schema_format: str = "spider"
    ):
        """保存格式化数据"""
        formatted = self.format_for_training(examples, format_type, schema_format)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)

        logger.info(f"保存 {len(formatted)} 条格式化数据到 {output_path}")

    def split_dataset(
        self,
        examples: List[SQLExample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by_db: bool = True
    ) -> Tuple[List[SQLExample], List[SQLExample], List[SQLExample]]:
        """分割数据集"""
        if not stratify_by_db:
            # 简单随机分割
            import random
            random.shuffle(examples)
            n = len(examples)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            return (
                examples[:train_end],
                examples[train_end:val_end],
                examples[val_end:]
            )

        # 按数据库分层分割
        from collections import defaultdict
        db_examples = defaultdict(list)
        for ex in examples:
            db_examples[ex.db_id or ex.database_id or "unknown"].append(ex)

        train, val, test = [], [], []

        for db_id, db_ex in db_examples.items():
            import random
            random.shuffle(db_ex)
            n = len(db_ex)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train.extend(db_ex[:train_end])
            val.extend(db_ex[train_end:val_end])
            test.extend(db_ex[val_end:])

        return train, val, test


def calculate_statistics(examples: List[SQLExample]) -> DatasetStats:
    """计算数据集统计"""
    stats = DatasetStats()
    stats.total_examples = len(examples)

    if not examples:
        return stats

    # 数据库分布
    db_set = set()
    db_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)

    question_lengths = []
    sql_lengths = []

    for ex in examples:
        db_id = ex.db_id or ex.database_id
        if db_id:
            db_set.add(db_id)
            db_counts[db_id] += 1

        if ex.difficulty:
            difficulty_counts[ex.difficulty] += 1

        question_lengths.append(len(ex.question))
        sql_lengths.append(len(ex.sql))

    stats.databases = len(db_set)
    stats.avg_question_length = sum(question_lengths) / len(question_lengths)
    stats.avg_sql_length = sum(sql_lengths) / len(sql_lengths)
    stats.difficulty_distribution = dict(difficulty_counts)
    stats.database_distribution = dict(db_counts)

    return stats


def create_data_loader(path: str) -> DataLoader:
    """自动创建合适的数据加载器"""
    if os.path.isdir(path) and "train.json" in os.listdir(path):
        return SpiderDataLoader()
    elif path.endswith(".json"):
        return JsonDataLoader()
    elif path.endswith(".csv"):
        return CSVDataLoader()
    else:
        return JsonDataLoader()


async def load_dataset(
    path: str,
    schema_path: Optional[str] = None,
    format_type: str = "alpaca"
) -> Tuple[List[Dict[str, Any]], DatasetStats]:
    """加载并格式化数据集"""
    # 加载数据
    loader = create_data_loader(path)
    examples = await loader.load(path)

    # 计算统计
    stats = calculate_statistics(examples)
    logger.info(f"数据集统计: {stats}")

    # 处理数据
    processor = DataProcessor(schema_path)
    formatter = DatasetFormatter(processor)

    # 格式化数据
    formatted = formatter.format_for_training(examples, format_type)

    return formatted, stats
