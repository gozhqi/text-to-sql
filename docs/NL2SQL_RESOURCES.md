# NL2SQL 论文与资源汇总

> **更新时间**: 2026-03-09

---

## 一、综述论文

### 1.1 必读综述

| 论文 | 发表 | 链接 | 核心内容 |
|------|------|------|----------|
| Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL | TKDE 2025 | [arXiv:2406.08426](https://arxiv.org/abs/2406.08426) | 首个系统性 LLM-based Text-to-SQL 综述 |
| A Survey of Text-to-SQL in the Era of LLMs | TKDE 2025 | [arXiv:2408.05109](https://arxiv.org/abs/2408.05109) | LLM 时代的 Text-to-SQL 综述 |
| Natural Language to SQL: State of the Art and Open Problems | VLDB 2025 | [PDF](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/VLDB25-NL2SQL.pdf) | 清华大学 NL2SQL 综述 |

### 1.2 GitHub 资源

- [Awesome-LLM-based-Text2SQL](https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL) - TKDE 2025 论文配套资源
- [Awesome-Text2SQL](https://github.com/topics/text-to-sql) - GitHub Topic 汇总

---

## 二、代表性方法论文

### 2.1 In-Context Learning 方法

#### 任务分解 (Decomposition)

| 论文 | 发表 | 核心贡献 |
|------|------|----------|
| [DIN-SQL](https://arxiv.org/abs/2304.11015) | NeurIPS 2023 | 四阶段流水线：Schema Linking → 分类 → 生成 → 修正 |
| [C3](https://arxiv.org/abs/2307.07306) | arXiv 2023 | Schema Linking + Calibration + Voting |
| [MAC-SQL](https://arxiv.org/abs/2312.11242) | arXiv 2023 | 多 Agent 协作框架 |
| [CHESS](https://arxiv.org/abs/2404.09533) | arXiv 2024 | Schema 选择 → 分解 → 精炼 |

#### 提示优化 (Prompt Optimization)

| 论文 | 发表 | 核心贡献 |
|------|------|----------|
| [DAIL-SQL](https://arxiv.org/abs/2308.10563) | VLDB 2024 | 问题-SQL 相似度选择示例，SOTA Spider |
| [ACT-SQL](https://arxiv.org/abs/2310.06902) | arXiv 2023 | 自适应提示构建 |
| [PET-SQL](https://arxiv.org/abs/2403.09732) | arXiv 2024 | 两阶段提示精炼 |

#### 执行修正 (Execution Refinement)

| 论文 | 发表 | 核心贡献 |
|------|------|----------|
| [Self-Debugging](https://arxiv.org/abs/2304.05128) | ICLR 2024 | SQL 执行错误自动修正 |
| [LEVER](https://arxiv.org/abs/2304.00899) | arXiv 2023 | 执行验证 + 错误反馈 |

### 2.2 Fine-tuning 方法

| 论文 | 发表 | 核心贡献 |
|------|------|----------|
| [CodeS](https://arxiv.org/abs/2402.16347) | ACL 2024 | SQL 专用预训练模型 |
| [DTS-SQL](https://arxiv.org/abs/2402.18347) | arXiv 2024 | 领域迁移学习 |
| [RESDSQL](https://arxiv.org/abs/2302.05965) | ACL 2023 | 解耦 Schema Linking 和 SQL 生成 |

---

## 三、基准数据集

### 3.1 标准基准

| 数据集 | 特点 | 链接 |
|--------|------|------|
| [Spider](https://yale-lily.github.io/spider) | 200 数据库，10K 问题 | 跨域标准基准 |
| [BIRD](https://bird-bench.github.io/) | 95 数据库，12K 问题 | 企业级复杂度 |
| [WikiSQL](https://github.com/salesforce/WikiSQL) | 265 表，80K 问题 | 单表查询 |
| [Spider 2.0](https://spider2-sql.github.io/) | 真实企业场景 | 长上下文挑战 |

### 3.2 BIRD Benchmark 特点

- **知识增强**: 外部领域知识标注
- **复杂推理**: 数值推理、同义词、领域知识
- **执行效率**: VES 指标评估 SQL 效率
- **BIRD-CRITIC**: 推理挑战测试集

### 3.3 Spider 2.0 特点

- **真实场景**: BigQuery、Snowflake 生产环境
- **长上下文**: 平均 SQL ~150 tokens
- **企业级 Schema**: 数百到数千列
- **多方言**: 不同 SQL 方言支持

---

## 四、开源项目

### 4.1 生产级框架

| 项目 | 特点 | 链接 |
|------|------|------|
| [Vanna.ai](https://github.com/vanna-ai/vanna) | RAG + 多数据库 + 多轮对话 | ⭐ 15K+ |
| [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) | 多模型 + 知识库 + 可视化 | ⭐ 12K+ |
| [LangChain SQL](https://python.langchain.com/docs/use_cases/sql) | Agent 架构 | 集成框架 |

### 4.2 研究/实验项目

| 项目 | 特点 | 链接 |
|------|------|------|
| [DIN-SQL](https://github.com/mohammadgpb/DIN-SQL) | 任务分解 | 论文实现 |
| [DAIL-SQL](https://github.com/HKUNLP/DAIL-SQL) | 提示优化 | 论文实现 |
| [CodeS](https://github.com/RUCKBReasoning/CodeS) | SQL 预训练 | 论文实现 |

---

## 五、博客与教程

### 5.1 技术博客

| 来源 | 标题 | 链接 |
|------|------|------|
| Vanna.ai | Text-to-SQL with Vanna-AI: A Developer's Guide | [typevar.dev](https://typevar.dev/articles/vanna-ai/vanna) |
| LangChain | SQL Agent Tutorial | [langchain.com](https://python.langchain.com/docs/use_cases/sql) |
| OpenAI | SQL Generation Examples | [platform.openai.com](https://platform.openai.com/examples/default-sql-translate) |

### 5.2 关键技术概念

1. **Schema Linking**: 将用户问题中的实体映射到数据库表/列
2. **RAG for SQL**: 检索相似 SQL 示例辅助生成
3. **Execution-Guided**: 执行 SQL 并根据结果/错误修正
4. **Self-Correction**: LLM 自我诊断和修复 SQL 错误

---

## 六、评估与对比

### 6.1 Spider 排行榜 (截至 2025)

| 方法 | EX | 特点 |
|------|-----|------|
| DIN-SQL + GPT-4 | 85.3% | 任务分解 |
| DAIL-SQL + GPT-4 | 86.2% | 提示优化 |
| CodeS-7B | 84.1% | 开源微调 |

### 6.2 BIRD 排行榜

| 方法 | EX | VES |
|------|-----|-----|
| GPT-4 + COT | 55.9% | 53.7% |
| Claude-3.5 Sonnet | 60.3% | 58.2% |

---

## 七、推荐阅读顺序

1. **入门**: TKDE 2025 综述 → Vanna.ai 文档
2. **方法**: DIN-SQL 论文 → DAIL-SQL 论文
3. **实践**: BIRD Benchmark → 本项目实现
4. **进阶**: CodeS 论文 → Self-Debugging 论文

---

**维护者**: Text-to-SQL Pro 项目组  
**最后更新**: 2026-03-09