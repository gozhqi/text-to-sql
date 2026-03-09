# Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL

> **论文来源**: [arXiv:2406.08426](https://arxiv.org/abs/2406.08426)  
> **作者**: Zijin Hong, Zheng Yuan, Qinggang Zhang, et al.  
> **发表**: IEEE TKDE 2025  
> **GitHub**: [Awesome-LLM-based-Text2SQL](https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL)

---

## 一、摘要

本文是首个系统性的 LLM-based Text-to-SQL 综述，全面回顾了从规则方法到 LLM 方法的演进过程，系统分析了最新的技术进展，并讨论了未来研究方向。

---

## 二、Text-to-SQL 技术挑战

### 2.1 语言复杂性和歧义
- 嵌套子句、指代消解、省略等复杂语言结构
- 自然语言固有的歧义性，需要深度语义理解

### 2.2 Schema 理解和表示
- 必须全面理解数据库 Schema（表名、列名、表间关系）
- Schema 结构复杂且跨域差异大

### 2.3 罕见和复杂的 SQL 操作
- 嵌套子查询、外连接、窗口函数等复杂操作
- 训练数据中罕见，难以泛化

### 2.4 跨域泛化
- 模型在特定领域训练后难以迁移到其他领域
- 词汇、Schema 结构、问题模式的差异

---

## 三、技术演进

```
Rule-based → Deep Learning → PLM-based → LLM-based
   (2014)       (2017)        (2020)       (2023)
```

### 3.1 Rule-based 方法
- **优点**: 语法正确性保证、输出一致
- **缺点**: 无法处理复杂/歧义问题、泛化能力差

### 3.2 Deep Learning 方法
- **技术**: Seq2Seq、Encoder-Decoder、GNN
- **代表**: RYANSQL
- **缺点**: 语法错误、复杂操作支持差

### 3.3 PLM-based 方法
- **技术**: BERT、RoBERTa 微调
- **代表**: TaBERT、GraPPa
- **缺点**: 跨域性能下降、需要大量微调

### 3.4 LLM-based 方法
- **技术**: In-Context Learning、Fine-tuning
- **代表**: DIN-SQL、DAIL-SQL、CodeS
- **优势**: 强语义理解、少样本学习、SOTA 准确率

---

## 四、基准数据集

### 4.1 跨域数据集
| 数据集 | 数据库数 | 问题数 | 特点 |
|--------|----------|--------|------|
| Spider | 200 | 10,181 | 标准基准 |
| BIRD | 95 | 12,751 | 企业级复杂度 |
| WikiSQL | 265 | 80,654 | 单表查询 |
| DuSQL | 200 | 23,797 | 中文 |

### 4.2 BIRD Benchmark 特点
- **知识增强**: 外部领域知识标注
- **复杂推理**: 数值推理、同义词、领域知识
- **执行效率**: VES 指标评估 SQL 效率

### 4.3 Spider 2.0 特点
- **长上下文**: 平均 SQL 长度 ~150 tokens
- **真实场景**: BigQuery、Snowflake 生产环境
- **企业级 Schema**: 数百到数千列

---

## 五、LLM-based 方法分类

### 5.1 In-Context Learning (ICL) 方法

#### C1 - Decomposition (任务分解)
| 方法 | 核心思想 |
|------|----------|
| DIN-SQL | 四阶段流水线：Schema Linking → 分类 → 生成 → 修正 |
| C3 | 三模块：Schema Linking → Calibration → Voting |
| MAC-SQL | 多 Agent 框架 |
| CHESS | Schema 选择 → 分解 → 精炼 |

#### C2 - Prompt Optimization (提示优化)
| 方法 | 核心思想 |
|------|----------|
| DAIL-SQL | 问题-SQL 相似度选择示例 |
| ACT-SQL | 自适应提示构建 |
| ODIS | 选择性示例检索 |

#### C3 - Reasoning Enhancement (推理增强)
| 方法 | 核心思想 |
|------|----------|
| Chain-of-Thought | 分步推理 |
| Least-to-Most | 从简单到复杂分解 |
| SQL-PaLM | 程序化推理 |

#### C4 - Execution Refinement (执行修正)
| 方法 | 核心思想 |
|------|----------|
| Self-Debugging | SQL 执行错误自动修正 |
| LEVER | 执行验证 + 错误反馈 |
| MBR-Exec | 基于执行的多数投票 |

### 5.2 Fine-tuning 方法

| 方法 | 核心技术 |
|------|----------|
| CodeS | SQL 专用预训练 |
| DAIL-SQL | 数据增强 + 多任务 |
| DTS-SQL | 领域迁移学习 |

---

## 六、评估指标

### 6.1 内容匹配指标
- **Component Matching (CM)**: SELECT/WHERE/GROUP BY/ORDER BY 分组件 F1
- **Exact Matching (EM)**: 完全匹配率

### 6.2 执行指标
- **Execution Accuracy (EX)**: 执行结果匹配
- **Valid Efficiency Score (VES)**: 执行效率 + 准确率

---

## 七、未来研究方向

1. **鲁棒性**: 处理 Schema 扰动、问题变体
2. **真实部署**: 多轮对话、用户反馈
3. **计算效率**: 推理延迟、成本优化
4. **领域适应**: 专用领域 (金融、医疗)
5. **隐私保护**: 敏感数据脱敏

---

## 八、关键论文推荐

1. **DIN-SQL** [NeurIPS 2023] - 任务分解范式
2. **DAIL-SQL** [VLDB 2024] - 提示优化
3. **CodeS** [ACL 2024] - SQL 专用预训练
4. **BIRD** [NeurIPS 2023] - 企业级基准

---

**更新时间**: 2026-03-09