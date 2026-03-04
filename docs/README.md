# Text-to-SQL 项目文档

## 目录

1. [架构设计](./architecture.md) - 系统架构与技术选型
2. [API文档](./api.md) - API接口说明
3. [部署指南](./deployment.md) - 部署与配置
4. [开发指南](./development.md) - 本地开发与测试
5. [资源链接](./resources.md) - 相关学习资源

## 快速链接

- **GitHub仓库**: 待创建
- **在线演示**: 待部署
- **问题反馈**: GitHub Issues

## 项目简介

Text-to-SQL 是一个基于大语言模型的智能数据库查询系统，支持：

- 单轮自然语言转SQL
- 多轮对话上下文理解
- Schema智能检索
- SQL安全校验

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| LLM | OpenAI GPT-4 / Azure OpenAI |
| 向量数据库 | ChromaDB |
| 关系数据库 | MySQL / PostgreSQL |
| 嵌入模型 | Sentence Transformers |
| 前端 | HTML + CSS + JavaScript |