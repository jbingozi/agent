# Agent 智能客服

一个基于 LangGraph + RAG + MCP 的扫地机器人智能客服小项目。

## 功能

- ReAct 多轮对话
- 记忆管理（Redis / MySQL）
- 向量检索问答（Milvus / Zilliz Cloud）
- MCP 工具调用（天气、搜索、POI）

## 目录

- `agent/`：Agent 主流程和工具编排
- `rag/`：知识库检索与向量存储
- `memory/`：短期/长期记忆
- `config/`：配置文件
- `prompts/`：系统提示词
- `data/`：知识库文档

## 环境

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置 `.env`

```env
AMAP_API_KEY=
TAVILY_API_KEY=
SERPER_API_KEY=
DASHSCOPE_API_KEY=
MILVUS_URI=./milvus_data/agent_knowledge.db
MILVUS_TOKEN=
```

## 启动

单轮模式：

```bash
python main.py "上海现在天气怎么样？"
```

交互模式：

```bash
python main.py
```

## 测试

```bash
python -m unittest discover -s tests
```

## 说明

- 项目会优先读取环境变量，其次读取 `.env`
- 日志、本地向量库、缓存文件已加入 `.gitignore`
- 如果要接 Zilliz Cloud，把 `MILVUS_URI` 和 `MILVUS_TOKEN` 写进 `.env`
