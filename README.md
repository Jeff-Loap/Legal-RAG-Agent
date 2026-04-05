# 法律 RAG 知识库助手

这是一个面向法律问答场景的本地 RAG 项目。主对话窗口使用 `PySide6` 桌面端，负责真实问答、流式输出、引用证据展示和历史会话管理；`Streamlit` 用作实时评测面板，直接读取桌面端产生的真实记录，持续统计 token、耗时、得分和召回质量。项目默认优先走本地知识库和结构化数据源，支持离线索引重建、双检索模式对比和持续可追溯评测。

## 这个项目能做什么

- 支持从本地法律文档、结构化数据和运行时索引构建知识库，并进行可追溯检索问答
- 支持 `llm_retrieval` 和 `hybrid` 两种模式，可在同题下直接比较 token、耗时和效果分数
- 支持带来源引用的回答输出，正文和证据卡片都能回到原文位置
- 支持 PySide6 桌面端流式回答、Markdown 渲染、引用面板和历史会话恢复
- 支持历史会话保存、回看、关键词筛选和补评未完成的记录
- 支持通过设置面板重建索引，适合新增资料后立即刷新知识库
- 支持基于本地缓存模型离线运行，便于无网或内网环境部署
- 支持 Streamlit 实时评测，只在检测到新记录或评测状态变化时刷新
- 支持真实问答驱动的持续评测，而不是依赖固定 benchmark 文本

## 目录结构

- `legal_rag_desktop.py`：PySide6 主对话窗口
- `app.py`：Streamlit 实时评测面板
- `run_legal_rag_harness.py`：命令行评测入口
- `legal_rag_harness_gui.py`：评测图形界面
- `start_rag_app.bat`：Windows 一键启动脚本
- `legal_agent/`：核心检索、记忆、存储和工作流实现
- `requirements.txt`：依赖清单
- `pdf_data/`：默认法律 PDF 数据源
- `raw_data/`：默认原始文档数据源
- `legal_agent_runtime/`：运行时索引与数据库文件
- `docs/`：项目架构和说明文档

## 安装步骤

### 1. 准备 Python 环境

建议使用 Python 3.10 或 3.11，并在项目目录下创建虚拟环境：

```bash
python -m venv .venv
```

Windows 激活虚拟环境：

```bash
.venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果后续要使用 OCR 功能，通常还需要额外安装系统组件：

- `Tesseract OCR`
- `Poppler`

### 3. 配置模型与密钥

项目通过 `config.ini` 保存大模型连接信息。仓库中只保留占位符，不要把真实密钥提交到仓库：

```ini
[llm]
base_url = https://api.example.com/v1
api_key = YOUR_API_KEY_HERE
model = your-model-name
retrieval_mode = llm_retrieval
answer_profile = quality
temperature = 0.0
max_tokens = 8192
```

也可以改用环境变量：

- `RAG_LLM_BASE_URL`
- `RAG_LLM_API_KEY`
- `RAG_LLM_MODEL`
- `RAG_RETRIEVAL_MODE`
- `RAG_ANSWER_PROFILE`
- `RAG_LLM_TEMPERATURE`
- `RAG_LLM_MAX_TOKENS`

### 4. 预下载本地模型

如果想先把本地缓存模型准备好，可以执行：

```bash
python download_local_models.py
```

### 5. 启动应用

#### 主对话窗口

双击 `start_rag_app.bat`，默认会打开 `PySide6` 主对话窗口。

#### 实时评测面板

`Streamlit` 页面会直接读取 `PySide6` 产生的真实问答记录，并持续展示最新回复的实时评测结果，不再依赖固定 benchmark 文本。

```bash
start_rag_app.bat eval-web
```

#### 评测 GUI

```bash
start_rag_app.bat eval-gui
```

#### 命令行评测

```bash
start_rag_app.bat eval-cli
```

## 数据与索引

项目默认会从以下目录收集知识源：

- `pdf_data/`
- `raw_data/`
- `external_data/`
- `external_db/`
- `db_data/`

运行时索引与数据库会写入：

- `legal_agent_runtime/rag_external.db`
- `legal_agent_runtime/legal_chunks.faiss`
- `legal_agent_runtime/legal_chunks_tfidf.pkl`
- `legal_agent_runtime/manifest.json`

## 如何添加新的知识库

如果要把新的法律知识加入检索，建议按下面流程做：

1. 把新文件放进默认扫描目录之一，例如 `pdf_data/` 或 `raw_data/`。
2. 如果你的文件不在默认目录里，可以在 `legal_agent/config.py` 的 `candidate_roots` 里补充新的目录。
3. 确认文件类型是项目支持的格式，例如 `.pdf`、`.docx`、`.jsonl`、`.csv`、`.db`、`.sqlite`、`.sqlite3`。
4. 启动 `start_rag_app.bat`，在右上角齿轮设置里点击 `重建索引`。
5. 等待索引重建完成后，新知识库才会进入检索结果。

如果只是把文件拷进目录，但不重建索引，应用不会自动读取到新内容。

## 主要依赖

- `PySide6`
- `streamlit`
- `pymupdf`
- `sentence-transformers`
- `faiss-cpu`
- `numpy`
- `scikit-learn`
- `pillow`
- `opencv-python`
- `pytesseract`
- `pdf2image`
- `pdfplumber`
- `huggingface_hub`
- `openai`
- `langchain-core`
- `langgraph`

## 设计特点

- 优先使用本地文件和本地索引，不依赖猜测式回答
- 检索和回答过程强调证据可追溯
- 对缺失配置和缺失资源采用明确报错，便于定位问题
- 支持长对话中的分层记忆与历史回看

## 相关文档

- [架构说明](docs/architecture.md)
- [法律 RAG 知识库助手说明](docs/legal_rag_agent_formal.md)
- [内部说明](docs/legal_rag_agent_internal.md)
