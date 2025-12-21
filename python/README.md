# rust_research_py

**rust-research-mcp** 的 Python 支持包，提供 PDF 处理、文本结构化抽取及学术论文下载等功能。

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 目录

- [功能概述](#功能概述)
- [安装](#安装)
- [快速开始](#快速开始)
- [模块详解](#模块详解)
  - [pdf2text - PDF 转结构化文本](#pdf2text---pdf-转结构化文本)
  - [text2table - 文本转表格](#text2table---文本转表格)
  - [plugins - 论文下载插件](#plugins---论文下载插件)
- [CLI 命令](#cli-命令)
- [Python API](#python-api)
- [开发者指南](#开发者指南)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 功能概述

`rust_research_py` 包含三个核心模块：

| 模块 | 功能 | 主要依赖 |
|------|------|----------|
| **pdf2text** | PDF → 结构化 JSON/Markdown，支持图表提取 | GROBID, scipdf |
| **text2table** | 自由文本 → TSV 表格（通过实体识别 + LLM） | GLiNER, vLLM |
| **plugins** | 多出版商论文 PDF 下载器 | Playwright, Selenium |

---

## 安装

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/Ladvien/sci_hub_mcp.git
cd sci_hub_mcp/python

# 安装包
pip install -e .
```

### 完整安装（包含所有依赖）

```bash
# 创建 conda 环境（推荐，pdf2text 需要 Java）
conda create -n research python=3.10 openjdk=11 -y
conda activate research

# 安装包及所有依赖
pip install -e .

# 安装 Playwright 浏览器（用于 plugins 模块）
playwright install
```

### 依赖项

```
aiohttp          # 异步 HTTP 客户端
beautifulsoup4   # HTML 解析
click            # CLI 框架
requests         # HTTP 请求
playwright       # 浏览器自动化
numpy, pandas    # 数据处理
pydantic         # 数据验证
selenium         # 浏览器自动化
webdriver-manager
scipdf           # PDF 图表提取
grobid-client-python  # GROBID 客户端
vllm             # LLM 推理服务
gliner           # 命名实体识别
```

---

## 快速开始

### 1. PDF 转文本（pdf2text）

```bash
# 转换单个 PDF
pdf2text pdf --pdf-file paper.pdf --output-dir ./output

# 批量转换目录
pdf2text pdf --pdf-dir ./pdfs --output-dir ./output
```

### 2. 文本转表格（text2table）

```bash
# 启动 text2table 服务
text2table-server --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 16384

# 运行提取
python -m rust_research_py.text2table.cli run \
  --server-url http://localhost:8000/v1 \
  --text-file data.txt \
  --label "Drug" --label "Disease" \
  --prompt "请输出表格"
```

### 3. 论文下载（plugins）

```python
from rust_research_py.plugins import download_with_detected_plugin

# 自动检测出版商并下载
result = download_with_detected_plugin(
    doi="10.1038/nature12373",
    output_dir="./papers"
)
print(result)
```

---

## 模块详解

### pdf2text - PDF 转结构化文本

将学术 PDF 转换为结构化 JSON 和 Markdown 格式，支持图表提取。

#### 功能特点

- 🔧 **GROBID 集成**：自动启动/管理本地 GROBID 服务
- 📄 **结构化输出**：提取标题、作者、摘要、正文、参考文献
- 🖼️ **图表提取**：通过 scipdf 提取图片和表格
- 📝 **Markdown 渲染**：生成可读的 Markdown 文档

#### 输出结构

```
output/
├── paper/
│   ├── paper.json    # 结构化元数据和文本
│   ├── paper.md      # Markdown 渲染（可选）
│   ├── paper.pdf     # 源 PDF 副本（可选）
│   ├── figures/      # 提取的图片
│   └── tables/       # 提取的表格
```

#### CLI 命令

```bash
# 基础转换
pdf2text pdf --pdf-dir ./pdfs --output-dir ./output

# 跳过 Markdown 或图表
pdf2text pdf --pdf-dir ./pdfs --output-dir ./output --no-markdown --no-figures

# 使用自定义 GROBID 服务
pdf2text pdf --pdf-dir ./pdfs --output-dir ./output --grobid-url http://localhost:8070

# GROBID 服务管理
pdf2text grobid start        # 启动服务
pdf2text grobid status       # 检查状态
pdf2text grobid stop         # 停止服务
```

#### Python API

```python
from rust_research_py.pdf2text import (
    extract_fulltext,
    extract_figures,
    save_markdown_from_json,
    list_pdfs
)

# 列出目录中的所有 PDF
pdfs = list_pdfs("./papers")

# 提取单个 PDF（包含图表）
extract_fulltext(
    pdf_file="paper.pdf",
    output_dir="./output",
    extract_figures=True,
    extract_tables=True,
    copy_pdf=True
)

# 仅提取图表
extract_figures("paper.pdf", "./output")

# 从 JSON 生成 Markdown
save_markdown_from_json("./output/paper/paper.json")
```

---

### text2table - 文本转表格

通过实体识别（GLiNER）和大型语言模型（vLLM）将自由文本转换为结构化 TSV 表格。

#### 功能特点

- 🏷️ **实体识别**：GLiNER 本地/服务模式
- 🤖 **LLM 生成**：支持 OpenAI 兼容 API（如 vLLM）
- ⚡ **异步处理**：支持批量并发处理
- ✅ **行验证**：可选的 LLM 验证模式

#### 工作流程

```
原始文本 → [GLiNER 实体识别] → [LLM 表格生成] → TSV 表格
```

#### CLI 命令

```bash
# 单文本处理
python -m rust_research_py.text2table.cli run \
  --server-url http://localhost:8000/v1 \
  --text-file input.txt \
  --label "Drug" --label "Disease" --label "ADE" \
  --output result.tsv

# 批量处理 JSONL 文件
python -m rust_research_py.text2table.cli run-batch \
  --input-jsonl samples.jsonl \
  --label "Drug" --label "ADE" \
  --server-url http://localhost:8000/v1 \
  --concurrency 4 \
  --dump-jsonl results.jsonl
```

#### 主要参数

| 参数 | 说明 |
|------|------|
| `--server-url` | vLLM 服务地址（必需） |
| `--gliner-url` | GLiNER 服务地址（可选，默认本地） |
| `--label` | 要提取的实体标签（可多次指定） |
| `--threshold` | GLiNER 置信度阈值（默认 0.5） |
| `--enable-thinking` | 启用 LLM 推理思考模式 |
| `--enable-row-validation` | 启用行验证 |

#### Python API

```python
import asyncio
from rust_research_py.text2table import AsyncText2Table, Text2Table

# 同步使用
t2t = Text2Table(
    labels=["Drug", "Disease", "ADE"],
    server_url="http://localhost:8000/v1",
    gliner_url="http://localhost:9001"  # 可选
)
table, entities = t2t.run("患者服用阿司匹林后出现头痛症状。")
print(table)
t2t.close()

# 异步使用
async def process():
    t2t = AsyncText2Table(
        labels=["Drug", "ADE"],
        server_url="http://localhost:8000/v1",
        enable_row_validation=True
    )
    table, entities = await t2t.run("示例文本")
    await t2t.close()
    return table

asyncio.run(process())
```

#### 批量处理

```python
from rust_research_py.text2table import AsyncText2Table, BatchItem

items = [
    BatchItem(text="文本 A", id="a"),
    BatchItem(text="文本 B", id="b"),
]

async def batch_process():
    t2t = AsyncText2Table(
        labels=["Drug", "ADE"],
        server_url="http://localhost:8000/v1"
    )
    results = await t2t.run_many(items, concurrency=4)
    await t2t.close()
    for res in results:
        print(f"{res.id}: {res.table}")

asyncio.run(batch_process())
```

---

### plugins - 论文下载插件

提供针对各主要学术出版商的专用 PDF 下载器。

#### 支持的出版商

| 插件 | 出版商 | 示例域名 |
|------|--------|----------|
| `NaturePDFDownloader` | Nature | nature.com |
| `WileyPDFDownloader` | Wiley | onlinelibrary.wiley.com |
| `MDPIPDFDownloader` | MDPI | mdpi.com |
| `SpringerPDFDownloader` | Springer | link.springer.com |
| `FrontiersPDFDownloader` | Frontiers | frontiersin.org |
| `PNASPDFDownloader` | PNAS | pnas.org |
| `PLOSPDFDownloader` | PLOS | plosone.org |
| `HindawiPDFDownloader` | Hindawi | hindawi.com |
| `BioRxivPDFDownloader` | bioRxiv | biorxiv.org |
| `OxfordPDFDownloader` | Oxford | academic.oup.com |

#### Python API

```python
from rust_research_py.plugins import (
    download_with_detected_plugin,
    detect_publisher_patterns,
    format_filename_from_doi,
    NaturePDFDownloader,
    WileyPDFDownloader,
)

# 自动检测出版商并下载
result = download_with_detected_plugin(
    doi="10.1038/nature12373",
    output_dir="./papers"
)

# 检测出版商
detection = detect_publisher_patterns("10.1002/example")
print(f"Publisher: {detection.publisher}")

# 格式化文件名
filename = format_filename_from_doi("10.1038/nature12373")

# 直接使用特定下载器
downloader = NaturePDFDownloader()
result = downloader.download(
    doi="10.1038/nature12373",
    output_path="./paper.pdf"
)
```

---

## CLI 命令

安装后可用的命令行工具：

| 命令 | 说明 |
|------|------|
| `pdf2text` | PDF 转换工具（JSON/Markdown/图表提取） |
| `text2table-server` | 启动 text2table vLLM 服务 |

### pdf2text 完整帮助

```bash
pdf2text --help

# 子命令
pdf2text pdf --help       # PDF 转换
pdf2text markdown --help  # Markdown 生成
pdf2text grobid --help    # GROBID 管理
```

### text2table 完整帮助

```bash
python -m rust_research_py.text2table.cli --help
python -m rust_research_py.text2table.cli run --help
python -m rust_research_py.text2table.cli run-batch --help
```

---

## Python API

### 模块导入

```python
# pdf2text 模块
from rust_research_py.pdf2text import (
    extract_fulltext,
    extract_figures,
    save_markdown_from_json,
    list_pdfs,
    GrobidServer
)

# text2table 模块
from rust_research_py.text2table import (
    Text2Table,
    AsyncText2Table,
    BatchItem,
    BatchResult,
    DEFAULT_USER_PROMPT
)

# plugins 模块
from rust_research_py.plugins import (
    download_with_detected_plugin,
    detect_publisher_patterns,
    format_filename_from_doi,
    PLUGIN_REGISTRY,
    # 各出版商下载器
    NaturePDFDownloader,
    WileyPDFDownloader,
    MDPIPDFDownloader,
    SpringerPDFDownloader,
    FrontiersPDFDownloader,
    PNASPDFDownloader,
    PLOSPDFDownloader,
    HindawiPDFDownloader,
    BioRxivPDFDownloader,
    OxfordPDFDownloader,
)
```

---

## 开发者指南

### 项目结构

```
python/
├── pyproject.toml              # 包配置
├── README.md                   # 本文档
├── examples/                   # 示例文件
└── rust_research_py/           # 主包
    ├── __init__.py
    ├── pdf2text/               # PDF 处理模块
    │   ├── __init__.py
    │   ├── cli.py              # CLI 入口
    │   ├── pdf2text.py         # 核心逻辑
    │   ├── grobid.py           # GROBID 管理
    │   └── models.py           # 数据模型
    ├── text2table/             # 文本转表格模块
    │   ├── __init__.py
    │   ├── cli.py              # CLI 入口
    │   ├── text2table.py       # 核心逻辑
    │   ├── client.py           # HTTP 客户端
    │   ├── server.py           # vLLM 服务包装
    │   └── prompts.py          # Prompt 模板
    └── plugins/                # 下载插件
        ├── __init__.py
        ├── common.py           # 基础类
        ├── utils.py            # 工具函数
        ├── plugin_runner.py    # 插件运行器
        └── downloader/         # 各出版商下载器
            ├── nature_pdf_downloader.py
            ├── wiley_pdf_downloader.py
            ├── mdpi_pdf_downloader.py
            └── ...
```

### 开发安装

```bash
# 开发模式安装
pip install -e ".[dev]"

# 运行测试
cd python/rust_research_py/text2table
pytest tests/
```

### 添加新的下载器插件

1. 在 `plugins/downloader/` 创建新文件 `{publisher}_pdf_downloader.py`
2. 继承 `BasePlugin` 类
3. 在 `plugins/__init__.py` 中注册插件

```python
from rust_research_py.plugins import BasePlugin, DownloadResult

class NewPublisherPDFDownloader(BasePlugin):
    """新出版商下载器"""
    
    def download(self, doi: str, output_path: str) -> DownloadResult:
        # 实现下载逻辑
        pass
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `TEXT2TABLE_VLLM_URL` | vLLM 服务地址 |
| `TEXT2TABLE_GLINER_URL` | GLiNER 服务地址 |
| `HUGGINGFACE_HUB_TOKEN` | Hugging Face 令牌（用于受限模型） |

---

## 常见问题

### Q: GROBID 启动失败？

确保已安装 Docker/Podman/Singularity：

```bash
# 检查 Docker
docker --version

# 手动启动 GROBID
docker run -p 8070:8070 lfoppiano/grobid:0.8.0
```

### Q: text2table 连接超时？

1. 确认 vLLM 服务正在运行
2. 检查服务 URL 是否正确
3. 调整超时参数：`--request-timeout 120`

### Q: 下载器需要登录？

部分出版商需要机构订阅。plugins 模块会尝试多种下载策略，但可能需要：

- 配置代理
- 使用机构网络
- 提供认证信息

### Q: 如何提高 text2table 准确率？

1. 使用更大的模型
2. 调整 GLiNER 阈值：`--threshold 0.3`
3. 启用思考模式：`--enable-thinking`
4. 启用行验证：`--enable-row-validation --row-validation-mode llm`

---

## 许可证

MIT License - 详见 [LICENSE](../LICENSE) 文件

---

## 相关资源

- **主项目**: [rust-research-mcp](https://github.com/Ladvien/sci_hub_mcp)
- **GROBID**: [https://github.com/kermitt2/grobid](https://github.com/kermitt2/grobid)
- **GLiNER**: [https://github.com/urchade/GLiNER](https://github.com/urchade/GLiNER)
- **vLLM**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---

*Made with ❤️ for the research community*
