# Technology Stack

**Project**: MONJYU (文殊)  
**Last Updated**: 2025-12-28  
**Status**: Decided  
**Version**: 3.1

---

## Overview

MONJYUは、学術論文（AI for Science）を対象とした Progressive GraphRAG システムです。
Microsoft GraphRAGと同等以上の機能を約1/100のコストで実現するPythonパッケージです。

**ターゲットドメイン**: 学術論文 (arXiv, PubMed, IEEE Xplore等)

---

## Core Technology Decisions

### Primary Language

| 項目 | 決定 | 理由 |
|------|------|------|
| **言語** | Python 3.10+ | GraphRAG互換、AI/ML エコシステム、async/await サポート |
| **型ヒント** | Required | mypy strict mode、PEP 561準拠 |
| **パッケージマネージャー** | uv / pip | 高速依存解決、標準互換 |

### LLM Integration

| 項目 | 決定 | 理由 |
|------|------|------|
| **Primary** | Azure OpenAI | エンタープライズ対応、高可用性 |
| **OpenAI Client** | openai >= 1.0 | 公式SDK、Streaming対応 |
| **Local LLM** | Ollama | ローカル開発、コスト削減 |
| **Embedding** | Azure OpenAI / text-embedding-3-small | 高品質、多言語対応 |

### PDF処理 (学術論文対応)

| 項目 | 決定 | 理由 |
|------|------|------|
| **Primary** | Azure Document Intelligence | 高精度レイアウト解析、表・数式対応 |
| **Fallback** | Unstructured | ローカル処理、オープンソース |
| **論文構造解析** | AcademicPaperParser | IMRaD構造、引用抽出 |

### Graph & NLP

| 項目 | 決定 | 理由 |
|------|------|------|
| **グラフライブラリ** | networkx | 軽量、標準的 |
| **コミュニティ検出** | graspologic (Leiden) | GraphRAG採用、階層的クラスタリング |
| **NLP (軽量)** | spaCy / NLTK | LLMコスト削減、キーワード/エンティティ抽出 |
| **トークナイザー** | tiktoken | OpenAI互換トークンカウント |

### Text Chunking (日本語対応)

| 項目 | 決定 | 理由 |
|------|------|------|
| **セマンティックチャンキング** | sentence-transformers | 埋め込みベース境界検出 |
| **日本語文分割** | bunkai / fugashi | 高精度日本語文分割 |
| **形態素解析** | MeCab (fugashi) | 日本語トークン化 |
| **多言語埋め込み** | multilingual-e5-* | 日英両対応、高精度 |
| **フォールバック** | regex (句読点) | 依存なし軽量版 |

### Data Storage

| 項目 | 決定 | 理由 |
|------|------|------|
| **テーブル形式** | Parquet (pyarrow) | GraphRAG互換、高圧縮、高速 |
| **メタデータ** | JSON / YAML | 人間可読、設定ファイル |
| **ベクトルストア (Primary)** | Azure AI Search | エンタープライズ、ハイブリッド検索 |
| **ベクトルストア (Local)** | LanceDB | 組み込み可能、開発用 |
| **キャッシュ** | Azure Cache for Redis | クエリキャッシュ、セッション管理 |

### External APIs (学術論文)

| 項目 | 決定 | 理由 |
|------|------|------|
| **Citation Network** | Semantic Scholar API | 無料、引用ネットワーク |
| **DOI解決** | CrossRef API | DOI→メタデータ変換 |
| **オープンアクセス** | OpenAlex API | オープンアクセス論文 |
| **PDF取得** | CORE / Unpaywall | フルテキスト取得 |

---

## Progressive GraphRAG (段階的GraphRAG)

| 項目 | 決定 | 理由 |
|------|------|------|
| **アーキテクチャ** | 単一インデックス + 段階的深化 | 二重管理排除、シンプル |
| **コスト制御** | 予算ベース (CostBudget) | 事前コスト指定、自動最適化 |
| **インデックス構築** | オンデマンド + バックグラウンド | 即時検索可能、徐々に深化 |

**インデックスレベル**:
| Level | 名称 | 内容 | コスト | 状態 |
|-------|------|------|--------|------|
| 0 | Raw | チャンク + 埋め込み | 低 | ✅ 実装済 |
| 1 | Lazy | 名詞句グラフ + コミュニティ | 低 (NLP) | ✅ 実装済 |
| 2 | Partial | エンティティ + 関係性 | 中 (LLM) | 🔲 計画中 |
| 3 | Full | コミュニティサマリー | 高 (LLM) | 🔲 計画中 |
| 4 | Enhanced | 事前抽出クレーム | 最高 (LLM) | 🔲 計画中 |

**コスト予算**:
| 予算 | 使用レベル | ユースケース |
|------|-----------|-------------|
| `MINIMAL` | 0-1 | 探索的、ワンオフ |
| `STANDARD` | 0-2 | 一般クエリ |
| `PREMIUM` | 0-3 | 高品質必要 |
| `UNLIMITED` | 0-4 | 最高品質 |

**統合インターフェース**:
```python
class MONJYU:
    """MONJYU Facade (03_components_v3.md準拠)"""
    
    def __init__(
        self,
        config: MONJYUConfig | None = None,
    ): ...
    
    async def search(
        self,
        query: str,
        method: SearchMethod = SearchMethod.AUTO,
        budget: CostBudget = CostBudget.STANDARD,
    ) -> SearchResult: ...
    
    async def index(
        self,
        documents: list[Document],
        target_level: IndexLevel = IndexLevel.LAZY,
    ) -> IndexResult: ...
```

**実装Phase**:
| Phase | 内容 | 状態 |
|-------|------|------|
| 1 | LazySearch単体 | ✅ 完了 |
| 2 | MONJYU Facade API | ✅ 完了 |
| 3 | Document Processing | ✅ 完了 |
| 4 | Citation Network | ✅ 完了 |
| 5 | ProgressiveIndex (Level 0-1) | ✅ 完了 |
| 6 | Query Router | ✅ 完了 |
| 7 | Level 2-4 実装 | 🔲 計画中 |
| 8 | MCP Server | ✅ 完了 |
| 9 | Streaming API | ✅ 完了 |
| 10 | HTTP Transport | ✅ 完了 |

### CLI Framework

| 項目 | 決定 | 理由 |
|------|------|------|
| **CLI** | typer | 型ヒント活用、自動ドキュメント |
| **リッチ出力** | rich | 進捗バー、テーブル、色付き出力 |

### Testing & Quality

| 項目 | 決定 | 理由 |
|------|------|------|
| **テストフレームワーク** | pytest | 標準、async対応 |
| **モック** | pytest-mock, responses | HTTPモック |
| **カバレッジ** | coverage.py | > 80% 目標 |
| **Linting** | ruff | 高速、Flake8/Black統合 |
| **型チェック** | mypy | strict mode |
| **フォーマッター** | ruff format | Black互換 |

```toml
[project]
name = "monjyu"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # LLM
    "openai>=1.0",
    "tiktoken>=0.5",
    "litellm>=1.0",  # Multi-provider support
    
    # Graph
    "networkx>=3.0",
    "graspologic>=3.0",  # Leiden algorithm
    
    # NLP (optional)
    "spacy>=3.7",
    
    # Data
    "pyarrow>=14.0",
    "pandas>=2.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    
    # Vector Store
    "lancedb>=0.3",  # Default embedded
    
    # CLI
    "typer>=0.9",
    "rich>=13.0",
    
    # Async
    "aiohttp>=3.9",
    "aiofiles>=23.0",
    
    # Document Preprocessing
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "nltk>=3.8",
]

[project.optional-dependencies]
# Document parsing (multi-format)
unstructured = [
    "unstructured[csv,doc,docx,email,html,md,msg,ppt,pptx,text,xlsx,xml]>=0.16",
]
pdf = [
    "unstructured[pdf]>=0.16",
    "pymupdf>=1.23",
]
# Azure Form Recognizer (高精度PDF解析)
azure-doc = [
    "azure-ai-formrecognizer>=3.3",
    "azure-identity>=1.15",
]
# Japanese NLP
nlp-ja = [
    "fugashi>=1.3",
    "bunkai>=1.5",
    "sentence-transformers>=2.2",
]
nlp = ["spacy>=3.7"]
faiss = ["faiss-cpu>=1.7"]
chroma = ["chromadb>=0.4"]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "mypy>=1.5",
    "ruff>=0.1",
]
# All preprocessing features
all = [
    "monjyu[unstructured,pdf,azure-doc,nlp-ja]",
]
```

### Model Downloads

```bash
# spaCy models (optional)
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm  # Japanese
```

---

## Architecture Pattern

### C4 Model - コンテナ図

```
┌────────────────────────────────────────────────────────────────────┐
│                           MONJYU System                             │
├────────────────────────────────────────────────────────────────────┤
│  【Presentation Layer】                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐│
│  │   CLI       │  │ Python API  │  │ MCP Server (Claude Desktop) ││
│  │  (Typer)    │  │   (async)   │  │    (stdio/SSE)              ││
│  └──────┬──────┘  └──────┬──────┘  └─────────────┬───────────────┘│
│         │                │                       │                 │
│         └────────────────┼───────────────────────┘                 │
│                          ▼                                          │
│  【Application Layer】                                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                      MONJYU Facade                           │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │  │
│  │  │UnifiedGraphRAG  │  │ProgressiveGraph │  │HybridGraph  │  │  │
│  │  │   Controller    │  │ RAG Controller  │  │RAG Controller│  │  │
│  │  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘  │  │
│  │           │                    │                   │         │  │
│  │           └────────────────────┼───────────────────┘         │  │
│  │                                ▼                              │  │
│  │                    ┌───────────────────────┐                 │  │
│  │                    │     Query Router      │                 │  │
│  │                    │ (AUTO/LAZY/GRAPH/VEC) │                 │  │
│  │                    └───────────────────────┘                 │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                ▼                                    │
│  【Domain Layer】                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │   Index Domain   │  │   Query Domain   │  │ Citation Domain  │ │
│  │  - Level0/1/2/3  │  │  - VectorSearch  │  │ - CitationNetwork│ │
│  │  - EntityExtract │  │  - LazySearch    │  │ - Co-citation    │ │
│  │  - Community     │  │  - GlobalSearch  │  │                  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                ▼                                    │
│  【Infrastructure Layer】                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ LLM Client  │  │  Embedding  │  │ PDF Process │  │ External  │ │
│  │- AzureOpenAI│  │- AzureOpenAI│  │- AzureDocInt│  │- Semantic │ │
│  │- Ollama     │  │- Ollama     │  │- Unstructured│ │  Scholar  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Strategy

### Environment Variables

```bash
# LLM Provider
MONJYU_LLM_PROVIDER=openai  # openai, azure, ollama
MONJYU_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Local LLM
OLLAMA_BASE_URL=http://localhost:11434

# Paths
MONJYU_ROOT=./monjyu_project
MONJYU_OUTPUT=./output
```

### Configuration File (settings.yaml)

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 4096

embedding:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536

index:
  chunk_size: 300
  chunk_overlap: 100
  entity_types:
    - person
    - organization
    - geo
    - event
    - concept

query:
  default_method: lazy  # global, local, drift, lazy, basic
  context_budget: 8000
  
storage:
  format: parquet
  vector_store: lancedb
```

---

## Performance Targets

| メトリクス | 目標 | 測定方法 |
|-----------|------|---------|
| クエリレイテンシ | < 3秒 (1M chunks) | pytest-benchmark |
| インデックス速度 | > 1000 chunks/秒 | time.perf_counter |
| メモリ使用量 | < 4GB (1M chunks) | memory_profiler |
| LLMコスト | GraphRAGの1/100 | API usage tracking |

---

## Development Environment

### Recommended Setup

```bash
# Clone repository
git clone https://github.com/your-org/monjyu.git
cd monjyu

# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=monjyu

# Type checking
mypy monjyu/

# Linting
ruff check monjyu/
ruff format monjyu/
```

### VS Code Extensions

- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)
- Even Better TOML (tamasfe.even-better-toml)

---

## GraphRAG Compatibility Matrix

| GraphRAG Feature | MONJYU Status | Notes |
|------------------|---------------|-------|
| Text Chunking | ✅ Compatible | tiktoken-based |
| Entity Extraction | ✅ Compatible | LLM-based |
| Relationship Extraction | ✅ Compatible | LLM-based |
| Leiden Community Detection | ✅ Compatible | graspologic |
| Community Reports | ✅ Compatible | LLM-generated |
| Global Search | ✅ Implemented | Map-reduce |
| Local Search | ✅ Implemented | Entity-based |
| DRIFT Search | 🔲 To implement | Hybrid |
| LazySearch | ✅ Implemented | MONJYU unique |
| Parquet Output | ✅ Compatible | pyarrow |
| Prompt Tuning | 🔲 To implement | Auto + Manual |

---

## Test Coverage (2025-12-28)

| Category | Tests | Status |
|----------|-------|--------|
| Unit Tests | 2200+ | ✅ Pass |
| Integration Tests | 165 | ✅ Pass |
| E2E Tests | 24 | ✅ Pass |
| **Total** | **2417** | **✅ All Pass** |

**Test Coverage**: 83% (目標80%達成 ✅)

| Module | Before | After |
|--------|--------|-------|
| mcp_server/handlers.py | 60% | 95% |
| lazy/relevance_tester.py | 59% | 100% |
| lazy/iterative_deepener.py | 59% | 98% |
| search/query_encoder.py | 49% | 84% |
| search/answer_synthesizer.py | 53% | 86% |
| index/manager.py | 51% | 98% |
| mcp_server/server.py | 68% | 83% |

**Test Framework**:
- pytest + pytest-asyncio
- pytest-cov (カバレッジ計測)
- unittest.mock (モック)

**Test Organization**:
```
tests/
├── unit/           # ユニットテスト (2200+)
├── integration/    # 統合テスト (165)
├── e2e/            # E2Eテスト (24)
└── benchmarks/     # パフォーマンステスト
```

---

**Powered by MUSUBI** - Technology Stack Documentation
