# 🦥 MONJYU - Progressive GraphRAG for Academic Papers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-2417%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)]()

**MONJYU**は、学術論文に特化した**Progressive GraphRAG**システムです。

> MONJYU（文殊）: 知恵の仏。「三人寄れば文殊の知恵」- 少ないリソースで大きな知恵を生み出す。

## ✨ Features

- 🚀 **Progressive Indexing** - Level 0 (Raw) → Level 4 (Full GraphRAG) の段階的構築
- 🔍 **Multiple Search Modes** - Vector, Lazy, Global, Local, Hybrid検索
- 📊 **RRF (Reciprocal Rank Fusion)** - 複数検索結果の統合
- 📚 **Citation Network** - 引用関係の分析
- �� **MCP Server** - Claude Desktop との連携
- 📡 **Streaming Response** - リアルタイム回答配信
- ☁️ **Azure AI Search** - エンタープライズベクトルストア
- 📦 **Library First** - Pythonライブラリとして利用可能
- 🔄 **Incremental Index** - 差分インデックス更新

## 📁 Architecture

\`\`\`
MONJYU/
├── monjyu/                    # メインパッケージ (136 files)
│   ├── api/                   # MONJYU Facade API
│   ├── cli/                   # CLI (Typer)
│   ├── controller/            # Unified/Progressive/Hybrid
│   ├── document/              # ドキュメント処理 (PDF, Word, Excel, PPT)
│   ├── embedding/             # Embedding (Azure OpenAI, Ollama)
│   ├── errors/                # Error Handling (Circuit Breaker, Retry)
│   ├── index/                 # Level0/Level1, Extractors, Incremental
│   ├── query/                 # Vector/Global/Local/Hybrid/Router
│   ├── lazy/                  # LazySearch Engine
│   ├── citation/              # Citation Network
│   ├── mcp_server/            # MCP Server
│   ├── observability/         # Metrics, Tracing
│   └── storage/               # Parquet, Cache
├── tests/                     # 2417 tests (80+ files)
│   ├── unit/                  # Unit Tests (2200+)
│   ├── integration/           # Integration Tests (165)
│   └── e2e/                   # E2E Tests (24)
└── specs/                     # 仕様書 (v3.1)
\`\`\`

## 🚀 Quick Start

### Installation

\`\`\`bash
# Clone
git clone https://github.com/your-org/MONJYU.git
cd MONJYU

# Install
pip install -e .

# Or with all dependencies
pip install -e ".[dev,docs]"
\`\`\`

### Python Library

\`\`\`python
from monjyu import MONJYU

# Initialize
monjyu = MONJYU()

# Index documents
await monjyu.index("/path/to/papers/")

# Search
result = await monjyu.search(
    query="What is GraphRAG?",
    mode="auto",  # auto, vector, lazy, hybrid
)

print(result.answer)
for doc in result.documents:
    print(f"- {doc.title} (score: {doc.score:.3f})")
\`\`\`

### CLI

\`\`\`bash
# Index documents
monjyu index /path/to/papers/ --level 0

# Search
monjyu search "What is GraphRAG?" --mode auto

# Interactive mode
monjyu chat
\`\`\`

### MCP Server (Claude Desktop)

\`\`\`bash
# Start MCP server (stdio mode - default)
monjyu-mcp

# Start MCP server (HTTP mode)
monjyu-mcp --http --port 8080
\`\`\`

Add to \`claude_desktop_config.json\`:

\`\`\`json
{
  "mcpServers": {
    "monjyu": {
      "command": "monjyu-mcp"
    }
  }
}
\`\`\`

**Available MCP Tools (7):**
- \`monjyu_search\` - 学術論文検索
- \`monjyu_get_document\` - ドキュメント詳細取得
- \`monjyu_list_documents\` - ドキュメント一覧
- \`monjyu_citation_chain\` - 引用チェーン取得
- \`monjyu_find_related\` - 関連論文検索
- \`monjyu_status\` - インデックスステータス
- \`monjyu_get_metrics\` - 引用メトリクス

**Available MCP Resources (6):**
- \`monjyu://index/status\` - インデックスステータス
- \`monjyu://documents\` - ドキュメント一覧
- \`monjyu://document/{id}\` - ドキュメント詳細
- \`monjyu://document/{id}/content\` - ドキュメント内容
- \`monjyu://document/{id}/citations\` - 引用情報
- \`monjyu://citation-network\` - 引用ネットワーク

**Available MCP Prompts (5):**
- \`literature_review\` - 文献レビュー生成
- \`paper_summary\` - 論文要約
- \`compare_papers\` - 論文比較
- \`research_question\` - 研究課題分析
- \`citation_analysis\` - 引用分析

---

## 📖 Usage Examples

### 1. Basic Document Indexing

\`\`\`python
from monjyu.document import DocumentPipeline
from monjyu.index.level0 import Level0IndexBuilder
from monjyu.embedding import AzureOpenAIEmbedder

# Setup
embedder = AzureOpenAIEmbedder(
    endpoint="https://your-endpoint.openai.azure.com/",
    api_key="your-key",
    deployment="text-embedding-3-small",
)

pipeline = DocumentPipeline()
builder = Level0IndexBuilder(embedder=embedder)

# Load and index documents
documents, text_units = await pipeline.process_directory("/path/to/papers/")
index = await builder.build(documents, text_units)

# Save index
index.save("./output/index")
\`\`\`

### 2. Vector Search

\`\`\`python
from monjyu.query.vector_search import VectorSearch, VectorSearchConfig

config = VectorSearchConfig(
    top_k=10,
    min_score=0.5,
    use_hybrid=True,
)

search = VectorSearch(embedder=embedder, config=config)
results = await search.search(query="GraphRAG architecture", index=index)

for hit in results.hits:
    print(f"Score: {hit.score:.3f} | {hit.content[:100]}...")
\`\`\`

### 3. Lazy Search (Query-time Graph Construction)

\`\`\`python
from monjyu.lazy import LazySearchEngine, LazySearchConfig

config = LazySearchConfig(
    max_iterations=3,
    relevance_threshold=0.7,
    max_claims=50,
)

lazy_engine = LazySearchEngine(config=config)
result = await lazy_engine.search(
    query="How does LazyGraphRAG compare to traditional GraphRAG?",
    text_units=text_units,
)

print(result.answer)
print(f"Claims extracted: {len(result.claims)}")
\`\`\`

### 4. Hybrid Search with RRF Fusion

\`\`\`python
from monjyu.query.hybrid_search import (
    HybridSearchEngine,
    HybridSearchConfig,
    SearchMethod,
    FusionMethod,
)

config = HybridSearchConfig(
    methods=[SearchMethod.VECTOR, SearchMethod.LAZY],
    fusion=FusionMethod.RRF,
    rrf_k=60,
    top_k=10,
    parallel=True,
)

hybrid = HybridSearchEngine(config=config)
results = await hybrid.search(
    query="What are the key innovations in recent RAG systems?",
    index=index,
)

for hit in results.hits:
    print(f"Score: {hit.score:.3f} | Sources: {hit.sources}")
\`\`\`

### 5. Incremental Index Update

\`\`\`python
from monjyu.index.incremental import (
    IncrementalIndexManager,
    IncrementalIndexConfig,
)

config = IncrementalIndexConfig(
    output_dir="./output/index",
    batch_size=50,
)

manager = IncrementalIndexManager(config)

# Detect changes
change_set = manager.detect_changes(documents, text_units)
print(f"Added: {change_set.added_count}")
print(f"Modified: {change_set.modified_count}")
print(f"Deleted: {change_set.deleted_count}")

# Apply changes
if change_set.total_changes > 0:
    result = await manager.update(documents, text_units, builder)
\`\`\`

### 6. Citation Network Analysis

\`\`\`python
from monjyu.citation import CitationNetworkBuilder, CoCitationAnalyzer

# Build citation network
builder = CitationNetworkBuilder()
network = await builder.build(documents)

# Analyze co-citations
analyzer = CoCitationAnalyzer(network)
pairs = analyzer.find_co_citation_pairs(min_count=3)

for pair in pairs[:10]:
    print(f"{pair.paper1} <-> {pair.paper2}: {pair.count} co-citations")
\`\`\`

### 7. Azure AI Search Integration

\`\`\`python
from monjyu.index.azure_search import (
    AzureAISearchIndexer,
    AzureSearchConfig,
    create_azure_search_indexer,
)

# Create indexer
indexer = create_azure_search_indexer(
    endpoint="https://your-search.search.windows.net",
    api_key="your-key",
    index_name="monjyu-papers",
)

# Create index (if not exists)
indexer.create_index_if_not_exists()

# Add documents
await indexer.add(text_units)

# Search
results = await indexer.search(
    query="machine learning",
    top=10,
    vector=query_embedding,
)
\`\`\`

### 8. Unified Controller (Auto Mode)

\`\`\`python
from monjyu.controller.unified import UnifiedController, UnifiedConfig

config = UnifiedConfig(
    default_mode="auto",
    enable_streaming=True,
)

controller = UnifiedController(config=config)

# Auto mode - automatically selects best search method
result = await controller.search(
    query="Explain the architecture of transformer models",
)

print(f"Mode used: {result.mode}")
print(f"Answer: {result.answer}")
\`\`\`

### 9. Streaming Response

\`\`\`python
from monjyu import MONJYU

monjyu = MONJYU()

# Streaming search
async for chunk in monjyu.search_stream(
    query="What are the benefits of GraphRAG?",
    mode="lazy",
):
    print(chunk.text, end="", flush=True)
\`\`\`

### 10. Error Handling with Circuit Breaker

\`\`\`python
from monjyu.errors import CircuitBreaker, with_retry

# Circuit breaker for external API calls
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
)

@with_retry(max_attempts=3, backoff_factor=2.0)
async def call_external_api():
    async with circuit_breaker:
        return await external_api.call()
\`\`\`

---

## ⚙️ Configuration

### Environment Variables

\`\`\`bash
# Required for Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

# Optional: Azure AI Search
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_API_KEY="your-key"

# Optional: Ollama (local)
export OLLAMA_HOST="http://localhost:11434"
\`\`\`

### Config File (monjyu.yaml)

\`\`\`yaml
# Storage
storage:
  base_path: "./storage"
  parquet_enabled: true

# Embedding
embedding:
  provider: "azure_openai"  # or "ollama"
  model: "text-embedding-3-small"

# Search
search:
  default_mode: "auto"
  max_results: 10

# Progressive Levels
progressive:
  auto_upgrade: true
  upgrade_threshold: 100

# Incremental Index
incremental:
  enabled: true
  batch_size: 50
\`\`\`

---

## 📊 Search Modes

| Mode | Description | Speed | Quality | Cost |
|------|-------------|-------|---------|------|
| \`vector\` | Vector similarity search | ⚡⚡⚡ | ★★☆ | 💰 |
| \`lazy\` | LazyGraphRAG (query-time graph) | ⚡⚡ | ★★★ | 💰💰 |
| \`global\` | Community-based global search | ⚡ | ★★★ | 💰💰💰 |
| \`local\` | Entity-based local search | ⚡⚡ | ★★★ | 💰💰 |
| \`hybrid\` | RRF fusion of multiple engines | ⚡⚡ | ★★★★ | 💰💰 |
| \`auto\` | Automatic selection | ⚡⚡ | ★★★ | 💰💰 |

---

## 🧪 Testing

\`\`\`bash
# Run all tests
pytest

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# With coverage
pytest --cov=monjyu --cov-report=html

# Run specific test
pytest tests/unit/test_incremental_index.py -v
\`\`\`

### Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| Unit Tests | 2,200+ | ✅ |
| Integration Tests | 165 | ✅ |
| E2E Tests | 24 | ✅ |
| **Total** | **2,417** | **✅ All Passed** |

**Coverage**: 83% (目標80%達成 ✅)

---

## 📚 Documentation

- [Requirements](specs/01_requirements_v3.md) - 機能要件 (v3.0 Approved)
- [Architecture](specs/02_architecture_v3.md) - アーキテクチャ設計 (v3.1 Approved)
- [Components](specs/03_components_v3.md) - コンポーネント仕様 (v3.1 Approved)
- [API Reference](specs/04_api_v3.md) - API仕様 (v3.0 Approved)
- [Directory Structure](specs/05_directory_structure_v3.md) - ディレクトリ構造

---

## 🔗 References

- [LazyGraphRAG (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [GraphRAG (Microsoft)](https://github.com/microsoft/graphrag)
- [Model Context Protocol (Anthropic)](https://www.anthropic.com/news/model-context-protocol)

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

**MONJYU v3.3.0** | 2025-12-28 | 2,417 tests passed | 83% coverage
