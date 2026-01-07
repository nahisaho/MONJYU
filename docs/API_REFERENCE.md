# MONJYU API Reference

**Last Updated**: 2026-01-07  
**Version**: 3.5.1

## Overview

MONJYU provides multiple layers of API for different use cases:

| API | Use Case | Level |
|-----|----------|-------|
| `MONJYU` Facade | Simple usage | High |
| Controllers | Advanced control | Medium |
| Query/Index modules | Custom pipelines | Low |
| CLI | Command line usage | User |
| MCP Server | Claude Desktop integration | External |

---

## Table of Contents

1. [MONJYU Facade API](#1-monjyu-facade-api)
2. [Search Modes](#2-search-modes)
3. [Hybrid Search API](#3-hybrid-search-api)
4. [Streaming API](#4-streaming-api)
5. [Index API](#5-index-api)
6. [Configuration](#6-configuration)
7. [CLI Commands](#7-cli-commands)
8. [MCP Server](#8-mcp-server)
9. [Type Definitions](#9-type-definitions)
10. [Error Handling](#10-error-handling)

---

## 1. MONJYU Facade API

The simplest way to use MONJYU.

### Basic Usage

```python
from monjyu import MONJYU

# Initialize
monjyu = MONJYU()

# Index documents
await monjyu.index("/path/to/papers/")

# Search
result = await monjyu.search(
    query="What is GraphRAG?",
    mode="auto"
)

print(result.answer)
```

### Class: MONJYU

```python
class MONJYU:
    """MONJYU Facade API."""
    
    def __init__(
        self,
        storage_path: str = "./storage",
        config: Optional[MONJYUConfig] = None,
    ):
        """Initialize MONJYU.
        
        Args:
            storage_path: Path to storage directory
            config: Configuration object
        """
    
    async def index(
        self,
        source: Union[str, Path, List[str]],
        level: int = 0,
        **kwargs,
    ) -> IndexResult:
        """Index documents.
        
        Args:
            source: Path to documents or list of paths
            level: Progressive index level (0-4)
            **kwargs: Additional options
            
        Returns:
            IndexResult with statistics
        """
    
    async def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
        **kwargs,
    ) -> SearchResult:
        """Search indexed documents.
        
        Args:
            query: Search query
            mode: Search mode (auto, vector, lazy, hybrid, global, local)
            top_k: Number of results
            **kwargs: Additional options
            
        Returns:
            SearchResult with answer and documents
        """
```

---

## 2. Search Modes

### Available Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `auto` | Automatic selection | General use |
| `vector` | Vector similarity | Fast retrieval |
| `lazy` | Query-time graph | Comprehensive answers |
| `hybrid` | Multi-engine fusion | Best quality |
| `global` | Community-based | Broad topics |
| `local` | Entity-centric | Specific entities |

### Usage Examples

```python
# Auto mode (recommended)
result = await monjyu.search("What is GraphRAG?", mode="auto")

# Vector mode (fast)
result = await monjyu.search("transformer architecture", mode="vector")

# Lazy mode (comprehensive)
result = await monjyu.search("Compare BERT and GPT", mode="lazy")

# Hybrid mode (best quality)
result = await monjyu.search("Latest NLP techniques", mode="hybrid")
```

---

## 3. Hybrid Search API

### monjyu.query.hybrid_search

```python
from monjyu.query.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    FusionMethod,
    SearchMethod,
    create_hybrid_search,
)

# Create with factory
search = create_hybrid_search(
    methods=["vector", "lazy"],
    fusion="rrf",
    top_k=10,
)

# Or with explicit config
config = HybridSearchConfig(
    methods=[SearchMethod.VECTOR, SearchMethod.LAZY],
    fusion=FusionMethod.RRF,
    rrf_k=60,
    top_k=10,
    parallel=True,
)
search = HybridSearch(config=config)

# Execute search
result = await search.search("query")
```

### Fusion Methods

| Method | Description | Formula |
|--------|-------------|---------|
| `rrf` | Reciprocal Rank Fusion | `1/(k + rank)` |
| `weighted` | Weighted average | `score * weight` |
| `max` | Maximum score | `max(scores)` |
| `combsum` | Score sum | `sum(scores)` |
| `combmnz` | CombMNZ | `sum(scores) * count` |

---

## 4. monjyu.search.hybrid

Advanced hybrid search with answer synthesis.

```python
from monjyu.search.hybrid import (
    HybridSearchEngine,
    HybridSearchConfig,
    create_hybrid_engine,
)

engine = create_hybrid_engine(
    methods=["vector", "lazy"],
    fusion="rrf",
)

result = await engine.search("query")
print(result.answer)
print(result.citations)
```

---

## 5. Controllers

### UnifiedController

Single entry point for all operations.

```python
from monjyu.controller.unified import UnifiedController

controller = UnifiedController()
result = await controller.search(
    query="What is GraphRAG?",
    mode="auto",
)
```

### ProgressiveController

Budget-aware progressive search.

```python
from monjyu.controller.progressive import ProgressiveController

controller = ProgressiveController()
result = await controller.search(
    query="Compare RAG approaches",
    budget=100,  # Token budget
    min_level=0,
    max_level=2,
)
```

### HybridController

Multi-engine hybrid search.

```python
from monjyu.controller.hybrid import HybridController

controller = HybridController()
result = await controller.search(
    query="Latest transformer models",
    engines=["vector", "lazy"],
    fusion="rrf",
)
```

---

## 6. MCP Server

MONJYU MCP Server は Model Context Protocol (MCP) 準拠のサーバーで、Claude Desktop や他の MCP クライアントから利用可能です。

### 起動モード

#### stdio モード（デフォルト）

```bash
# 標準起動
monjyu-mcp

# または
python -m monjyu.mcp_server
```

#### HTTP モード（Streamable HTTP Transport）

```bash
# デフォルト (127.0.0.1:8080)
monjyu-mcp --http

# カスタムホスト/ポート
monjyu-mcp --http --host 0.0.0.0 --port 9000
```

### バージョン情報

```bash
monjyu-mcp --version
# => monjyu-mcp 0.2.0
```

### Available Tools (7 Tools)

| Tool | Description | Parameters |
|------|-------------|------------|
| `monjyu_search` | 学術論文検索 | query, mode (auto/vector/lazy), top_k |
| `monjyu_get_document` | ドキュメント詳細取得 | document_id |
| `monjyu_list_documents` | ドキュメント一覧 | limit, offset |
| `monjyu_citation_chain` | 引用チェーン取得 | document_id, depth |
| `monjyu_find_related` | 関連論文検索 | document_id, limit |
| `monjyu_status` | インデックスステータス | - |
| `monjyu_get_metrics` | 引用メトリクス取得 | - |

### Available Resources (6 Resources)

| Resource URI | Description |
|-------------|-------------|
| `monjyu://index/status` | インデックスステータス |
| `monjyu://documents` | ドキュメント一覧 |
| `monjyu://documents/{id}` | 個別ドキュメント詳細 |
| `monjyu://documents/{id}/content` | ドキュメント全文 |
| `monjyu://documents/{id}/citations` | ドキュメントの引用 |
| `monjyu://citation-network` | 引用ネットワーク概要 |

### Available Prompts (5 Prompts)

| Prompt | Description | Arguments |
|--------|-------------|-----------|
| `literature_review` | 文献レビュー生成 | topic |
| `paper_summary` | 論文要約 | document_id, length |
| `compare_papers` | 論文比較 | document_ids |
| `research_question` | 研究質問の探索 | topic |
| `citation_analysis` | 引用分析 | document_id |

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "monjyu": {
      "command": "monjyu-mcp"
    }
  }
}
```

### HTTP モードでの接続例

```python
import httpx

async with httpx.AsyncClient() as client:
    # MCP JSON-RPC リクエスト
    response = await client.post(
        "http://127.0.0.1:8080/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "monjyu_search",
                "arguments": {
                    "query": "GraphRAG",
                    "mode": "auto"
                }
            },
            "id": 1
        }
    )
```

---

## 7. Streaming API

MONJYU はストリーミングレスポンスをサポートし、大規模な検索結果をリアルタイムで返すことができます。

### 基本的な使用法

```python
from monjyu import MONJYU
from monjyu.api.streaming import StreamingService, StreamingConfig

monjyu = MONJYU()
service = StreamingService(source=monjyu)

# ストリーミング検索
async for chunk in service.stream_search("GraphRAG"):
    if chunk.is_text:
        print(chunk.content, end="", flush=True)
    elif chunk.is_done:
        print(f"\nCompleted: {chunk.content}")
```

### StreamingConfig

```python
from monjyu.api.streaming import StreamingConfig

config = StreamingConfig(
    buffer_size=1024,        # バッファサイズ
    timeout_seconds=30.0,    # タイムアウト
    include_citations=True,  # 引用情報を含める
)

service = StreamingService(source=monjyu, config=config)
```

### StreamingChunk

ストリーミングの各チャンクは `StreamingChunk` オブジェクトで表現されます。

```python
from monjyu.api.streaming import StreamingChunk, ChunkType

# チャンクタイプ
ChunkType.TEXT      # テキストトークン
ChunkType.CITATION  # 引用情報
ChunkType.METADATA  # メタデータ
ChunkType.PROGRESS  # 進捗情報
ChunkType.ERROR     # エラー
ChunkType.DONE      # 完了マーカー

# プロパティ
chunk.is_text   # テキストチャンクか
chunk.is_done   # 完了チャンクか
chunk.is_error  # エラーチャンクか

# 変換
chunk.to_dict()  # 辞書に変換
StreamingChunk.from_dict(data)  # 辞書から生成

# ファクトリメソッド
StreamingChunk.text("content")
StreamingChunk.citation("text", citation_data)
StreamingChunk.progress("Processing...", percentage=50.0)
StreamingChunk.done("Completed with 5 citations")
StreamingChunk.error("Timeout", error_code="TIMEOUT")
```

### StreamingStatus

```python
from monjyu.api.streaming import StreamingStatus

StreamingStatus.PENDING     # 開始待ち
StreamingStatus.STREAMING   # ストリーミング中
StreamingStatus.COMPLETED   # 完了
StreamingStatus.CANCELLED   # キャンセル
StreamingStatus.ERROR       # エラー
```

### ストリーミング結果の収集

```python
# すべてのチャンクを収集
result = await service.collect_search_result("GraphRAG")
print(f"Text: {result['text']}")
print(f"Citations: {result['citations']}")
print(f"Total chunks: {result['total_chunks']}")
```

### コールバックの使用

```python
async def on_progress(chunk):
    print(f"Progress: {chunk.data.get('percentage', 0)}%")

async for chunk in service.stream_search(
    "GraphRAG",
    callbacks=[on_progress]
):
    print(chunk.content)
```

### キャンセル処理

```python
import asyncio

async def search_with_timeout():
    try:
        async with asyncio.timeout(10):
            async for chunk in service.stream_search("query"):
                process(chunk)
    except asyncio.TimeoutError:
        print("Search timed out")
```

### エラーハンドリング

```python
from monjyu.api.streaming import (
    StreamingError,
    StreamingCancelledError,
    StreamingTimeoutError,
)

try:
    async for chunk in service.stream_search("query"):
        if chunk.is_error:
            print(f"Error: {chunk.content}")
            break
        print(chunk.content)
except StreamingTimeoutError:
    print("Timeout occurred")
except StreamingCancelledError:
    print("Stream was cancelled")
except StreamingError as e:
    print(f"Streaming error: {e}")
```

---

## 8. External APIs

### Semantic Scholar

```python
from monjyu.external import SemanticScholarClient

client = SemanticScholarClient()

# Search papers
papers = await client.search("GraphRAG", limit=10)

# Get paper details
paper = await client.get_paper("paper_id")

# Get citations
citations = await client.get_citations("paper_id")
```

### CrossRef

```python
from monjyu.external import CrossRefClient

client = CrossRefClient()

# Search by DOI
result = await client.get_by_doi("10.1234/example")

# Search works
works = await client.search("machine learning", limit=10)
```

---

## 9. Configuration

### Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Ollama
OLLAMA_HOST=http://localhost:11434

# Storage
MONJYU_STORAGE_PATH=./storage

# MCP Server
MONJYU_CONFIG=./monjyu.yaml
```

### Config File (monjyu.yaml)

```yaml
storage:
  base_path: "./storage"
  parquet_enabled: true

embedding:
  provider: "azure_openai"
  model: "text-embedding-3-small"
  
search:
  default_mode: "auto"
  max_results: 10
  
hybrid:
  fusion: "rrf"
  rrf_k: 60
  parallel: true

progressive:
  auto_upgrade: true
  upgrade_threshold: 100
```

---

## 9. Error Handling

### Exception Types

```python
from monjyu.errors import (
    MONJYUError,          # Base exception
    IndexError,           # Index-related errors
    SearchError,          # Search-related errors
    ConfigurationError,   # Configuration errors
    StorageError,         # Storage errors
)
from monjyu.api.streaming import (
    StreamingError,           # Streaming base error
    StreamingCancelledError,  # Stream cancelled
    StreamingTimeoutError,    # Timeout occurred
)```

### Example

```python
from monjyu import MONJYU
from monjyu.errors import SearchError

monjyu = MONJYU()

try:
    result = await monjyu.search("query")
except SearchError as e:
    print(f"Search failed: {e}")
```

---

## 10. Type Definitions

### SearchResult

```python
@dataclass
class SearchResult:
    query: str
    answer: str
    documents: List[Document]
    citations: List[Citation]
    mode: str
    processing_time_ms: float
```

### Document

```python
@dataclass
class Document:
    id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]
```

### Citation

```python
@dataclass
class Citation:
    source_id: str
    text: str
    relevance: float
```

### StreamingChunk

```python
@dataclass
class StreamingChunk:
    content: str
    chunk_type: ChunkType
    chunk_id: str
    stream_id: str
    sequence: int
    timestamp: float
    data: Dict[str, Any]
```

### StreamingState

```python
@dataclass
class StreamingState:
    stream_id: str
    status: StreamingStatus
    chunks_sent: int
    total_tokens: int
    started_at: float
    updated_at: float
```

---

For more examples, see the [examples/](examples/) directory.
