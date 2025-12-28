# Project Structure

**Project**: MONJYU (文殊)  
**Last Updated**: 2025-12-28  
**Version**: 3.2

---

## Architecture Pattern

**Primary Pattern**: Hexagonal Architecture (Ports & Adapters) + Library-First

MONJYUは、学術論文（AI for Science）を対象とした Progressive GraphRAG システムです。
Pythonベースのモジュラーアーキテクチャを採用し、各コンポーネントが独立して
テスト・置換可能な設計を維持しています。

**対応アーキテクチャパターン**:
| パターン | 説明 | 状態 |
|---------|------|------|
| Baseline RAG | チャンク検索 + 生成 | ✅ 実装済 |
| LazyGraphRAG | 遅延グラフ + 動的抽出 | ✅ 実装済 |
| GraphRAG | グラフ構築 + 検索 | ✅ 実装済 |
| Unified GraphRAG | Query Router による動的選択 | ✅ 実装済 |
| Progressive GraphRAG | 段階的インデックス + 予算制御 | ✅ 実装済 |
| Hybrid GraphRAG | 複数エンジン並列実行 + マージ | ✅ 実装済 |

---

## Architecture Layers

### Layer 1: Presentation

**Purpose**: エントリーポイント（CLI, API, MCP）
**Location**: `monjyu/cli/`, `monjyu/api/`, `monjyu/mcp_server/`
**Rules**:
- Applicationレイヤーのみに依存
- 入力バリデーション、出力フォーマット

**Components**:
| ディレクトリ | 役割 |
|-------------|------|
| `monjyu/cli/` | CLIインターフェース (Typer) |
| `monjyu/api/` | MONJYU Facade API |
| `monjyu/mcp_server/` | MCP Server (Claude Desktop連携) |

### Layer 2: Application / Controller

**Purpose**: ユースケース実装、オーケストレーション、Query Router
**Location**: `monjyu/controller/`
**Rules**:
- Domainレイヤーのみに依存
- ポート/インターフェースを通じたI/O

**Components**:
| ディレクトリ | 役割 |
|-------------|------|
| `monjyu/controller/unified/` | Unified GraphRAG Controller |
| `monjyu/controller/budget/` | Budget Controller (コスト予算制御) |
| `monjyu/controller/progressive/` | Progressive Controller (段階的検索制御) || `monjyu/controller/hybrid/` | Hybrid Controller (並列検索+RRFマージ) |
### Layer 3: Domain Layer

**Purpose**: ビジネスロジック、ドメインモデル
**Location**: `monjyu/index/`, `monjyu/query/`, `monjyu/citation/`

**Index Domain**:
| ディレクトリ/ファイル | 役割 |
|---------------------|------|
| `monjyu/index/level0/` | Level 0: Raw (チャンク + 埋め込み) |
| `monjyu/index/level1/` | Level 1: Lazy (名詞句グラフ + コミュニティ) |
| `monjyu/index/entity_extractor/` | エンティティ抽出 |
| `monjyu/index/relationship_extractor/` | 関係性抽出 |
| `monjyu/index/community_detector/` | Leidenコミュニティ検出 |
| `monjyu/index/community_report_generator/` | コミュニティレポート生成 |

**Query Domain**:
| ディレクトリ | 役割 |
|-------------|------|
| `monjyu/query/vector_search/` | ベクトル検索 |
| `monjyu/query/global_search/` | グローバル検索 (Map-Reduce) |
| `monjyu/query/local_search/` | ローカル検索 (エンティティベース) |
| `monjyu/query/router/` | Query Router (AUTO/LAZY/GRAPH/VEC) |

**Citation Domain**:
| ファイル | 役割 |
|---------|------|
| `monjyu/citation/builder.py` | CitationNetworkBuilder |
| `monjyu/citation/analyzer.py` | CoCitationAnalyzer |
| `monjyu/citation/resolver.py` | 参照解決 |
| `monjyu/citation/metrics.py` | 引用メトリクス |

### Layer 4: Infrastructure / Adapters

**Purpose**: 外部統合（LLM、ストレージ、ベクトルDB、PDF処理）
**Location**: `monjyu/embedding/`, `monjyu/storage/`, `monjyu/document/`
**Rules**:
- Domainレイヤーのポートを実装
- すべてのI/O操作をここに集約

**Components**:
| ディレクトリ | 役割 |
|-------------|------|
| `monjyu/embedding/` | Azure OpenAI / Ollama 埋め込み |
| `monjyu/storage/` | ファイル / Azure Blob ストレージ |
| `monjyu/document/` | PDF処理 (Azure DI / Unstructured) |
| `monjyu/nlp/` | NLP処理 (spaCy / MeCab) |
| `monjyu/graph/` | グラフ操作 (NetworkX) |

### Layer 5: Legacy LazySearch (リファクタリング対象)

**Purpose**: 初期実装のLazySearch
**Location**: `lazy_search/`
**Status**: `monjyu/lazy/` への移行中

**Components**:
| ファイル | 役割 |
|----------|------|
| `lazy_search/search.py` | メイン検索エンジン (LazySearch) |
| `lazy_search/core/` | コアユーティリティ |

---

## Current Directory Structure

```
MONJYU/
├── __init__.py               # パッケージエントリーポイント
├── AGENTS.md                 # MUSUBI SDD エージェント設定
├── README.md                 # プロジェクト概要
├── pyproject.toml            # プロジェクト設定
│
├── monjyu/                   # 🔵 メインパッケージ (v3)
│   ├── __init__.py
│   │
│   ├── api/                  # 🎯 MONJYU Facade API
│   │   ├── base.py           # ベース定義
│   │   ├── config.py         # API設定
│   │   ├── factory.py        # ファクトリー
│   │   ├── monjyu.py         # メインFacade
│   │   ├── streaming.py      # StreamingService
│   │   └── state.py          # 状態管理
│   │
│   ├── cli/                  # 🖥️ CLI (Typer)
│   │   ├── main.py
│   │   └── commands/
│   │
│   ├── controller/           # 🎮 Controller Layer
│   │   ├── unified/          # Unified GraphRAG Controller
│   │   ├── budget/           # Budget Controller (CostBudget制御)
│   │   ├── progressive/      # Progressive Controller (段階的検索)
│   │   └── hybrid/           # Hybrid Controller (RRFマージ)
│   │
│   ├── index/                # 📊 Index Domain
│   │   ├── base.py           # インデックスベース
│   │   ├── azure_search.py   # Azure AI Search
│   │   ├── lancedb.py        # LanceDB
│   │   ├── level0/           # Level 0: Raw
│   │   ├── level1/           # Level 1: Lazy
│   │   ├── entity_extractor/ # エンティティ抽出
│   │   ├── relationship_extractor/  # 関係性抽出
│   │   ├── community_detector/      # コミュニティ検出
│   │   └── community_report_generator/  # レポート生成
│   │
│   ├── query/                # 🔍 Query Domain
│   │   ├── vector_search/    # ベクトル検索
│   │   ├── global_search/    # グローバル検索
│   │   ├── local_search/     # ローカル検索
│   │   └── router/           # Query Router
│   │
│   ├── citation/             # 📚 Citation Domain
│   │   ├── base.py
│   │   ├── builder.py        # CitationNetworkBuilder
│   │   ├── analyzer.py       # CoCitationAnalyzer
│   │   ├── resolver.py       # 参照解決
│   │   ├── metrics.py        # 引用メトリクス
│   │   └── manager.py        # 引用管理
│   │
│   ├── document/             # 📄 Document Processing
│   │   ├── loader.py         # ドキュメントローダー
│   │   ├── parser.py         # パーサー
│   │   ├── chunker.py        # TextChunker
│   │   ├── models.py         # データモデル
│   │   ├── pipeline.py       # 処理パイプライン
│   │   └── pdf/              # PDF処理 (Azure DI / Unstructured)
│   │
│   ├── embedding/            # 🧬 Embedding
│   │   ├── base.py
│   │   ├── azure_openai.py   # Azure OpenAI Embedding
│   │   └── ollama.py         # Ollama Embedding
│   │
│   ├── graph/                # 🕸️ Graph Operations
│   │   ├── base.py
│   │   ├── noun_phrase_graph.py  # 名詞句グラフ
│   │   └── community_detector.py # Leiden検出
│   │
│   ├── nlp/                  # 🗣️ NLP Processing
│   │
│   ├── storage/              # 💾 Storage Layer
│   │   ├── parquet.py        # Parquet ストレージ
│   │   └── cache.py          # CacheManager (LRU/Redis)
│   │
│   ├── observability/        # 📊 Observability (Telemetry/Logging)
│   │   └── __init__.py       # Tracer, Metrics, Logger
│   │
│   ├── errors/               # ⚠️ Error Handling Framework
│   │   └── __init__.py       # Exceptions, Retry, CircuitBreaker
│   │
│   ├── lazy/                 # 🦥 LazySearch (v3)
│   │   └── base.py
│   │
│   ├── search/               # 🔎 Search (統合予定)
│   │
│   └── mcp_server/           # 🔌 MCP Server
│
├── lazy_search/              # 🔵 LazySearch ライブラリ (Legacy)
│   ├── __init__.py           # Public API exports
│   ├── search.py             # メイン検索エンジン
│   ├── query_expander.py     # クエリ展開
│   ├── relevance_tester.py   # 関連性テスト
│   ├── claim_extractor.py    # クレーム抽出
│   ├── iterative_deepener.py # 反復的深化
│   ├── context.py            # コンテキストビルダー
│   ├── state.py              # 検索状態管理
│   └── core/                 # コアユーティリティ
│       ├── chat_model.py     # LLMインターフェース
│       ├── conversation.py   # 会話履歴
│       ├── search_result.py  # 検索結果型
│       ├── text_utils.py     # テキストユーティリティ
│       └── tokenizer.py      # トークナイザー
│
├── config/                   # 🔧 設定
│   └── lazy_search_config.py # LazySearch設定
│
├── prompts/                  # 📝 プロンプト
│   └── lazy_search_system_prompt.py
│
├── tests/                    # 🧪 テスト
│   ├── unit/                 # ユニットテスト
│   ├── integration/          # 統合テスト
│   ├── e2e/                  # E2Eテスト
│   └── benchmarks/           # ベンチマーク
│
├── docs/                     # 📚 ドキュメント
│   ├── lazy_search.md        # 技術ドキュメント
│   ├── lazy_search.ipynb     # Jupyterノートブック
│   └── qiita_lazygraphrag.md # Qiita記事
│
├── specs/                    # 📋 仕様書
│   ├── 01_requirements_v3.md # 要件定義 v3 ✅
│   ├── 02_architecture_v3.md # アーキテクチャ設計 v3
│   ├── 03_components_v3.md   # コンポーネント設計 v3
│   ├── 04_api_v3.md          # API仕様 v3
│   └── 05_directory_structure_v3.md
│
├── steering/                 # 🎯 プロジェクトメモリ
│   ├── product.ja.md         # プロダクトコンテキスト
│   ├── structure.ja.md       # このファイル
│   ├── tech.ja.md            # 技術スタック
│   ├── project.yml           # プロジェクト設定
│   └── rules/
│       └── constitution.md   # 憲法 (9条)
│
├── storage/                  # 💾 SDD成果物
│   ├── specs/                # 仕様書ストレージ
│   ├── changes/              # 差分仕様
│   └── features/             # 機能トラッキング
│
├── templates/                # 📄 テンプレート
│
├── output/                   # 📤 出力ファイル
│
└── References/               # 📖 参照資料
    ├── Spec-LazyGraphRAG.md
    ├── graphrag/             # GraphRAGソースコード
    └── PubSec-Info-Assistant/
```

---

## Implementation Status

### ✅ Implemented Modules

| モジュール | ステータス | 説明 |
|-----------|-----------|------|
| `lazy_search/` | ✅ 完了 | LazyGraphRAG検索エンジン (Legacy) |
| `monjyu/api/` | ✅ 完了 | MONJYU Facade API |
| `monjyu/document/` | ✅ 完了 | ドキュメント処理 |
| `monjyu/embedding/` | ✅ 完了 | Embedding (Azure/Ollama) |
| `monjyu/citation/` | ✅ 完了 | Citation Network |
| `monjyu/graph/` | ✅ 完了 | Graph Operations |
| `monjyu/index/level0/` | ✅ 完了 | Level 0 インデックス |
| `monjyu/index/level1/` | ✅ 完了 | Level 1 インデックス |
| `monjyu/index/entity_extractor/` | ✅ 完了 | LLMエンティティ抽出 |
| `monjyu/index/relationship_extractor/` | ✅ 完了 | LLM関係性抽出 |
| `monjyu/index/community_detector/` | ✅ 完了 | Leidenコミュニティ検出 |
| `monjyu/index/community_report_generator/` | ✅ 完了 | コミュニティレポート生成 |
| `monjyu/query/router/` | ✅ 完了 | Query Router |
| `monjyu/query/vector_search/` | ✅ 完了 | ベクトル検索 |
| `monjyu/query/global_search/` | ✅ 完了 | グローバル検索 (Map-Reduce) |
| `monjyu/query/local_search/` | ✅ 完了 | ローカル検索 (エンティティベース) |
| `monjyu/cli/` | ✅ 完了 | CLIインターフェース |
| `monjyu/mcp_server/` | ✅ 完了 | MCP Server (7ツール) |
| `monjyu/index/azure_search.py` | ✅ 完了 | Azure AI Search 統合 (本番用) |
| `monjyu/index/lancedb.py` | ✅ 完了 | LanceDB 統合 (ローカル開発用) |
| `monjyu/index/manager.py` | ✅ 完了 | Progressive Index Manager |
| `monjyu/controller/unified/` | ✅ 完了 | Unified Controller (32テスト) |
| `monjyu/controller/budget/` | ✅ 完了 | Budget Controller (43テスト) |
| `monjyu/controller/progressive/` | ✅ 完了 | Progressive Controller (28テスト) |
| `monjyu/controller/hybrid/` | ✅ 完了 | Hybrid Controller (38テスト) |
| `monjyu/storage/cache.py` | ✅ 完了 | CacheManager (44テスト) |
| `monjyu/api/streaming.py` | ✅ 完了 | StreamingService (45テスト) |
| `monjyu/observability/` | ✅ 完了 | Observability (59テスト) |
| `monjyu/errors/` | ✅ 完了 | Error Handling (63テスト) |
| `tests/integration/` | ✅ 完了 | 統合テスト (Controller 24 + Search 24 = 48テスト) |
| `tests/e2e/` | ✅ 完了 | E2Eテスト (16テスト) |
| `config/` | ✅ 完了 | 設定管理 |
| `prompts/` | ✅ 完了 | プロンプトテンプレート |
| `tests/` | ✅ 完了 | テストスイート (1238テスト) |

### 🔧 In Progress Modules

| モジュール | ステータス | 説明 |
|-----------|-----------|------|
| `monjyu/lazy/` | 🔧 実装中 | LazySearch v3 移行 |

### 🔲 Planned Modules

| モジュール | ステータス | 説明 |
|-----------|-----------|------|
| `monjyu/search/` | 🔲 計画中 | 統合検索インターフェース |

---

## Progressive Index Levels

| Level | 名称 | 内容 | コスト | 状態 |
|-------|------|------|--------|------|
| 0 | Raw | チャンク + 埋め込み | 低 | ✅ 完了 |
| 1 | Lazy | 名詞句グラフ + コミュニティ | 低 (NLP) | ✅ 完了 |
| 2 | Partial | エンティティ + 関係性 | 中 (LLM) | ✅ 完了 |
| 3 | Full | コミュニティサマリー | 高 (LLM) | ✅ 完了 |
| 4 | Enhanced | 事前抽出クレーム | 最高 (LLM) | ✅ 完了 (lazy/) |

---

## Library-First Pattern (Article I)

### Current Libraries

#### `monjyu` Library (v3)

```
monjyu/
├── __init__.py           # Public API
├── api/                  # MONJYU Facade
├── index/                # Progressive Index
├── query/                # Query Domain
├── citation/             # Citation Domain
├── document/             # Document Processing
├── embedding/            # Embedding Adapters
├── graph/                # Graph Operations
├── nlp/                  # NLP Processing
├── storage/              # Storage Adapters
├── controller/           # Controllers
├── cli/                  # CLI
└── mcp_server/           # MCP Server
```

**Public API** (`monjyu/api/monjyu.py`):
```python
from monjyu.api.monjyu import MONJYU, MONJYUConfig
from monjyu.api.state import MONJYUState
```

#### `lazy_search` Library (Legacy)

```
lazy_search/
├── __init__.py           # Public API: LazySearch, LazySearchConfig, etc.
├── search.py             # Main entry: LazySearch class
├── [components].py       # Internal components
└── core/                 # Core utilities
```

**Public API** (`__init__.py`):
```python
from MONJYU.lazy_search.search import LazySearch, LazySearchResult, LazySearchData
from MONJYU.config.lazy_search_config import LazySearchConfig
```

### Component Categories (03_components_v3.md)

| カテゴリ | コンポーネント数 | 状態 |
|---------|----------------|------|
| ドキュメント処理 | 3 | ✅/🔧 |
| Index | 10 | 🔧 |
| Query | 6 | 🔧 |
| Controller | 3 | 🔧 |
| Citation | 2 | ✅ |
| Storage | 3 | 🔧 |
| LLM | 2 | ✅ |
| API | 3 | 🔧 |
| External | 2 | 🔲 |

| ライブラリ | 目的 |
|-----------|------|
| `monjyu-index` | インデックス作成 |
| `monjyu-query` | クエリ処理 |
| `monjyu-storage` | ストレージ抽象化 |

---

## Naming Conventions

### Python Files

| 種別 | 規則 | 例 |
|------|------|-----|
| モジュール | `snake_case.py` | `lazy_search.py` |
| クラス | `PascalCase` | `LazySearch` |
| 関数 | `snake_case` | `expand_query` |
| 定数 | `SCREAMING_SNAKE_CASE` | `MAX_ITERATIONS` |
| 型エイリアス | `PascalCase` | `SearchResult` |

### Directory

| 種別 | 規則 | 例 |
|------|------|-----|
| パッケージ | `snake_case` | `lazy_search/` |
| テスト | `test_*.py` | `test_search.py` |
| 設定 | `*_config.py` | `lazy_search_config.py` |

---

## Test Organization

### Test Structure (Current)

```
tests/
├── unit/                          # ユニットテスト
│   ├── test_components.py         # コンポーネント単体テスト
│   ├── test_config.py             # 設定テスト
│   ├── test_state.py              # 状態管理テスト
│   ├── test_query_router.py       # Query Router テスト
│   ├── test_local_search.py       # Local Search テスト
│   ├── test_cli.py                # CLI テスト
│   ├── test_mcp_server.py         # MCP Server テスト
│   ├── test_entity_extractor.py   # Entity Extractor テスト
│   ├── test_unified_controller.py # Unified Controller テスト (32)
│   ├── test_budget_controller.py  # Budget Controller テスト (43)
│   ├── test_progressive_controller.py # Progressive Controller テスト (28)
│   ├── test_hybrid_controller.py  # Hybrid Controller テスト (38)
│   ├── test_cache_manager.py      # Cache Manager テスト (44)
│   ├── test_streaming_service.py  # Streaming Service テスト (45)
│   ├── test_observability.py      # Observability テスト (59)
│   └── test_error_handling.py     # Error Handling テスト (63)
├── integration/                   # 統合テスト (24テスト)
│   ├── test_lazy_search_integration.py
│   ├── test_controller_integration.py  # Controller統合テスト (24)
│   └── test_e2e_pipeline.py       # E2Eパイプラインテスト
├── e2e/                           # E2Eテスト
│   └── test_lazy_search_e2e.py
└── benchmarks/                    # パフォーマンステスト
    └── lazy_search_benchmark.py
```

### Test Guidelines

- **Article III**: テストファースト (Red-Green-Blue)
- **Article IX**: 統合テストは実サービス使用
- **Coverage**: 80%以上

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
```

### LazySearchConfig

```python
from MONJYU.config.lazy_search_config import LazySearchConfig

config = LazySearchConfig(
    budget_name="Z500",           # Z100, Z500, Z1500
    context_budget=8000,          # トークン制限
    max_iterations=5,             # 最大反復回数
    min_relevance_score=5,        # 最小関連性スコア
)
```

---

## Import Structure

### Internal Imports

```python
# Core → Domain (allowed)
from MONJYU.lazy_search.core.search_result import SearchResult

# Application → Domain (allowed)  
from MONJYU.lazy_search.core.tokenizer import get_tokenizer
from MONJYU.lazy_search.state import LazySearchState

# Infrastructure → Application (allowed)
from MONJYU.lazy_search.search import LazySearch
```

### External Usage

```python
# Public API
from MONJYU.lazy_search import LazySearch, LazySearchConfig
from MONJYU.lazy_search import LazySearchData, LazySearchResult

# Direct usage
search = LazySearch(config=config, data=data)
result = await search.search("Your question here")
```

---

## Version Control

### Branch Strategy

- `main` - プロダクションブランチ
- `develop` - 開発ブランチ
- `feature/*` - 機能ブランチ
- `hotfix/*` - ホットフィックス

### Commit Convention

```
<type>(<scope>): <subject>

Types: feat, fix, docs, refactor, test, chore
Example: feat(lazy-search): add streaming response support
```

---

## Test Structure (2025-12-27)

### Test Organization

```
tests/
├── conftest.py              # 共通フィクスチャ
├── mock_provider.py         # MockChatLLM, MockEmbedding
├── unit/                    # ユニットテスト (1086)
│   ├── test_api.py          # MONJYU Facade API
│   ├── test_chunker.py      # TextChunker
│   ├── test_parser.py       # DocumentParser
│   ├── test_pipeline.py     # DocumentPipeline
│   ├── test_loader.py       # FileLoader
│   ├── test_storage.py      # ParquetStorage
│   ├── test_embedding.py    # Embedding Clients
│   ├── test_lazy_search.py  # LazySearch
│   ├── test_graph.py        # Graph Operations
│   ├── test_citation.py     # Citation Network
│   └── ...
├── integration/             # 統合テスト (165)
│   ├── test_document_processing.py
│   ├── test_index_level0.py
│   ├── test_index_level1.py
│   ├── test_lazy_search.py
│   └── ...
└── e2e/                     # E2Eテスト (17)
    ├── conftest.py          # E2E用フィクスチャ
    ├── test_lazy_search_e2e.py
    └── test_monjyu_e2e.py
```

### Test Results

| Category | Count | Status |
|----------|-------|--------|
| Unit | 1086 | ✅ Pass |
| Integration | 165 | ✅ Pass |
| E2E | 17 | ✅ Pass (4 skipped) |
| **Total** | **1268** | **All Pass** |

---

## Constitutional Compliance

この構造は以下を順守しています：

- **Article I**: ライブラリファースト (`lazy_search/`)
- **Article II**: CLI インターフェース (計画中)
- **Article III**: テストファースト (`tests/`)
- **Article VI**: Steeringファイルでプロジェクトメモリ維持

---

**Powered by MUSUBI** - Project Structure Documentation
