# MONJYU ディレクトリ構造設計書 v3.0

**バージョン**: 3.0.0  
**作成日**: 2025-12-25  
**ステータス**: Approved  
**対応要件**: REQ-MONJYU-001 v3.0

---

## 目次

1. [概要](#1-概要)
2. [プロジェクト構造](#2-プロジェクト構造)
3. [モジュール詳細](#3-モジュール詳細)
4. [出力ディレクトリ構造](#4-出力ディレクトリ構造)
5. [設定ファイル構造](#5-設定ファイル構造)
6. [テスト構造](#6-テスト構造)
7. [ドキュメント構造](#7-ドキュメント構造)
8. [インポート設計](#8-インポート設計)
9. [移行計画](#9-移行計画)
10. [変更履歴](#10-変更履歴)

---

## 1. 概要

### 1.1 設計原則

| 原則 | 説明 |
|------|------|
| **レイヤー分離** | Core / Domain / Application / Infrastructure の明確な分離 |
| **依存性逆転** | 抽象（Protocol）への依存、具象への非依存 |
| **環境透過性** | Local / Production 環境の透過的切り替え |
| **プラグイン対応** | PDF処理、LLM、Storage の実装差し替え可能 |
| **学術論文特化** | 論文構造（IMRaD）、引用ネットワーク対応 |

### 1.2 レイヤー構成

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                   │
│   ┌─────────┐  ┌─────────────┐  ┌──────────────────┐   │
│   │   CLI   │  │ MCP Server  │  │    REST API      │   │
│   └─────────┘  └─────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    Application Layer                    │
│            ┌─────────────────────────┐                 │
│            │      MONJYUFacade       │                 │
│            └─────────────────────────┘                 │
├─────────────────────────────────────────────────────────┤
│                      Domain Layer                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐  │
│  │  Index  │ │  Query  │ │Citation │ │   Document   │  │
│  │ Manager │ │  Router │ │ Network │ │   Parser     │  │
│  └─────────┘ └─────────┘ └─────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐  │
│  │   LLM   │ │ Storage │ │   PDF   │ │    Cache     │  │
│  │ Clients │ │         │ │Processor│ │              │  │
│  └─────────┘ └─────────┘ └─────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│                       Core Layer                        │
│     Protocols / Exceptions / Types / Config             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. プロジェクト構造

### 2.1 完全ディレクトリ構造

```
MONJYU/
├── __init__.py                 # パッケージエントリーポイント
├── __main__.py                 # CLIエントリーポイント
├── py.typed                    # PEP 561 型情報マーカー
├── facade.py                   # MONJYUFacade
│
├── core/                       # コア抽象・共通
│   ├── __init__.py
│   ├── protocols.py            # 全プロトコル定義
│   ├── exceptions.py           # 例外階層
│   ├── types.py                # 共通型定義
│   ├── tokenizer.py            # Tokenizer
│   └── text_utils.py           # テキストユーティリティ
│
├── config/                     # 設定管理
│   ├── __init__.py
│   ├── config.py               # MONJYUConfig メイン
│   ├── environments.py         # 環境別設定（local/production）
│   ├── lazy_search_config.py   # LazySearch設定
│   └── defaults.py             # デフォルト値
│
├── domain/                     # ドメインモデル
│   ├── __init__.py
│   ├── paper.py                # AcademicPaper, Author, Section
│   ├── chunk.py                # TextChunk, ChunkMetadata
│   ├── index.py                # ProgressiveIndex, IndexLevel
│   ├── query.py                # SearchResult, QueryType
│   └── citation.py             # CitationEdge, CitationNetwork
│
├── document/                   # ドキュメント処理
│   ├── __init__.py
│   ├── protocols.py            # PDFProcessorProtocol, ParserProtocol
│   ├── pdf/                    # PDF処理
│   │   ├── __init__.py
│   │   ├── azure_di.py         # Azure Document Intelligence
│   │   └── unstructured.py     # unstructured（ローカル）
│   ├── parser/                 # 学術論文パーサー
│   │   ├── __init__.py
│   │   ├── academic_parser.py  # IMRaD構造解析
│   │   └── reference_parser.py # 参照文献解析
│   └── pipeline.py             # PreprocessingPipeline
│
├── index/                      # インデックス構築
│   ├── __init__.py
│   ├── protocols.py            # インデックス関連プロトコル
│   ├── manager.py              # ProgressiveIndexManager
│   ├── chunker/                # テキストチャンク分割
│   │   ├── __init__.py
│   │   └── tiktoken_chunker.py # tiktoken実装
│   ├── embedder/               # エンベディング
│   │   ├── __init__.py
│   │   ├── protocol.py         # EmbedderProtocol
│   │   ├── azure_openai.py     # Azure OpenAI
│   │   └── ollama.py           # Ollama（nomic-embed-text）
│   ├── nlp/                    # NLP特徴抽出
│   │   ├── __init__.py
│   │   ├── protocol.py         # NLPExtractorProtocol
│   │   └── spacy_extractor.py  # spaCy + NounGraph
│   ├── entity/                 # エンティティ抽出
│   │   ├── __init__.py
│   │   └── llm_extractor.py    # LLM-based抽出
│   └── community/              # コミュニティ検出
│       ├── __init__.py
│       └── leiden.py           # Leidenアルゴリズム
│
├── query/                      # クエリ処理
│   ├── __init__.py
│   ├── protocols.py            # クエリ関連プロトコル
│   ├── router.py               # QueryRouter
│   ├── vector/                 # ベクトル検索
│   │   ├── __init__.py
│   │   ├── lancedb.py          # LanceDB（ローカル）
│   │   └── azure_search.py     # Azure AI Search（本番）
│   ├── lazy/                   # LazySearch
│   │   ├── __init__.py
│   │   ├── search.py           # LazySearch メイン
│   │   ├── query_expander.py   # クエリ拡張
│   │   ├── relevance_tester.py # 関連性テスト
│   │   ├── claim_extractor.py  # クレーム抽出
│   │   ├── context.py          # コンテキストビルダー
│   │   └── state.py            # 状態管理
│   └── hybrid/                 # ハイブリッド検索
│       ├── __init__.py
│       └── rrf.py              # RRF融合
│
├── citation/                   # 引用ネットワーク
│   ├── __init__.py
│   ├── builder.py              # CitationNetworkBuilder
│   ├── analyzer.py             # CoCitationAnalyzer
│   └── graph.py                # NetworkX統合
│
├── storage/                    # ストレージ
│   ├── __init__.py
│   ├── protocols.py            # StorageProtocol, CacheProtocol
│   ├── file_storage.py         # Parquetファイル
│   ├── cache/                  # キャッシュ
│   │   ├── __init__.py
│   │   ├── local_cache.py      # ローカルメモリ
│   │   └── redis_cache.py      # Redis（本番）
│   └── vector_store/           # ベクトルストア
│       ├── __init__.py
│       ├── lancedb_store.py    # LanceDB
│       └── azure_store.py      # Azure AI Search
│
├── llm/                        # LLMクライアント
│   ├── __init__.py
│   ├── protocols.py            # ChatModelProtocol
│   ├── factory.py              # LLMFactory
│   ├── azure_openai.py         # Azure OpenAI
│   └── ollama.py               # Ollama
│
├── mcp/                        # MCP Server
│   ├── __init__.py
│   ├── server.py               # MCPServer
│   ├── tools.py                # ツール定義
│   └── handlers.py             # リクエストハンドラー
│
├── cli/                        # CLI
│   ├── __init__.py
│   ├── main.py                 # typer アプリケーション
│   ├── commands/               # コマンド実装
│   │   ├── __init__.py
│   │   ├── index_cmd.py        # index
│   │   ├── search_cmd.py       # search
│   │   ├── upgrade_cmd.py      # upgrade
│   │   ├── serve_cmd.py        # serve
│   │   ├── papers_cmd.py       # papers
│   │   └── config_cmd.py       # config
│   └── utils.py                # CLI ユーティリティ
│
├── prompts/                    # プロンプトテンプレート
│   ├── __init__.py
│   ├── lazy_search.py          # LazySearch用
│   ├── query_expansion.py      # クエリ拡張用
│   ├── relevance_test.py       # 関連性テスト用
│   ├── claim_extraction.py     # クレーム抽出用
│   ├── entity_extraction.py    # エンティティ抽出用
│   └── summarization.py        # 要約用
│
├── analysis/                   # 分析・生成
│   ├── __init__.py
│   ├── summarizer.py           # 論文要約
│   ├── comparator.py           # 論文比較
│   └── surveyor.py             # サーベイ生成
│
├── api/                        # REST API（オプション）
│   ├── __init__.py
│   ├── app.py                  # FastAPI アプリケーション
│   ├── routes/                 # エンドポイント
│   │   ├── __init__.py
│   │   ├── search.py
│   │   ├── index.py
│   │   └── papers.py
│   └── schemas.py              # Pydantic スキーマ
│
├── specs/                      # 仕様書
│   ├── 01_requirements.md
│   ├── 01_requirements_v3.md
│   ├── 02_architecture.md
│   ├── 02_architecture_v3.md
│   ├── 03_components.md
│   ├── 03_components_v3.md
│   ├── 04_api.md
│   ├── 04_api_v3.md
│   ├── 05_directory_structure.md
│   └── 05_directory_structure_v3.md
│
├── tests/                      # テスト
│   ├── __init__.py
│   ├── conftest.py             # pytest設定・共通fixture
│   ├── unit/                   # ユニットテスト
│   ├── integration/            # 統合テスト
│   ├── e2e/                    # E2Eテスト
│   ├── benchmarks/             # ベンチマーク
│   └── fixtures/               # テストデータ
│
├── docs/                       # ドキュメント
│   ├── getting_started.md
│   ├── configuration.md
│   ├── api_reference.md
│   └── examples/
│
├── config/                     # 設定ファイル（ランタイム）
│   ├── local.yaml              # ローカル環境
│   └── production.yaml         # 本番環境
│
├── pyproject.toml              # プロジェクト設定
├── README.md                   # プロジェクトREADME
├── AGENTS.md                   # エージェント設定（MUSUBI）
├── steering/                   # MUSUBI steering files
└── storage/                    # MUSUBI storage files
```

---

## 3. モジュール詳細

### 3.1 core/ - コアモジュール

抽象化、共通型、例外定義。依存関係なし。

```
core/
├── __init__.py
├── protocols.py        # 全プロトコル（ChatModelProtocol, EmbedderProtocol, etc.）
├── exceptions.py       # 例外階層（MONJYUError, IndexError, etc.）
├── types.py            # 共通型（IndexLevel, SearchMode, QueryType）
├── tokenizer.py        # TiktokenTokenizer
└── text_utils.py       # テキストユーティリティ
```

**protocols.py**
```python
from typing import Protocol, List, Dict, Any, AsyncIterator, Optional
import numpy as np


class ChatModelProtocol(Protocol):
    async def chat(self, prompt: str, history: Optional[List[Dict]] = None) -> str: ...
    async def chat_stream(self, prompt: str) -> AsyncIterator[str]: ...


class EmbedderProtocol(Protocol):
    async def embed(self, text: str) -> np.ndarray: ...
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]: ...


class PDFProcessorProtocol(Protocol):
    async def process(self, pdf_path: str) -> 'PDFDocument': ...


class StorageProtocol(Protocol):
    async def save_index(self, index: 'ProgressiveIndex') -> None: ...
    async def load_index(self, index_id: str) -> 'ProgressiveIndex': ...


class CacheProtocol(Protocol):
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...


class VectorSearchProtocol(Protocol):
    async def search(self, query: str, top_k: int = 10) -> List['SearchHit']: ...
    async def search_by_vector(self, vector: List[float], top_k: int = 10) -> List['SearchHit']: ...
```

---

### 3.2 domain/ - ドメインモデル

ビジネスロジックに関わるデータモデル。

```
domain/
├── __init__.py
├── paper.py            # AcademicPaper, Author, Section, Reference
├── chunk.py            # TextChunk, ChunkMetadata
├── index.py            # ProgressiveIndex, IndexLevel, LevelStatus
├── query.py            # SearchResult, SearchHit, QueryType
└── citation.py         # CitationEdge, CitationNetwork
```

**paper.py（主要部分）**
```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date
from enum import Enum


class SectionType(Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"


@dataclass
class Author:
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


@dataclass
class Section:
    type: SectionType
    title: str
    content: str
    subsections: List['Section'] = field(default_factory=list)


@dataclass
class Reference:
    ref_id: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


@dataclass
class AcademicPaper:
    id: str
    title: str
    authors: List[Author]
    abstract: str
    sections: List[Section]
    references: List[Reference]
    
    # メタデータ
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    publication_date: Optional[date] = None
    venue: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # 処理情報
    source_path: Optional[str] = None
    processed_at: Optional[str] = None
```

---

### 3.3 document/ - ドキュメント処理

PDF読み取りと学術論文構造解析。

```
document/
├── __init__.py
├── protocols.py            # インターフェース定義
├── pdf/
│   ├── __init__.py
│   ├── azure_di.py         # Azure Document Intelligence（本番）
│   └── unstructured.py     # unstructured（ローカル）
├── parser/
│   ├── __init__.py
│   ├── academic_parser.py  # IMRaD構造解析
│   └── reference_parser.py # 参照文献解析（GROBID風）
└── pipeline.py             # 前処理パイプライン
```

---

### 3.4 index/ - インデックス構築

Progressive GraphRAG インデックス（Level 0-4）。

```
index/
├── __init__.py
├── protocols.py            # インデックス関連プロトコル
├── manager.py              # ProgressiveIndexManager
├── chunker/
│   ├── __init__.py
│   └── tiktoken_chunker.py # 学術論文対応チャンカー
├── embedder/
│   ├── __init__.py
│   ├── protocol.py
│   ├── azure_openai.py     # text-embedding-3-large
│   └── ollama.py           # nomic-embed-text
├── nlp/
│   ├── __init__.py
│   ├── protocol.py
│   └── spacy_extractor.py  # NounGraph構築
├── entity/
│   ├── __init__.py
│   └── llm_extractor.py    # LLMによるエンティティ抽出
└── community/
    ├── __init__.py
    └── leiden.py           # Leidenコミュニティ検出
```

---

### 3.5 query/ - クエリ処理

検索エンジン群とルーティング。

```
query/
├── __init__.py
├── protocols.py            # VectorSearchProtocol, LazySearchProtocol
├── router.py               # QueryRouter（クエリタイプ分類）
├── vector/
│   ├── __init__.py
│   ├── lancedb.py          # LanceDB（ローカル）
│   └── azure_search.py     # Azure AI Search（本番）
├── lazy/
│   ├── __init__.py
│   ├── search.py           # LazySearch メイン
│   ├── query_expander.py   # サブクエリ生成
│   ├── relevance_tester.py # LLM関連性評価
│   ├── claim_extractor.py  # クレーム抽出
│   ├── context.py          # コンテキスト構築
│   └── state.py            # 検索状態管理
└── hybrid/
    ├── __init__.py
    └── rrf.py              # Reciprocal Rank Fusion
```

---

### 3.6 citation/ - 引用ネットワーク

引用関係の構築と分析。

```
citation/
├── __init__.py
├── builder.py              # CitationNetworkBuilder
├── analyzer.py             # CoCitationAnalyzer, BibliographicCoupling
└── graph.py                # NetworkX統合、可視化
```

---

### 3.7 storage/ - ストレージ

永続化とキャッシュ。

```
storage/
├── __init__.py
├── protocols.py            # StorageProtocol, CacheProtocol
├── file_storage.py         # Parquetベース（ローカル）
├── cache/
│   ├── __init__.py
│   ├── local_cache.py      # LRUメモリキャッシュ
│   └── redis_cache.py      # Redis（本番）
└── vector_store/
    ├── __init__.py
    ├── lancedb_store.py    # LanceDB
    └── azure_store.py      # Azure AI Search
```

---

### 3.8 llm/ - LLMクライアント

LLMプロバイダー抽象化。

```
llm/
├── __init__.py
├── protocols.py            # ChatModelProtocol
├── factory.py              # LLMFactory（環境自動検出）
├── azure_openai.py         # Azure OpenAI（GPT-4o, text-embedding-3-large）
└── ollama.py               # Ollama（llama3.2, nomic-embed-text）
```

---

### 3.9 mcp/ - MCP Server

Model Context Protocol サーバー。

```
mcp/
├── __init__.py
├── server.py               # MCPServer（stdio/SSE対応）
├── tools.py                # 7ツール定義
└── handlers.py             # リクエストハンドラー
```

---

### 3.10 cli/ - CLI

typerベースのコマンドラインインターフェース。

```
cli/
├── __init__.py
├── main.py                 # typer.Typer() アプリケーション
├── commands/
│   ├── __init__.py
│   ├── index_cmd.py        # monjyu index
│   ├── search_cmd.py       # monjyu search
│   ├── upgrade_cmd.py      # monjyu upgrade
│   ├── serve_cmd.py        # monjyu serve
│   ├── papers_cmd.py       # monjyu papers
│   └── config_cmd.py       # monjyu config
└── utils.py                # 進捗表示、出力フォーマット
```

---

## 4. 出力ディレクトリ構造

インデックス構築後の出力構造（Progressive対応）:

```
output/                             # デフォルト出力ディレクトリ
├── metadata.json                   # インデックスメタデータ
│   {
│     "id": "idx_20251225_001",
│     "name": "transformer_papers",
│     "version": "3.0.0",
│     "current_level": 2,
│     "level_status": {
│       "0": {"is_built": true, "built_at": "2025-12-25T10:00:00Z", ...},
│       "1": {"is_built": true, ...},
│       "2": {"is_built": true, ...},
│       "3": {"is_built": false, ...},
│       "4": {"is_built": false, ...}
│     },
│     "paper_count": 150,
│     "chunk_count": 4500,
│     "created_at": "2025-12-25T10:00:00Z"
│   }
│
├── papers/                         # 論文データ
│   ├── papers.parquet              # 論文メタデータ
│   │   columns: [id, title, authors, abstract, doi, arxiv_id, 
│   │             publication_date, venue, keywords, source_path]
│   └── sections.parquet            # セクションデータ
│       columns: [paper_id, section_id, type, title, content]
│
├── chunks/                         # チャンクデータ
│   ├── chunks.parquet              # テキストチャンク
│   │   columns: [id, content, paper_id, section_id, section_type,
│   │             start_offset, end_offset, token_count, metadata]
│   └── chunk_index.parquet         # チャンクインデックス
│       columns: [chunk_id, paper_id, position]
│
├── nlp/                            # NLP特徴（Level 1+）
│   ├── noun_graph.parquet          # 名詞グラフ
│   │   columns: [chunk_id, nouns, edges, frequencies]
│   └── keywords.parquet            # キーワード
│       columns: [chunk_id, keywords, scores]
│
├── embeddings/                     # 埋め込み（Level 2+）
│   ├── embeddings.lance/           # LanceDBストア
│   └── embeddings_meta.json        # 埋め込みメタデータ
│       {"model": "nomic-embed-text", "dimensions": 768}
│
├── entities/                       # エンティティ（Level 3+）
│   ├── entities.parquet            # エンティティ
│   │   columns: [id, name, type, description, source_chunk_ids]
│   └── relationships.parquet       # 関係
│       columns: [source_id, target_id, relation_type, description]
│
├── communities/                    # コミュニティ（Level 3+）
│   ├── communities.parquet         # コミュニティ
│   │   columns: [id, level, entity_ids, summary]
│   └── reports.parquet             # コミュニティレポート
│       columns: [community_id, title, summary, key_findings]
│
├── citations/                      # 引用ネットワーク
│   ├── citations.parquet           # 引用エッジ
│   │   columns: [citing_id, cited_id, context, section]
│   └── references.parquet          # 参照文献
│       columns: [paper_id, ref_id, title, authors, year, doi]
│
├── claims/                         # クレーム（Level 4）
│   └── claims.parquet              # 事前抽出クレーム
│       columns: [id, text, confidence, source_chunk_id, topics]
│
└── cache/                          # キャッシュ
    ├── query_cache.json            # クエリキャッシュ
    └── embedding_cache.json        # 埋め込みキャッシュ
```

---

## 5. 設定ファイル構造

### 5.1 ローカル環境設定 (config/local.yaml)

```yaml
# MONJYU Local Configuration
version: "3.0"
environment: local

# LLM Configuration
llm:
  provider: ollama
  host: "http://192.168.224.1:11434"  # WSL → Windows
  model: llama3.2
  embedding_model: nomic-embed-text
  temperature: 0.0
  max_tokens: 4096

# Storage Configuration
storage:
  type: file
  base_path: ./output
  vector_store: lancedb
  cache:
    type: local
    max_size: 1000

# PDF Processor Configuration
pdf_processor:
  type: unstructured
  strategy: hi_res
  languages: [eng, jpn]

# Index Configuration
index:
  chunk_size: 300
  chunk_overlap: 100
  encoding_model: cl100k_base
  default_level: 1

# Query Configuration
query:
  default_mode: auto
  rate_query: 1
  rate_relevancy: 32
  rate_cluster: 2
  query_expansion_limit: 8
  max_context_tokens: 8000

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 5.2 本番環境設定 (config/production.yaml)

```yaml
# MONJYU Production Configuration
version: "3.0"
environment: production

# LLM Configuration
llm:
  provider: azure_openai
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
  deployment: gpt-4o
  embedding_deployment: text-embedding-3-large
  api_version: "2024-02-15-preview"
  temperature: 0.0
  max_tokens: 4096

# Storage Configuration
storage:
  type: azure
  blob_connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  vector_store: azure_ai_search
  search_endpoint: ${AZURE_SEARCH_ENDPOINT}
  search_api_key: ${AZURE_SEARCH_API_KEY}
  index_name: monjyu-papers
  cache:
    type: redis
    host: ${REDIS_HOST}
    port: 6379
    password: ${REDIS_PASSWORD}

# PDF Processor Configuration
pdf_processor:
  type: azure_document_intelligence
  endpoint: ${AZURE_DI_ENDPOINT}
  api_key: ${AZURE_DI_KEY}

# Index Configuration
index:
  chunk_size: 300
  chunk_overlap: 100
  encoding_model: cl100k_base
  default_level: 2

# Query Configuration
query:
  default_mode: auto
  rate_query: 1
  rate_relevancy: 32
  rate_cluster: 2
  query_expansion_limit: 8
  max_context_tokens: 8000

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: /var/log/monjyu/monjyu.log
```

---

## 6. テスト構造

```
tests/
├── __init__.py
├── conftest.py                     # 共通fixture
│
├── unit/                           # ユニットテスト
│   ├── __init__.py
│   ├── core/
│   │   ├── test_tokenizer.py
│   │   └── test_text_utils.py
│   ├── document/
│   │   ├── test_pdf_processor.py
│   │   └── test_academic_parser.py
│   ├── index/
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   ├── test_nlp_extractor.py
│   │   └── test_progressive_manager.py
│   ├── query/
│   │   ├── test_vector_search.py
│   │   ├── test_lazy_search.py
│   │   ├── test_query_router.py
│   │   └── test_hybrid_search.py
│   ├── citation/
│   │   └── test_citation_network.py
│   ├── storage/
│   │   └── test_file_storage.py
│   └── llm/
│       └── test_llm_clients.py
│
├── integration/                    # 統合テスト
│   ├── __init__.py
│   ├── test_index_pipeline.py      # インデックス構築パイプライン
│   ├── test_search_pipeline.py     # 検索パイプライン
│   ├── test_progressive_upgrade.py # レベルアップグレード
│   └── test_mcp_server.py          # MCPサーバー
│
├── e2e/                            # E2Eテスト
│   ├── __init__.py
│   ├── test_cli.py                 # CLI全コマンド
│   ├── test_local_workflow.py      # ローカル環境ワークフロー
│   └── test_azure_workflow.py      # Azure環境ワークフロー（CI/CD）
│
├── benchmarks/                     # ベンチマーク
│   ├── __init__.py
│   ├── bench_search_speed.py       # 検索速度
│   ├── bench_index_build.py        # インデックス構築速度
│   └── bench_memory_usage.py       # メモリ使用量
│
└── fixtures/                       # テストデータ
    ├── sample_papers/              # サンプルPDF
    │   ├── attention_is_all_you_need.pdf
    │   └── vit.pdf
    ├── sample_index/               # 構築済みサンプルインデックス
    │   └── level_2/
    └── mock_responses/             # LLMモックレスポンス
        ├── query_expansion.json
        └── claim_extraction.json
```

---

## 7. ドキュメント構造

```
docs/
├── getting_started.md              # クイックスタート
├── installation.md                 # インストール手順
├── configuration.md                # 設定ガイド
├── api_reference.md                # APIリファレンス
├── cli_reference.md                # CLIリファレンス
├── mcp_integration.md              # MCP統合ガイド
├── progressive_index.md            # Progressive GraphRAG解説
├── academic_paper_parsing.md       # 学術論文パース解説
├── deployment/                     # デプロイメント
│   ├── local_setup.md              # ローカル環境構築
│   └── azure_deployment.md         # Azure Container Apps
└── examples/                       # コード例
    ├── basic_usage.py
    ├── streaming_search.py
    ├── batch_indexing.py
    ├── mcp_client.py
    └── citation_analysis.py
```

---

## 8. インポート設計

### 8.1 公開API

```python
# トップレベルインポート
from monjyu import MONJYUFacade
from monjyu import MONJYUConfig

# ドメインモデル
from monjyu.domain import AcademicPaper, TextChunk, ProgressiveIndex

# 検索結果
from monjyu.domain import SearchResult, SearchHit

# 例外
from monjyu.core.exceptions import (
    MONJYUError,
    ConfigurationError,
    IndexError,
    QueryError,
    LLMError,
)

# 設定
from monjyu.config import LocalConfig, ProductionConfig
```

### 8.2 コンポーネント別インポート

```python
# インデックス
from monjyu.index import ProgressiveIndexManager, TiktokenChunker

# クエリ
from monjyu.query import QueryRouter, LazySearch, HybridSearch

# LLM
from monjyu.llm import LLMFactory, AzureOpenAIClient, OllamaClient

# ストレージ
from monjyu.storage import FileStorage, LanceDBStore

# 引用ネットワーク
from monjyu.citation import CitationNetworkBuilder, CoCitationAnalyzer
```

### 8.3 __init__.py 設計

**monjyu/__init__.py**
```python
"""
MONJYU - Academic Paper RAG System

A Progressive GraphRAG implementation specialized for academic papers.
Supports local development (Ollama) and production deployment (Azure).
"""

__version__ = "3.0.0"
__author__ = "MONJYU Team"

# Main Facade
from monjyu.facade import MONJYUFacade

# Configuration
from monjyu.config import MONJYUConfig

# Domain Models
from monjyu.domain import (
    AcademicPaper,
    TextChunk,
    ProgressiveIndex,
    SearchResult,
    SearchHit,
)

# Exceptions
from monjyu.core.exceptions import (
    MONJYUError,
    ConfigurationError,
    IndexError,
    QueryError,
    LLMError,
    StorageError,
    PDFProcessError,
)

__all__ = [
    # Main
    "MONJYUFacade",
    "MONJYUConfig",
    # Domain
    "AcademicPaper",
    "TextChunk",
    "ProgressiveIndex",
    "SearchResult",
    "SearchHit",
    # Exceptions
    "MONJYUError",
    "ConfigurationError",
    "IndexError",
    "QueryError",
    "LLMError",
    "StorageError",
    "PDFProcessError",
    # Version
    "__version__",
]
```

---

## 9. 移行計画

### 9.1 現在の状態

```
MONJYU/
├── __init__.py
├── config/
│   └── lazy_search_config.py   # 既存
├── lazy_search/                # 既存（旧構造）
│   ├── core/
│   └── ...
├── prompts/
│   └── lazy_search_system_prompt.py  # 既存
├── specs/                      # 既存
├── tests/                      # 既存
└── docs/                       # 既存
```

### 9.2 移行フェーズ

#### Phase 1: コア再構成

```bash
# 1. 新ディレクトリ作成
mkdir -p monjyu/{core,domain,config}

# 2. coreモジュール作成
touch monjyu/core/{__init__,protocols,exceptions,types,tokenizer,text_utils}.py

# 3. domainモジュール作成
touch monjyu/domain/{__init__,paper,chunk,index,query,citation}.py

# 4. 既存lazy_search/coreからコピー
cp lazy_search/core/tokenizer.py monjyu/core/
cp lazy_search/core/text_utils.py monjyu/core/
```

#### Phase 2: インフラストラクチャ

```bash
# 1. document/（PDF処理）
mkdir -p monjyu/document/{pdf,parser}

# 2. index/
mkdir -p monjyu/index/{chunker,embedder,nlp,entity,community}

# 3. storage/
mkdir -p monjyu/storage/{cache,vector_store}

# 4. llm/
mkdir -p monjyu/llm
```

#### Phase 3: クエリ・引用

```bash
# 1. query/
mkdir -p monjyu/query/{vector,lazy,hybrid}

# 2. citation/
mkdir -p monjyu/citation

# 3. 既存lazy_search移行
mv lazy_search/*.py monjyu/query/lazy/
```

#### Phase 4: プレゼンテーション

```bash
# 1. cli/
mkdir -p monjyu/cli/commands

# 2. mcp/
mkdir -p monjyu/mcp

# 3. api/（オプション）
mkdir -p monjyu/api/routes
```

#### Phase 5: Facade統合

```bash
# 1. facade.py作成
touch monjyu/facade.py

# 2. __main__.py作成
touch monjyu/__main__.py

# 3. インポート整理
```

### 9.3 後方互換性

移行期間中のエイリアス:

```python
# monjyu/__init__.py
import warnings

# 旧パスからのインポートサポート（非推奨）
def __getattr__(name):
    if name == "lazy_search":
        warnings.warn(
            "Importing from monjyu.lazy_search is deprecated. "
            "Use monjyu.query.lazy instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from monjyu.query import lazy
        return lazy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## 10. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-01-06 | 初版作成（LazyGraphRAGベース） |
| 3.0.0 | 2025-12-25 | v3.0要件対応、学術論文特化、レイヤー分離、Progressive対応、MCP/REST追加 |
