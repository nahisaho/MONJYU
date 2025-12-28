# MONJYU ディレクトリ構造設計書

**バージョン**: 1.0.0  
**作成日**: 2025-01-06  
**ステータス**: Draft

---

## 1. 概要

本書は、MONJYUパッケージの完全なディレクトリ構造を定義します。

---

## 2. 目標ディレクトリ構造

```
MONJYU/
├── __init__.py                 # パッケージエントリーポイント
├── __main__.py                 # CLIエントリーポイント
├── py.typed                    # PEP 561 型情報マーカー
├── README.md                   # パッケージドキュメント
├── pyproject.toml              # パッケージ設定（将来分離時）
│
├── core/                       # コア抽象・インターフェース
│   ├── __init__.py
│   ├── protocols.py            # 全プロトコル定義
│   ├── chat_model.py           # ChatModelProtocol（既存）
│   ├── tokenizer.py            # Tokenizer（既存）
│   ├── search_result.py        # SearchResult（既存）
│   ├── conversation.py         # ConversationHistory（既存）
│   ├── text_utils.py           # テキストユーティリティ（既存）
│   ├── exceptions.py           # 例外クラス定義
│   └── types.py                # 型定義
│
├── config/                     # 設定管理
│   ├── __init__.py
│   ├── config.py               # MONJYUConfig クラス
│   ├── lazy_search_config.py   # LazySearchConfig（既存）
│   ├── index_config.py         # IndexConfig
│   ├── storage_config.py       # StorageConfig
│   ├── llm_config.py           # LLMConfig
│   └── defaults.py             # デフォルト値定義
│
├── index/                      # インデックス構築
│   ├── __init__.py
│   ├── builder.py              # IndexBuilder オーケストレーター
│   ├── chunker/                # テキストチャンク分割
│   │   ├── __init__.py
│   │   ├── protocol.py         # TextChunkerProtocol
│   │   ├── tiktoken_chunker.py # tiktoken実装
│   │   └── models.py           # TextChunk データモデル
│   │
│   ├── nlp/                    # NLP特徴抽出
│   │   ├── __init__.py
│   │   ├── protocol.py         # NLPExtractorProtocol
│   │   ├── basic_extractor.py  # 基本実装
│   │   ├── spacy_extractor.py  # spaCy実装（オプション）
│   │   └── models.py           # NLPFeatures データモデル
│   │
│   └── embedder/               # エンベディング（オプション）
│       ├── __init__.py
│       ├── protocol.py         # EmbedderProtocol
│       ├── openai_embedder.py  # OpenAI実装
│       ├── sentence_transformer.py # SentenceTransformer実装
│       └── models.py           # EmbeddingResult データモデル
│
├── query/                      # クエリ処理（LazySearch）
│   ├── __init__.py
│   ├── lazy_search/            # LazySearch実装（既存を移動）
│   │   ├── __init__.py
│   │   ├── search.py           # メイン検索（既存）
│   │   ├── state.py            # 状態管理（既存）
│   │   ├── context.py          # コンテキストビルダー（既存）
│   │   ├── query_expander.py   # クエリ拡張（既存）
│   │   ├── relevance.py        # 関連性評価（既存）
│   │   ├── claims.py           # クレーム抽出（既存）
│   │   ├── deepening.py        # 反復深化（既存）
│   │   └── answer_builder.py   # 回答構築（既存）
│   │
│   └── result.py               # QueryResult データモデル
│
├── storage/                    # ストレージ
│   ├── __init__.py
│   ├── protocol.py             # StorageProtocol
│   ├── file_storage.py         # ファイルベース実装
│   ├── cache.py                # キャッシュ管理
│   └── vector_store/           # ベクトルストア（将来拡張）
│       ├── __init__.py
│       ├── protocol.py         # VectorStoreProtocol
│       ├── lancedb.py          # LanceDB アダプター
│       └── chroma.py           # ChromaDB アダプター
│
├── llm/                        # LLMクライアント
│   ├── __init__.py
│   ├── factory.py              # LLMFactory
│   ├── openai_client.py        # OpenAI クライアント
│   ├── azure_client.py         # Azure OpenAI クライアント
│   └── ollama_client.py        # Ollama クライアント
│
├── prompts/                    # プロンプトテンプレート
│   ├── __init__.py
│   ├── lazy_search_system_prompt.py  # LazySearch用（既存）
│   ├── query_expansion.py      # クエリ拡張用
│   ├── relevance_test.py       # 関連性テスト用
│   └── claim_extraction.py     # クレーム抽出用
│
├── cli/                        # コマンドラインインターフェース
│   ├── __init__.py
│   ├── main.py                 # CLIエントリーポイント
│   ├── index_cmd.py            # index コマンド
│   ├── query_cmd.py            # query コマンド
│   ├── config_cmd.py           # config コマンド
│   └── utils.py                # CLI ユーティリティ
│
├── facade.py                   # MONJYU ファサードクラス
│
├── specs/                      # 仕様書（既存）
│   ├── 01_requirements.md
│   ├── 02_architecture.md
│   ├── 03_components.md
│   ├── 04_api.md
│   └── 05_directory_structure.md
│
├── tests/                      # テスト（既存を拡張）
│   ├── __init__.py
│   ├── conftest.py             # pytest設定
│   ├── unit/                   # ユニットテスト
│   │   ├── __init__.py
│   │   ├── test_chunker.py
│   │   ├── test_nlp_extractor.py
│   │   ├── test_embedder.py
│   │   ├── test_storage.py
│   │   ├── test_llm_clients.py
│   │   └── test_lazy_search/   # 既存テスト
│   │
│   ├── integration/            # 統合テスト
│   │   ├── __init__.py
│   │   ├── test_index_pipeline.py
│   │   ├── test_query_pipeline.py
│   │   └── test_end_to_end.py
│   │
│   ├── e2e/                    # E2Eテスト
│   │   ├── __init__.py
│   │   └── test_cli.py
│   │
│   ├── benchmarks/             # ベンチマーク
│   │   ├── __init__.py
│   │   └── test_performance.py
│   │
│   └── fixtures/               # テストデータ
│       ├── sample_documents/
│       ├── sample_index/
│       └── mock_responses/
│
└── docs/                       # ドキュメント
    ├── getting_started.md
    ├── configuration.md
    ├── api_reference.md
    └── examples/
        ├── basic_usage.py
        ├── streaming.py
        └── custom_llm.py
```

---

## 3. 現在の状態からの移行

### 3.1 既存ファイルのマッピング

| 現在のパス | 新しいパス | 変更内容 |
|-----------|-----------|---------|
| `lazy_search/core/chat_model.py` | `core/chat_model.py` | ディレクトリ移動 |
| `lazy_search/core/tokenizer.py` | `core/tokenizer.py` | ディレクトリ移動 |
| `lazy_search/core/search_result.py` | `core/search_result.py` | ディレクトリ移動 |
| `lazy_search/core/conversation.py` | `core/conversation.py` | ディレクトリ移動 |
| `lazy_search/core/text_utils.py` | `core/text_utils.py` | ディレクトリ移動 |
| `lazy_search/search.py` | `query/lazy_search/search.py` | ディレクトリ移動 |
| `lazy_search/state.py` | `query/lazy_search/state.py` | ディレクトリ移動 |
| `lazy_search/context.py` | `query/lazy_search/context.py` | ディレクトリ移動 |
| `lazy_search/query_expander.py` | `query/lazy_search/query_expander.py` | ディレクトリ移動 |
| `lazy_search/relevance.py` | `query/lazy_search/relevance.py` | ディレクトリ移動 |
| `lazy_search/claims.py` | `query/lazy_search/claims.py` | ディレクトリ移動 |
| `lazy_search/deepening.py` | `query/lazy_search/deepening.py` | ディレクトリ移動 |
| `lazy_search/answer_builder.py` | `query/lazy_search/answer_builder.py` | ディレクトリ移動 |
| `config/lazy_search_config.py` | `config/lazy_search_config.py` | 維持 |
| `prompts/lazy_search_system_prompt.py` | `prompts/lazy_search_system_prompt.py` | 維持 |

### 3.2 新規作成ファイル

```
新規作成が必要なファイル:
├── core/
│   ├── protocols.py            # 新規
│   ├── exceptions.py           # 新規
│   └── types.py                # 新規
│
├── config/
│   ├── config.py               # 新規
│   ├── index_config.py         # 新規
│   ├── storage_config.py       # 新規
│   ├── llm_config.py           # 新規
│   └── defaults.py             # 新規
│
├── index/                      # 全て新規
│   └── ...
│
├── storage/                    # 全て新規
│   └── ...
│
├── llm/                        # 全て新規
│   └── ...
│
├── cli/                        # 全て新規
│   └── ...
│
└── facade.py                   # 新規
```

---

## 4. インポートパス設計

### 4.1 公開API

```python
# トップレベルインポート
from MONJYU import MONJYU
from MONJYU import MONJYUConfig

# コンポーネント別インポート
from MONJYU.core import ChatModelProtocol, Tokenizer
from MONJYU.index import IndexBuilder, TextChunker
from MONJYU.storage import FileStorage
from MONJYU.llm import OpenAIClient, AzureOpenAIClient, OllamaClient
from MONJYU.query import LazySearch
```

### 4.2 内部インポート規則

```python
# 推奨: 絶対インポート
from MONJYU.core.tokenizer import TiktokenTokenizer
from MONJYU.config.lazy_search_config import LazySearchConfig

# 非推奨: 相対インポート（同一パッケージ内のみ許可）
from .tokenizer import TiktokenTokenizer  # 同一ディレクトリ内のみ
```

---

## 5. __init__.py 設計

### 5.1 トップレベル `MONJYU/__init__.py`

```python
"""
MONJYU - Standalone LazyGraphRAG Implementation

A lightweight, efficient implementation of LazyGraphRAG that operates
independently without requiring the full GraphRAG framework.
"""

__version__ = "1.0.0"
__author__ = "MONJYU Team"

# メインファサード
from MONJYU.facade import MONJYU

# 設定
from MONJYU.config.config import MONJYUConfig

# 結果型
from MONJYU.query.result import QueryResult
from MONJYU.index.builder import IndexResult

# 例外
from MONJYU.core.exceptions import (
    MONJYUError,
    ConfigurationError,
    IndexError,
    QueryError,
    LLMError,
    StorageError,
)

__all__ = [
    # Main
    "MONJYU",
    "MONJYUConfig",
    # Results
    "QueryResult",
    "IndexResult",
    # Exceptions
    "MONJYUError",
    "ConfigurationError",
    "IndexError",
    "QueryError",
    "LLMError",
    "StorageError",
    # Version
    "__version__",
]
```

### 5.2 `MONJYU/core/__init__.py`

```python
"""Core protocols and abstractions."""

from MONJYU.core.protocols import (
    ChatModelProtocol,
    TokenizerProtocol,
    StorageProtocol,
    EmbedderProtocol,
)
from MONJYU.core.tokenizer import TiktokenTokenizer
from MONJYU.core.search_result import SearchResult
from MONJYU.core.conversation import ConversationHistory

__all__ = [
    "ChatModelProtocol",
    "TokenizerProtocol",
    "StorageProtocol",
    "EmbedderProtocol",
    "TiktokenTokenizer",
    "SearchResult",
    "ConversationHistory",
]
```

---

## 6. 出力ディレクトリ構造

インデックス構築後の出力ディレクトリ構造:

```
output/                         # デフォルト出力ディレクトリ
├── metadata.json               # インデックスメタデータ
│   {
│     "version": "1.0.0",
│     "created_at": "2025-01-06T12:00:00Z",
│     "document_count": 150,
│     "chunk_count": 2450,
│     "config": { ... }
│   }
│
├── chunks.parquet              # テキストチャンク
│   columns: [id, content, source_id, start_offset, end_offset, token_count, metadata]
│
├── nlp_features.parquet        # NLP特徴（オプション）
│   columns: [chunk_id, keywords, entities, summary]
│
├── embeddings.parquet          # エンベディング（オプション）
│   columns: [chunk_id, vector, model, dimensions]
│
├── sources/                    # 元文書情報
│   └── documents.parquet       # 文書メタデータ
│       columns: [id, filename, path, size, created_at, hash]
│
└── cache/                      # キャッシュ
    └── query_cache.json        # クエリキャッシュ
```

---

## 7. 設定ファイル構造

### 7.1 monjyu.yaml

```yaml
# MONJYU Configuration File
version: "1.0"

# Index Configuration
index:
  chunker:
    chunk_size: 300
    chunk_overlap: 100
    encoding_model: "cl100k_base"
  
  nlp:
    enabled: true
    extractor: "basic"
  
  embeddings:
    enabled: false
    provider: "openai"
    model: "text-embedding-3-small"

# Storage Configuration
storage:
  type: "file"
  path: "./output"
  format: "parquet"

# LLM Configuration
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 4096
  
  # Provider-specific settings
  openai:
    api_key: "${OPENAI_API_KEY}"
  
  azure_openai:
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    deployment: "${AZURE_OPENAI_DEPLOYMENT}"
    api_version: "2024-02-15-preview"
  
  ollama:
    host: "http://localhost:11434"

# Query Configuration
query:
  rate_query: 1
  rate_relevancy: 32
  rate_cluster: 2
  query_expansion_limit: 8
  max_context_tokens: 8000

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null  # or path to log file
```

---

## 8. 移行計画

### 8.1 Phase 1: コア再構成

```bash
# 1. core/ ディレクトリ作成と移動
mkdir -p MONJYU/core
mv MONJYU/lazy_search/core/*.py MONJYU/core/

# 2. 新規ファイル作成
touch MONJYU/core/protocols.py
touch MONJYU/core/exceptions.py
touch MONJYU/core/types.py
```

### 8.2 Phase 2: クエリモジュール再構成

```bash
# 1. query/lazy_search/ ディレクトリ作成
mkdir -p MONJYU/query/lazy_search

# 2. ファイル移動
mv MONJYU/lazy_search/*.py MONJYU/query/lazy_search/

# 3. インポートパス更新
# すべてのファイルで from MONJYU.lazy_search → from MONJYU.query.lazy_search
```

### 8.3 Phase 3: 新規コンポーネント追加

```bash
# 1. index/ モジュール作成
mkdir -p MONJYU/index/{chunker,nlp,embedder}

# 2. storage/ モジュール作成
mkdir -p MONJYU/storage/vector_store

# 3. llm/ モジュール作成
mkdir -p MONJYU/llm

# 4. cli/ モジュール作成
mkdir -p MONJYU/cli
```

---

## 9. 互換性

### 9.1 後方互換性

移行期間中、以下のエイリアスを維持:

```python
# MONJYU/__init__.py
# 後方互換性のためのエイリアス（非推奨警告付き）
import warnings

def _lazy_import_lazy_search():
    warnings.warn(
        "Importing from MONJYU.lazy_search is deprecated. "
        "Use MONJYU.query.lazy_search instead.",
        DeprecationWarning,
        stacklevel=3
    )
    from MONJYU.query import lazy_search
    return lazy_search

# 動的インポートで後方互換性を維持
```

### 9.2 非推奨スケジュール

| バージョン | 状態 |
|-----------|------|
| 1.0.x | 旧パス利用可能（警告あり） |
| 1.1.x | 旧パス利用可能（警告あり） |
| 2.0.0 | 旧パス削除 |

---

## 10. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-01-06 | 初版作成 |
