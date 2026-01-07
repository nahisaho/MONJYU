# MONJYU API仕様書 v3.1

**バージョン**: 3.1.0  
**作成日**: 2026-01-07  
**ステータス**: Approved  
**対応要件**: REQ-MONJYU-001 v3.1  
**テスト状況**: 2,417テスト / 83% カバレッジ (2026-01-07時点)  
**実装状況**: 全API実装完了

---

## 目次

1. [概要](#1-概要)
2. [Python API](#2-python-api)
3. [MCP Server API](#3-mcp-server-api)
4. [REST API（オプション）](#4-rest-apiオプション)
5. [CLI API](#5-cli-api)
6. [エラーハンドリング](#6-エラーハンドリング)
7. [環境変数](#7-環境変数)
8. [使用例](#8-使用例)
9. [変更履歴](#9-変更履歴)

---

## 1. 概要

### 1.1 APIレイヤー構成

```
┌─────────────────────────────────────────────────────────┐
│                   クライアント                           │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │   CLI   │  │  MCP Client │  │   REST Client    │    │
│  └────┬────┘  └──────┬──────┘  └────────┬─────────┘    │
└───────┼──────────────┼──────────────────┼──────────────┘
        │              │                  │
┌───────┼──────────────┼──────────────────┼──────────────┐
│       ▼              ▼                  ▼              │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │CLI Layer│  │ MCP Server  │  │    REST API      │   │
│  └────┬────┘  └──────┬──────┘  └────────┬─────────┘   │
│       │              │                  │              │
│       └──────────────┼──────────────────┘              │
│                      ▼                                 │
│            ┌─────────────────┐                         │
│            │  MONJYUFacade   │  ← 統一エントリーポイント│
│            └────────┬────────┘                         │
│                     │                                  │
│  ┌──────────────────┼──────────────────┐              │
│  │                  ▼                  │              │
│  │  ┌───────┐ ┌───────┐ ┌───────────┐  │              │
│  │  │Index  │ │Query  │ │ Citation  │  │  Core APIs  │
│  │  │Manager│ │Router │ │ Network   │  │              │
│  │  └───────┘ └───────┘ └───────────┘  │              │
│  └─────────────────────────────────────┘              │
└───────────────────────────────────────────────────────┘
```

### 1.2 API設計原則

| 原則 | 説明 |
|------|------|
| **統一インターフェース** | MONJYUFacadeを通じた統一アクセス |
| **非同期優先** | async/await による非同期設計 |
| **型安全性** | dataclass + Protocol による型定義 |
| **環境非依存** | Local/Production透過的切り替え |
| **ストリーミング対応** | AsyncIterator によるストリーム |

---

## 2. Python API

### 2.1 MONJYUFacade クラス

#### 2.1.1 クラス定義

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator
from pathlib import Path


class MONJYUFacade:
    """
    MONJYUの統一ファサードクラス
    
    すべての機能への統一的なエントリーポイントを提供します。
    学術論文RAGシステムの全機能を集約。
    
    Example:
        >>> from monjyu import MONJYUFacade
        >>> 
        >>> # 初期化（環境自動検出）
        >>> facade = MONJYUFacade.from_environment()
        >>> 
        >>> # 論文インデックス構築
        >>> result = await facade.index(
        ...     paths=["./papers"],
        ...     level=2
        ... )
        >>> 
        >>> # 検索
        >>> answer = await facade.search(
        ...     "Transformerの注意機構について説明して",
        ...     mode="auto"
        ... )
        >>> print(answer.response)
    """
    
    def __init__(
        self,
        config: 'MONJYUConfig',
        index_manager: 'ProgressiveIndexManager',
        query_router: 'QueryRouter',
        vector_search: 'VectorSearchProtocol',
        lazy_search: 'LazySearch',
        citation_builder: 'CitationNetworkBuilder',
        llm_client: 'ChatModelProtocol',
        storage: 'StorageProtocol'
    ):
        """
        MONJYUFacadeインスタンスを初期化
        
        Args:
            config: MONJYU設定
            index_manager: プログレッシブインデックスマネージャー
            query_router: クエリルーター
            vector_search: ベクトル検索
            lazy_search: LazySearch
            citation_builder: 引用ネットワークビルダー
            llm_client: LLMクライアント
            storage: ストレージ
        """
        self.config = config
        self.index_manager = index_manager
        self.query_router = query_router
        self.vector_search = vector_search
        self.lazy_search = lazy_search
        self.citation_builder = citation_builder
        self.llm_client = llm_client
        self.storage = storage
        self._index: Optional['ProgressiveIndex'] = None
```

#### 2.1.2 ファクトリーメソッド

```python
@classmethod
def from_config(cls, config_path: str | Path) -> "MONJYUFacade":
    """
    設定ファイルからMONJYUFacadeインスタンスを作成
    
    Args:
        config_path: YAML設定ファイルのパス
    
    Returns:
        MONJYUFacade: 初期化済みインスタンス
    
    Raises:
        ConfigurationError: 設定ファイルの読み込みエラー
        FileNotFoundError: 設定ファイルが見つからない
    
    Example:
        >>> facade = MONJYUFacade.from_config("config/local.yaml")
    """
    ...


@classmethod
def from_environment(cls) -> "MONJYUFacade":
    """
    環境変数からMONJYUFacadeインスタンスを作成
    
    自動検出ロジック:
    - AZURE_OPENAI_ENDPOINT が設定されている場合 → Azure環境
    - それ以外 → Ollama（ローカル）環境
    
    Returns:
        MONJYUFacade: 初期化済みインスタンス
    
    Example:
        >>> facade = MONJYUFacade.from_environment()
    """
    ...


@classmethod
def create_local(
    cls,
    ollama_host: str = "http://192.168.224.1:11434",
    index_path: str = "./output"
) -> "MONJYUFacade":
    """
    ローカル開発環境用インスタンスを作成
    
    Args:
        ollama_host: Ollamaホスト（WSL環境）
        index_path: インデックス保存パス
    
    Returns:
        MONJYUFacade: ローカル環境用インスタンス
    
    Example:
        >>> facade = MONJYUFacade.create_local()
    """
    ...


@classmethod
def create_azure(
    cls,
    endpoint: str,
    api_key: str,
    search_endpoint: str,
    search_api_key: str
) -> "MONJYUFacade":
    """
    Azure本番環境用インスタンスを作成
    
    Args:
        endpoint: Azure OpenAI エンドポイント
        api_key: Azure OpenAI APIキー
        search_endpoint: Azure AI Search エンドポイント
        search_api_key: Azure AI Search APIキー
    
    Returns:
        MONJYUFacade: Azure環境用インスタンス
    """
    ...
```

---

### 2.2 インデックスAPI

#### 2.2.1 index メソッド

```python
async def index(
    self,
    paths: List[str | Path],
    output_path: Optional[str | Path] = None,
    level: int = 1,
    recursive: bool = True,
    file_patterns: List[str] = ["*.pdf"],
    callback: Optional[Callable[['IndexProgress'], None]] = None
) -> 'IndexResult':
    """
    学術論文からインデックスを構築
    
    Args:
        paths: 入力パス（ファイルまたはディレクトリ）のリスト
        output_path: 出力ディレクトリ（省略時は設定値を使用）
        level: インデックスレベル（0-4）
            - 0: Raw（チャンクのみ）
            - 1: Lazy（NLP抽出、NounGraph）
            - 2: Partial（埋め込みベクトル追加）
            - 3: Full（エンティティ、コミュニティ）
            - 4: Enhanced（クレーム事前抽出）
        recursive: サブディレクトリを再帰的に処理するか
        file_patterns: ファイルパターン（glob形式）
        callback: 進捗コールバック関数
    
    Returns:
        IndexResult: インデックス構築結果
    
    Raises:
        IndexError: インデックス構築エラー
        PDFProcessError: PDF処理エラー
        FileNotFoundError: 入力パスが見つからない
    
    Example:
        >>> result = await facade.index(
        ...     paths=["./papers/2024"],
        ...     level=2,
        ...     file_patterns=["*.pdf"]
        ... )
        >>> print(f"Indexed {result.paper_count} papers, {result.chunk_count} chunks")
    """
    ...
```

#### 2.2.2 upgrade_index メソッド

```python
async def upgrade_index(
    self,
    target_level: int,
    callback: Optional[Callable[['IndexProgress'], None]] = None
) -> 'IndexResult':
    """
    インデックスをより高いレベルにアップグレード
    
    Args:
        target_level: 目標レベル（現在より大きい必要がある）
        callback: 進捗コールバック関数
    
    Returns:
        IndexResult: アップグレード結果
    
    Raises:
        IndexError: 現在のレベルより低いレベルを指定した場合
    
    Example:
        >>> # Level 1 → Level 3 へアップグレード
        >>> result = await facade.upgrade_index(target_level=3)
    """
    ...
```

#### 2.2.3 IndexResult データクラス

```python
@dataclass
class IndexResult:
    """インデックス構築結果"""
    success: bool
    paper_count: int          # 処理した論文数
    chunk_count: int          # チャンク数
    level: int                # インデックスレベル
    elapsed_time: float       # 処理時間（秒）
    output_path: Path         # 出力パス
    
    # 詳細情報
    entity_count: int = 0     # エンティティ数（Level 3+）
    community_count: int = 0  # コミュニティ数（Level 3+）
    embedding_dim: int = 0    # 埋め込み次元（Level 2+）
    
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IndexProgress:
    """インデックス構築進捗"""
    stage: str              # "pdf_processing" | "chunking" | "nlp" | "embedding" | "entity" | "community"
    current: int            # 現在の処理数
    total: int              # 合計数
    percentage: float       # 進捗率（0-100）
    message: str            # 進捗メッセージ
    level: int              # 現在のレベル
```

---

### 2.3 検索API

#### 2.3.1 search メソッド

```python
async def search(
    self,
    query: str,
    mode: str = "auto",
    top_k: int = 10,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    **kwargs
) -> 'SearchResult':
    """
    学術論文を検索し、回答を生成
    
    Args:
        query: クエリ文字列
        mode: 検索モード
            - "auto": 自動選択（QueryRouter使用）
            - "vector": ベクトル検索のみ
            - "lazy": LazySearch
            - "hybrid": ハイブリッド検索
        top_k: 返却するチャンク数
        conversation_history: 会話履歴（マルチターン対応）
        **kwargs: 追加パラメータ
    
    Returns:
        SearchResult: 検索結果
    
    Raises:
        QueryError: クエリ実行エラー
        LLMError: LLM呼び出しエラー
    
    Example:
        >>> result = await facade.search(
        ...     "Transformerモデルの計算量削減手法を比較して",
        ...     mode="lazy"
        ... )
        >>> print(result.response)
    """
    ...


async def search_stream(
    self,
    query: str,
    mode: str = "auto",
    **kwargs
) -> AsyncIterator[str]:
    """
    ストリーミング検索を実行
    
    Args:
        query: クエリ文字列
        mode: 検索モード
        **kwargs: 追加パラメータ
    
    Yields:
        str: 回答の部分文字列
    
    Example:
        >>> async for chunk in facade.search_stream("主要な発見は？"):
        ...     print(chunk, end="", flush=True)
    """
    ...
```

#### 2.3.2 SearchResult データクラス

```python
@dataclass
class SearchResult:
    """検索結果"""
    query: str
    response: str                    # 生成された回答
    sources: List['SourceChunk']     # 参照ソース
    mode_used: str                   # 使用された検索モード
    metadata: 'SearchMetadata'


@dataclass
class SourceChunk:
    """参照ソースチャンク"""
    chunk_id: str
    paper_id: str
    paper_title: str
    section_type: Optional[str]      # "abstract" | "introduction" | "method" | ...
    content: str
    relevance_score: float
    
    # 論文メタデータ
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    doi: Optional[str] = None


@dataclass
class SearchMetadata:
    """検索メタデータ"""
    query_expansion: List[str]       # 拡張されたクエリ
    query_type: str                  # "survey" | "factoid" | "comparison" | "exploration"
    chunks_evaluated: int            # 評価されたチャンク数
    chunks_selected: int             # 選択されたチャンク数
    claims_extracted: int            # 抽出されたクレーム数（LazySearch時）
    llm_calls: int                   # LLM呼び出し回数
    total_tokens: int                # 合計トークン数
    elapsed_time_ms: float           # 処理時間（ミリ秒）
```

---

### 2.4 論文管理API

#### 2.4.1 get_paper メソッド

```python
async def get_paper(
    self,
    paper_id: str
) -> 'AcademicPaper':
    """
    論文詳細を取得
    
    Args:
        paper_id: 論文ID
    
    Returns:
        AcademicPaper: 論文データ
    
    Raises:
        NotFoundError: 論文が見つからない
    
    Example:
        >>> paper = await facade.get_paper("arxiv:2406.12345")
        >>> print(paper.title)
        >>> print(paper.abstract)
    """
    ...


async def list_papers(
    self,
    filter: Optional['PaperFilter'] = None,
    sort_by: str = "date",
    limit: int = 100
) -> List['AcademicPaperSummary']:
    """
    論文一覧を取得
    
    Args:
        filter: フィルター条件
        sort_by: ソートキー（"date" | "title" | "citations"）
        limit: 最大件数
    
    Returns:
        List[AcademicPaperSummary]: 論文サマリーリスト
    """
    ...


@dataclass
class PaperFilter:
    """論文フィルター"""
    authors: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    keywords: Optional[List[str]] = None
    venue: Optional[str] = None
```

---

### 2.5 引用ネットワークAPI

#### 2.5.1 get_citations メソッド

```python
async def get_citations(
    self,
    paper_id: str,
    depth: int = 1,
    direction: str = "both"
) -> 'CitationNetwork':
    """
    論文の引用ネットワークを取得
    
    Args:
        paper_id: 論文ID
        depth: 探索深度
        direction: 方向（"citing" | "cited" | "both"）
    
    Returns:
        CitationNetwork: 引用ネットワーク
    
    Example:
        >>> network = await facade.get_citations("arxiv:2406.12345", depth=2)
        >>> print(f"被引用数: {len(network.cited_by)}")
        >>> print(f"参照数: {len(network.references)}")
    """
    ...


async def get_co_citations(
    self,
    paper_id: str,
    min_count: int = 2
) -> List['CoCitationPair']:
    """
    共引用論文を取得
    
    Args:
        paper_id: 論文ID
        min_count: 最小共引用数
    
    Returns:
        List[CoCitationPair]: 共引用ペアリスト
    """
    ...


@dataclass
class CitationNetwork:
    """引用ネットワーク"""
    paper_id: str
    cited_by: List['CitationEdge']      # この論文を引用している論文
    references: List['CitationEdge']     # この論文が引用している論文
    
    def to_networkx(self) -> 'nx.DiGraph':
        """NetworkXグラフに変換"""
        ...
```

---

### 2.6 分析・生成API

#### 2.6.1 summarize メソッド

```python
async def summarize(
    self,
    paper_id: str,
    section: Optional[str] = None,
    style: str = "academic"
) -> str:
    """
    論文を要約
    
    Args:
        paper_id: 論文ID
        section: セクション指定（省略時は全体要約）
        style: 要約スタイル（"academic" | "simple" | "bullet"）
    
    Returns:
        str: 要約テキスト
    
    Example:
        >>> summary = await facade.summarize("arxiv:2406.12345", section="method")
        >>> print(summary)
    """
    ...
```

#### 2.6.2 compare メソッド

```python
async def compare(
    self,
    paper_ids: List[str],
    aspect: Optional[str] = None
) -> str:
    """
    複数論文を比較
    
    Args:
        paper_ids: 比較する論文IDリスト
        aspect: 比較観点（"method" | "result" | "contribution"）
    
    Returns:
        str: 比較結果テキスト
    
    Example:
        >>> comparison = await facade.compare(
        ...     ["arxiv:2406.12345", "arxiv:2405.67890"],
        ...     aspect="method"
        ... )
    """
    ...
```

#### 2.6.3 survey メソッド

```python
async def survey(
    self,
    topic: str,
    max_papers: int = 10
) -> str:
    """
    トピックに関するサーベイを生成
    
    Args:
        topic: サーベイトピック
        max_papers: 使用する最大論文数
    
    Returns:
        str: サーベイテキスト
    
    Example:
        >>> survey = await facade.survey(
        ...     "Vision Transformerの効率化手法",
        ...     max_papers=20
        ... )
    """
    ...
```

---

## 3. MCP Server API

### 3.1 概要

Model Context Protocol (MCP) サーバーとして、AIアシスタント（Claude、GitHub Copilot等）からのツール呼び出しに対応。

### 3.2 ツール一覧

| ツール名 | 説明 | 対応Facadeメソッド |
|---------|------|-------------------|
| `monjyu_search` | 論文検索 | `search()` |
| `monjyu_index` | インデックス構築 | `index()` |
| `monjyu_get_paper` | 論文詳細取得 | `get_paper()` |
| `monjyu_citations` | 引用ネットワーク | `get_citations()` |
| `monjyu_summarize` | 論文要約 | `summarize()` |
| `monjyu_compare` | 論文比較 | `compare()` |
| `monjyu_survey` | サーベイ生成 | `survey()` |

### 3.3 ツールスキーマ

#### 3.3.1 monjyu_search

```json
{
  "name": "monjyu_search",
  "description": "Search academic papers and generate answers using RAG",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query in natural language"
      },
      "mode": {
        "type": "string",
        "enum": ["auto", "vector", "lazy", "hybrid"],
        "default": "auto",
        "description": "Search mode"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "description": "Number of results to return"
      }
    },
    "required": ["query"]
  }
}
```

#### 3.3.2 monjyu_index

```json
{
  "name": "monjyu_index",
  "description": "Build index from academic papers (PDF files)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "paths": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Paths to PDF files or directories"
      },
      "level": {
        "type": "integer",
        "minimum": 0,
        "maximum": 4,
        "default": 1,
        "description": "Index level (0=Raw, 1=Lazy, 2=Partial, 3=Full, 4=Enhanced)"
      }
    },
    "required": ["paths"]
  }
}
```

#### 3.3.3 monjyu_get_paper

```json
{
  "name": "monjyu_get_paper",
  "description": "Get detailed information about a specific paper",
  "inputSchema": {
    "type": "object",
    "properties": {
      "paper_id": {
        "type": "string",
        "description": "Paper ID (e.g., arxiv:2406.12345)"
      }
    },
    "required": ["paper_id"]
  }
}
```

#### 3.3.4 monjyu_citations

```json
{
  "name": "monjyu_citations",
  "description": "Get citation network for a paper",
  "inputSchema": {
    "type": "object",
    "properties": {
      "paper_id": {
        "type": "string",
        "description": "Paper ID"
      },
      "depth": {
        "type": "integer",
        "default": 1,
        "description": "Depth of citation network traversal"
      },
      "direction": {
        "type": "string",
        "enum": ["citing", "cited", "both"],
        "default": "both"
      }
    },
    "required": ["paper_id"]
  }
}
```

#### 3.3.5 monjyu_summarize

```json
{
  "name": "monjyu_summarize",
  "description": "Generate summary of a paper or specific section",
  "inputSchema": {
    "type": "object",
    "properties": {
      "paper_id": {
        "type": "string",
        "description": "Paper ID"
      },
      "section": {
        "type": "string",
        "enum": ["abstract", "introduction", "method", "results", "conclusion"],
        "description": "Specific section to summarize (optional)"
      },
      "style": {
        "type": "string",
        "enum": ["academic", "simple", "bullet"],
        "default": "academic"
      }
    },
    "required": ["paper_id"]
  }
}
```

#### 3.3.6 monjyu_compare

```json
{
  "name": "monjyu_compare",
  "description": "Compare multiple papers on specific aspects",
  "inputSchema": {
    "type": "object",
    "properties": {
      "paper_ids": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "description": "List of paper IDs to compare"
      },
      "aspect": {
        "type": "string",
        "enum": ["method", "result", "contribution", "limitation"],
        "description": "Comparison aspect"
      }
    },
    "required": ["paper_ids"]
  }
}
```

#### 3.3.7 monjyu_survey

```json
{
  "name": "monjyu_survey",
  "description": "Generate a survey on a specific topic",
  "inputSchema": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "Survey topic"
      },
      "max_papers": {
        "type": "integer",
        "default": 10,
        "description": "Maximum number of papers to include"
      }
    },
    "required": ["topic"]
  }
}
```

### 3.4 MCP設定例

```json
{
  "mcpServers": {
    "monjyu": {
      "command": "python",
      "args": ["-m", "monjyu.mcp_server"],
      "env": {
        "MONJYU_INDEX_PATH": "./output",
        "OLLAMA_HOST": "http://192.168.224.1:11434"
      }
    }
  }
}
```

---

## 4. REST API（オプション）

### 4.1 エンドポイント一覧

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/v1/search` | 検索実行 |
| POST | `/api/v1/index` | インデックス構築 |
| GET | `/api/v1/papers` | 論文一覧 |
| GET | `/api/v1/papers/{paper_id}` | 論文詳細 |
| GET | `/api/v1/papers/{paper_id}/citations` | 引用ネットワーク |
| POST | `/api/v1/papers/{paper_id}/summarize` | 論文要約 |
| POST | `/api/v1/compare` | 論文比較 |
| POST | `/api/v1/survey` | サーベイ生成 |
| GET | `/api/v1/health` | ヘルスチェック |

### 4.2 リクエスト/レスポンス例

#### 4.2.1 検索

**Request:**
```http
POST /api/v1/search HTTP/1.1
Content-Type: application/json

{
  "query": "Transformerの注意機構の計算量削減手法",
  "mode": "auto",
  "top_k": 10
}
```

**Response:**
```json
{
  "query": "Transformerの注意機構の計算量削減手法",
  "response": "Transformerの注意機構の計算量削減には主に以下の手法があります...",
  "sources": [
    {
      "chunk_id": "chunk_001",
      "paper_id": "arxiv:2009.14794",
      "paper_title": "Efficient Transformers: A Survey",
      "section_type": "method",
      "content": "...",
      "relevance_score": 0.95
    }
  ],
  "metadata": {
    "query_type": "survey",
    "mode_used": "lazy",
    "chunks_evaluated": 150,
    "chunks_selected": 12,
    "elapsed_time_ms": 2340
  }
}
```

#### 4.2.2 エラーレスポンス

```json
{
  "error": {
    "code": "E005",
    "message": "Index not found",
    "details": {
      "path": "./output"
    }
  }
}
```

---

## 5. CLI API

### 5.1 コマンド一覧

| コマンド | 説明 |
|---------|------|
| `monjyu index` | インデックス構築 |
| `monjyu search` | 検索実行 |
| `monjyu upgrade` | インデックスアップグレード |
| `monjyu serve` | MCPサーバー起動 |
| `monjyu papers` | 論文一覧・詳細 |
| `monjyu config` | 設定管理 |
| `monjyu version` | バージョン表示 |

### 5.2 index コマンド

```bash
monjyu index [OPTIONS] INPUT_PATH
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `INPUT_PATH` | | PATH | 必須 | 入力パス |
| `--output` | `-o` | PATH | `./output` | 出力ディレクトリ |
| `--level` | `-l` | INT | `1` | インデックスレベル（0-4） |
| `--recursive` | `-r` | FLAG | `true` | 再帰的処理 |
| `--pattern` | `-p` | TEXT | `*.pdf` | ファイルパターン |
| `--verbose` | `-v` | FLAG | `false` | 詳細ログ |

```bash
# 基本使用
monjyu index ./papers --level 2

# 複数パターン
monjyu index ./papers -p "*.pdf" -p "*.txt" -l 1

# 出力先指定
monjyu index ./papers -o ./my_index -l 3 -v
```

### 5.3 search コマンド

```bash
monjyu search [OPTIONS] QUERY
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `QUERY` | | TEXT | 必須 | クエリ文字列 |
| `--index` | `-i` | PATH | `./output` | インデックスパス |
| `--mode` | `-m` | TEXT | `auto` | 検索モード |
| `--top-k` | `-k` | INT | `10` | 結果数 |
| `--stream` | `-s` | FLAG | `false` | ストリーミング |
| `--json` | `-j` | FLAG | `false` | JSON出力 |
| `--interactive` | | FLAG | `false` | 対話モード |

```bash
# 基本検索
monjyu search "Transformerの効率化手法について"

# ストリーミング
monjyu search "主要な発見は？" --stream

# 対話モード
monjyu search --interactive
```

### 5.4 upgrade コマンド

```bash
monjyu upgrade [OPTIONS] TARGET_LEVEL
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `TARGET_LEVEL` | | INT | 必須 | 目標レベル |
| `--index` | `-i` | PATH | `./output` | インデックスパス |

```bash
# Level 1 → Level 3 へアップグレード
monjyu upgrade 3 -i ./output
```

### 5.5 serve コマンド

```bash
monjyu serve [OPTIONS]
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `--transport` | `-t` | TEXT | `stdio` | トランスポート（stdio/sse） |
| `--host` | | TEXT | `127.0.0.1` | ホスト（SSE時） |
| `--port` | `-p` | INT | `8000` | ポート（SSE時） |
| `--index` | `-i` | PATH | `./output` | インデックスパス |

```bash
# stdio（デフォルト）
monjyu serve

# SSE
monjyu serve -t sse -p 8080
```

### 5.6 papers コマンド

```bash
monjyu papers [OPTIONS] [PAPER_ID]
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `PAPER_ID` | | TEXT | - | 論文ID（指定時は詳細表示） |
| `--index` | `-i` | PATH | `./output` | インデックスパス |
| `--limit` | `-n` | INT | `20` | 表示件数 |
| `--json` | `-j` | FLAG | `false` | JSON出力 |

```bash
# 論文一覧
monjyu papers -n 50

# 論文詳細
monjyu papers arxiv:2406.12345
```

---

## 6. エラーハンドリング

### 6.1 例外階層

```python
class MONJYUError(Exception):
    """MONJYU基底例外"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ConfigurationError(MONJYUError):
    """設定エラー（E001-E009）"""
    pass


class IndexError(MONJYUError):
    """インデックスエラー（E010-E019）"""
    pass


class QueryError(MONJYUError):
    """クエリエラー（E020-E029）"""
    pass


class LLMError(MONJYUError):
    """LLMエラー（E030-E039）"""
    pass


class StorageError(MONJYUError):
    """ストレージエラー（E040-E049）"""
    pass


class PDFProcessError(MONJYUError):
    """PDF処理エラー（E050-E059）"""
    pass


class NotFoundError(MONJYUError):
    """リソース未検出エラー（E060-E069）"""
    pass


class ExternalAPIError(MONJYUError):
    """外部APIエラー（E070-E079）"""
    pass
```

### 6.2 エラーコード一覧

| コード | 名前 | 説明 |
|--------|------|------|
| E001 | CONFIG_NOT_FOUND | 設定ファイルが見つからない |
| E002 | CONFIG_INVALID | 設定ファイルが無効 |
| E003 | CONFIG_MISSING_REQUIRED | 必須設定が不足 |
| E010 | INDEX_NOT_FOUND | インデックスが見つからない |
| E011 | INDEX_INVALID | インデックスが無効 |
| E012 | INDEX_LEVEL_ERROR | レベル指定エラー |
| E013 | INDEX_BUILD_FAILED | インデックス構築失敗 |
| E020 | QUERY_EMPTY | クエリが空 |
| E021 | QUERY_TOO_LONG | クエリが長すぎる |
| E022 | QUERY_ROUTING_FAILED | ルーティング失敗 |
| E030 | LLM_AUTH_ERROR | LLM認証エラー |
| E031 | LLM_RATE_LIMIT | LLMレート制限 |
| E032 | LLM_TIMEOUT | LLMタイムアウト |
| E033 | LLM_INVALID_RESPONSE | LLM無効応答 |
| E040 | STORAGE_READ_ERROR | ストレージ読み込みエラー |
| E041 | STORAGE_WRITE_ERROR | ストレージ書き込みエラー |
| E050 | PDF_PARSE_ERROR | PDF解析エラー |
| E051 | PDF_ENCRYPTED | PDFが暗号化されている |
| E060 | PAPER_NOT_FOUND | 論文が見つからない |
| E061 | CHUNK_NOT_FOUND | チャンクが見つからない |
| E070 | AZURE_API_ERROR | Azure APIエラー |
| E071 | OLLAMA_CONNECTION_ERROR | Ollama接続エラー |

---

## 7. 環境変数

### 7.1 共通設定

| 変数名 | 説明 | デフォルト |
|--------|------|----------|
| `MONJYU_CONFIG_PATH` | 設定ファイルパス | `monjyu.yaml` |
| `MONJYU_INDEX_PATH` | インデックスパス | `./output` |
| `MONJYU_LOG_LEVEL` | ログレベル | `INFO` |
| `MONJYU_ENVIRONMENT` | 環境（local/production） | `local` |

### 7.2 LLM設定

| 変数名 | 説明 | デフォルト |
|--------|------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI エンドポイント | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI APIキー | - |
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI デプロイメント | `gpt-4o` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | 埋め込みデプロイメント | `text-embedding-3-large` |
| `OLLAMA_HOST` | Ollamaホスト | `http://192.168.224.1:11434` |
| `OLLAMA_MODEL` | Ollamaモデル | `llama3.2` |
| `OLLAMA_EMBEDDING_MODEL` | 埋め込みモデル | `nomic-embed-text` |

### 7.3 ストレージ設定

| 変数名 | 説明 | デフォルト |
|--------|------|----------|
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Storage接続文字列 | - |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search エンドポイント | - |
| `AZURE_SEARCH_API_KEY` | Azure AI Search APIキー | - |
| `REDIS_HOST` | Redisホスト | - |
| `REDIS_PASSWORD` | Redisパスワード | - |

### 7.4 PDF処理設定

| 変数名 | 説明 | デフォルト |
|--------|------|----------|
| `AZURE_DI_ENDPOINT` | Azure Document Intelligence エンドポイント | - |
| `AZURE_DI_KEY` | Azure Document Intelligence キー | - |

### 7.5 優先順位

1. コマンドライン引数
2. 設定ファイル（YAML）
3. 環境変数
4. デフォルト値

---

## 8. 使用例

### 8.1 基本ワークフロー

```python
import asyncio
from monjyu import MONJYUFacade

async def main():
    # 1. 初期化（ローカル環境）
    facade = MONJYUFacade.create_local()
    
    # 2. 論文インデックス構築
    result = await facade.index(
        paths=["./papers/transformer"],
        level=2,
        file_patterns=["*.pdf"]
    )
    print(f"✅ Indexed {result.paper_count} papers, {result.chunk_count} chunks")
    
    # 3. 検索
    search_result = await facade.search(
        "Vision Transformerの主要な改良点は何ですか？",
        mode="lazy"
    )
    print(f"\n📝 回答:\n{search_result.response}")
    
    # 4. ソース確認
    print(f"\n📚 参照元 ({len(search_result.sources)} chunks):")
    for src in search_result.sources[:3]:
        print(f"  - [{src.relevance_score:.2f}] {src.paper_title}")


asyncio.run(main())
```

### 8.2 対話型検索

```python
async def interactive_search():
    facade = MONJYUFacade.from_environment()
    
    history = []
    
    while True:
        query = input("\n❓ 質問 (qで終了): ")
        if query.lower() == 'q':
            break
        
        result = await facade.search(
            query,
            conversation_history=history
        )
        
        print(f"\n💡 回答:\n{result.response}")
        
        # 履歴更新
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result.response})
```

### 8.3 ストリーミング出力

```python
async def streaming_example():
    facade = MONJYUFacade.from_environment()
    
    print("🤖 回答: ", end="")
    async for chunk in facade.search_stream(
        "Transformerの注意機構を簡潔に説明してください"
    ):
        print(chunk, end="", flush=True)
    print()
```

### 8.4 サーベイ生成

```python
async def generate_survey():
    facade = MONJYUFacade.from_environment()
    
    survey = await facade.survey(
        "大規模言語モデルの効率化手法",
        max_papers=15
    )
    
    print(survey)
```

### 8.5 論文比較

```python
async def compare_papers():
    facade = MONJYUFacade.from_environment()
    
    comparison = await facade.compare(
        paper_ids=[
            "arxiv:1706.03762",  # Transformer
            "arxiv:2010.11929",  # ViT
            "arxiv:2103.14030"   # Swin Transformer
        ],
        aspect="method"
    )
    
    print(comparison)
```

### 8.6 進捗表示付きインデックス構築

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

async def index_with_progress():
    facade = MONJYUFacade.from_environment()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Indexing...", total=None)
        
        def callback(p: IndexProgress):
            progress.update(
                task,
                description=f"{p.stage}: {p.message} ({p.percentage:.1f}%)"
            )
        
        result = await facade.index(
            paths=["./papers"],
            level=3,
            callback=callback
        )
    
    print(f"✅ Complete: {result.paper_count} papers indexed")
```

---

## 9. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-01-06 | 初版作成（LazyGraphRAGベース） |
| 3.0.0 | 2025-12-25 | v3.0要件対応、学術論文特化、MCP Server API追加、Progressive Index対応 |
| 3.1.0 | 2026-01-07 | ドキュメント更新：実装完了ステータス追加、参照要件をv3.1に更新 |
