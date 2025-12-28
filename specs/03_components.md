# MONJYU コンポーネント仕様書

**バージョン**: 1.0.0  
**作成日**: 2025-01-06  
**ステータス**: Draft

---

## 1. 概要

本書は、MONJYUパッケージを構成する各コンポーネントの詳細仕様を定義します。

---

## 2. Indexコンポーネント群

### 2.1 TextChunker

テキストを適切なサイズのチャンクに分割するコンポーネント。

#### 2.1.1 クラス定義

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TextChunk:
    """テキストチャンクのデータモデル"""
    id: str                    # ユニークID (UUID)
    content: str               # チャンクテキスト
    source_id: str             # 元文書ID
    start_offset: int          # 元文書内の開始位置
    end_offset: int            # 元文書内の終了位置
    token_count: int           # トークン数
    metadata: dict             # メタデータ（オプション）


class TextChunkerProtocol(ABC):
    """テキストチャンク分割のプロトコル"""
    
    @abstractmethod
    def chunk(self, text: str, source_id: str) -> List[TextChunk]:
        """テキストをチャンクに分割"""
        ...
    
    @abstractmethod
    def chunk_documents(self, documents: List[dict]) -> List[TextChunk]:
        """複数文書をチャンクに分割"""
        ...
```

#### 2.1.2 実装クラス

```python
@dataclass
class ChunkerConfig:
    """チャンカー設定"""
    chunk_size: int = 300       # チャンクサイズ（トークン数）
    chunk_overlap: int = 100    # オーバーラップ（トークン数）
    encoding_model: str = "cl100k_base"  # トークナイザーモデル


class TiktokenChunker(TextChunkerProtocol):
    """tiktoken基づくチャンク分割実装"""
    
    def __init__(self, config: ChunkerConfig):
        self.config = config
        self._tokenizer = TiktokenTokenizer(config.encoding_model)
    
    def chunk(self, text: str, source_id: str) -> List[TextChunk]:
        """テキストをチャンクに分割"""
        # 実装詳細：
        # 1. テキストをトークン化
        # 2. chunk_size単位で分割
        # 3. chunk_overlap分のオーバーラップを維持
        # 4. TextChunkオブジェクトのリストを返却
        ...
```

#### 2.1.3 処理フロー

```
入力テキスト
    ↓
[トークン化] → トークンリスト
    ↓
[ウィンドウ分割] → (size=300, overlap=100)
    ↓
[TextChunk生成] → IDとメタデータ付与
    ↓
List[TextChunk]
```

---

### 2.2 NLPExtractor

NLPベースの軽量インデックス抽出コンポーネント。

#### 2.2.1 クラス定義

```python
@dataclass
class NLPFeatures:
    """NLP抽出結果"""
    chunk_id: str
    keywords: List[str]        # キーワードリスト
    entities: List[dict]       # エンティティ（名詞句等）
    summary: Optional[str]     # 要約（オプション）


class NLPExtractorProtocol(ABC):
    """NLP特徴抽出のプロトコル"""
    
    @abstractmethod
    def extract(self, chunk: TextChunk) -> NLPFeatures:
        """単一チャンクからNLP特徴を抽出"""
        ...
    
    @abstractmethod
    def extract_batch(self, chunks: List[TextChunk]) -> List[NLPFeatures]:
        """複数チャンクからNLP特徴を一括抽出"""
        ...
```

#### 2.2.2 実装クラス

```python
class BasicNLPExtractor(NLPExtractorProtocol):
    """基本的なNLP抽出実装（外部依存なし）"""
    
    def __init__(self, language: str = "en"):
        self.language = language
    
    def extract(self, chunk: TextChunk) -> NLPFeatures:
        """
        正規表現とヒューリスティクスによる軽量抽出
        - キーワード：TF-IDF風のスコアリング
        - エンティティ：大文字始まりの単語列
        """
        ...


class SpacyNLPExtractor(NLPExtractorProtocol):
    """spaCyを使用したNLP抽出（オプション拡張）"""
    
    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model)
    
    def extract(self, chunk: TextChunk) -> NLPFeatures:
        """spaCyによる高精度抽出"""
        ...
```

---

### 2.3 Embedder

ベクトルエンベディング生成コンポーネント（オプション）。

#### 2.3.1 クラス定義

```python
@dataclass
class EmbeddingResult:
    """エンベディング結果"""
    chunk_id: str
    vector: List[float]        # エンベディングベクトル
    model: str                 # 使用モデル名
    dimensions: int            # 次元数


class EmbedderProtocol(ABC):
    """エンベディングのプロトコル"""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """単一テキストをエンベディング"""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """複数テキストを一括エンベディング"""
        ...
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """エンベディング次元数"""
        ...
```

#### 2.3.2 実装クラス

```python
class OpenAIEmbedder(EmbedderProtocol):
    """OpenAI Embeddings API実装"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536
    ):
        self.client = OpenAIClient(api_key)
        self.model = model
        self._dimensions = dimensions


class SentenceTransformerEmbedder(EmbedderProtocol):
    """SentenceTransformers実装（ローカル）"""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
```

---

### 2.4 IndexBuilder

インデックス構築のオーケストレーター。

#### 2.4.1 クラス定義

```python
@dataclass
class IndexConfig:
    """インデックス構築設定"""
    chunker_config: ChunkerConfig
    enable_nlp: bool = True
    enable_embeddings: bool = False
    nlp_extractor: str = "basic"    # "basic" | "spacy"
    embedder: Optional[str] = None  # "openai" | "sentence_transformer"
    output_path: str = "./output"


class IndexBuilder:
    """インデックス構築オーケストレーター"""
    
    def __init__(
        self,
        config: IndexConfig,
        chunker: TextChunkerProtocol,
        nlp_extractor: Optional[NLPExtractorProtocol] = None,
        embedder: Optional[EmbedderProtocol] = None,
        storage: StorageProtocol
    ):
        self.config = config
        self.chunker = chunker
        self.nlp_extractor = nlp_extractor
        self.embedder = embedder
        self.storage = storage
    
    def build(self, documents: List[dict]) -> IndexResult:
        """
        インデックス構築メインフロー
        
        Args:
            documents: [{"id": str, "text": str, "metadata": dict}, ...]
        
        Returns:
            IndexResult: 構築結果
        """
        ...
    
    def build_incremental(
        self,
        documents: List[dict],
        existing_index_path: str
    ) -> IndexResult:
        """増分インデックス構築"""
        ...
```

#### 2.4.2 処理フロー

```
Documents
    ↓
[TextChunker.chunk_documents]
    ↓
List[TextChunk] ──────────────────┐
    ↓                             │
[NLPExtractor.extract_batch]      │ (並列処理可能)
    ↓                             │
List[NLPFeatures]                 │
    ↓                             │
[Embedder.embed_batch] (optional) │
    ↓                             ↓
[Storage.save] ←── 統合 ←─────────┘
    ↓
IndexResult
```

---

## 3. Storageコンポーネント群

### 3.1 StorageProtocol

#### 3.1.1 インターフェース定義

```python
class StorageProtocol(ABC):
    """ストレージの抽象インターフェース"""
    
    @abstractmethod
    def save_chunks(self, chunks: List[TextChunk]) -> None:
        """チャンクを保存"""
        ...
    
    @abstractmethod
    def load_chunks(self) -> List[TextChunk]:
        """チャンクを読み込み"""
        ...
    
    @abstractmethod
    def save_nlp_features(self, features: List[NLPFeatures]) -> None:
        """NLP特徴を保存"""
        ...
    
    @abstractmethod
    def load_nlp_features(self) -> List[NLPFeatures]:
        """NLP特徴を読み込み"""
        ...
    
    @abstractmethod
    def save_embeddings(self, embeddings: List[EmbeddingResult]) -> None:
        """エンベディングを保存"""
        ...
    
    @abstractmethod
    def save_metadata(self, metadata: dict) -> None:
        """メタデータを保存"""
        ...
    
    @abstractmethod
    def load_metadata(self) -> dict:
        """メタデータを読み込み"""
        ...
```

---

### 3.2 FileStorage

ファイルベースのストレージ実装。

#### 3.2.1 実装詳細

```python
@dataclass
class FileStorageConfig:
    """ファイルストレージ設定"""
    base_path: str = "./output"
    chunks_file: str = "chunks.parquet"
    nlp_file: str = "nlp_features.parquet"
    embeddings_file: str = "embeddings.parquet"
    metadata_file: str = "metadata.json"
    format: str = "parquet"     # "parquet" | "json"


class FileStorage(StorageProtocol):
    """ファイルベースストレージ実装"""
    
    def __init__(self, config: FileStorageConfig):
        self.config = config
        self._ensure_directory()
    
    def save_chunks(self, chunks: List[TextChunk]) -> None:
        """Parquet形式でチャンクを保存"""
        df = pd.DataFrame([asdict(c) for c in chunks])
        df.to_parquet(self._chunks_path)
    
    def load_chunks(self) -> List[TextChunk]:
        """Parquet形式からチャンクを読み込み"""
        df = pd.read_parquet(self._chunks_path)
        return [TextChunk(**row) for row in df.to_dict('records')]
```

#### 3.2.2 ディレクトリ構造

```
output/
├── metadata.json           # インデックスメタデータ
├── chunks.parquet         # テキストチャンク
├── nlp_features.parquet   # NLP特徴
├── embeddings.parquet     # エンベディング（オプション）
└── cache/                 # キャッシュディレクトリ
    └── query_cache.json
```

---

### 3.3 VectorStoreAdapter

ベクトルストア連携アダプター。

#### 3.3.1 インターフェース

```python
class VectorStoreProtocol(ABC):
    """ベクトルストアの抽象インターフェース"""
    
    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[dict]] = None
    ) -> None:
        """ベクトルを挿入/更新"""
        ...
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[dict] = None
    ) -> List[dict]:
        """類似ベクトル検索"""
        ...
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """ベクトルを削除"""
        ...
```

#### 3.3.2 実装クラス（将来拡張）

```python
class LanceDBAdapter(VectorStoreProtocol):
    """LanceDB アダプター"""
    ...

class ChromaDBAdapter(VectorStoreProtocol):
    """ChromaDB アダプター"""
    ...

class QdrantAdapter(VectorStoreProtocol):
    """Qdrant アダプター"""
    ...
```

---

## 4. LLMクライアントコンポーネント群

### 4.1 ChatModelProtocol

既存の `lazy_search/core/chat_model.py` を基盤とする。

#### 4.1.1 拡張定義

```python
@dataclass
class ChatModelConfig:
    """チャットモデル共通設定"""
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 3
    streaming: bool = False


class ChatModelProtocol(Protocol):
    """チャットモデルプロトコル（既存）"""
    
    async def chat(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs
    ) -> str:
        """チャット応答を生成"""
        ...
```

---

### 4.2 OpenAIClient

OpenAI API クライアント実装。

```python
class OpenAIClient(ChatModelProtocol):
    """OpenAI API クライアント"""
    
    def __init__(
        self,
        api_key: str,
        config: ChatModelConfig,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key
        self.config = config
        self.base_url = base_url or "https://api.openai.com/v1"
        self._client = self._create_client()
    
    async def chat(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs
    ) -> str:
        """OpenAI Chat Completion API呼び出し"""
        messages = self._build_messages(prompt, history)
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    async def chat_stream(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        """ストリーミング応答"""
        ...
```

---

### 4.3 AzureOpenAIClient

Azure OpenAI Service クライアント実装。

```python
class AzureOpenAIClient(ChatModelProtocol):
    """Azure OpenAI Service クライアント"""
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = "2024-02-15-preview",
        config: ChatModelConfig
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version
        self.config = config
    
    async def chat(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs
    ) -> str:
        """Azure OpenAI Chat Completion API呼び出し"""
        ...
```

---

### 4.4 OllamaClient

Ollama（ローカルLLM）クライアント実装。

```python
class OllamaClient(ChatModelProtocol):
    """Ollama ローカルLLMクライアント"""
    
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        config: ChatModelConfig
    ):
        self.model = model
        self.host = host
        self.config = config
    
    async def chat(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs
    ) -> str:
        """Ollama API呼び出し"""
        ...
```

---

### 4.5 LLMFactory

LLMクライアントのファクトリークラス。

```python
class LLMFactory:
    """LLMクライアントファクトリー"""
    
    _providers = {
        "openai": OpenAIClient,
        "azure_openai": AzureOpenAIClient,
        "ollama": OllamaClient,
    }
    
    @classmethod
    def create(
        cls,
        provider: str,
        config: dict
    ) -> ChatModelProtocol:
        """
        プロバイダーに応じたLLMクライアントを生成
        
        Args:
            provider: "openai" | "azure_openai" | "ollama"
            config: プロバイダー固有の設定
        
        Returns:
            ChatModelProtocol実装インスタンス
        """
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        return cls._providers[provider](**config)
    
    @classmethod
    def register(
        cls,
        name: str,
        client_class: Type[ChatModelProtocol]
    ) -> None:
        """カスタムプロバイダーを登録"""
        cls._providers[name] = client_class
```

---

## 5. CLIコンポーネント

### 5.1 コマンド構造

```
monjyu
├── index          # インデックス構築
│   ├── --input    # 入力ディレクトリ
│   ├── --output   # 出力ディレクトリ
│   ├── --config   # 設定ファイル
│   └── --verbose  # 詳細ログ
│
├── query          # クエリ実行
│   ├── --index    # インデックスパス
│   ├── --query    # クエリ文字列
│   ├── --config   # 設定ファイル
│   └── --stream   # ストリーミング出力
│
├── config         # 設定管理
│   ├── init       # 設定ファイル初期化
│   └── show       # 現在の設定表示
│
└── version        # バージョン表示
```

### 5.2 実装

```python
import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """CLIパーサーを作成"""
    parser = argparse.ArgumentParser(
        prog="monjyu",
        description="MONJYU - Standalone LazyGraphRAG Implementation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # index コマンド
    index_parser = subparsers.add_parser("index", help="Build index")
    index_parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing documents"
    )
    index_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output"),
        help="Output directory for index"
    )
    index_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file path"
    )
    
    # query コマンド
    query_parser = subparsers.add_parser("query", help="Execute query")
    query_parser.add_argument(
        "--index", "-i",
        type=Path,
        required=True,
        help="Index directory path"
    )
    query_parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Query string"
    )
    query_parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable streaming output"
    )
    
    return parser


def main():
    """CLIエントリーポイント"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "index":
        run_index(args)
    elif args.command == "query":
        run_query(args)
    # ...
```

---

## 6. コンポーネント依存関係

### 6.1 依存関係図

```
┌─────────────────────────────────────────────────────┐
│                      CLI                             │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   MONJYU Facade                      │
│  (index(), query(), configure())                     │
└─────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────┐
│    IndexBuilder      │    │      LazySearch          │
│  ┌────────────────┐  │    │  ┌────────────────────┐  │
│  │  TextChunker   │  │    │  │  QueryExpander     │  │
│  │  NLPExtractor  │  │    │  │  RelevanceTester   │  │
│  │  Embedder      │  │    │  │  ClaimExtractor    │  │
│  └────────────────┘  │    │  │  IterativeDeepener │  │
└──────────────────────┘    │  │  LazyContextBuilder│  │
          │                 │  └────────────────────┘  │
          ▼                 └──────────────────────────┘
┌──────────────────────┐               │
│      Storage         │               │
│  ┌────────────────┐  │               │
│  │  FileStorage   │  │◄──────────────┘
│  │  VectorStore   │  │
│  └────────────────┘  │
└──────────────────────┘
          ▲
          │
┌──────────────────────┐
│    LLM Clients       │
│  ┌────────────────┐  │
│  │  OpenAI        │  │
│  │  AzureOpenAI   │  │
│  │  Ollama        │  │
│  └────────────────┘  │
└──────────────────────┘
```

### 6.2 レイヤー依存ルール

| レイヤー | 依存可能なレイヤー |
|---------|-------------------|
| CLI | Facade |
| Facade | Index, Query, Storage, LLM |
| Index | Storage, Core |
| Query | Storage, LLM, Core |
| Storage | Core |
| LLM | Core |
| Core | なし（外部依存のみ） |

---

## 7. エラーハンドリング

### 7.1 例外階層

```python
class MONJYUError(Exception):
    """MONJYU基底例外"""
    pass


class ConfigurationError(MONJYUError):
    """設定エラー"""
    pass


class IndexError(MONJYUError):
    """インデックスエラー"""
    pass


class StorageError(MONJYUError):
    """ストレージエラー"""
    pass


class LLMError(MONJYUError):
    """LLMエラー"""
    pass


class QueryError(MONJYUError):
    """クエリエラー"""
    pass
```

---

## 8. 設定スキーマ

### 8.1 YAML設定ファイル

```yaml
# monjyu.yaml
version: "1.0"

index:
  chunker:
    chunk_size: 300
    chunk_overlap: 100
    encoding_model: "cl100k_base"
  
  nlp:
    enabled: true
    extractor: "basic"  # basic | spacy
  
  embeddings:
    enabled: false
    provider: "openai"
    model: "text-embedding-3-small"

storage:
  type: "file"  # file | lancedb | chroma
  path: "./output"
  format: "parquet"  # parquet | json

llm:
  provider: "openai"  # openai | azure_openai | ollama
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 4096
  
  # OpenAI specific
  api_key: "${OPENAI_API_KEY}"
  
  # Azure specific (when provider: azure_openai)
  # endpoint: "${AZURE_OPENAI_ENDPOINT}"
  # deployment: "${AZURE_OPENAI_DEPLOYMENT}"
  # api_version: "2024-02-15-preview"
  
  # Ollama specific (when provider: ollama)
  # host: "http://localhost:11434"

query:
  rate_query: 1
  rate_relevancy: 32
  rate_cluster: 2
  query_expansion_limit: 8
  max_context_tokens: 8000
```

---

## 9. テスト戦略

### 9.1 コンポーネント別テスト

| コンポーネント | テスト種別 | カバレッジ目標 |
|--------------|----------|--------------|
| TextChunker | Unit | 90% |
| NLPExtractor | Unit | 85% |
| Embedder | Unit + Mock | 80% |
| IndexBuilder | Integration | 80% |
| FileStorage | Unit + E2E | 90% |
| LLM Clients | Unit + Mock | 85% |
| CLI | E2E | 75% |

### 9.2 モック戦略

```python
# テスト用モックLLMクライアント
class MockChatModel(ChatModelProtocol):
    """テスト用モックLLMクライアント"""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
    
    async def chat(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs
    ) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
```

---

## 10. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-01-06 | 初版作成 |
