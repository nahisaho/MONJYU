# MONJYU API仕様書

**バージョン**: 1.0.0  
**作成日**: 2025-01-06  
**ステータス**: Draft

---

## 1. 概要

本書は、MONJYUパッケージが提供するAPIの詳細仕様を定義します。

---

## 2. Python API

### 2.1 MONJYU Facade クラス

#### 2.1.1 クラス定義

```python
class MONJYU:
    """
    MONJYUのメインファサードクラス
    
    すべての機能への統一的なエントリーポイントを提供します。
    
    Example:
        >>> from MONJYU import MONJYU
        >>> 
        >>> # 初期化
        >>> monjyu = MONJYU.from_config("monjyu.yaml")
        >>> 
        >>> # インデックス構築
        >>> monjyu.index(input_dir="./documents")
        >>> 
        >>> # クエリ実行
        >>> result = await monjyu.query("What is the main topic?")
        >>> print(result.response)
    """
    
    def __init__(
        self,
        config: MONJYUConfig,
        chat_model: ChatModelProtocol,
        storage: StorageProtocol,
        index_builder: Optional[IndexBuilder] = None
    ):
        """
        MONJYUインスタンスを初期化
        
        Args:
            config: MONJYU設定
            chat_model: LLMクライアント
            storage: ストレージ実装
            index_builder: インデックスビルダー（オプション）
        """
        ...
```

#### 2.1.2 クラスメソッド

```python
@classmethod
def from_config(cls, config_path: str | Path) -> "MONJYU":
    """
    設定ファイルからMONJYUインスタンスを作成
    
    Args:
        config_path: YAML設定ファイルのパス
    
    Returns:
        MONJYU: 初期化済みインスタンス
    
    Raises:
        ConfigurationError: 設定ファイルの読み込みエラー
        FileNotFoundError: 設定ファイルが見つからない
    
    Example:
        >>> monjyu = MONJYU.from_config("monjyu.yaml")
    """
    ...


@classmethod
def from_env(cls) -> "MONJYU":
    """
    環境変数からMONJYUインスタンスを作成
    
    環境変数:
        MONJYU_LLM_PROVIDER: LLMプロバイダー
        MONJYU_LLM_MODEL: モデル名
        OPENAI_API_KEY: OpenAI APIキー
        AZURE_OPENAI_ENDPOINT: Azure OpenAIエンドポイント
        AZURE_OPENAI_API_KEY: Azure OpenAI APIキー
        MONJYU_OUTPUT_PATH: 出力ディレクトリ
    
    Returns:
        MONJYU: 初期化済みインスタンス
    
    Example:
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>> monjyu = MONJYU.from_env()
    """
    ...
```

---

### 2.2 インデックスAPI

#### 2.2.1 index メソッド

```python
def index(
    self,
    input_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    file_pattern: str = "*.txt",
    recursive: bool = True,
    callback: Optional[Callable[[IndexProgress], None]] = None
) -> IndexResult:
    """
    ドキュメントからインデックスを構築
    
    Args:
        input_dir: 入力ドキュメントディレクトリ
        output_dir: 出力ディレクトリ（省略時は設定値を使用）
        file_pattern: ファイルパターン（glob形式）
        recursive: サブディレクトリを再帰的に処理するか
        callback: 進捗コールバック関数
    
    Returns:
        IndexResult: インデックス構築結果
            - success: bool - 成功したか
            - chunk_count: int - チャンク数
            - document_count: int - ドキュメント数
            - elapsed_time: float - 処理時間（秒）
            - output_path: Path - 出力パス
    
    Raises:
        IndexError: インデックス構築エラー
        FileNotFoundError: 入力ディレクトリが見つからない
    
    Example:
        >>> result = monjyu.index(
        ...     input_dir="./documents",
        ...     file_pattern="*.md",
        ...     recursive=True
        ... )
        >>> print(f"Indexed {result.chunk_count} chunks")
    """
    ...
```

#### 2.2.2 IndexResult データクラス

```python
@dataclass
class IndexResult:
    """インデックス構築結果"""
    success: bool
    chunk_count: int
    document_count: int
    elapsed_time: float
    output_path: Path
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IndexProgress:
    """インデックス構築進捗"""
    stage: str              # "chunking" | "nlp" | "embedding" | "saving"
    current: int            # 現在の処理数
    total: int              # 合計数
    percentage: float       # 進捗率（0-100）
    message: str            # 進捗メッセージ
```

---

### 2.3 クエリAPI

#### 2.3.1 query メソッド

```python
async def query(
    self,
    query: str,
    conversation_history: Optional[List[dict]] = None,
    context_budget: Optional[int] = None,
    **kwargs
) -> QueryResult:
    """
    LazyGraphRAGクエリを実行
    
    Args:
        query: クエリ文字列
        conversation_history: 会話履歴（オプション）
        context_budget: コンテキストトークン予算（オプション）
        **kwargs: 追加のクエリパラメータ
    
    Returns:
        QueryResult: クエリ結果
            - response: str - 生成された回答
            - context: str - 使用されたコンテキスト
            - sources: List[Source] - 参照ソース
            - metadata: dict - メタデータ
    
    Raises:
        QueryError: クエリ実行エラー
        LLMError: LLM呼び出しエラー
    
    Example:
        >>> result = await monjyu.query(
        ...     "What are the main findings?",
        ...     context_budget=8000
        ... )
        >>> print(result.response)
    """
    ...


async def query_stream(
    self,
    query: str,
    conversation_history: Optional[List[dict]] = None,
    **kwargs
) -> AsyncIterator[str]:
    """
    ストリーミングクエリを実行
    
    Args:
        query: クエリ文字列
        conversation_history: 会話履歴（オプション）
        **kwargs: 追加のクエリパラメータ
    
    Yields:
        str: 回答の部分文字列
    
    Example:
        >>> async for chunk in monjyu.query_stream("Summarize the document"):
        ...     print(chunk, end="", flush=True)
    """
    ...
```

#### 2.3.2 QueryResult データクラス

```python
@dataclass
class QueryResult:
    """クエリ結果"""
    response: str
    context: str
    sources: List[Source]
    metadata: QueryMetadata


@dataclass
class Source:
    """参照ソース"""
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float


@dataclass
class QueryMetadata:
    """クエリメタデータ"""
    query_expansion: List[str]      # 拡張されたクエリ
    chunks_evaluated: int           # 評価されたチャンク数
    chunks_selected: int            # 選択されたチャンク数
    llm_calls: int                  # LLM呼び出し回数
    total_tokens: int               # 合計トークン数
    elapsed_time: float             # 処理時間（秒）
```

---

### 2.4 設定API

#### 2.4.1 MONJYUConfig データクラス

```python
@dataclass
class MONJYUConfig:
    """MONJYU設定"""
    
    # インデックス設定
    index: IndexConfig = field(default_factory=IndexConfig)
    
    # クエリ設定
    query: QueryConfig = field(default_factory=QueryConfig)
    
    # ストレージ設定
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # LLM設定
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "MONJYUConfig":
        """YAMLファイルから設定を読み込み"""
        ...
    
    @classmethod
    def from_dict(cls, data: dict) -> "MONJYUConfig":
        """辞書から設定を作成"""
        ...
    
    def to_yaml(self, path: str | Path) -> None:
        """YAMLファイルに設定を保存"""
        ...
    
    def validate(self) -> List[str]:
        """設定を検証し、エラーリストを返す"""
        ...


@dataclass
class IndexConfig:
    """インデックス設定"""
    chunk_size: int = 300
    chunk_overlap: int = 100
    encoding_model: str = "cl100k_base"
    enable_nlp: bool = True
    enable_embeddings: bool = False
    output_path: str = "./output"


@dataclass
class QueryConfig:
    """クエリ設定"""
    rate_query: int = 1
    rate_relevancy: int = 32
    rate_cluster: int = 2
    query_expansion_limit: int = 8
    max_context_tokens: int = 8000


@dataclass
class StorageConfig:
    """ストレージ設定"""
    type: str = "file"  # "file" | "lancedb" | "chroma"
    path: str = "./output"
    format: str = "parquet"


@dataclass
class LLMConfig:
    """LLM設定"""
    provider: str = "openai"  # "openai" | "azure_openai" | "ollama"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    # Azure specific
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    # Ollama specific
    host: str = "http://localhost:11434"
```

---

## 3. CLI API

### 3.1 コマンド一覧

| コマンド | 説明 |
|---------|------|
| `monjyu index` | インデックス構築 |
| `monjyu query` | クエリ実行 |
| `monjyu config init` | 設定ファイル初期化 |
| `monjyu config show` | 設定表示 |
| `monjyu version` | バージョン表示 |

---

### 3.2 index コマンド

```bash
monjyu index [OPTIONS]
```

#### オプション

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `--input` | `-i` | PATH | 必須 | 入力ディレクトリ |
| `--output` | `-o` | PATH | `./output` | 出力ディレクトリ |
| `--config` | `-c` | PATH | `monjyu.yaml` | 設定ファイル |
| `--pattern` | `-p` | TEXT | `*.txt` | ファイルパターン |
| `--recursive` | `-r` | FLAG | `true` | 再帰的処理 |
| `--verbose` | `-v` | FLAG | `false` | 詳細ログ |
| `--dry-run` | | FLAG | `false` | ドライラン |

#### 使用例

```bash
# 基本的な使用法
monjyu index --input ./documents --output ./index

# Markdownファイルのみを処理
monjyu index -i ./docs -o ./index -p "*.md"

# 設定ファイルを指定
monjyu index -i ./documents -c custom_config.yaml

# ドライラン（実際には実行しない）
monjyu index -i ./documents --dry-run
```

#### 出力

```
MONJYU Index Builder v1.0.0
===========================
Input:  ./documents
Output: ./index

Processing documents...
  [████████████████████████████████] 100% (150/150 files)

Chunking text...
  [████████████████████████████████] 100% (2,450 chunks)

Extracting NLP features...
  [████████████████████████████████] 100% (2,450 chunks)

Index built successfully!
  Documents: 150
  Chunks:    2,450
  Time:      12.3s
  Output:    ./index/
```

---

### 3.3 query コマンド

```bash
monjyu query [OPTIONS]
```

#### オプション

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `--index` | `-i` | PATH | 必須 | インデックスディレクトリ |
| `--query` | `-q` | TEXT | 必須* | クエリ文字列 |
| `--config` | `-c` | PATH | `monjyu.yaml` | 設定ファイル |
| `--stream` | `-s` | FLAG | `false` | ストリーミング出力 |
| `--json` | `-j` | FLAG | `false` | JSON出力 |
| `--interactive` | | FLAG | `false` | 対話モード |

*`--interactive`使用時は不要

#### 使用例

```bash
# 基本的なクエリ
monjyu query --index ./index --query "What is the main topic?"

# ストリーミング出力
monjyu query -i ./index -q "Summarize the document" --stream

# JSON形式で出力
monjyu query -i ./index -q "List the key points" --json

# 対話モード
monjyu query -i ./index --interactive
```

#### 出力

```
MONJYU Query v1.0.0
===================

Query: What is the main topic?

Response:
---------
The main topic of the document is the implementation of a graph-based
retrieval augmented generation (RAG) system. The document discusses...

Sources:
--------
1. doc1.txt (relevance: 0.95)
2. doc3.txt (relevance: 0.87)
3. doc2.txt (relevance: 0.82)

Metadata:
---------
  Chunks evaluated: 150
  Chunks selected:  12
  LLM calls:        8
  Time:             2.3s
```

#### JSON出力形式

```json
{
  "response": "The main topic of the document is...",
  "context": "...",
  "sources": [
    {
      "chunk_id": "abc123",
      "document_id": "doc1.txt",
      "content": "...",
      "relevance_score": 0.95
    }
  ],
  "metadata": {
    "query_expansion": ["topic", "main theme", "subject"],
    "chunks_evaluated": 150,
    "chunks_selected": 12,
    "llm_calls": 8,
    "total_tokens": 4500,
    "elapsed_time": 2.3
  }
}
```

---

### 3.4 config コマンド

#### config init

```bash
monjyu config init [OPTIONS]
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `--output` | `-o` | PATH | `monjyu.yaml` | 出力ファイル |
| `--provider` | `-p` | TEXT | `openai` | LLMプロバイダー |
| `--force` | `-f` | FLAG | `false` | 上書き確認なし |

```bash
# デフォルト設定ファイルを作成
monjyu config init

# Azure OpenAI用の設定を作成
monjyu config init --provider azure_openai

# カスタムパスに作成
monjyu config init -o ./config/my_config.yaml
```

#### config show

```bash
monjyu config show [OPTIONS]
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
|-----------|-------|-----|----------|------|
| `--config` | `-c` | PATH | `monjyu.yaml` | 設定ファイル |
| `--json` | `-j` | FLAG | `false` | JSON形式で出力 |

```bash
# 現在の設定を表示
monjyu config show

# JSON形式で出力
monjyu config show --json
```

---

### 3.5 version コマンド

```bash
monjyu version
```

出力:
```
MONJYU v1.0.0
Python 3.11.4
Platform: Linux-x86_64
```

---

## 4. エラーコード

### 4.1 エラーコード一覧

| コード | 名前 | 説明 |
|--------|------|------|
| E001 | CONFIG_NOT_FOUND | 設定ファイルが見つからない |
| E002 | CONFIG_INVALID | 設定ファイルが無効 |
| E003 | INPUT_NOT_FOUND | 入力ディレクトリが見つからない |
| E004 | INPUT_EMPTY | 入力ドキュメントがない |
| E005 | INDEX_NOT_FOUND | インデックスが見つからない |
| E006 | INDEX_INVALID | インデックスが無効 |
| E010 | LLM_AUTH_ERROR | LLM認証エラー |
| E011 | LLM_RATE_LIMIT | LLMレート制限 |
| E012 | LLM_TIMEOUT | LLMタイムアウト |
| E013 | LLM_INVALID_RESPONSE | LLM無効応答 |
| E020 | STORAGE_READ_ERROR | ストレージ読み込みエラー |
| E021 | STORAGE_WRITE_ERROR | ストレージ書き込みエラー |
| E030 | QUERY_EMPTY | クエリが空 |
| E031 | QUERY_TOO_LONG | クエリが長すぎる |

### 4.2 エラー応答形式

```python
@dataclass
class MONJYUError(Exception):
    """MONJYU例外基底クラス"""
    code: str
    message: str
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details
            }
        }
```

---

## 5. 環境変数

### 5.1 環境変数一覧

| 変数名 | 説明 | デフォルト |
|--------|------|----------|
| `MONJYU_CONFIG_PATH` | 設定ファイルパス | `monjyu.yaml` |
| `MONJYU_OUTPUT_PATH` | 出力ディレクトリ | `./output` |
| `MONJYU_LOG_LEVEL` | ログレベル | `INFO` |
| `MONJYU_LLM_PROVIDER` | LLMプロバイダー | `openai` |
| `MONJYU_LLM_MODEL` | LLMモデル | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI APIキー | - |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAIエンドポイント | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI APIキー | - |
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAIデプロイメント | - |

### 5.2 優先順位

1. コマンドライン引数
2. 設定ファイル（YAML）
3. 環境変数
4. デフォルト値

---

## 6. 使用例

### 6.1 基本的なワークフロー

```python
import asyncio
from MONJYU import MONJYU

async def main():
    # 1. 初期化
    monjyu = MONJYU.from_config("monjyu.yaml")
    
    # 2. インデックス構築
    index_result = monjyu.index(
        input_dir="./documents",
        output_dir="./index",
        file_pattern="*.md"
    )
    print(f"Indexed {index_result.chunk_count} chunks")
    
    # 3. クエリ実行
    query_result = await monjyu.query(
        "What are the main findings of the research?"
    )
    print(query_result.response)
    
    # 4. ソース確認
    for source in query_result.sources:
        print(f"- {source.document_id}: {source.relevance_score:.2f}")


asyncio.run(main())
```

### 6.2 会話履歴の使用

```python
async def conversation_example():
    monjyu = MONJYU.from_config("monjyu.yaml")
    
    history = []
    
    # 最初の質問
    result1 = await monjyu.query(
        "What is the document about?",
        conversation_history=history
    )
    print(f"A: {result1.response}")
    history.append({"role": "user", "content": "What is the document about?"})
    history.append({"role": "assistant", "content": result1.response})
    
    # フォローアップ質問
    result2 = await monjyu.query(
        "Can you elaborate on the second point?",
        conversation_history=history
    )
    print(f"A: {result2.response}")
```

### 6.3 ストリーミング出力

```python
async def streaming_example():
    monjyu = MONJYU.from_config("monjyu.yaml")
    
    print("Response: ", end="")
    async for chunk in monjyu.query_stream(
        "Summarize the key findings"
    ):
        print(chunk, end="", flush=True)
    print()
```

### 6.4 進捗コールバック

```python
def progress_callback(progress: IndexProgress):
    bar = "█" * int(progress.percentage / 5) + "░" * (20 - int(progress.percentage / 5))
    print(f"\r[{bar}] {progress.percentage:.1f}% {progress.message}", end="")


def index_with_progress():
    monjyu = MONJYU.from_config("monjyu.yaml")
    
    result = monjyu.index(
        input_dir="./documents",
        callback=progress_callback
    )
    print(f"\nCompleted: {result.chunk_count} chunks")
```

---

## 7. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-01-06 | 初版作成 |
