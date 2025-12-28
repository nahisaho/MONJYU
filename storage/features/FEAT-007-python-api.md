# FEAT-007: Python API (MONJYU Facade)

**フィーチャーID**: FEAT-007  
**名称**: Python API (MONJYU Facade)  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

MONJYUのすべての機能を統合する高レベルPython API。インデックス構築から検索まで、シンプルなインターフェースで利用可能にする。

### 1.1 スコープ

```python
from monjyu import MONJYU

monjyu = MONJYU("./config.yaml")
monjyu.index("./papers/")
result = monjyu.search("What is Transformer?")
```

- **入力**: 設定ファイル、ドキュメント、クエリ
- **処理**: 全コンポーネントの統合・オーケストレーション
- **出力**: 検索結果、インデックス状態、分析結果
- **特徴**: Facade パターンによるシンプルなAPI

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-EXT-PY-001 | MONJYU クラス | P0 |
| FR-EXT-PY-002 | インデックスメソッド | P0 |
| FR-EXT-PY-003 | 検索メソッド | P0 |
| FR-EXT-PY-004 | 設定読み込み | P0 |
| FR-EXT-PY-005 | 状態管理 | P0 |
| FR-EXT-PY-006 | 非同期API | P1 |

### 1.3 依存関係

- **依存**: FEAT-001~006 (全コア機能)
- **被依存**: FEAT-008 (CLI), FEAT-009 (MCP Server)

---

## 2. アーキテクチャ

### 2.1 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MONJYU Facade                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      Public API                                 │ │
│  │  - index(path)                                                  │ │
│  │  - search(query, mode)                                          │ │
│  │  - get_document(doc_id)                                         │ │
│  │  - get_citation_network()                                       │ │
│  │  - get_status()                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                    │                                 │
│                                    ▼                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   Internal Components                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │ │
│  │  │ ConfigManager│  │ StateManager │  │ ComponentFactory     │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                    │                                 │
│                                    ▼                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   Core Components                               │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │ │
│  │  │ Document   │ │ Index      │ │ Search     │ │ Citation   │  │ │
│  │  │ Processor  │ │ Builders   │ │ Engines    │ │ Network    │  │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 クラス図

```python
from dataclasses import dataclass, field
from typing import Protocol, overload
from pathlib import Path
from enum import Enum

# === Enums ===

class SearchMode(Enum):
    """検索モード"""
    VECTOR = "vector"      # Level 0: ベクトル検索のみ
    LAZY = "lazy"          # Level 0-1: LazyGraphRAG
    # GRAPH = "graph"      # Level 2-4: Future
    AUTO = "auto"          # 自動選択

class IndexLevel(Enum):
    """インデックスレベル"""
    LEVEL_0 = 0  # Baseline (Vector)
    LEVEL_1 = 1  # Lazy (NLP Graph)
    # LEVEL_2 = 2  # Future
    # LEVEL_3 = 3  # Future
    # LEVEL_4 = 4  # Future

class IndexStatus(Enum):
    """インデックス状態"""
    NOT_BUILT = "not_built"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"

# === Data Classes ===

@dataclass
class MONJYUConfig:
    """MONJYU設定"""
    # 基本設定
    output_path: Path = Path("./output")
    
    # 環境設定
    environment: str = "local"  # "local" | "azure"
    
    # インデックス設定
    index_levels: list[IndexLevel] = field(default_factory=lambda: [IndexLevel.LEVEL_0, IndexLevel.LEVEL_1])
    
    # 検索設定
    default_search_mode: SearchMode = SearchMode.LAZY
    default_top_k: int = 10
    
    # ドキュメント処理設定
    chunk_size: int = 1200
    chunk_overlap: int = 100
    
    # LLM設定
    llm_model: str = "llama3:8b-instruct-q4_K_M"
    embedding_model: str = "nomic-embed-text"
    
    # ローカル設定
    ollama_base_url: str = "http://192.168.224.1:11434"
    
    # Azure設定（オプション）
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None

@dataclass
class MONJYUStatus:
    """MONJYUステータス"""
    # インデックス状態
    index_status: IndexStatus = IndexStatus.NOT_BUILT
    index_levels_built: list[IndexLevel] = field(default_factory=list)
    
    # 統計
    document_count: int = 0
    text_unit_count: int = 0
    
    # Level 1 (NLP Graph)
    noun_phrase_count: int = 0
    community_count: int = 0
    
    # Citation Network
    citation_edge_count: int = 0
    
    # エラー
    last_error: str | None = None

@dataclass
class SearchResult:
    """検索結果"""
    query: str
    answer: str
    citations: list["Citation"]
    
    # メタデータ
    search_mode: SearchMode
    search_level: int
    
    # パフォーマンス
    total_time_ms: float
    llm_calls: int
    
    # デバッグ情報（オプション）
    raw_results: list[dict] | None = None

@dataclass
class DocumentInfo:
    """ドキュメント情報"""
    id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    
    # 統計
    chunk_count: int
    
    # 引用メトリクス
    citation_count: int
    reference_count: int
    influence_score: float
```

---

## 3. 詳細設計

### 3.1 ConfigManager

```python
import yaml
from pathlib import Path

class ConfigManager:
    """設定マネージャー"""
    
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: MONJYUConfig | None = None
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConfigManager":
        """YAMLファイルから読み込み"""
        manager = cls(path)
        manager.load()
        return manager
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ConfigManager":
        """辞書から作成"""
        manager = cls()
        manager._config = cls._parse_config(config_dict)
        return manager
    
    def load(self) -> MONJYUConfig:
        """設定を読み込み"""
        if not self.config_path or not self.config_path.exists():
            self._config = MONJYUConfig()
            return self._config
        
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        
        self._config = self._parse_config(data)
        return self._config
    
    @staticmethod
    def _parse_config(data: dict) -> MONJYUConfig:
        """設定をパース"""
        return MONJYUConfig(
            output_path=Path(data.get("output_path", "./output")),
            environment=data.get("environment", "local"),
            index_levels=[
                IndexLevel(l) for l in data.get("index_levels", [0, 1])
            ],
            default_search_mode=SearchMode(data.get("default_search_mode", "lazy")),
            default_top_k=data.get("default_top_k", 10),
            chunk_size=data.get("chunk_size", 1200),
            chunk_overlap=data.get("chunk_overlap", 100),
            llm_model=data.get("llm_model", "llama3:8b-instruct-q4_K_M"),
            embedding_model=data.get("embedding_model", "nomic-embed-text"),
            ollama_base_url=data.get("ollama_base_url", "http://192.168.224.1:11434"),
            azure_openai_endpoint=data.get("azure_openai_endpoint"),
            azure_openai_api_key=data.get("azure_openai_api_key"),
            azure_search_endpoint=data.get("azure_search_endpoint"),
            azure_search_api_key=data.get("azure_search_api_key"),
        )
    
    @property
    def config(self) -> MONJYUConfig:
        """設定を取得"""
        if self._config is None:
            self._config = self.load()
        return self._config
```

### 3.2 ComponentFactory

```python
class ComponentFactory:
    """コンポーネントファクトリー"""
    
    def __init__(self, config: MONJYUConfig):
        self.config = config
        self._cached_components = {}
    
    def get_embedding_client(self) -> "EmbeddingClientProtocol":
        """埋め込みクライアントを取得"""
        if "embedding" not in self._cached_components:
            if self.config.environment == "azure":
                from monjyu.core.embedding import AzureOpenAIEmbeddingClient
                self._cached_components["embedding"] = AzureOpenAIEmbeddingClient(
                    endpoint=self.config.azure_openai_endpoint,
                    api_key=self.config.azure_openai_api_key
                )
            else:
                from monjyu.core.embedding import OllamaEmbeddingClient
                self._cached_components["embedding"] = OllamaEmbeddingClient(
                    base_url=self.config.ollama_base_url,
                    model=self.config.embedding_model
                )
        return self._cached_components["embedding"]
    
    def get_llm_client(self) -> "LLMClientProtocol":
        """LLMクライアントを取得"""
        if "llm" not in self._cached_components:
            if self.config.environment == "azure":
                from monjyu.core.llm import AzureOpenAIClient
                self._cached_components["llm"] = AzureOpenAIClient(
                    endpoint=self.config.azure_openai_endpoint,
                    api_key=self.config.azure_openai_api_key
                )
            else:
                from monjyu.core.llm import OllamaClient
                self._cached_components["llm"] = OllamaClient(
                    base_url=self.config.ollama_base_url,
                    model=self.config.llm_model
                )
        return self._cached_components["llm"]
    
    def get_document_processor(self) -> "DocumentProcessor":
        """ドキュメントプロセッサーを取得"""
        if "doc_processor" not in self._cached_components:
            from monjyu.document import DocumentProcessor
            self._cached_components["doc_processor"] = DocumentProcessor(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        return self._cached_components["doc_processor"]
    
    def get_level0_builder(self) -> "Level0IndexBuilder":
        """Level 0 ビルダーを取得"""
        if "level0" not in self._cached_components:
            from monjyu.index import Level0IndexBuilder
            self._cached_components["level0"] = Level0IndexBuilder(
                embedding_client=self.get_embedding_client(),
                output_path=self.config.output_path / "index" / "level_0"
            )
        return self._cached_components["level0"]
    
    def get_level1_builder(self) -> "Level1IndexBuilder":
        """Level 1 ビルダーを取得"""
        if "level1" not in self._cached_components:
            from monjyu.index import Level1IndexBuilder
            self._cached_components["level1"] = Level1IndexBuilder(
                output_path=self.config.output_path / "index" / "level_1"
            )
        return self._cached_components["level1"]
    
    def get_vector_search_engine(self) -> "VectorSearchEngine":
        """ベクトル検索エンジンを取得"""
        if "vector_search" not in self._cached_components:
            from monjyu.search import VectorSearchEngine, LanceDBVectorSearcher
            
            searcher = LanceDBVectorSearcher(
                db_path=str(self.config.output_path / "index" / "level_0" / "vector_index")
            )
            
            self._cached_components["vector_search"] = VectorSearchEngine(
                embedding_client=self.get_embedding_client(),
                vector_searcher=searcher,
                llm_client=self.get_llm_client()
            )
        return self._cached_components["vector_search"]
    
    def get_lazy_search_engine(self) -> "LazySearchEngine":
        """Lazy検索エンジンを取得"""
        if "lazy_search" not in self._cached_components:
            from monjyu.search import LazySearchEngine
            
            # Level 1 インデックスを読み込み
            level1_builder = self.get_level1_builder()
            level1_index = level1_builder.load()
            
            self._cached_components["lazy_search"] = LazySearchEngine(
                vector_search_engine=self.get_vector_search_engine(),
                level1_index=level1_index,
                embedding_client=self.get_embedding_client(),
                llm_client=self.get_llm_client()
            )
        return self._cached_components["lazy_search"]
    
    def get_citation_network_manager(self) -> "CitationNetworkManager":
        """引用ネットワークマネージャーを取得"""
        if "citation" not in self._cached_components:
            from monjyu.citation import CitationNetworkManager
            self._cached_components["citation"] = CitationNetworkManager(
                output_path=self.config.output_path / "citation"
            )
        return self._cached_components["citation"]
```

### 3.3 StateManager

```python
import json
from pathlib import Path

class StateManager:
    """状態マネージャー"""
    
    STATE_FILE = "monjyu_state.json"
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.state_file = output_path / self.STATE_FILE
        self._status: MONJYUStatus | None = None
    
    def load(self) -> MONJYUStatus:
        """状態を読み込み"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
            self._status = MONJYUStatus(
                index_status=IndexStatus(data.get("index_status", "not_built")),
                index_levels_built=[
                    IndexLevel(l) for l in data.get("index_levels_built", [])
                ],
                document_count=data.get("document_count", 0),
                text_unit_count=data.get("text_unit_count", 0),
                noun_phrase_count=data.get("noun_phrase_count", 0),
                community_count=data.get("community_count", 0),
                citation_edge_count=data.get("citation_edge_count", 0),
                last_error=data.get("last_error")
            )
        else:
            self._status = MONJYUStatus()
        
        return self._status
    
    def save(self):
        """状態を保存"""
        if self._status is None:
            return
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "index_status": self._status.index_status.value,
            "index_levels_built": [l.value for l in self._status.index_levels_built],
            "document_count": self._status.document_count,
            "text_unit_count": self._status.text_unit_count,
            "noun_phrase_count": self._status.noun_phrase_count,
            "community_count": self._status.community_count,
            "citation_edge_count": self._status.citation_edge_count,
            "last_error": self._status.last_error
        }
        
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def update(self, **kwargs):
        """状態を更新"""
        if self._status is None:
            self.load()
        
        for key, value in kwargs.items():
            if hasattr(self._status, key):
                setattr(self._status, key, value)
        
        self.save()
    
    @property
    def status(self) -> MONJYUStatus:
        """状態を取得"""
        if self._status is None:
            self.load()
        return self._status
```

### 3.4 MONJYU (Main Facade)

```python
from pathlib import Path
from typing import overload, Literal
import time

class MONJYU:
    """MONJYU メインAPI"""
    
    def __init__(
        self,
        config: str | Path | dict | MONJYUConfig | None = None
    ):
        """
        MONJYU を初期化
        
        Args:
            config: 設定ファイルパス、辞書、またはMONJYUConfigオブジェクト
        """
        # 設定読み込み
        if config is None:
            self._config_manager = ConfigManager()
        elif isinstance(config, (str, Path)):
            self._config_manager = ConfigManager.from_yaml(config)
        elif isinstance(config, dict):
            self._config_manager = ConfigManager.from_dict(config)
        elif isinstance(config, MONJYUConfig):
            self._config_manager = ConfigManager()
            self._config_manager._config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        # コンポーネント初期化
        self._factory = ComponentFactory(self.config)
        self._state_manager = StateManager(self.config.output_path)
        
        # ドキュメントキャッシュ
        self._documents: list["AcademicPaperDocument"] = []
    
    @property
    def config(self) -> MONJYUConfig:
        """設定を取得"""
        return self._config_manager.config
    
    def get_status(self) -> MONJYUStatus:
        """ステータスを取得"""
        return self._state_manager.status
    
    # === インデックス構築 ===
    
    def index(
        self,
        path: str | Path,
        levels: list[IndexLevel] | None = None,
        rebuild: bool = False,
        show_progress: bool = True
    ) -> MONJYUStatus:
        """
        ドキュメントをインデックス化
        
        Args:
            path: ドキュメントのパス（ファイルまたはディレクトリ）
            levels: 構築するインデックスレベル（デフォルト: config設定）
            rebuild: 既存インデックスを再構築するか
            show_progress: 進捗を表示するか
        
        Returns:
            MONJYUStatus: 更新後のステータス
        """
        try:
            self._state_manager.update(index_status=IndexStatus.BUILDING)
            
            levels = levels or self.config.index_levels
            path = Path(path)
            
            # 1. ドキュメント処理
            processor = self._factory.get_document_processor()
            self._documents = processor.process_directory(path)
            
            # 2. Level 0 インデックス構築
            if IndexLevel.LEVEL_0 in levels:
                level0 = self._factory.get_level0_builder()
                
                # TextUnitsを取得
                text_units = []
                for doc in self._documents:
                    text_units.extend(doc.text_units)
                
                result = level0.build(text_units, show_progress=show_progress)
                
                self._state_manager.update(
                    document_count=len(self._documents),
                    text_unit_count=len(text_units)
                )
            
            # 3. Level 1 インデックス構築
            if IndexLevel.LEVEL_1 in levels:
                level1 = self._factory.get_level1_builder()
                
                # TextUnitsを読み込み
                import pyarrow.parquet as pq
                tu_table = pq.read_table(
                    self.config.output_path / "index" / "level_0" / "text_units.parquet"
                )
                
                result = level1.build(tu_table.to_pylist(), show_progress=show_progress)
                
                self._state_manager.update(
                    noun_phrase_count=result.node_count,
                    community_count=result.community_count
                )
            
            # 4. 引用ネットワーク構築
            citation_manager = self._factory.get_citation_network_manager()
            citation_result = citation_manager.build(self._documents)
            
            self._state_manager.update(
                citation_edge_count=citation_result.edge_count
            )
            
            # 完了
            self._state_manager.update(
                index_status=IndexStatus.READY,
                index_levels_built=levels
            )
            
            return self.get_status()
        
        except Exception as e:
            self._state_manager.update(
                index_status=IndexStatus.ERROR,
                last_error=str(e)
            )
            raise
    
    # === 検索 ===
    
    @overload
    def search(
        self,
        query: str,
        mode: Literal[SearchMode.VECTOR] = ...,
        top_k: int = ...,
        synthesize: bool = ...
    ) -> SearchResult: ...
    
    @overload
    def search(
        self,
        query: str,
        mode: Literal[SearchMode.LAZY],
        max_level: int = ...,
        max_llm_calls: int = ...
    ) -> SearchResult: ...
    
    def search(
        self,
        query: str,
        mode: SearchMode | None = None,
        **kwargs
    ) -> SearchResult:
        """
        検索を実行
        
        Args:
            query: 検索クエリ
            mode: 検索モード（デフォルト: config設定）
            **kwargs: モード固有のパラメータ
        
        Returns:
            SearchResult: 検索結果
        """
        mode = mode or self.config.default_search_mode
        top_k = kwargs.get("top_k", self.config.default_top_k)
        
        start_time = time.time()
        
        if mode == SearchMode.VECTOR:
            engine = self._factory.get_vector_search_engine()
            response = engine.search(
                query,
                top_k=top_k,
                synthesize=kwargs.get("synthesize", True)
            )
            
            result = SearchResult(
                query=query,
                answer=response.answer.answer,
                citations=response.answer.citations,
                search_mode=SearchMode.VECTOR,
                search_level=0,
                total_time_ms=(time.time() - start_time) * 1000,
                llm_calls=1 if kwargs.get("synthesize", True) else 0
            )
        
        elif mode == SearchMode.LAZY:
            engine = self._factory.get_lazy_search_engine()
            response = engine.search(
                query,
                max_level=kwargs.get("max_level", 1)
            )
            
            result = SearchResult(
                query=query,
                answer=response.answer,
                citations=response.citations,
                search_mode=SearchMode.LAZY,
                search_level=response.search_level_reached.value,
                total_time_ms=response.total_time_ms,
                llm_calls=response.llm_calls
            )
        
        elif mode == SearchMode.AUTO:
            # 自動選択（クエリの複雑さで判断）
            if self._is_complex_query(query):
                return self.search(query, mode=SearchMode.LAZY, **kwargs)
            else:
                return self.search(query, mode=SearchMode.VECTOR, **kwargs)
        
        else:
            raise ValueError(f"Unsupported search mode: {mode}")
        
        return result
    
    def _is_complex_query(self, query: str) -> bool:
        """クエリの複雑さを判定"""
        # 簡易判定：単語数、疑問詞の数
        words = query.split()
        complex_indicators = ["why", "how", "explain", "compare", "difference", "relationship"]
        
        return len(words) > 10 or any(ind in query.lower() for ind in complex_indicators)
    
    # === ドキュメント操作 ===
    
    def get_document(self, document_id: str) -> DocumentInfo | None:
        """ドキュメント情報を取得"""
        # Parquetから読み込み
        import pyarrow.parquet as pq
        
        doc_table = pq.read_table(
            self.config.output_path / "index" / "level_0" / "documents.parquet"
        )
        
        for row in doc_table.to_pylist():
            if row["id"] == document_id:
                # 引用メトリクス取得
                citation_manager = self._factory.get_citation_network_manager()
                citation_manager.load()
                metrics = citation_manager.get_metrics(document_id)
                
                return DocumentInfo(
                    id=row["id"],
                    title=row["title"],
                    authors=row.get("authors", []),
                    year=row.get("year"),
                    doi=row.get("doi"),
                    chunk_count=row.get("chunk_count", 0),
                    citation_count=metrics.citation_count if metrics else 0,
                    reference_count=metrics.reference_count if metrics else 0,
                    influence_score=metrics.influence_score if metrics else 0.0
                )
        
        return None
    
    def list_documents(self, limit: int = 100) -> list[DocumentInfo]:
        """ドキュメント一覧を取得"""
        import pyarrow.parquet as pq
        
        doc_table = pq.read_table(
            self.config.output_path / "index" / "level_0" / "documents.parquet"
        )
        
        results = []
        for row in doc_table.to_pylist()[:limit]:
            results.append(DocumentInfo(
                id=row["id"],
                title=row["title"],
                authors=row.get("authors", []),
                year=row.get("year"),
                doi=row.get("doi"),
                chunk_count=row.get("chunk_count", 0),
                citation_count=0,
                reference_count=0,
                influence_score=0.0
            ))
        
        return results
    
    # === 引用ネットワーク ===
    
    def get_citation_network(self) -> "CitationNetworkManager":
        """引用ネットワークマネージャーを取得"""
        manager = self._factory.get_citation_network_manager()
        manager.load()
        return manager
    
    def find_related_papers(
        self,
        document_id: str,
        top_k: int = 10
    ) -> list["RelatedPaper"]:
        """関連論文を検索"""
        manager = self.get_citation_network()
        return manager.find_related_papers(document_id, top_k)
    
    def get_citation_chain(
        self,
        document_id: str,
        depth: int = 2
    ) -> dict[str, list[str]]:
        """引用チェーンを取得"""
        manager = self.get_citation_network()
        return manager.get_citation_chain(document_id, depth)
```

### 3.5 非同期API（オプション）

```python
import asyncio
from typing import AsyncIterator

class AsyncMONJYU:
    """非同期MONJYU API"""
    
    def __init__(self, monjyu: MONJYU):
        self._monjyu = monjyu
    
    async def search(
        self,
        query: str,
        mode: SearchMode | None = None,
        **kwargs
    ) -> SearchResult:
        """非同期検索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._monjyu.search(query, mode, **kwargs)
        )
    
    async def search_stream(
        self,
        query: str,
        mode: SearchMode | None = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """ストリーミング検索（将来実装）"""
        # TODO: LLMストリーミング対応
        result = await self.search(query, mode, **kwargs)
        for char in result.answer:
            yield char
            await asyncio.sleep(0.01)
    
    async def index(
        self,
        path: str | Path,
        **kwargs
    ) -> MONJYUStatus:
        """非同期インデックス構築"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._monjyu.index(path, **kwargs)
        )
```

---

## 4. 設定

```yaml
# config/monjyu.yaml

# 基本設定
output_path: ./output
environment: local  # local | azure

# インデックス設定
index_levels: [0, 1]  # 構築するレベル

# 検索設定
default_search_mode: lazy  # vector | lazy | auto
default_top_k: 10

# ドキュメント処理
chunk_size: 1200
chunk_overlap: 100

# ローカル環境 (Ollama)
llm_model: llama3:8b-instruct-q4_K_M
embedding_model: nomic-embed-text
ollama_base_url: http://192.168.224.1:11434

# Azure環境（オプション）
# azure_openai_endpoint: https://xxx.openai.azure.com/
# azure_openai_api_key: ${AZURE_OPENAI_API_KEY}
# azure_search_endpoint: https://xxx.search.windows.net/
# azure_search_api_key: ${AZURE_SEARCH_API_KEY}
```

---

## 5. 使用例

```python
from monjyu import MONJYU, SearchMode

# === 基本的な使用 ===

# 初期化
monjyu = MONJYU("./config/monjyu.yaml")

# インデックス構築
monjyu.index("./papers/")

# 検索
result = monjyu.search("What is Transformer architecture?")
print(result.answer)

# === 詳細な使用 ===

# ステータス確認
status = monjyu.get_status()
print(f"Documents: {status.document_count}")
print(f"Index status: {status.index_status}")

# 検索モード指定
vector_result = monjyu.search("BERT model", mode=SearchMode.VECTOR)
lazy_result = monjyu.search("How does attention work?", mode=SearchMode.LAZY)

# ドキュメント情報
doc = monjyu.get_document("doc_001")
print(f"Title: {doc.title}")
print(f"Citations: {doc.citation_count}")

# 関連論文
related = monjyu.find_related_papers("doc_001", top_k=5)
for paper in related:
    print(f"- {paper.title} ({paper.relationship_type})")

# 引用チェーン
chain = monjyu.get_citation_chain("doc_001", depth=2)
print(f"Cites: {chain['cites']}")
print(f"Cited by: {chain['cited_by']}")

# === 非同期使用 ===

import asyncio
from monjyu import AsyncMONJYU

async def main():
    monjyu = MONJYU("./config/monjyu.yaml")
    async_monjyu = AsyncMONJYU(monjyu)
    
    result = await async_monjyu.search("What is GPT?")
    print(result.answer)

asyncio.run(main())
```

---

## 6. テスト計画

### 6.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_config_from_yaml | ConfigManager | 設定読み込み成功 |
| test_config_from_dict | ConfigManager | 辞書から作成成功 |
| test_component_factory | ComponentFactory | 各コンポーネント生成 |
| test_state_save_load | StateManager | 状態の保存・復元 |

### 6.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_full_index | MONJYU.index | 全レベルインデックス構築 |
| test_vector_search | MONJYU.search(VECTOR) | ベクトル検索成功 |
| test_lazy_search | MONJYU.search(LAZY) | Lazy検索成功 |
| test_document_info | MONJYU.get_document | ドキュメント情報取得 |
| test_citation_network | MONJYU.get_citation_network | 引用情報取得 |

### 6.3 E2Eテスト

| テストケース | 条件 | 期待結果 |
|-------------|------|---------|
| test_e2e_workflow | index → search | 全フロー成功 |
| test_async_api | AsyncMONJYU | 非同期動作 |

---

## 7. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-007-01 | ConfigManager 実装 | 2h | - |
| TASK-007-02 | StateManager 実装 | 1h | - |
| TASK-007-03 | ComponentFactory 実装 | 3h | FEAT-001~006 |
| TASK-007-04 | MONJYU Facade 実装 | 4h | TASK-007-01~03 |
| TASK-007-05 | AsyncMONJYU 実装 | 2h | TASK-007-04 |
| TASK-007-06 | 単体テスト作成 | 2h | TASK-007-01~05 |
| TASK-007-07 | 統合テスト作成 | 2h | TASK-007-06 |
| TASK-007-08 | ドキュメント作成 | 2h | TASK-007-04 |
| **合計** | | **18h** | |

---

## 8. 受入基準

- [ ] YAMLまたは辞書から設定を読み込める
- [ ] `monjyu.index(path)` でインデックスを構築できる
- [ ] `monjyu.search(query)` で検索できる
- [ ] 検索モード（VECTOR, LAZY, AUTO）を切り替えられる
- [ ] ドキュメント情報を取得できる
- [ ] 引用ネットワーク情報を取得できる
- [ ] 状態を永続化・復元できる
- [ ] 非同期APIが動作する
- [ ] 使用例のコードが実際に動作する
