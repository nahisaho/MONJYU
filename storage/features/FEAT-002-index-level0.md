# FEAT-002: Index Level 0 (Baseline)

**フィーチャーID**: FEAT-002  
**名称**: インデックス Level 0 (Baseline RAG 基盤)  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

ベクトル検索の基盤となるLevel 0インデックスを構築するフィーチャー。TextUnitのベクトル埋め込みを生成し、Parquet形式で永続化する。

### 1.1 スコープ

```
TextUnit[] → Embedding生成 → Vector Index構築 → Parquet永続化
```

- **入力**: FEAT-001 で生成された TextUnit[]
- **処理**: ベクトル埋め込み生成、インデックス構築
- **出力**: Parquetファイル、ベクトルDBインデックス

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-IDX-L0-001 | TextUnit永続化 | P0 |
| FR-IDX-L0-002 | ベクトル埋め込み生成 | P0 |
| FR-IDX-L0-003 | ベクトルインデックス構築 | P0 |
| FR-IDX-L0-004 | 文書メタデータ永続化 | P0 |

### 1.3 依存関係

- **依存**: FEAT-001 (Document Processing)
- **被依存**: FEAT-003 (Index Level 1), FEAT-004 (Vector Search)

---

## 2. アーキテクチャ

### 2.1 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Index Level 0 Builder                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ EmbeddingClient  │    │ VectorIndexer    │    │ ParquetWriter │ │
│  │                  │    │                  │    │               │ │
│  │ - embed()        │───▶│ - build()        │───▶│ - write()     │ │
│  │ - embed_batch()  │    │ - add()          │    │ - append()    │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
│         │                        │                       │          │
│         ▼                        ▼                       ▼          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ EmbeddingModel   │    │ VectorDB         │    │ Parquet Files │ │
│  │ (Azure/Ollama)   │    │ (LanceDB/FAISS)  │    │               │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Embedding Strategy                         │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐   │  │
│  │  │ Azure OpenAI        │  │ Ollama (Local)              │   │  │
│  │  │ text-embedding-3    │  │ nomic-embed-text            │   │  │
│  │  └─────────────────────┘  └─────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 出力ディレクトリ構造

```
output/
└── index/
    └── level_0/
        ├── documents.parquet        # 文書メタデータ
        ├── text_units.parquet       # チャンクテキスト
        ├── embeddings.parquet       # ベクトル埋め込み
        └── vector_index/            # ベクトルDBインデックス
            ├── lancedb/             # LanceDB (ローカル)
            │   └── text_units.lance/
            └── faiss/               # FAISS (オプション)
                └── index.faiss
```

### 2.3 クラス図

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol
import numpy as np

# === Protocols ===

class EmbeddingClientProtocol(Protocol):
    """埋め込みクライアントプロトコル"""
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...

class VectorIndexerProtocol(Protocol):
    """ベクトルインデクサープロトコル"""
    def build(self, embeddings: list[list[float]], ids: list[str]) -> None: ...
    def add(self, embeddings: list[list[float]], ids: list[str]) -> None: ...
    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

class StorageWriterProtocol(Protocol):
    """ストレージライタープロトコル"""
    def write_documents(self, documents: list["AcademicPaperDocument"]) -> None: ...
    def write_text_units(self, text_units: list["TextUnit"]) -> None: ...
    def write_embeddings(self, embeddings: list["EmbeddingRecord"]) -> None: ...

# === Data Classes ===

@dataclass
class EmbeddingRecord:
    """埋め込みレコード"""
    id: str
    text_unit_id: str
    vector: list[float]
    model: str
    dimensions: int

@dataclass
class Level0Index:
    """Level 0 インデックス"""
    documents: list["AcademicPaperDocument"]
    text_units: list["TextUnit"]
    embeddings: list[EmbeddingRecord]
    vector_index_path: str
    
    @property
    def document_count(self) -> int:
        return len(self.documents)
    
    @property
    def text_unit_count(self) -> int:
        return len(self.text_units)
```

---

## 3. 詳細設計

### 3.1 EmbeddingClient

```python
from abc import ABC, abstractmethod
import asyncio

class EmbeddingClient(ABC):
    """埋め込みクライアント抽象基底クラス"""
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
    
    @abstractmethod
    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]: ...
    
    @property
    @abstractmethod
    def dimensions(self) -> int: ...
    
    @property
    @abstractmethod
    def model_name(self) -> str: ...


class OllamaEmbeddingClient(EmbeddingClient):
    """Ollama 埋め込みクライアント（ローカル開発用）"""
    
    DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://192.168.224.1:11434"
    ):
        self.model = model
        self.base_url = base_url
        self._dimensions = self.DIMENSIONS.get(model, 768)
    
    async def embed(self, text: str) -> list[float]:
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
    
    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(*[self.embed(text) for text in batch])
            results.extend(batch_results)
        return results
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self.model


class AzureOpenAIEmbeddingClient(EmbeddingClient):
    """Azure OpenAI 埋め込みクライアント（本番用）"""
    
    DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        deployment: str = "text-embedding-3-large",
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-08-01-preview"
    ):
        import os
        from openai import AsyncAzureOpenAI
        
        self.deployment = deployment
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version
        )
        self._dimensions = self.DIMENSIONS.get(deployment, 3072)
    
    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            input=text,
            model=self.deployment
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.deployment
            )
            results.extend([d.embedding for d in response.data])
        return results
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self.deployment
```

### 3.2 VectorIndexer

```python
from abc import ABC, abstractmethod
from pathlib import Path

class VectorIndexer(ABC):
    """ベクトルインデクサー抽象基底クラス"""
    
    @abstractmethod
    def build(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] = None) -> None: ...
    
    @abstractmethod
    def add(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] = None) -> None: ...
    
    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float, dict]]: ...
    
    @abstractmethod
    def save(self, path: Path) -> None: ...
    
    @abstractmethod
    def load(self, path: Path) -> None: ...


class LanceDBIndexer(VectorIndexer):
    """LanceDB インデクサー（ローカル開発用）"""
    
    def __init__(self, db_path: str = "./storage/lancedb"):
        import lancedb
        self.db = lancedb.connect(db_path)
        self.table = None
        self.table_name = "text_units"
    
    def build(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] = None) -> None:
        import pyarrow as pa
        
        data = []
        for i, (emb, id_) in enumerate(zip(embeddings, ids)):
            record = {
                "id": id_,
                "vector": emb,
            }
            if metadata and i < len(metadata):
                record.update(metadata[i])
            data.append(record)
        
        self.table = self.db.create_table(self.table_name, data, mode="overwrite")
    
    def add(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] = None) -> None:
        if self.table is None:
            self.build(embeddings, ids, metadata)
            return
        
        data = []
        for i, (emb, id_) in enumerate(zip(embeddings, ids)):
            record = {
                "id": id_,
                "vector": emb,
            }
            if metadata and i < len(metadata):
                record.update(metadata[i])
            data.append(record)
        
        self.table.add(data)
    
    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float, dict]]:
        if self.table is None:
            raise ValueError("Index not built")
        
        results = self.table.search(query_embedding).limit(top_k).to_list()
        
        return [
            (r["id"], r["_distance"], {k: v for k, v in r.items() if k not in ["id", "vector", "_distance"]})
            for r in results
        ]
    
    def save(self, path: Path) -> None:
        # LanceDBは自動永続化
        pass
    
    def load(self, path: Path) -> None:
        import lancedb
        self.db = lancedb.connect(str(path))
        self.table = self.db.open_table(self.table_name)


class AzureAISearchIndexer(VectorIndexer):
    """Azure AI Search インデクサー（本番用）"""
    
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str = "monjyu-text-units"
    ):
        import os
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        self.endpoint = endpoint or os.environ["AZURE_SEARCH_ENDPOINT"]
        self.api_key = api_key or os.environ["AZURE_SEARCH_KEY"]
        self.index_name = index_name
        
        self.client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.api_key)
        )
    
    def build(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] = None) -> None:
        # インデックス作成（要: 事前にAzureでインデックス定義）
        documents = []
        for i, (emb, id_) in enumerate(zip(embeddings, ids)):
            doc = {
                "id": id_,
                "vector": emb,
            }
            if metadata and i < len(metadata):
                doc.update(metadata[i])
            documents.append(doc)
        
        # バッチアップロード
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.client.upload_documents(documents=batch)
    
    def add(self, embeddings: list[list[float]], ids: list[str], metadata: list[dict] = None) -> None:
        self.build(embeddings, ids, metadata)  # 同じ処理（upsert）
    
    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float, dict]]:
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="vector"
        )
        
        results = self.client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k
        )
        
        return [
            (r["id"], r["@search.score"], {k: v for k, v in r.items() if k not in ["id", "vector", "@search.score"]})
            for r in results
        ]
    
    def save(self, path: Path) -> None:
        # Azure AI Search は自動永続化
        pass
    
    def load(self, path: Path) -> None:
        # 接続情報のみ復元
        pass
```

### 3.3 ParquetStorage

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

class ParquetStorage:
    """Parquet ストレージ"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_documents(self, documents: list["AcademicPaperDocument"]) -> Path:
        """文書メタデータを永続化"""
        records = []
        for doc in documents:
            records.append({
                "file_name": doc.file_name,
                "file_type": doc.file_type,
                "title": doc.title,
                "doi": doc.doi,
                "arxiv_id": doc.arxiv_id,
                "pmid": doc.pmid,
                "publication_year": doc.publication_year,
                "venue": doc.venue,
                "venue_type": doc.venue_type,
                "abstract": doc.abstract,
                "keywords": doc.keywords or [],
                "language": doc.language,
                "page_count": doc.page_count,
                "author_names": [a.name for a in doc.authors] if doc.authors else [],
            })
        
        table = pa.Table.from_pylist(records)
        path = self.base_path / "documents.parquet"
        pq.write_table(table, path)
        return path
    
    def write_text_units(self, text_units: list["TextUnit"]) -> Path:
        """テキストユニットを永続化"""
        records = []
        for tu in text_units:
            records.append({
                "id": tu.id,
                "text": tu.text,
                "n_tokens": tu.n_tokens,
                "document_id": tu.document_id,
                "chunk_index": tu.chunk_index,
                "start_char": tu.start_char,
                "end_char": tu.end_char,
                "section_type": tu.section_type,
                "page_numbers": tu.page_numbers or [],
            })
        
        table = pa.Table.from_pylist(records)
        path = self.base_path / "text_units.parquet"
        pq.write_table(table, path)
        return path
    
    def write_embeddings(self, embeddings: list["EmbeddingRecord"]) -> Path:
        """埋め込みを永続化"""
        records = []
        for emb in embeddings:
            records.append({
                "id": emb.id,
                "text_unit_id": emb.text_unit_id,
                "vector": emb.vector,
                "model": emb.model,
                "dimensions": emb.dimensions,
            })
        
        table = pa.Table.from_pylist(records)
        path = self.base_path / "embeddings.parquet"
        pq.write_table(table, path)
        return path
    
    def read_documents(self) -> list[dict]:
        """文書メタデータを読み込み"""
        path = self.base_path / "documents.parquet"
        table = pq.read_table(path)
        return table.to_pylist()
    
    def read_text_units(self) -> list[dict]:
        """テキストユニットを読み込み"""
        path = self.base_path / "text_units.parquet"
        table = pq.read_table(path)
        return table.to_pylist()
    
    def read_embeddings(self) -> list[dict]:
        """埋め込みを読み込み"""
        path = self.base_path / "embeddings.parquet"
        table = pq.read_table(path)
        return table.to_pylist()
```

### 3.4 Level0IndexBuilder (Facade)

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Level0BuildResult:
    """Level 0 インデックス構築結果"""
    document_count: int
    text_unit_count: int
    embedding_count: int
    output_path: Path
    vector_index_path: Path


class Level0IndexBuilder:
    """Level 0 インデックスビルダー"""
    
    def __init__(
        self,
        embedding_client: EmbeddingClientProtocol,
        vector_indexer: VectorIndexerProtocol,
        output_path: Path = Path("./output/index/level_0")
    ):
        self.embedding_client = embedding_client
        self.vector_indexer = vector_indexer
        self.output_path = Path(output_path)
        self.storage = ParquetStorage(self.output_path)
    
    async def build(
        self,
        documents: list["AcademicPaperDocument"],
        text_units: list["TextUnit"],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Level0BuildResult:
        """Level 0 インデックスを構築"""
        
        # 1. 文書メタデータ永続化
        self.storage.write_documents(documents)
        
        # 2. テキストユニット永続化
        self.storage.write_text_units(text_units)
        
        # 3. 埋め込み生成
        texts = [tu.text for tu in text_units]
        embeddings_vectors = await self.embedding_client.embed_batch(texts, batch_size)
        
        # 4. 埋め込みレコード作成
        embedding_records = []
        for tu, vector in zip(text_units, embeddings_vectors):
            embedding_records.append(EmbeddingRecord(
                id=f"emb_{tu.id}",
                text_unit_id=tu.id,
                vector=vector,
                model=self.embedding_client.model_name,
                dimensions=self.embedding_client.dimensions
            ))
        
        # 5. 埋め込み永続化
        self.storage.write_embeddings(embedding_records)
        
        # 6. ベクトルインデックス構築
        metadata = [
            {
                "text": tu.text,
                "document_id": tu.document_id,
                "section_type": tu.section_type,
            }
            for tu in text_units
        ]
        self.vector_indexer.build(
            embeddings_vectors,
            [tu.id for tu in text_units],
            metadata
        )
        
        vector_index_path = self.output_path / "vector_index"
        vector_index_path.mkdir(exist_ok=True)
        self.vector_indexer.save(vector_index_path)
        
        return Level0BuildResult(
            document_count=len(documents),
            text_unit_count=len(text_units),
            embedding_count=len(embedding_records),
            output_path=self.output_path,
            vector_index_path=vector_index_path
        )
    
    async def add(
        self,
        documents: list["AcademicPaperDocument"],
        text_units: list["TextUnit"]
    ) -> Level0BuildResult:
        """既存インデックスに追加"""
        # 増分追加の実装
        ...
```

---

## 4. 設定

```yaml
# config/index_level0.yaml

index_level0:
  output_path: ./output/index/level_0
  
  # 埋め込み設定
  embedding:
    provider: ollama  # ollama | azure_openai
    
    ollama:
      model: nomic-embed-text
      base_url: http://192.168.224.1:11434
      batch_size: 100
    
    azure_openai:
      deployment: text-embedding-3-large
      api_version: "2024-08-01-preview"
      batch_size: 100
  
  # ベクトルDB設定
  vector_store:
    provider: lancedb  # lancedb | azure_ai_search
    
    lancedb:
      db_path: ./storage/lancedb
      table_name: text_units
    
    azure_ai_search:
      index_name: monjyu-text-units
  
  # Parquet設定
  parquet:
    compression: snappy
    row_group_size: 10000
```

---

## 5. テスト計画

### 5.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_ollama_embed | OllamaEmbeddingClient.embed | 768次元ベクトルを返す |
| test_ollama_embed_batch | OllamaEmbeddingClient.embed_batch | バッチサイズ分のベクトルを返す |
| test_azure_embed | AzureOpenAIEmbeddingClient.embed | 3072次元ベクトルを返す |
| test_lancedb_build | LanceDBIndexer.build | インデックスを作成 |
| test_lancedb_search | LanceDBIndexer.search | Top-K結果を返す |
| test_parquet_write_documents | ParquetStorage.write_documents | Parquetファイルを作成 |
| test_parquet_read_documents | ParquetStorage.read_documents | 書き込んだデータを読み込み |

### 5.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_level0_build | Level0IndexBuilder.build | 全アーティファクトを生成 |
| test_level0_search | Level0IndexBuilder + VectorIndexer | 検索結果を返す |

### 5.3 パフォーマンステスト

| テストケース | 条件 | 期待結果 |
|-------------|------|---------|
| test_embedding_throughput | 1000 TextUnits | > 50 units/sec (Ollama) |
| test_index_build_throughput | 10000 TextUnits | < 10 min |
| test_search_latency | 1000 queries | < 50ms (p95) |

---

## 6. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-002-01 | EmbeddingClient Protocol 定義 | 1h | - |
| TASK-002-02 | OllamaEmbeddingClient 実装 | 2h | TASK-002-01 |
| TASK-002-03 | AzureOpenAIEmbeddingClient 実装 | 2h | TASK-002-01 |
| TASK-002-04 | VectorIndexer Protocol 定義 | 1h | - |
| TASK-002-05 | LanceDBIndexer 実装 | 3h | TASK-002-04 |
| TASK-002-06 | AzureAISearchIndexer 実装 | 3h | TASK-002-04 |
| TASK-002-07 | ParquetStorage 実装 | 2h | - |
| TASK-002-08 | Level0IndexBuilder 実装 | 3h | TASK-002-01~07 |
| TASK-002-09 | 単体テスト作成 | 3h | TASK-002-01~08 |
| TASK-002-10 | 統合テスト作成 | 2h | TASK-002-09 |
| **合計** | | **22h** | |

---

## 7. 受入基準

- [ ] Ollama / Azure OpenAI の両方で埋め込み生成できる
- [ ] LanceDB / Azure AI Search の両方でベクトル検索できる
- [ ] Parquet形式で documents, text_units, embeddings を永続化できる
- [ ] 1000件のTextUnitを10分以内にインデックス構築できる
- [ ] ベクトル検索のレイテンシが50ms以下（p95）
- [ ] 設定ファイルでプロバイダーを切り替えられる
