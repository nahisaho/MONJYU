# Level 0 Index Builder
"""
Builder for Level 0 (Baseline RAG) index.

Coordinates embedding generation, vector indexing, and storage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from monjyu.embedding.base import EmbeddingClient, EmbeddingClientProtocol, EmbeddingRecord
from monjyu.embedding.ollama import OllamaEmbeddingClient
from monjyu.index.base import VectorIndexer, VectorIndexerProtocol
from monjyu.index.lancedb import LanceDBIndexer
from monjyu.storage.parquet import ParquetStorage

if TYPE_CHECKING:
    from monjyu.document.models import AcademicPaperDocument, TextUnit


@dataclass
class Level0IndexConfig:
    """Level 0 インデックス設定
    
    Attributes:
        output_dir: 出力ディレクトリ
        embedding_strategy: 埋め込み戦略 ("ollama", "azure")
        index_strategy: インデックス戦略 ("lancedb", "azure_search")
        ollama_model: Ollamaモデル名
        ollama_base_url: Ollama APIベースURL
        azure_openai_deployment: Azure OpenAIデプロイメント名
        azure_openai_endpoint: Azure OpenAIエンドポイント
        azure_search_endpoint: Azure AI Searchエンドポイント
        azure_search_index_name: Azure AI Searchインデックス名
        batch_size: バッチサイズ
        show_progress: 進捗表示
    """
    output_dir: str | Path = "./output/index/level_0"
    embedding_strategy: Literal["ollama", "azure"] = "ollama"
    index_strategy: Literal["lancedb", "azure_search"] = "lancedb"
    
    # Ollama settings
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    
    # Azure OpenAI settings
    azure_openai_deployment: str = "text-embedding-3-large"
    azure_openai_endpoint: str | None = None
    
    # Azure AI Search settings
    azure_search_endpoint: str | None = None
    azure_search_index_name: str = "monjyu-text-units"
    
    # Processing settings
    batch_size: int = 100
    show_progress: bool = True


@dataclass
class Level0Index:
    """Level 0 インデックス
    
    Attributes:
        documents: ドキュメントのリスト
        text_units: TextUnitのリスト
        embeddings: 埋め込みレコードのリスト
        output_dir: 出力ディレクトリ
        embedding_model: 使用した埋め込みモデル
        embedding_dimensions: 埋め込み次元数
    """
    documents: list["AcademicPaperDocument"]
    text_units: list["TextUnit"]
    embeddings: list[EmbeddingRecord]
    output_dir: Path
    embedding_model: str
    embedding_dimensions: int
    
    @property
    def document_count(self) -> int:
        """ドキュメント数"""
        return len(self.documents)
    
    @property
    def text_unit_count(self) -> int:
        """TextUnit数"""
        return len(self.text_units)
    
    @property
    def embedding_count(self) -> int:
        """埋め込み数"""
        return len(self.embeddings)


class Level0IndexBuilder:
    """Level 0 インデックスビルダー
    
    TextUnitからベクトル埋め込みを生成し、
    ベクトルインデックスとParquetストレージに保存する。
    
    Example:
        >>> config = Level0IndexConfig(
        ...     output_dir="./output/index/level_0",
        ...     embedding_strategy="ollama",
        ... )
        >>> builder = Level0IndexBuilder(config)
        >>> index = await builder.build(documents, text_units)
        >>> print(f"Built index with {index.text_unit_count} units")
    
    Attributes:
        config: インデックス設定
    """
    
    def __init__(self, config: Level0IndexConfig | None = None) -> None:
        """初期化
        
        Args:
            config: インデックス設定
        """
        self.config = config or Level0IndexConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # コンポーネント初期化
        self._embedding_client: EmbeddingClientProtocol | None = None
        self._vector_indexer: VectorIndexerProtocol | None = None
        self._storage: ParquetStorage | None = None
    
    @property
    def embedding_client(self) -> EmbeddingClientProtocol:
        """埋め込みクライアントを取得（遅延初期化）"""
        if self._embedding_client is None:
            self._embedding_client = self._create_embedding_client()
        return self._embedding_client
    
    @property
    def vector_indexer(self) -> VectorIndexerProtocol:
        """ベクトルインデクサーを取得（遅延初期化）"""
        if self._vector_indexer is None:
            self._vector_indexer = self._create_vector_indexer()
        return self._vector_indexer
    
    @property
    def storage(self) -> ParquetStorage:
        """ストレージを取得（遅延初期化）"""
        if self._storage is None:
            self._storage = ParquetStorage(self.output_dir)
        return self._storage
    
    def _create_embedding_client(self) -> EmbeddingClientProtocol:
        """埋め込みクライアントを作成"""
        if self.config.embedding_strategy == "azure":
            from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
            return AzureOpenAIEmbeddingClient(
                deployment=self.config.azure_openai_deployment,
                endpoint=self.config.azure_openai_endpoint,
            )
        else:
            return OllamaEmbeddingClient(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
            )
    
    def _create_vector_indexer(self) -> VectorIndexerProtocol:
        """ベクトルインデクサーを作成"""
        if self.config.index_strategy == "azure_search":
            from monjyu.index.azure_search import AzureAISearchIndexer
            return AzureAISearchIndexer(
                endpoint=self.config.azure_search_endpoint,
                index_name=self.config.azure_search_index_name,
            )
        else:
            return LanceDBIndexer(
                db_path=self.output_dir / "vector_index" / "lancedb",
            )
    
    async def build(
        self,
        documents: list["AcademicPaperDocument"],
        text_units: list["TextUnit"],
    ) -> Level0Index:
        """Level 0 インデックスを構築
        
        Args:
            documents: ドキュメントのリスト
            text_units: TextUnitのリスト
            
        Returns:
            構築されたインデックス
        """
        if self.config.show_progress:
            print(f"Building Level 0 index...")
            print(f"  Documents: {len(documents)}")
            print(f"  Text Units: {len(text_units)}")
        
        # 1. 埋め込み生成
        embeddings = await self._generate_embeddings(text_units)
        
        # 2. ベクトルインデックス構築
        self._build_vector_index(embeddings, text_units)
        
        # 3. Parquet保存
        self._save_to_parquet(documents, text_units, embeddings)
        
        if self.config.show_progress:
            print(f"Level 0 index built successfully!")
            print(f"  Output: {self.output_dir}")
        
        return Level0Index(
            documents=documents,
            text_units=text_units,
            embeddings=embeddings,
            output_dir=self.output_dir,
            embedding_model=self.embedding_client.model_name,
            embedding_dimensions=self.embedding_client.dimensions,
        )
    
    async def _generate_embeddings(
        self,
        text_units: list["TextUnit"],
    ) -> list[EmbeddingRecord]:
        """埋め込みを生成
        
        Args:
            text_units: TextUnitのリスト
            
        Returns:
            埋め込みレコードのリスト
        """
        if self.config.show_progress:
            print(f"  Generating embeddings...")
        
        # テキストを抽出
        texts = [unit.text for unit in text_units]
        
        # バッチで埋め込み生成
        vectors = await self.embedding_client.embed_batch(
            texts,
            batch_size=self.config.batch_size,
        )
        
        # EmbeddingRecordを作成
        embeddings = []
        for unit, vector in zip(text_units, vectors, strict=True):
            record = EmbeddingRecord(
                id=f"emb_{unit.id}",
                text_unit_id=unit.id,
                vector=vector,
                model=self.embedding_client.model_name,
                dimensions=self.embedding_client.dimensions,
            )
            embeddings.append(record)
        
        if self.config.show_progress:
            print(f"    Generated {len(embeddings)} embeddings")
        
        return embeddings
    
    def _build_vector_index(
        self,
        embeddings: list[EmbeddingRecord],
        text_units: list["TextUnit"],
    ) -> None:
        """ベクトルインデックスを構築
        
        Args:
            embeddings: 埋め込みレコードのリスト
            text_units: TextUnitのリスト
        """
        if self.config.show_progress:
            print(f"  Building vector index...")
        
        # メタデータを準備
        unit_map = {unit.id: unit for unit in text_units}
        
        vectors = []
        ids = []
        metadata = []
        
        for emb in embeddings:
            vectors.append(emb.vector)
            ids.append(emb.text_unit_id)
            
            unit = unit_map.get(emb.text_unit_id)
            if unit:
                meta = {
                    "text": unit.text[:500],  # 検索結果表示用
                    "document_id": unit.document_id or "",
                    "section_type": unit.section_type or "",
                    "n_tokens": unit.n_tokens,
                }
                metadata.append(meta)
            else:
                metadata.append({})
        
        # インデックス構築
        self.vector_indexer.build(vectors, ids, metadata)
        
        if self.config.show_progress:
            print(f"    Vector index built with {len(vectors)} vectors")
    
    def _save_to_parquet(
        self,
        documents: list["AcademicPaperDocument"],
        text_units: list["TextUnit"],
        embeddings: list[EmbeddingRecord],
    ) -> None:
        """Parquetに保存
        
        Args:
            documents: ドキュメントのリスト
            text_units: TextUnitのリスト
            embeddings: 埋め込みレコードのリスト
        """
        if self.config.show_progress:
            print(f"  Saving to Parquet...")
        
        # ドキュメント保存
        if documents:
            self.storage.write_documents(documents)
        
        # TextUnit保存
        if text_units:
            self.storage.write_text_units(text_units)
        
        # 埋め込み保存
        if embeddings:
            self.storage.write_embeddings(embeddings)
        
        if self.config.show_progress:
            stats = self.storage.get_stats()
            print(f"    Saved {stats['text_units_count']} text units")
            print(f"    Saved {stats['embeddings_count']} embeddings")
    
    async def add(
        self,
        documents: list["AcademicPaperDocument"],
        text_units: list["TextUnit"],
    ) -> Level0Index:
        """既存インデックスにデータを追加
        
        Args:
            documents: 追加するドキュメントのリスト
            text_units: 追加するTextUnitのリスト
            
        Returns:
            更新されたインデックス
        """
        if self.config.show_progress:
            print(f"Adding to Level 0 index...")
        
        # 埋め込み生成
        embeddings = await self._generate_embeddings(text_units)
        
        # ベクトルインデックスに追加
        unit_map = {unit.id: unit for unit in text_units}
        vectors = [emb.vector for emb in embeddings]
        ids = [emb.text_unit_id for emb in embeddings]
        metadata = [
            {
                "text": unit_map[id_].text[:500],
                "document_id": unit_map[id_].document_id or "",
                "section_type": unit_map[id_].section_type or "",
            }
            for id_ in ids
        ]
        
        self.vector_indexer.add(vectors, ids, metadata)
        
        # Parquetに追記
        if documents:
            self.storage.write_documents(documents, append=True)
        if text_units:
            self.storage.write_text_units(text_units, append=True)
        if embeddings:
            self.storage.write_embeddings(embeddings, append=True)
        
        # 全データを読み込んで返す
        all_docs = documents  # 簡略化: 追加分のみ返す
        all_units = text_units
        all_embeddings = embeddings
        
        return Level0Index(
            documents=all_docs,
            text_units=all_units,
            embeddings=all_embeddings,
            output_dir=self.output_dir,
            embedding_model=self.embedding_client.model_name,
            embedding_dimensions=self.embedding_client.dimensions,
        )
    
    def load(self) -> Level0Index | None:
        """既存インデックスを読み込み
        
        Returns:
            読み込んだインデックス（存在しない場合はNone）
        """
        if not self.storage.exists("text_units"):
            return None
        
        # Parquetから読み込み
        from monjyu.document.models import TextUnit
        
        text_unit_data = self.storage.read_text_units()
        embedding_data = self.storage.read_embeddings()
        
        # TextUnitを復元
        text_units = []
        for data in text_unit_data:
            unit = TextUnit(
                id=data["id"],
                text=data["text"],
                n_tokens=data.get("n_tokens", 0),
                document_id=data.get("document_id"),
                section_type=data.get("section_type"),
                chunk_index=data.get("chunk_index", 0),
                start_char=data.get("start_char", 0),
                end_char=data.get("end_char", len(data["text"])),
            )
            text_units.append(unit)
        
        # EmbeddingRecordを復元
        embeddings = []
        for data in embedding_data:
            record = EmbeddingRecord(
                id=data["id"],
                text_unit_id=data["text_unit_id"],
                vector=data["vector"],
                model=data.get("model", ""),
                dimensions=data.get("dimensions", 0),
            )
            embeddings.append(record)
        
        # ベクトルインデクサーを読み込み
        vector_index_path = self.output_dir / "vector_index" / "lancedb"
        if vector_index_path.exists():
            self.vector_indexer.load(vector_index_path)
        
        model_name = embeddings[0].model if embeddings else ""
        dimensions = embeddings[0].dimensions if embeddings else 0
        
        return Level0Index(
            documents=[],  # 簡略化
            text_units=text_units,
            embeddings=embeddings,
            output_dir=self.output_dir,
            embedding_model=model_name,
            embedding_dimensions=dimensions,
        )
    
    def get_stats(self) -> dict[str, Any]:
        """インデックスの統計情報を取得
        
        Returns:
            統計情報
        """
        stats = self.storage.get_stats()
        stats["vector_index_count"] = self.vector_indexer.count()
        return stats
