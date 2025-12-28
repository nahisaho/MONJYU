# Index Level 0 Integration Tests
"""
Integration tests for Level 0 index building.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monjyu.embedding.base import EmbeddingClient
from monjyu.index.level0 import Level0IndexBuilder, Level0IndexConfig, Level0Index


@dataclass
class MockAcademicPaperDocument:
    """モック学術論文ドキュメント"""
    id: str
    title: str
    source_path: str
    abstract: str = ""
    
    # AcademicPaperDocument互換フィールド
    @property
    def file_name(self) -> str:
        return Path(self.source_path).name if self.source_path else ""
    
    @property
    def file_type(self) -> str:
        return Path(self.source_path).suffix.lstrip(".") if self.source_path else "txt"
    
    @property
    def authors(self) -> list:
        return []
    
    @property
    def doi(self) -> str | None:
        return None
    
    @property
    def arxiv_id(self) -> str | None:
        return None
    
    @property
    def language(self) -> str:
        return "en"
    
    @property
    def page_count(self) -> int:
        return 0
    
    @property
    def keywords(self) -> list:
        return []
    
    @property
    def sections(self) -> list:
        return []
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source_path": self.source_path,
            "abstract": self.abstract,
            "file_name": self.file_name,
            "file_type": self.file_type,
        }


@dataclass
class MockTextUnit:
    """モックTextUnit"""
    id: str
    text: str
    n_tokens: int
    document_id: str | None = None
    section_type: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] | None = None
    start_char: int = 0
    end_char: int | None = None
    
    def __post_init__(self):
        if self.end_char is None:
            self.end_char = len(self.text)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "document_id": self.document_id,
            "section_type": self.section_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata or {},
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class MockEmbeddingClient(EmbeddingClient):
    """モック埋め込みクライアント"""
    
    def __init__(self, dimensions: int = 768):
        self._dimensions = dimensions
        self._model_name = "mock-embedding"
        self.call_count = 0
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    async def embed(self, text: str) -> list[float]:
        """テキストを埋め込み"""
        self.call_count += 1
        # テキストの長さに基づいた決定論的なベクトル生成
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [((hash_val >> i) & 0xFF) / 255.0 for i in range(self._dimensions)]


class TestLevel0IndexConfig:
    """Level0IndexConfigのテスト"""
    
    def test_default_config(self):
        """デフォルト設定"""
        config = Level0IndexConfig()
        
        assert config.embedding_strategy == "ollama"
        assert config.index_strategy == "lancedb"
        assert config.batch_size == 100
    
    def test_custom_config(self):
        """カスタム設定"""
        config = Level0IndexConfig(
            output_dir="/custom/path",
            embedding_strategy="azure",
            index_strategy="azure_search",
            batch_size=50,
        )
        
        assert config.output_dir == "/custom/path"
        assert config.embedding_strategy == "azure"
        assert config.index_strategy == "azure_search"
        assert config.batch_size == 50


class TestLevel0IndexBuilder:
    """Level0IndexBuilderの統合テスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "output"
    
    @pytest.fixture
    def sample_documents(self):
        """サンプルドキュメント"""
        return [
            MockAcademicPaperDocument(
                id="doc_001",
                title="Deep Learning for NLP",
                source_path="/papers/nlp.pdf",
                abstract="This paper presents a novel approach...",
            ),
            MockAcademicPaperDocument(
                id="doc_002",
                title="Graph Neural Networks",
                source_path="/papers/gnn.pdf",
                abstract="We propose a new architecture...",
            ),
        ]
    
    @pytest.fixture
    def sample_text_units(self):
        """サンプルTextUnits"""
        return [
            MockTextUnit(
                id="tu_001",
                text="Deep learning has revolutionized natural language processing.",
                n_tokens=8,
                document_id="doc_001",
                section_type="abstract",
            ),
            MockTextUnit(
                id="tu_002",
                text="Our model achieves state-of-the-art results on multiple benchmarks.",
                n_tokens=10,
                document_id="doc_001",
                section_type="body",
            ),
            MockTextUnit(
                id="tu_003",
                text="Graph neural networks learn representations of graph-structured data.",
                n_tokens=9,
                document_id="doc_002",
                section_type="abstract",
            ),
            MockTextUnit(
                id="tu_004",
                text="We evaluate our approach on node classification and link prediction.",
                n_tokens=11,
                document_id="doc_002",
                section_type="body",
            ),
        ]
    
    def test_builder_init(self, temp_output_dir):
        """ビルダー初期化"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        assert builder.config == config
        assert builder.output_dir == temp_output_dir
    
    @pytest.mark.asyncio
    async def test_build_index_with_mock(
        self,
        temp_output_dir,
        sample_documents,
        sample_text_units,
    ):
        """モッククライアントでインデックス構築"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        # モッククライアントを注入
        mock_client = MockEmbeddingClient(dimensions=768)
        builder._embedding_client = mock_client
        
        # インデックス構築
        index = await builder.build(sample_documents, sample_text_units)
        
        # 検証
        assert isinstance(index, Level0Index)
        assert index.document_count == 2
        assert index.text_unit_count == 4
        assert index.embedding_count == 4
        assert index.embedding_model == "mock-embedding"
        assert index.embedding_dimensions == 768
        
        # 埋め込みクライアントが呼ばれたことを確認
        assert mock_client.call_count == 4
    
    @pytest.mark.asyncio
    async def test_build_creates_storage(
        self,
        temp_output_dir,
        sample_documents,
        sample_text_units,
    ):
        """ストレージファイルが作成される"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        builder._embedding_client = MockEmbeddingClient()
        
        await builder.build(sample_documents, sample_text_units)
        
        # Parquetファイルが作成されていることを確認
        assert (temp_output_dir / "documents.parquet").exists()
        assert (temp_output_dir / "text_units.parquet").exists()
        assert (temp_output_dir / "embeddings.parquet").exists()
    
    @pytest.mark.asyncio
    async def test_build_creates_vector_index(
        self,
        temp_output_dir,
        sample_documents,
        sample_text_units,
    ):
        """ベクトルインデックスが作成される"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        builder._embedding_client = MockEmbeddingClient()
        
        await builder.build(sample_documents, sample_text_units)
        
        # LanceDBが作成されていることを確認
        vector_index_dir = temp_output_dir / "vector_index" / "lancedb"
        assert vector_index_dir.exists()
    
    @pytest.mark.asyncio
    async def test_search_after_build(
        self,
        temp_output_dir,
        sample_documents,
        sample_text_units,
    ):
        """インデックス構築後に検索できる"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        mock_client = MockEmbeddingClient(dimensions=768)
        builder._embedding_client = mock_client
        
        await builder.build(sample_documents, sample_text_units)
        
        # 検索
        query_vector = await mock_client.embed("deep learning NLP")
        results = builder.vector_indexer.search(query_vector, top_k=2)
        
        assert len(results) == 2
        assert all(r.id.startswith("tu_") for r in results)
    
    @pytest.mark.asyncio
    async def test_add_to_existing_index(
        self,
        temp_output_dir,
        sample_documents,
        sample_text_units,
    ):
        """既存インデックスへの追加"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        builder._embedding_client = MockEmbeddingClient()
        
        # 最初の2つで構築
        await builder.build(
            sample_documents[:1],
            sample_text_units[:2],
        )
        
        # 残りを追加
        updated_index = await builder.add(
            sample_documents[1:],
            sample_text_units[2:],
        )
        
        # 追加分のみ返される
        assert updated_index.text_unit_count == 2
    
    def test_get_stats(
        self,
        temp_output_dir,
        sample_documents,
        sample_text_units,
    ):
        """統計情報取得"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        builder._embedding_client = MockEmbeddingClient()
        
        # 空の状態
        stats = builder.get_stats()
        assert stats["text_units_count"] == 0
        
        # 構築後
        asyncio.run(builder.build(sample_documents, sample_text_units))
        
        stats = builder.get_stats()
        assert stats["text_units_count"] == 4
        assert stats["embeddings_count"] == 4
        assert stats["vector_index_count"] == 4


class TestLevel0IndexE2E:
    """Level 0インデックスのE2Eテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "output"
    
    @pytest.mark.asyncio
    async def test_full_pipeline_simulation(self, temp_output_dir):
        """フルパイプラインのシミュレーション"""
        # シミュレート: ドキュメント処理 → インデックス構築 → 検索
        
        # 1. ドキュメント準備
        documents = [
            MockAcademicPaperDocument(
                id="paper_001",
                title="Attention Is All You Need",
                source_path="/papers/transformer.pdf",
                abstract="We propose a new architecture based solely on attention mechanisms.",
            ),
        ]
        
        text_units = [
            MockTextUnit(
                id="chunk_001",
                text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                n_tokens=15,
                document_id="paper_001",
                section_type="introduction",
            ),
            MockTextUnit(
                id="chunk_002",
                text="Attention mechanisms have become an integral part of compelling sequence modeling.",
                n_tokens=12,
                document_id="paper_001",
                section_type="introduction",
            ),
            MockTextUnit(
                id="chunk_003",
                text="The Transformer model achieves state-of-the-art results on machine translation benchmarks.",
                n_tokens=13,
                document_id="paper_001",
                section_type="results",
            ),
        ]
        
        # 2. インデックス構築
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        builder._embedding_client = MockEmbeddingClient(dimensions=384)
        
        index = await builder.build(documents, text_units)
        
        # 3. 検証
        assert index.document_count == 1
        assert index.text_unit_count == 3
        
        # 4. 検索シミュレーション
        query = "transformer attention mechanism"
        query_vector = await builder._embedding_client.embed(query)
        results = builder.vector_indexer.search(query_vector, top_k=2)
        
        assert len(results) == 2
        
        # 5. 永続化確認
        assert builder.storage.exists("documents")
        assert builder.storage.exists("text_units")
        assert builder.storage.exists("embeddings")
        
        # 6. 統計確認
        stats = builder.get_stats()
        assert stats["documents_count"] == 1
        assert stats["text_units_count"] == 3
        assert stats["embeddings_count"] == 3


class TestLevel0IndexLoad:
    """Level 0インデックスの読み込みテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "output"
    
    @pytest.mark.asyncio
    async def test_load_existing_index(self, temp_output_dir):
        """既存インデックスの読み込み"""
        # 1. インデックス構築
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        builder._embedding_client = MockEmbeddingClient()
        
        documents = [MockAcademicPaperDocument("doc_001", "Test", "/path")]
        text_units = [MockTextUnit("tu_001", "Test text", 2)]
        
        await builder.build(documents, text_units)
        
        # 2. 新しいビルダーで読み込み
        new_builder = Level0IndexBuilder(config)
        loaded_index = new_builder.load()
        
        assert loaded_index is not None
        assert loaded_index.text_unit_count == 1
        assert loaded_index.embedding_count == 1
    
    def test_load_nonexistent_index(self, temp_output_dir):
        """存在しないインデックスの読み込み"""
        config = Level0IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        loaded_index = builder.load()
        
        assert loaded_index is None
