# Parquet Storage Unit Tests
"""
Unit tests for Parquet storage.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from monjyu.document.models import AcademicPaperDocument, TextUnit
from monjyu.embedding.base import EmbeddingRecord
from monjyu.storage.parquet import ParquetStorage


def create_test_document(doc_id: str, title: str) -> AcademicPaperDocument:
    """テスト用ドキュメントを作成"""
    return AcademicPaperDocument(
        file_name=f"{doc_id}.pdf",
        file_type="pdf",
        title=title,
        authors=[],
        abstract="Test abstract",
        sections=[],
        raw_text=f"Raw text for {title}",
    )


def create_test_text_unit(
    unit_id: str,
    text: str,
    n_tokens: int,
    document_id: str = "doc_001",
    section_type: str = "body",
    chunk_index: int = 0,
) -> TextUnit:
    """テスト用TextUnitを作成"""
    return TextUnit(
        id=unit_id,
        text=text,
        n_tokens=n_tokens,
        document_id=document_id,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(text),
        section_type=section_type,
        page_numbers=[],
        metadata={"test": True},
    )


class TestParquetStorage:
    """ParquetStorageのテスト"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """一時ストレージディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_init(self, temp_storage_dir):
        """初期化"""
        storage = ParquetStorage(temp_storage_dir)
        
        assert storage.output_dir == temp_storage_dir
        assert not storage.exists("documents")
        assert not storage.exists("text_units")
        assert not storage.exists("embeddings")
    
    def test_write_documents(self, temp_storage_dir):
        """ドキュメント書き込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        docs = [
            create_test_document("doc_001", "Title 1"),
            create_test_document("doc_002", "Title 2"),
        ]
        
        storage.write_documents(docs)
        
        assert storage.exists("documents")
    
    def test_read_documents(self, temp_storage_dir):
        """ドキュメント読み込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        docs = [
            create_test_document("doc_001", "Title 1"),
            create_test_document("doc_002", "Title 2"),
        ]
        
        storage.write_documents(docs)
        data = storage.read_documents()
        
        assert len(data) == 2
        # ドキュメントはfile_nameで識別
        file_names = {d["file_name"] for d in data}
        assert "doc_001.pdf" in file_names
        assert "doc_002.pdf" in file_names
    
    def test_write_text_units(self, temp_storage_dir):
        """TextUnit書き込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        units = [
            create_test_text_unit("tu_001", "Text 1", 100, section_type="abstract"),
            create_test_text_unit("tu_002", "Text 2", 150, section_type="body"),
        ]
        
        storage.write_text_units(units)
        
        assert storage.exists("text_units")
    
    def test_read_text_units(self, temp_storage_dir):
        """TextUnit読み込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        units = [
            create_test_text_unit("tu_001", "Text 1", 100, section_type="abstract"),
            create_test_text_unit("tu_002", "Text 2", 150, section_type="body"),
        ]
        
        storage.write_text_units(units)
        data = storage.read_text_units()
        
        assert len(data) == 2
        assert data[0]["id"] == "tu_001"
        assert data[0]["text"] == "Text 1"
        assert data[0]["n_tokens"] == 100
    
    def test_read_text_units_by_ids(self, temp_storage_dir):
        """ID指定TextUnit読み込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        units = [
            create_test_text_unit("tu_001", "Text 1", 100),
            create_test_text_unit("tu_002", "Text 2", 150),
            create_test_text_unit("tu_003", "Text 3", 200),
        ]
        
        storage.write_text_units(units)
        data = storage.read_text_units_by_ids(["tu_001", "tu_003"])
        
        assert len(data) == 2
        ids = {d["id"] for d in data}
        assert ids == {"tu_001", "tu_003"}
    
    def test_write_embeddings(self, temp_storage_dir):
        """埋め込み書き込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        embeddings = [
            EmbeddingRecord("emb_001", "tu_001", [0.1, 0.2, 0.3], "nomic", 3),
            EmbeddingRecord("emb_002", "tu_002", [0.4, 0.5, 0.6], "nomic", 3),
        ]
        
        storage.write_embeddings(embeddings)
        
        assert storage.exists("embeddings")
    
    def test_read_embeddings(self, temp_storage_dir):
        """埋め込み読み込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        embeddings = [
            EmbeddingRecord("emb_001", "tu_001", [0.1, 0.2, 0.3], "nomic", 3),
            EmbeddingRecord("emb_002", "tu_002", [0.4, 0.5, 0.6], "nomic", 3),
        ]
        
        storage.write_embeddings(embeddings)
        data = storage.read_embeddings()
        
        assert len(data) == 2
        assert data[0]["id"] == "emb_001"
        assert data[0]["vector"] == [0.1, 0.2, 0.3]
    
    def test_read_embeddings_by_text_unit_ids(self, temp_storage_dir):
        """TextUnit ID指定埋め込み読み込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        embeddings = [
            EmbeddingRecord("emb_001", "tu_001", [0.1, 0.2, 0.3], "nomic", 3),
            EmbeddingRecord("emb_002", "tu_002", [0.4, 0.5, 0.6], "nomic", 3),
            EmbeddingRecord("emb_003", "tu_003", [0.7, 0.8, 0.9], "nomic", 3),
        ]
        
        storage.write_embeddings(embeddings)
        data = storage.read_embeddings_by_text_unit_ids(["tu_001", "tu_003"])
        
        assert len(data) == 2
        tu_ids = {d["text_unit_id"] for d in data}
        assert tu_ids == {"tu_001", "tu_003"}
    
    def test_append_mode(self, temp_storage_dir):
        """追記モード"""
        storage = ParquetStorage(temp_storage_dir)
        
        # 最初の書き込み
        units1 = [create_test_text_unit("tu_001", "Text 1", 100)]
        storage.write_text_units(units1)
        
        # 追記
        units2 = [create_test_text_unit("tu_002", "Text 2", 150)]
        storage.write_text_units(units2, append=True)
        
        data = storage.read_text_units()
        
        assert len(data) == 2
    
    def test_overwrite_mode(self, temp_storage_dir):
        """上書きモード（デフォルト）"""
        storage = ParquetStorage(temp_storage_dir)
        
        # 最初の書き込み
        units1 = [
            create_test_text_unit("tu_001", "Text 1", 100),
            create_test_text_unit("tu_002", "Text 2", 150),
        ]
        storage.write_text_units(units1)
        
        # 上書き
        units2 = [create_test_text_unit("tu_003", "Text 3", 200)]
        storage.write_text_units(units2, append=False)
        
        data = storage.read_text_units()
        
        assert len(data) == 1
        assert data[0]["id"] == "tu_003"
    
    def test_get_stats(self, temp_storage_dir):
        """統計情報取得"""
        storage = ParquetStorage(temp_storage_dir)
        
        # 空の状態
        stats = storage.get_stats()
        assert stats["documents_count"] == 0
        assert stats["text_units_count"] == 0
        assert stats["embeddings_count"] == 0
        
        # データ追加後
        storage.write_documents([create_test_document("doc_001", "Title")])
        storage.write_text_units([create_test_text_unit("tu_001", "Text", 100)])
        storage.write_embeddings([
            EmbeddingRecord("emb_001", "tu_001", [0.1, 0.2, 0.3], "nomic", 3)
        ])
        
        stats = storage.get_stats()
        assert stats["documents_count"] == 1
        assert stats["text_units_count"] == 1
        assert stats["embeddings_count"] == 1
    
    def test_clear(self, temp_storage_dir):
        """クリア"""
        storage = ParquetStorage(temp_storage_dir)
        
        storage.write_documents([create_test_document("doc_001", "Title")])
        storage.write_text_units([create_test_text_unit("tu_001", "Text", 100)])
        
        storage.clear()
        
        assert not storage.exists("documents")
        assert not storage.exists("text_units")
        assert not storage.exists("embeddings")
    
    def test_read_empty(self, temp_storage_dir):
        """空の状態で読み込み"""
        storage = ParquetStorage(temp_storage_dir)
        
        docs = storage.read_documents()
        units = storage.read_text_units()
        embeddings = storage.read_embeddings()
        
        assert docs == []
        assert units == []
        assert embeddings == []
