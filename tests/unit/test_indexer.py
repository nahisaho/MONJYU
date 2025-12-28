# Vector Indexer Unit Tests
"""
Unit tests for vector indexers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from monjyu.index.base import SearchResult, VectorIndexer
from monjyu.index.lancedb import LanceDBIndexer


class TestSearchResult:
    """SearchResultのテスト"""
    
    def test_create_result(self):
        """結果作成"""
        result = SearchResult(
            id="tu_001",
            score=0.95,
            metadata={"text": "Sample text", "document_id": "doc_001"},
        )
        
        assert result.id == "tu_001"
        assert result.score == 0.95
        assert result.metadata["text"] == "Sample text"
    
    def test_to_dict(self):
        """辞書変換"""
        result = SearchResult(
            id="tu_001",
            score=0.95,
            metadata={"text": "Sample"},
        )
        
        data = result.to_dict()
        
        assert data["id"] == "tu_001"
        assert data["score"] == 0.95
        assert data["metadata"]["text"] == "Sample"


class TestLanceDBIndexer:
    """LanceDBIndexerのテスト"""
    
    @pytest.fixture
    def temp_db_path(self):
        """一時DBパス"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_lancedb"
    
    def test_init(self, temp_db_path):
        """初期化"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        assert indexer.db_path == temp_db_path
        assert indexer.table_name == "text_units"
        assert indexer.count() == 0
    
    def test_build_index(self, temp_db_path):
        """インデックス構築"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        ids = ["tu_001", "tu_002", "tu_003"]
        metadata = [
            {"text": "Text 1"},
            {"text": "Text 2"},
            {"text": "Text 3"},
        ]
        
        indexer.build(vectors, ids, metadata)
        
        assert indexer.count() == 3
    
    def test_add_vectors(self, temp_db_path):
        """ベクトル追加"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        # 初期構築
        indexer.build(
            [[0.1, 0.2, 0.3]],
            ["tu_001"],
            [{"text": "Text 1"}],
        )
        
        # 追加
        indexer.add(
            [[0.4, 0.5, 0.6]],
            ["tu_002"],
            [{"text": "Text 2"}],
        )
        
        assert indexer.count() == 2
    
    def test_search(self, temp_db_path):
        """検索"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        ids = ["tu_001", "tu_002", "tu_003"]
        metadata = [
            {"text": "X axis"},
            {"text": "Y axis"},
            {"text": "Z axis"},
        ]
        
        indexer.build(vectors, ids, metadata)
        
        # X軸に近いベクトルで検索
        query_vector = [0.9, 0.1, 0.0]
        results = indexer.search(query_vector, top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "tu_001"  # X軸が最も近い
        assert results[0].score < results[1].score  # L2距離なので小さいほうが近い
    
    def test_search_with_filter(self, temp_db_path):
        """フィルター付き検索"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        vectors = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ]
        ids = ["tu_001", "tu_002", "tu_003"]
        metadata = [
            {"text": "Doc A", "document_id": "doc_A"},
            {"text": "Doc A also", "document_id": "doc_A"},
            {"text": "Doc B", "document_id": "doc_B"},
        ]
        
        indexer.build(vectors, ids, metadata)
        
        # Doc_Bのみで検索
        query_vector = [1.0, 0.0, 0.0]
        results = indexer.search(
            query_vector,
            top_k=5,
            filter_expr="document_id = 'doc_B'",
        )
        
        assert len(results) == 1
        assert results[0].id == "tu_003"
    
    def test_get_by_id(self, temp_db_path):
        """ID指定取得"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        indexer.build(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            ["tu_001", "tu_002"],
            [{"text": "Text 1"}, {"text": "Text 2"}],
        )
        
        result = indexer.get_by_id("tu_001")
        
        assert result is not None
        assert result.id == "tu_001"
    
    def test_get_by_ids(self, temp_db_path):
        """複数ID指定取得"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        indexer.build(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            ["tu_001", "tu_002", "tu_003"],
            [{"text": "Text 1"}, {"text": "Text 2"}, {"text": "Text 3"}],
        )
        
        results = indexer.get_by_ids(["tu_001", "tu_003"])
        
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"tu_001", "tu_003"}
    
    def test_delete(self, temp_db_path):
        """削除"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        indexer.build(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            ["tu_001", "tu_002"],
            [{"text": "Text 1"}, {"text": "Text 2"}],
        )
        
        indexer.delete(["tu_001"])
        
        assert indexer.count() == 1
        assert indexer.get_by_id("tu_001") is None
    
    def test_clear(self, temp_db_path):
        """クリア"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        indexer.build(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            ["tu_001", "tu_002"],
            [{"text": "Text 1"}, {"text": "Text 2"}],
        )
        
        indexer.clear()
        
        assert indexer.count() == 0
    
    def test_save_and_load(self, temp_db_path):
        """保存と読み込み"""
        indexer = LanceDBIndexer(db_path=temp_db_path)
        
        indexer.build(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ["tu_001", "tu_002"],
            [{"text": "Text 1"}, {"text": "Text 2"}],
        )
        
        # 保存（LanceDBは自動永続化されるので実質no-op）
        indexer.save(temp_db_path)
        
        # 新しいインデクサーで読み込み
        # LanceDBはコンストラクタで既存テーブルを自動的に開く
        new_indexer = LanceDBIndexer(db_path=temp_db_path)
        
        assert new_indexer.count() == 2
        
        # 検索も機能する
        results = new_indexer.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0].id == "tu_001"


class TestAzureAISearchIndexer:
    """AzureAISearchIndexerのテスト"""
    
    def test_init_requires_endpoint(self):
        """エンドポイント必須"""
        try:
            from monjyu.index.azure_search import AzureAISearchIndexer
            
            with pytest.raises(ValueError):
                AzureAISearchIndexer()
        except ImportError:
            pytest.skip("Azure Search SDK not installed")
    
    def test_init_with_endpoint(self):
        """エンドポイント指定"""
        try:
            from monjyu.index.azure_search import AzureAISearchIndexer
            
            with pytest.raises(Exception):  # 接続エラー
                AzureAISearchIndexer(
                    endpoint="https://test.search.windows.net/",
                    index_name="test-index",
                )
        except ImportError:
            pytest.skip("Azure Search SDK not installed")
