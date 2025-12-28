"""Vector Searcher カバレッジ向上テスト"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from monjyu.search.base import SearchHit, SearchResults
from monjyu.search.vector_searcher import (
    VectorSearcherConfig,
    LanceDBVectorSearcher,
    AzureAISearchVectorSearcher,
    InMemoryVectorSearcher,
)


# ========== VectorSearcherConfig テスト ==========


class TestVectorSearcherConfig:
    """VectorSearcherConfig テスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = VectorSearcherConfig()
        
        assert config.db_path == "./output/index/level_0/vector_index"
        assert config.table_name == "embeddings"
        assert config.default_top_k == 10
        assert config.default_threshold == 0.0
    
    def test_custom_config(self):
        """カスタム設定のテスト"""
        config = VectorSearcherConfig(
            db_path="/custom/path",
            table_name="custom_table",
            default_top_k=20,
            default_threshold=0.5,
        )
        
        assert config.db_path == "/custom/path"
        assert config.table_name == "custom_table"
        assert config.default_top_k == 20
        assert config.default_threshold == 0.5


# ========== LanceDBVectorSearcher テスト ==========


class TestLanceDBVectorSearcher:
    """LanceDBVectorSearcher テスト"""
    
    def test_init(self):
        """初期化テスト"""
        searcher = LanceDBVectorSearcher(
            db_path="./test/db",
            table_name="test_table",
        )
        
        assert str(searcher.db_path) == "test/db"
        assert searcher.table_name == "test_table"
        assert searcher._db is None
        assert searcher._table is None
    
    def test_db_property_import_error(self):
        """lancedbインポートエラーのテスト"""
        searcher = LanceDBVectorSearcher()
        
        with patch.dict("sys.modules", {"lancedb": None}):
            with patch("builtins.__import__", side_effect=ImportError("no lancedb")):
                with pytest.raises(RuntimeError, match="lancedb package not installed"):
                    _ = searcher.db
    
    def test_db_property_connection_error(self):
        """DB接続エラーのテスト"""
        searcher = LanceDBVectorSearcher(db_path="/nonexistent/path")
        
        mock_lancedb = MagicMock()
        mock_lancedb.connect.side_effect = Exception("Connection failed")
        
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            with pytest.raises(RuntimeError, match="Failed to connect to LanceDB"):
                _ = searcher.db
    
    def test_table_property_error(self):
        """テーブルオープンエラーのテスト"""
        searcher = LanceDBVectorSearcher()
        
        mock_db = MagicMock()
        mock_db.open_table.side_effect = Exception("Table not found")
        searcher._db = mock_db
        
        with pytest.raises(RuntimeError, match="Failed to open table"):
            _ = searcher.table
    
    def test_search_with_mock(self):
        """モックを使った検索テスト"""
        searcher = LanceDBVectorSearcher()
        
        # モックテーブルを設定
        mock_table = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.limit.return_value.to_list.return_value = [
            {
                "text_unit_id": "tu_001",
                "document_id": "doc_001",
                "text": "Test text content",
                "document_title": "Test Document",
                "chunk_index": 0,
                "_distance": 0.1,  # cosine distance
            },
            {
                "text_unit_id": "tu_002",
                "document_id": "doc_002",
                "text": "Another text",
                "document_title": "Another Document",
                "chunk_index": 1,
                "_distance": 0.3,
            },
        ]
        mock_table.search.return_value = mock_search_result
        searcher._table = mock_table
        
        query_vector = [0.1] * 768
        results = searcher.search(query_vector, top_k=10, threshold=0.0)
        
        assert isinstance(results, SearchResults)
        assert len(results.hits) == 2
        assert results.hits[0].text_unit_id == "tu_001"
        assert results.hits[0].score == 0.9  # 1.0 - 0.1
        assert results.hits[1].score == 0.7  # 1.0 - 0.3
        assert results.search_time_ms > 0
    
    def test_search_with_threshold(self):
        """閾値を使った検索テスト"""
        searcher = LanceDBVectorSearcher()
        
        mock_table = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.limit.return_value.to_list.return_value = [
            {"text_unit_id": "tu_001", "_distance": 0.1},  # score 0.9
            {"text_unit_id": "tu_002", "_distance": 0.5},  # score 0.5
            {"text_unit_id": "tu_003", "_distance": 0.8},  # score 0.2
        ]
        mock_table.search.return_value = mock_search_result
        searcher._table = mock_table
        
        results = searcher.search([0.1] * 768, threshold=0.6)
        
        # 閾値0.6以上のみ（score 0.9のみ）
        assert len(results.hits) == 1
        assert results.hits[0].text_unit_id == "tu_001"
    
    def test_hybrid_search_with_mock(self):
        """ハイブリッド検索テスト"""
        searcher = LanceDBVectorSearcher()
        
        mock_table = MagicMock()
        
        # ベクトル検索結果
        mock_vector_result = MagicMock()
        mock_vector_result.limit.return_value.to_list.return_value = [
            {"text_unit_id": "tu_001", "text": "Vector result", "_distance": 0.1},
            {"text_unit_id": "tu_002", "text": "Both result", "_distance": 0.2},
        ]
        
        # FTS検索結果
        mock_fts_result = MagicMock()
        mock_fts_result.limit.return_value.to_list.return_value = [
            {"text_unit_id": "tu_002", "text": "Both result", "_score": 0.8},
            {"text_unit_id": "tu_003", "text": "FTS only result", "_score": 0.6},
        ]
        
        def search_side_effect(*args, **kwargs):
            if kwargs.get("query_type") == "fts":
                return mock_fts_result
            return mock_vector_result
        
        mock_table.search.side_effect = search_side_effect
        searcher._table = mock_table
        
        results = searcher.hybrid_search(
            query_text="test query",
            query_vector=[0.1] * 768,
            top_k=10,
            alpha=0.5,
        )
        
        assert isinstance(results, SearchResults)
        assert len(results.hits) == 3  # tu_001, tu_002, tu_003
        
        # tu_002は両方にあるので統合スコアが高い
        tu_002 = next(h for h in results.hits if h.text_unit_id == "tu_002")
        assert tu_002.vector_score == 0.8  # 1.0 - 0.2
        assert tu_002.keyword_score == 0.8
    
    def test_hybrid_search_fts_unavailable(self):
        """FTSが利用不可の場合のハイブリッド検索テスト"""
        searcher = LanceDBVectorSearcher()
        
        mock_table = MagicMock()
        
        mock_vector_result = MagicMock()
        mock_vector_result.limit.return_value.to_list.return_value = [
            {"text_unit_id": "tu_001", "text": "Vector result", "_distance": 0.1},
        ]
        
        def search_side_effect(*args, **kwargs):
            if kwargs.get("query_type") == "fts":
                raise Exception("FTS not available")
            return mock_vector_result
        
        mock_table.search.side_effect = search_side_effect
        searcher._table = mock_table
        
        results = searcher.hybrid_search(
            query_text="test",
            query_vector=[0.1] * 768,
        )
        
        # FTSエラーでもベクトル検索結果は返される
        assert len(results.hits) == 1
    
    def test_get_stats_success(self):
        """統計情報取得成功テスト"""
        searcher = LanceDBVectorSearcher(db_path="./test/db", table_name="test_table")
        
        mock_table = MagicMock()
        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=100)
        mock_table.to_pandas.return_value = mock_df
        searcher._table = mock_table
        
        stats = searcher.get_stats()
        
        assert stats["db_path"] == "test/db"
        assert stats["table_name"] == "test_table"
        assert stats["row_count"] == 100
    
    def test_get_stats_error(self):
        """統計情報取得エラーテスト"""
        searcher = LanceDBVectorSearcher(db_path="./test/db")
        
        mock_table = MagicMock()
        mock_table.to_pandas.side_effect = Exception("Error")
        searcher._table = mock_table
        
        stats = searcher.get_stats()
        
        assert stats["row_count"] == 0


# ========== AzureAISearchVectorSearcher テスト ==========


class TestAzureAISearchVectorSearcher:
    """AzureAISearchVectorSearcher テスト"""
    
    def test_init(self):
        """初期化テスト"""
        searcher = AzureAISearchVectorSearcher(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        
        assert searcher.endpoint == "https://test.search.windows.net"
        assert searcher.index_name == "test-index"
        assert searcher._client is None
    
    def test_client_property_import_error(self):
        """azure-search-documentsインポートエラーのテスト"""
        searcher = AzureAISearchVectorSearcher(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        
        with patch.dict("sys.modules", {"azure.search.documents": None, "azure.core.credentials": None}):
            with patch("builtins.__import__", side_effect=ImportError("no azure")):
                with pytest.raises(RuntimeError, match="azure-search-documents not installed"):
                    _ = searcher.client
    
    def test_search_with_mock(self):
        """モックを使った検索テスト"""
        searcher = AzureAISearchVectorSearcher(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        
        # SearchClientをモック
        mock_client = MagicMock()
        mock_client.search.return_value = [
            {
                "text_unit_id": "tu_001",
                "document_id": "doc_001",
                "text": "Test content",
                "document_title": "Test Doc",
                "chunk_index": 0,
                "@search.score": 0.95,
            },
            {
                "text_unit_id": "tu_002",
                "document_id": "doc_002",
                "text": "Another content",
                "document_title": "Another Doc",
                "chunk_index": 1,
                "@search.score": 0.85,
            },
        ]
        searcher._client = mock_client
        
        # VectorizedQueryをモック（sys.modulesに追加）
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"azure.search.documents.models": mock_models}):
            results = searcher.search([0.1] * 768, top_k=10)
        
        assert isinstance(results, SearchResults)
        assert len(results.hits) == 2
        assert results.hits[0].score == 0.95
    
    def test_search_with_threshold(self):
        """閾値付き検索テスト"""
        searcher = AzureAISearchVectorSearcher(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        
        mock_client = MagicMock()
        mock_client.search.return_value = [
            {"text_unit_id": "tu_001", "@search.score": 0.95},
            {"text_unit_id": "tu_002", "@search.score": 0.50},
            {"text_unit_id": "tu_003", "@search.score": 0.30},
        ]
        searcher._client = mock_client
        
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"azure.search.documents.models": mock_models}):
            results = searcher.search([0.1] * 768, threshold=0.6)
        
        assert len(results.hits) == 1
        assert results.hits[0].text_unit_id == "tu_001"
    
    def test_hybrid_search_with_mock(self):
        """ハイブリッド検索テスト"""
        searcher = AzureAISearchVectorSearcher(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        
        mock_client = MagicMock()
        mock_client.search.return_value = [
            {
                "text_unit_id": "tu_001",
                "document_id": "doc_001",
                "text": "Reranked content",
                "document_title": "Reranked Doc",
                "chunk_index": 0,
                "@search.score": 0.85,
                "@search.reranker_score": 0.95,  # Semantic Rankerスコア
            },
            {
                "text_unit_id": "tu_002",
                "document_id": "doc_002",
                "text": "Normal content",
                "document_title": "Normal Doc",
                "chunk_index": 1,
                "@search.score": 0.80,
                # reranker_scoreなし
            },
        ]
        searcher._client = mock_client
        
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"azure.search.documents.models": mock_models}):
            results = searcher.hybrid_search(
                query_text="test query",
                query_vector=[0.1] * 768,
                top_k=10,
            )
        
        assert len(results.hits) == 2
        # reranker_scoreが優先される
        assert results.hits[0].score == 0.95
        # reranker_scoreがない場合はsearch.scoreを使用
        assert results.hits[1].score == 0.80


# ========== InMemoryVectorSearcher テスト ==========


class TestInMemoryVectorSearcher:
    """InMemoryVectorSearcher テスト"""
    
    def test_init(self):
        """初期化テスト"""
        searcher = InMemoryVectorSearcher()
        assert len(searcher) == 0
    
    def test_add_single(self):
        """単一ベクトル追加テスト"""
        searcher = InMemoryVectorSearcher()
        
        searcher.add(
            text_unit_id="tu_001",
            vector=[1.0, 0.0, 0.0],
            text="Test text",
            document_id="doc_001",
            document_title="Test Doc",
            chunk_index=0,
        )
        
        assert len(searcher) == 1
    
    def test_add_batch(self):
        """バッチ追加テスト"""
        searcher = InMemoryVectorSearcher()
        
        items = [
            {
                "text_unit_id": "tu_001",
                "vector": [1.0, 0.0, 0.0],
                "text": "Text 1",
            },
            {
                "text_unit_id": "tu_002",
                "vector": [0.0, 1.0, 0.0],
                "text": "Text 2",
            },
            {
                "text_unit_id": "tu_003",
                "vector": [0.0, 0.0, 1.0],
                "text": "Text 3",
            },
        ]
        
        searcher.add_batch(items)
        
        assert len(searcher) == 3
    
    def test_search_cosine_similarity(self):
        """コサイン類似度検索テスト"""
        searcher = InMemoryVectorSearcher()
        
        # 直交するベクトル
        searcher.add(text_unit_id="tu_x", vector=[1.0, 0.0, 0.0], text="X axis")
        searcher.add(text_unit_id="tu_y", vector=[0.0, 1.0, 0.0], text="Y axis")
        searcher.add(text_unit_id="tu_z", vector=[0.0, 0.0, 1.0], text="Z axis")
        
        # X軸方向のクエリ
        results = searcher.search([1.0, 0.0, 0.0], top_k=3)
        
        assert len(results.hits) == 3
        assert results.hits[0].text_unit_id == "tu_x"
        assert results.hits[0].score == pytest.approx(1.0, abs=0.001)
        assert results.hits[1].score == pytest.approx(0.0, abs=0.001)
    
    def test_search_with_threshold(self):
        """閾値付き検索テスト"""
        searcher = InMemoryVectorSearcher()
        
        searcher.add(text_unit_id="tu_1", vector=[1.0, 0.0, 0.0])
        searcher.add(text_unit_id="tu_2", vector=[0.9, 0.1, 0.0])
        searcher.add(text_unit_id="tu_3", vector=[0.0, 1.0, 0.0])
        
        results = searcher.search([1.0, 0.0, 0.0], threshold=0.5)
        
        # 閾値0.5以上のみ
        assert all(h.score >= 0.5 for h in results.hits)
    
    def test_search_top_k(self):
        """Top-K制限テスト"""
        searcher = InMemoryVectorSearcher()
        
        for i in range(10):
            searcher.add(text_unit_id=f"tu_{i}", vector=[1.0, 0.0, 0.0])
        
        results = searcher.search([1.0, 0.0, 0.0], top_k=3)
        
        assert len(results.hits) == 3
    
    def test_search_zero_vector(self):
        """ゼロベクトルの検索テスト"""
        searcher = InMemoryVectorSearcher()
        
        searcher.add(text_unit_id="tu_1", vector=[1.0, 0.0, 0.0])
        searcher.add(text_unit_id="tu_zero", vector=[0.0, 0.0, 0.0])  # ゼロベクトル
        
        # ゼロベクトルでクエリ
        results = searcher.search([0.0, 0.0, 0.0])
        
        # ゼロベクトルとの類似度は0
        assert all(h.score == 0.0 for h in results.hits)
    
    def test_hybrid_search_fallback(self):
        """ハイブリッド検索（ベクトル検索へのフォールバック）テスト"""
        searcher = InMemoryVectorSearcher()
        
        searcher.add(text_unit_id="tu_1", vector=[1.0, 0.0, 0.0], text="Test")
        
        # hybrid_searchはベクトル検索と同じ結果を返す
        results = searcher.hybrid_search(
            query_text="test",
            query_vector=[1.0, 0.0, 0.0],
        )
        
        assert len(results.hits) == 1
    
    def test_clear(self):
        """クリアテスト"""
        searcher = InMemoryVectorSearcher()
        
        searcher.add(text_unit_id="tu_1", vector=[1.0, 0.0, 0.0])
        searcher.add(text_unit_id="tu_2", vector=[0.0, 1.0, 0.0])
        
        assert len(searcher) == 2
        
        searcher.clear()
        
        assert len(searcher) == 0
    
    def test_search_timing(self):
        """検索時間計測テスト"""
        searcher = InMemoryVectorSearcher()
        
        for i in range(100):
            searcher.add(text_unit_id=f"tu_{i}", vector=[float(i % 10), 0.0, 0.0])
        
        results = searcher.search([1.0, 0.0, 0.0])
        
        assert results.search_time_ms > 0
    
    def test_search_empty(self):
        """空の検索テスト"""
        searcher = InMemoryVectorSearcher()
        
        results = searcher.search([1.0, 0.0, 0.0])
        
        assert len(results.hits) == 0
        assert results.total_count == 0


# ========== 追加のエッジケーステスト ==========


class TestEdgeCases:
    """エッジケーステスト"""
    
    def test_lancedb_search_missing_fields(self):
        """LanceDB検索で欠落フィールドのテスト"""
        searcher = LanceDBVectorSearcher()
        
        mock_table = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.limit.return_value.to_list.return_value = [
            {
                "text_unit_id": "tu_001",
                # 他のフィールドは欠落
            },
        ]
        mock_table.search.return_value = mock_search_result
        searcher._table = mock_table
        
        results = searcher.search([0.1] * 768)
        
        assert len(results.hits) == 1
        assert results.hits[0].document_id == ""  # デフォルト値
        assert results.hits[0].text == ""
        assert results.hits[0].score == 1.0  # _distance欠落時は0
    
    def test_azure_search_missing_fields(self):
        """Azure検索で欠落フィールドのテスト"""
        searcher = AzureAISearchVectorSearcher(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        
        mock_client = MagicMock()
        mock_client.search.return_value = [
            {
                "text_unit_id": "tu_001",
                "@search.score": 0.9,
                # 他のフィールドは欠落
            },
        ]
        searcher._client = mock_client
        
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"azure.search.documents.models": mock_models}):
            results = searcher.search([0.1] * 768)
        
        assert len(results.hits) == 1
        assert results.hits[0].document_id == ""
    
    def test_inmemory_high_dimensional_vectors(self):
        """高次元ベクトルのテスト"""
        searcher = InMemoryVectorSearcher()
        
        # 768次元ベクトル（異なる方向）
        dim = 768
        # クエリと同じ方向のベクトル
        searcher.add(text_unit_id="tu_1", vector=[1.0] + [0.0] * (dim - 1))
        # 直交するベクトル
        searcher.add(text_unit_id="tu_2", vector=[0.0, 1.0] + [0.0] * (dim - 2))
        
        results = searcher.search([1.0] + [0.0] * (dim - 1))
        
        assert len(results.hits) == 2
        assert results.hits[0].text_unit_id == "tu_1"
        assert results.hits[0].score > results.hits[1].score
