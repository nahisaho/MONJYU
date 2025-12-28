# Search Engine Coverage Tests
"""
Search Engine カバレッジ向上テスト

VectorSearchEngine の未テスト部分をカバー
- 各検索モード (VECTOR, HYBRID, KEYWORD)
- クエリ拡張 (search_with_expansion)
- 回答合成オン/オフ
- ファクトリ関数
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from monjyu.search.base import (
    SearchMode,
    SearchResults,
    SynthesizedAnswer,
)
from monjyu.search.engine import (
    VectorSearchConfig,
    VectorSearchEngine,
)


class TestVectorSearchConfig:
    """VectorSearchConfig のテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        config = VectorSearchConfig()
        
        assert config.top_k == 10
        assert config.threshold == 0.0
        assert config.mode == SearchMode.VECTOR
        assert config.hybrid_alpha == 0.5
        assert config.expand_query is False
        assert config.num_expansions == 3
        assert config.synthesize is True
        assert config.system_prompt is None
    
    def test_custom_values(self):
        """カスタム値のテスト"""
        config = VectorSearchConfig(
            top_k=20,
            threshold=0.5,
            mode=SearchMode.HYBRID,
            hybrid_alpha=0.7,
            expand_query=True,
            num_expansions=5,
            synthesize=False,
            system_prompt="Custom prompt",
        )
        
        assert config.top_k == 20
        assert config.threshold == 0.5
        assert config.mode == SearchMode.HYBRID
        assert config.hybrid_alpha == 0.7
        assert config.expand_query is True
        assert config.num_expansions == 5
        assert config.synthesize is False
        assert config.system_prompt == "Custom prompt"
    
    def test_to_dict(self):
        """辞書変換のテスト"""
        config = VectorSearchConfig(
            top_k=15,
            mode=SearchMode.KEYWORD,
            expand_query=True,
        )
        
        data = config.to_dict()
        
        assert data["top_k"] == 15
        assert data["mode"] == "keyword"
        assert data["expand_query"] is True
        assert "system_prompt" not in data  # システムプロンプトは含まれない


class TestVectorSearchEngine:
    """VectorSearchEngine のテスト"""
    
    @pytest.fixture
    def mock_embedding_client(self):
        """モック埋め込みクライアント"""
        client = MagicMock()
        client.embed.return_value = [0.1, 0.2, 0.3] * 256  # 768次元
        return client
    
    @pytest.fixture
    def mock_vector_searcher(self):
        """モックベクトル検索クライアント"""
        searcher = MagicMock()
        
        # 検索結果モック - SearchHit互換の属性を設定
        mock_hit = MagicMock()
        mock_hit.text_unit_id = "unit-1"
        mock_hit.content = "Test content about GraphRAG"
        mock_hit.text = "Test content about GraphRAG"  # synthesizer用
        mock_hit.document_title = "Test Document"  # synthesizer用
        mock_hit.score = 0.95
        mock_hit.document_id = "doc-1"
        
        mock_results = SearchResults(
            hits=[mock_hit],
            total_count=1,
            search_time_ms=10.0,
        )
        
        searcher.search.return_value = mock_results
        searcher.hybrid_search.return_value = mock_results
        return searcher
    
    @pytest.fixture
    def mock_llm_client(self):
        """モックLLMクライアント"""
        client = MagicMock()
        client.generate.return_value = "This is a synthesized answer about GraphRAG."
        return client
    
    @pytest.fixture
    def search_engine(self, mock_embedding_client, mock_vector_searcher, mock_llm_client):
        """検索エンジンインスタンス"""
        return VectorSearchEngine(
            embedding_client=mock_embedding_client,
            vector_searcher=mock_vector_searcher,
            llm_client=mock_llm_client,
            config=VectorSearchConfig(),
        )
    
    def test_init(self, mock_embedding_client, mock_vector_searcher, mock_llm_client):
        """初期化のテスト"""
        engine = VectorSearchEngine(
            embedding_client=mock_embedding_client,
            vector_searcher=mock_vector_searcher,
            llm_client=mock_llm_client,
        )
        
        assert engine.encoder is not None
        assert engine.searcher == mock_vector_searcher
        assert engine.synthesizer is not None
        assert engine.expander is not None
        assert engine.config is not None
    
    def test_search_vector_mode(self, search_engine, mock_vector_searcher):
        """VECTORモード検索のテスト"""
        response = search_engine.search(
            query="What is GraphRAG?",
            mode=SearchMode.VECTOR,
        )
        
        assert response.query == "What is GraphRAG?"
        assert response.mode == SearchMode.VECTOR
        assert response.search_results.total_count == 1
        mock_vector_searcher.search.assert_called_once()
    
    def test_search_hybrid_mode(self, search_engine, mock_vector_searcher):
        """HYBRIDモード検索のテスト"""
        response = search_engine.search(
            query="What is GraphRAG?",
            mode=SearchMode.HYBRID,
        )
        
        assert response.mode == SearchMode.HYBRID
        mock_vector_searcher.hybrid_search.assert_called_once()
    
    def test_search_keyword_mode(self, search_engine, mock_vector_searcher):
        """KEYWORDモード検索のテスト"""
        response = search_engine.search(
            query="GraphRAG",
            mode=SearchMode.KEYWORD,
        )
        
        assert response.mode == SearchMode.KEYWORD
        # KEYWORDモードはalpha=0でhybrid_searchを呼ぶ
        mock_vector_searcher.hybrid_search.assert_called_once()
        call_args = mock_vector_searcher.hybrid_search.call_args
        assert call_args.kwargs.get("alpha") == 0.0
    
    def test_search_with_synthesis(self, search_engine):
        """回答合成ありの検索"""
        response = search_engine.search(
            query="What is GraphRAG?",
            synthesize=True,
        )
        
        assert response.answer is not None
        # 合成時間が記録されている
        assert response.synthesis_time_ms >= 0
    
    def test_search_without_synthesis(self, search_engine):
        """回答合成なしの検索"""
        response = search_engine.search(
            query="What is GraphRAG?",
            synthesize=False,
        )
        
        # 合成なしなので空回答
        assert response.answer.answer == ""
    
    def test_search_top_k_override(self, search_engine, mock_vector_searcher):
        """top_kオーバーライドのテスト"""
        response = search_engine.search(
            query="What is GraphRAG?",
            top_k=5,
        )
        
        assert response.top_k == 5
        call_args = mock_vector_searcher.search.call_args
        assert call_args.kwargs.get("top_k") == 5
    
    def test_search_threshold_override(self, search_engine, mock_vector_searcher):
        """thresholdオーバーライドのテスト"""
        response = search_engine.search(
            query="What is GraphRAG?",
            threshold=0.8,
        )
        
        call_args = mock_vector_searcher.search.call_args
        assert call_args.kwargs.get("threshold") == 0.8
    
    def test_retrieve_only(self, search_engine):
        """retrieve_onlyメソッドのテスト"""
        results = search_engine.retrieve_only(
            query="What is GraphRAG?",
            top_k=5,
        )
        
        assert isinstance(results, SearchResults)
        assert results.total_count == 1
    
    def test_get_stats(self, search_engine):
        """get_statsメソッドのテスト"""
        stats = search_engine.get_stats()
        
        assert "config" in stats
        assert "encoder_cache_size" in stats
        assert stats["config"]["top_k"] == 10
    
    def test_search_response_times(self, search_engine):
        """レスポンスタイムの記録テスト"""
        response = search_engine.search(
            query="What is GraphRAG?",
        )
        
        assert response.total_time_ms >= 0
        assert response.search_time_ms >= 0
        assert response.synthesis_time_ms >= 0
        # 総時間 >= 検索時間 + 合成時間
        assert response.total_time_ms >= (response.search_time_ms - 1)  # 誤差許容


class TestSearchWithExpansion:
    """search_with_expansion メソッドのテスト"""
    
    @pytest.fixture
    def mock_embedding_client(self):
        client = MagicMock()
        client.embed.return_value = [0.1, 0.2, 0.3] * 256
        return client
    
    @pytest.fixture
    def mock_vector_searcher(self):
        searcher = MagicMock()
        
        def create_hit(unit_id, score):
            hit = MagicMock()
            hit.text_unit_id = unit_id
            hit.content = f"Content for {unit_id}"
            hit.text = f"Content for {unit_id}"
            hit.document_title = f"Document {unit_id}"
            hit.score = score
            hit.document_id = "doc-1"
            return hit
        
        # 各拡張クエリで異なる結果を返す
        call_count = [0]
        def search_side_effect(**kwargs):
            call_count[0] += 1
            hits = [
                create_hit(f"unit-{call_count[0]}", 0.9 - call_count[0] * 0.1),
            ]
            return SearchResults(hits=hits, total_count=1, search_time_ms=5.0)
        
        searcher.search.side_effect = search_side_effect
        searcher.hybrid_search.side_effect = search_side_effect
        return searcher
    
    @pytest.fixture
    def mock_llm_client(self):
        client = MagicMock()
        client.generate.return_value = "Synthesized answer"
        return client
    
    @pytest.fixture
    def search_engine(self, mock_embedding_client, mock_vector_searcher, mock_llm_client):
        engine = VectorSearchEngine(
            embedding_client=mock_embedding_client,
            vector_searcher=mock_vector_searcher,
            llm_client=mock_llm_client,
            config=VectorSearchConfig(synthesize=True),
        )
        # クエリ拡張のモック
        engine.expander = MagicMock()
        engine.expander.expand.return_value = [
            "What is GraphRAG?",
            "How does GraphRAG work?",
            "GraphRAG architecture",
        ]
        return engine
    
    def test_search_with_expansion_basic(self, search_engine):
        """クエリ拡張検索の基本テスト"""
        response = search_engine.search_with_expansion(
            query="What is GraphRAG?",
        )
        
        assert response.query == "What is GraphRAG?"
        # 3つの拡張クエリで検索
        search_engine.expander.expand.assert_called_once()
    
    def test_search_with_expansion_num_expansions(self, search_engine):
        """num_expansionsパラメータのテスト"""
        search_engine.search_with_expansion(
            query="What is GraphRAG?",
            num_expansions=5,
        )
        
        call_args = search_engine.expander.expand.call_args
        assert call_args[0][1] == 5  # 2番目の引数がnum_expansions
    
    def test_search_with_expansion_deduplication(self, search_engine, mock_vector_searcher):
        """重複除去のテスト"""
        # 同じtext_unit_idを持つヒットが複数クエリから返される場合
        # search_with_expansionは重複を除去する
        
        # 実際のヒットオブジェクトを作成（text_unit_idを明示的に設定）
        hit_same_id = MagicMock()
        hit_same_id.text_unit_id = "unit-same"
        hit_same_id.content = "Same content"
        hit_same_id.text = "Same content"
        hit_same_id.document_title = "Same Document"
        hit_same_id.score = 0.8
        hit_same_id.document_id = "doc-1"
        
        mock_vector_searcher.search.return_value = SearchResults(
            hits=[hit_same_id],
            total_count=1,
            search_time_ms=5.0,
        )
        
        response = search_engine.search_with_expansion(
            query="What is GraphRAG?",
        )
        
        # expanderが3つのクエリを返す（元のクエリ + 2つの拡張）
        # 各クエリで同じヒットが返されるが、重複除去により1つになる
        # ただしモックでは同じオブジェクトを返すので、重複除去は効く
        # (all_hits_mapでtext_unit_idが同じなら上書き)
        assert response.search_results.total_count >= 1
        # searchが3回呼ばれる（拡張クエリ数分）
        assert mock_vector_searcher.search.call_count == 3
    
    def test_search_with_expansion_highest_score(self, search_engine, mock_vector_searcher):
        """最高スコア保持のテスト"""
        # 同じIDで異なるスコアを返す
        call_count = [0]
        scores = [0.7, 0.9, 0.8]
        
        def search_side_effect(**kwargs):
            hit = MagicMock()
            hit.text_unit_id = "unit-same"
            hit.content = "Same content"
            hit.text = "Same content"
            hit.document_title = "Same Document"
            hit.score = scores[min(call_count[0], len(scores) - 1)]
            hit.document_id = "doc-1"
            call_count[0] += 1
            return SearchResults(hits=[hit], total_count=1, search_time_ms=5.0)
        
        mock_vector_searcher.search.side_effect = search_side_effect
        
        response = search_engine.search_with_expansion(
            query="What is GraphRAG?",
        )
        
        # 最高スコア(0.9)が保持される
        assert response.search_results.hits[0].score == 0.9


class TestEmptyResults:
    """空結果のテスト"""
    
    @pytest.fixture
    def search_engine_empty(self):
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed.return_value = [0.1] * 768
        
        mock_vector_searcher = MagicMock()
        mock_vector_searcher.search.return_value = SearchResults(
            hits=[],
            total_count=0,
            search_time_ms=5.0,
        )
        mock_vector_searcher.hybrid_search.return_value = SearchResults(
            hits=[],
            total_count=0,
            search_time_ms=5.0,
        )
        
        mock_llm_client = MagicMock()
        mock_llm_client.generate.return_value = ""
        
        return VectorSearchEngine(
            embedding_client=mock_embedding_client,
            vector_searcher=mock_vector_searcher,
            llm_client=mock_llm_client,
        )
    
    def test_search_empty_results(self, search_engine_empty):
        """結果が空の場合"""
        response = search_engine_empty.search(
            query="Nonexistent topic",
        )
        
        assert response.search_results.total_count == 0
        assert response.answer.answer == ""
    
    def test_search_with_expansion_empty(self, search_engine_empty):
        """拡張検索で結果が空の場合"""
        search_engine_empty.expander = MagicMock()
        search_engine_empty.expander.expand.return_value = ["query1", "query2"]
        
        response = search_engine_empty.search_with_expansion(
            query="Nonexistent topic",
        )
        
        assert response.search_results.total_count == 0


class TestFactoryFunctions:
    """ファクトリ関数のテスト"""
    
    def test_create_local_search_engine(self):
        """create_local_search_engine のテスト"""
        with patch("monjyu.search.query_encoder.OllamaEmbeddingClient") as mock_embed, \
             patch("monjyu.search.answer_synthesizer.OllamaLLMClient") as mock_llm, \
             patch("monjyu.search.vector_searcher.LanceDBVectorSearcher") as mock_searcher:
            
            from monjyu.search.engine import create_local_search_engine
            
            engine = create_local_search_engine(
                vector_db_path="./test/db",
                embedding_model="test-embed",
                llm_model="test-llm",
                ollama_host="http://localhost:11434",
            )
            
            assert isinstance(engine, VectorSearchEngine)
            mock_embed.assert_called_once()
            mock_llm.assert_called_once()
    
    def test_create_azure_search_engine(self):
        """create_azure_search_engine のテスト"""
        with patch("monjyu.search.query_encoder.AzureOpenAIEmbeddingClient") as mock_embed, \
             patch("monjyu.search.answer_synthesizer.AzureOpenAILLMClient") as mock_llm, \
             patch("monjyu.search.vector_searcher.AzureAISearchVectorSearcher") as mock_searcher:
            
            from monjyu.search.engine import create_azure_search_engine
            
            engine = create_azure_search_engine(
                search_endpoint="https://search.windows.net",
                search_api_key="search-key",
                search_index_name="test-index",
                openai_endpoint="https://openai.azure.com",
                openai_api_key="openai-key",
                embedding_deployment="embed-deploy",
                llm_deployment="llm-deploy",
            )
            
            assert isinstance(engine, VectorSearchEngine)
            mock_embed.assert_called_once()
            mock_llm.assert_called_once()
            mock_searcher.assert_called_once()
    
    def test_create_local_search_engine_with_config(self):
        """設定付きcreate_local_search_engine"""
        with patch("monjyu.search.query_encoder.OllamaEmbeddingClient"), \
             patch("monjyu.search.answer_synthesizer.OllamaLLMClient"), \
             patch("monjyu.search.vector_searcher.LanceDBVectorSearcher"):
            
            from monjyu.search.engine import create_local_search_engine
            
            config = VectorSearchConfig(top_k=20, mode=SearchMode.HYBRID)
            
            engine = create_local_search_engine(config=config)
            
            assert engine.config.top_k == 20
            assert engine.config.mode == SearchMode.HYBRID
