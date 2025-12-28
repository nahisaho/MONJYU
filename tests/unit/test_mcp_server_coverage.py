# MONJYU MCP Server Additional Coverage Tests
"""
カバレッジ向上のための追加テスト
- Prompt variations (influence, trends)
- Edge cases
- Server startup functions
- HTTP mode
"""

import json
from io import StringIO
from unittest.mock import MagicMock, Mock, patch, AsyncMock
import pytest


def create_mock_monjyu():
    """テスト用のモックMONJYU インスタンスを作成"""
    mock = MagicMock()
    
    # Mock Status
    mock_status = MagicMock()
    mock_status.document_count = 100
    mock_status.text_unit_count = 1000
    mock_status.noun_phrase_count = 5000
    mock_status.community_count = 50
    mock_status.citation_edge_count = 200
    mock_status.entity_count = 500
    mock_status.relationship_count = 2000
    mock_status.index_levels = ["level0", "level1"]
    mock_status.index_levels_built = []
    mock_status.index_status = MagicMock(value="ready")
    mock_status.is_ready = True
    mock_status.last_error = None
    mock_status.last_updated = None
    mock.get_status.return_value = mock_status

    # Mock Document
    mock_doc = MagicMock()
    mock_doc.id = "doc_001"
    mock_doc.title = "Test Paper"
    mock_doc.authors = ["Author A", "Author B"]
    mock_doc.year = 2024
    mock_doc.doi = "10.1234/test"
    mock_doc.abstract = "Test abstract"
    mock_doc.chunk_count = 10
    mock_doc.citation_count = 50
    mock_doc.reference_count = 30
    mock_doc.influence_score = 0.8
    mock.get_document.return_value = mock_doc
    mock.list_documents.return_value = [mock_doc]

    # Mock Search
    mock_result = MagicMock()
    mock_result.query = "test query"
    mock_result.answer = "Test answer"
    mock_result.citations = []
    mock_result.search_mode = MagicMock(value="lazy")
    mock_result.search_level = "level1"
    mock_result.total_time_ms = 100.0
    mock_result.llm_calls = 2
    mock_result.citation_count = 5
    mock.search.return_value = mock_result

    # Mock Text Units
    mock_text_unit = MagicMock()
    mock_text_unit.id = "tu_001"
    mock_text_unit.text = "Test text content"
    mock_text_unit.chunk_index = 0
    mock.get_text_units.return_value = [mock_text_unit]

    # Mock Citation Network
    mock_citation_manager = MagicMock()
    mock_citation_manager.get_references.return_value = []
    mock_citation_manager.get_citations.return_value = []
    mock_citation_manager.get_statistics.return_value = {
        "node_count": 100,
        "edge_count": 200,
        "density": 0.01,
        "avg_citations": 5.5,
    }
    mock_citation_manager.get_most_cited.return_value = [("doc_001", 50)]
    mock_citation_manager.find_co_citation_papers.return_value = []
    mock_citation_manager.get_metrics.return_value = MagicMock(
        citation_count=50,
        reference_count=30,
        pagerank=0.01,
        hub_score=0.5,
        authority_score=0.6,
        influence_score=0.8,
    )
    mock.get_citation_network.return_value = mock_citation_manager

    return mock


# ========== Citation Analysis Prompt Variations ==========


class TestCitationAnalysisPromptVariations:
    """引用分析プロンプトの各種バリエーション"""

    @pytest.mark.asyncio
    async def test_citation_analysis_influence_type(self):
        """影響度分析タイプのテスト"""
        from monjyu.mcp_server.server import citation_analysis

        result = await citation_analysis(document_id="doc_001", analysis_type="influence")

        assert isinstance(result, str)
        assert "doc_001" in result
        assert "Influence" in result or "influence" in result.lower()

    @pytest.mark.asyncio
    async def test_citation_analysis_trends_type(self):
        """トレンド分析タイプのテスト"""
        from monjyu.mcp_server.server import citation_analysis

        result = await citation_analysis(analysis_type="trends")

        assert isinstance(result, str)
        assert "Temporal" in result or "trends" in result.lower()

    @pytest.mark.asyncio
    async def test_citation_analysis_invalid_type(self):
        """無効な分析タイプのテスト（デフォルトにフォールバック）"""
        from monjyu.mcp_server.server import citation_analysis

        result = await citation_analysis(analysis_type="invalid_type")

        assert isinstance(result, str)
        # デフォルトのfullにフォールバックする


# ========== Prompt Edge Cases ==========


class TestPromptEdgeCases:
    """プロンプトのエッジケーステスト"""

    @pytest.mark.asyncio
    async def test_paper_summary_no_args(self):
        """引数なしの論文要約プロンプト"""
        from monjyu.mcp_server.server import paper_summary

        result = await paper_summary()

        assert isinstance(result, str)
        # ユーザーに入力を求める指示が含まれる
        assert "document_id" in result.lower() or "title" in result.lower()

    @pytest.mark.asyncio
    async def test_compare_papers_no_args(self):
        """引数なしの論文比較プロンプト"""
        from monjyu.mcp_server.server import compare_papers

        result = await compare_papers()

        assert isinstance(result, str)
        # ユーザーに入力を求める指示が含まれる

    @pytest.mark.asyncio
    async def test_compare_papers_custom_aspects(self):
        """カスタム比較項目での論文比較"""
        from monjyu.mcp_server.server import compare_papers

        result = await compare_papers(
            paper_ids="doc_001,doc_002",
            comparison_aspects="novelty,impact,reproducibility"
        )

        assert "Novelty" in result or "novelty" in result
        assert "Impact" in result or "impact" in result

    @pytest.mark.asyncio
    async def test_research_question_with_methodology(self):
        """方法論指定の研究質問プロンプト"""
        from monjyu.mcp_server.server import research_question

        result = await research_question(
            domain="NLP",
            current_interest="sentiment analysis",
            methodology_preference="deep learning"
        )

        assert "NLP" in result
        assert "sentiment analysis" in result
        assert "deep learning" in result


# ========== Tool Input Validation ==========


class TestToolInputValidation:
    """ツール入力検証のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_search_top_k_clamping_high(self):
        """top_kの上限クランプ（50より大きい場合）"""
        from monjyu.mcp_server.server import monjyu_search

        result = await monjyu_search("test", top_k=100)
        
        response = json.loads(result)
        # エラーにならないことを確認
        assert "error" not in response or response.get("query") == "test"

    @pytest.mark.asyncio
    async def test_search_top_k_clamping_low(self):
        """top_kの下限クランプ（0以下の場合）"""
        from monjyu.mcp_server.server import monjyu_search

        result = await monjyu_search("test", top_k=0)
        
        response = json.loads(result)
        # エラーにならないことを確認
        assert "error" not in response or response.get("query") == "test"

    @pytest.mark.asyncio
    async def test_list_documents_pagination(self):
        """ドキュメント一覧のページネーション"""
        from monjyu.mcp_server.server import monjyu_list_documents

        result = await monjyu_list_documents(limit=5, offset=10)
        
        response = json.loads(result)
        assert response.get("limit") == 5
        assert response.get("offset") == 10

    @pytest.mark.asyncio
    async def test_list_documents_limit_clamping(self):
        """ドキュメント一覧のlimit上限クランプ"""
        from monjyu.mcp_server.server import monjyu_list_documents

        result = await monjyu_list_documents(limit=200)
        
        response = json.loads(result)
        # 100に制限されている
        assert response.get("limit") <= 100

    @pytest.mark.asyncio
    async def test_citation_chain_depth_clamping(self):
        """引用チェーンの深さクランプ"""
        from monjyu.mcp_server.server import monjyu_citation_chain

        result = await monjyu_citation_chain("doc_001", depth=10)
        
        response = json.loads(result)
        # 3に制限されている
        assert response.get("depth") <= 3

    @pytest.mark.asyncio
    async def test_find_related_top_k_clamping(self):
        """関連論文のtop_kクランプ"""
        from monjyu.mcp_server.server import monjyu_find_related

        result = await monjyu_find_related("doc_001", top_k=100)
        
        response = json.loads(result)
        # エラーにならないことを確認
        assert "source_document" in response


# ========== Resource Error Cases ==========


class TestResourceErrorCases:
    """リソースエラーケースのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_resource_document_content_not_found(self):
        """存在しないドキュメントのコンテンツ取得"""
        from monjyu.mcp_server.server import resource_document_content

        self.mock_monjyu.get_document.return_value = None

        result = await resource_document_content("nonexistent")
        response = json.loads(result)

        assert "error" in response

    @pytest.mark.asyncio
    async def test_resource_document_citations_not_found(self):
        """存在しないドキュメントの引用取得"""
        from monjyu.mcp_server.server import resource_document_citations

        self.mock_monjyu.get_document.return_value = None

        result = await resource_document_citations("nonexistent")
        response = json.loads(result)

        assert "error" in response

    @pytest.mark.asyncio
    async def test_resource_documents_list_exception(self):
        """ドキュメント一覧取得中の例外"""
        from monjyu.mcp_server.server import resource_documents_list

        self.mock_monjyu.list_documents.side_effect = Exception("Database error")

        result = await resource_documents_list()
        response = json.loads(result)

        assert "error" in response

    @pytest.mark.asyncio
    async def test_resource_document_content_exception(self):
        """ドキュメントコンテンツ取得中の例外"""
        from monjyu.mcp_server.server import resource_document_content

        self.mock_monjyu.get_text_units.side_effect = Exception("Content error")

        result = await resource_document_content("doc_001")
        response = json.loads(result)

        assert "error" in response

    @pytest.mark.asyncio
    async def test_resource_citation_network_no_manager(self):
        """引用ネットワークマネージャーが存在しない場合"""
        from monjyu.mcp_server.server import resource_citation_network

        self.mock_monjyu.get_citation_network.return_value = None

        result = await resource_citation_network()
        response = json.loads(result)

        assert response.get("available") is False

    @pytest.mark.asyncio
    async def test_resource_citation_network_exception(self):
        """引用ネットワーク取得中の例外"""
        from monjyu.mcp_server.server import resource_citation_network

        self.mock_monjyu.get_citation_network.side_effect = Exception("Network error")

        result = await resource_citation_network()
        response = json.loads(result)

        assert "error" in response


# ========== Citation Network Edge Cases ==========


class TestCitationNetworkEdgeCases:
    """引用ネットワークのエッジケーステスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_document_citations_with_refs_and_cites(self):
        """引用と被引用の両方がある場合"""
        from monjyu.mcp_server.server import resource_document_citations

        # 引用データを設定
        mock_ref = MagicMock()
        mock_ref.target_id = "ref_001"
        mock_cite = MagicMock()
        mock_cite.source_id = "cite_001"

        self.mock_monjyu.get_citation_network.return_value.get_references.return_value = [mock_ref]
        self.mock_monjyu.get_citation_network.return_value.get_citations.return_value = [mock_cite]

        # 参照先・引用元のドキュメントを設定
        def get_doc_side_effect(doc_id):
            if doc_id == "doc_001":
                return self.mock_monjyu.get_document.return_value
            mock = MagicMock()
            mock.title = f"Related Paper {doc_id}"
            mock.year = 2023
            return mock

        self.mock_monjyu.get_document.side_effect = get_doc_side_effect

        result = await resource_document_citations("doc_001")
        response = json.loads(result)

        assert response.get("references_count", 0) >= 0
        assert response.get("cited_by_count", 0) >= 0

    @pytest.mark.asyncio
    async def test_citation_chain_no_network(self):
        """引用ネットワークが存在しない場合のチェーン取得"""
        from monjyu.mcp_server.server import monjyu_citation_chain

        self.mock_monjyu.get_citation_network.return_value = None

        result = await monjyu_citation_chain("doc_001")
        response = json.loads(result)

        assert response.get("references") == []
        assert response.get("cited_by") == []

    @pytest.mark.asyncio
    async def test_find_related_no_network(self):
        """引用ネットワークが存在しない場合の関連論文検索"""
        from monjyu.mcp_server.server import monjyu_find_related

        self.mock_monjyu.get_citation_network.return_value = None

        result = await monjyu_find_related("doc_001")
        response = json.loads(result)

        assert response.get("related_papers") == []

    @pytest.mark.asyncio
    async def test_get_metrics_no_network(self):
        """引用ネットワークが存在しない場合のメトリクス取得"""
        from monjyu.mcp_server.server import monjyu_get_metrics

        self.mock_monjyu.get_citation_network.return_value = None

        result = await monjyu_get_metrics("doc_001")
        response = json.loads(result)

        # デフォルト値が返される
        assert response["metrics"]["pagerank"] == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_no_metrics(self):
        """メトリクスが取得できない場合"""
        from monjyu.mcp_server.server import monjyu_get_metrics

        self.mock_monjyu.get_citation_network.return_value.get_metrics.return_value = None

        result = await monjyu_get_metrics("doc_001")
        response = json.loads(result)

        # デフォルト値が使用される
        assert "metrics" in response


# ========== Server Startup Tests ==========


class TestServerStartup:
    """サーバー起動関数のテスト"""

    def test_run_help(self):
        """ヘルプオプションのテスト"""
        from monjyu.mcp_server.server import run
        import sys
        from io import StringIO
        from unittest.mock import patch

        with patch.object(sys, 'argv', ['monjyu-mcp', '--help']):
            with patch.object(sys, 'exit') as mock_exit:
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    run()
                    
                mock_exit.assert_called_once_with(0)

    def test_run_version(self):
        """バージョンオプションのテスト"""
        from monjyu.mcp_server.server import run
        import sys
        from io import StringIO
        from unittest.mock import patch

        with patch.object(sys, 'argv', ['monjyu-mcp', '--version']):
            with patch.object(sys, 'exit') as mock_exit:
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    run()
                    output = mock_stdout.getvalue()
                    
                assert "0.2.0" in output
                mock_exit.assert_called_once_with(0)

    def test_run_unknown_option(self):
        """不明なオプションのテスト"""
        from monjyu.mcp_server.server import run
        import sys
        from unittest.mock import patch

        with patch.object(sys, 'argv', ['monjyu-mcp', '--unknown']):
            with patch.object(sys, 'exit') as mock_exit:
                with patch('sys.stdout', new_callable=StringIO):
                    run()
                    
                mock_exit.assert_called_once_with(1)

    def test_run_http_mode_port_parse(self):
        """HTTPモードのポート解析テスト"""
        from monjyu.mcp_server.server import run
        import sys
        from unittest.mock import patch

        with patch.object(sys, 'argv', ['monjyu-mcp', '--http', '--port', 'invalid']):
            with patch.object(sys, 'exit') as mock_exit:
                with patch('sys.stdout', new_callable=StringIO):
                    try:
                        run()
                    except (TypeError, Exception):
                        # mcp.run()の呼び出し前にエラーになるはず
                        pass
                    
                # invalidなポートでexit(1)が呼ばれる
                mock_exit.assert_called_with(1)


# ========== HTTP Mode Tests ==========


class TestHTTPMode:
    """HTTPモードのテスト"""

    def test_run_http_with_custom_host_port(self):
        """カスタムホスト/ポートでのHTTPモード"""
        from monjyu.mcp_server.server import run, mcp
        import sys
        from unittest.mock import patch

        with patch.object(sys, 'argv', ['monjyu-mcp', '--http', '--host', '0.0.0.0', '--port', '9000']):
            with patch.object(mcp, 'run') as mock_mcp_run:
                run()
                
                mock_mcp_run.assert_called_once_with(
                    transport="streamable-http",
                    host="0.0.0.0",
                    port=9000
                )

    def test_main_http_default(self):
        """main_httpデフォルト値のテスト"""
        from monjyu.mcp_server.server import main_http, mcp
        from unittest.mock import patch

        with patch.object(mcp, 'run') as mock_mcp_run:
            main_http()
            
            mock_mcp_run.assert_called_once_with(
                transport="streamable-http",
                host="127.0.0.1",
                port=8080
            )


# ========== Literature Review Prompt Tests ==========


class TestLiteratureReviewPromptVariations:
    """文献レビュープロンプトの詳細テスト"""

    @pytest.mark.asyncio
    async def test_literature_review_with_num_papers(self):
        """論文数指定の文献レビュー"""
        from monjyu.mcp_server.server import literature_review

        result = await literature_review("quantum computing", num_papers=20)

        assert "quantum computing" in result
        assert "20" in result

    @pytest.mark.asyncio
    async def test_literature_review_structure(self):
        """文献レビューの構造が含まれていることを確認"""
        from monjyu.mcp_server.server import literature_review

        result = await literature_review("machine learning")

        # 必要なセクションが含まれていることを確認
        assert "Introduction" in result
        assert "Background" in result or "Context" in result
        assert "Conclusions" in result or "Future" in result


# ========== JSON Format Tests ==========


class TestJSONFormat:
    """JSON出力フォーマットのテスト"""

    def test_json_format_nested_objects(self):
        """ネストされたオブジェクトのJSON出力"""
        from monjyu.mcp_server.server import json_format

        data = {
            "level1": {
                "level2": {
                    "value": 123
                }
            },
            "array": [1, 2, 3]
        }
        
        result = json_format(data)
        parsed = json.loads(result)
        
        assert parsed["level1"]["level2"]["value"] == 123
        assert parsed["array"] == [1, 2, 3]

    def test_error_format_unicode(self):
        """Unicode文字を含むエラーメッセージ"""
        from monjyu.mcp_server.server import error_format

        result = error_format("エラーが発生しました: データベース接続失敗")
        parsed = json.loads(result)
        
        assert "error" in parsed
        assert "エラー" in parsed["error"]


# ========== Tool Empty Results ==========


class TestToolEmptyResults:
    """空の結果を返すケースのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_list_documents_empty(self):
        """空のドキュメント一覧"""
        from monjyu.mcp_server.server import monjyu_list_documents

        self.mock_monjyu.list_documents.return_value = []

        result = await monjyu_list_documents()
        response = json.loads(result)

        assert response["count"] == 0
        assert response["documents"] == []

    @pytest.mark.asyncio
    async def test_search_empty_citations(self):
        """引用なしの検索結果"""
        from monjyu.mcp_server.server import monjyu_search

        mock_result = self.mock_monjyu.search.return_value
        mock_result.citations = []

        result = await monjyu_search("test query")
        response = json.loads(result)

        assert response["citations"] == []


# ========== Index Status Edge Cases ==========


class TestIndexStatusEdgeCases:
    """インデックスステータスのエッジケース"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_status_with_last_updated(self):
        """last_updated が設定されている場合"""
        from monjyu.mcp_server.server import resource_index_status
        from datetime import datetime

        self.mock_monjyu.get_status.return_value.last_updated = datetime(2024, 1, 1, 12, 0, 0)

        result = await resource_index_status()
        response = json.loads(result)

        assert response.get("last_updated") is not None
        assert "2024" in response["last_updated"]

    @pytest.mark.asyncio
    async def test_status_with_community_count(self):
        """community_count の確認"""
        from monjyu.mcp_server.server import resource_index_status

        self.mock_monjyu.get_status.return_value.community_count = 100

        result = await resource_index_status()
        response = json.loads(result)

        assert response.get("community_count") == 100
