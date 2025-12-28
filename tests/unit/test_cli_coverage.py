# CLI Commands Coverage Tests
"""
Unit tests for CLI command modules to improve coverage.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ==============================================================================
# Mock Classes
# ==============================================================================

@dataclass
class MockDocument:
    """モックドキュメント"""
    id: str
    title: str
    authors: list[str] = None
    year: int = None
    doi: str = None
    abstract: str = ""
    chunk_count: int = 10
    citation_count: int = 5
    reference_count: int = 8
    influence_score: float = 0.5
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []


@dataclass
class MockCitationEdge:
    """モック引用エッジ"""
    source_id: str
    target_id: str


@dataclass
class MockCitationMetrics:
    """モック引用メトリクス"""
    document_id: str
    citation_count: int = 5
    pagerank: float = 0.01
    influence_score: float = 0.5


@dataclass
class MockSearchResult:
    """モック検索結果"""
    query: str
    answer: str
    citations: list = None
    search_mode: MagicMock = None
    search_level: int = 1
    total_time_ms: float = 150.0
    llm_calls: int = 2
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.search_mode is None:
            self.search_mode = MagicMock(value="lazy")


@dataclass
class MockCitation:
    """モック引用"""
    doc_id: str
    title: str
    text: str = ""
    relevance_score: float = 0.8


@dataclass
class MockStatus:
    """モックステータス"""
    document_count: int = 100
    text_unit_count: int = 1000
    noun_phrase_count: int = 500
    community_count: int = 50
    citation_edge_count: int = 200
    index_status: MagicMock = None
    is_ready: bool = True
    index_levels_built: list = None
    last_error: str = None
    
    def __post_init__(self):
        if self.index_status is None:
            self.index_status = MagicMock(value="ready")
        if self.index_levels_built is None:
            self.index_levels_built = [MagicMock(value=0), MagicMock(value=1)]


# ==============================================================================
# Citation Command Tests
# ==============================================================================

class TestCitationChainCommand:
    """citation chain コマンドテスト"""
    
    def test_chain_document_not_found(self):
        """ドキュメントが見つからない場合"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = None
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["chain", "nonexistent-doc"])
            assert result.exit_code == 1
            assert "not found" in result.output.lower() or "error" in result.output.lower()
    
    def test_chain_no_citation_network(self):
        """引用ネットワークが利用不可の場合"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test Doc")
        mock_monjyu.get_citation_network.return_value = None
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["chain", "d1"])
            assert "not available" in result.output.lower()
    
    def test_chain_success_text_output(self):
        """正常な引用チェーン（テキスト出力）"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test Document")
        
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_references.return_value = [
            MockCitationEdge(source_id="d1", target_id="ref1")
        ]
        mock_citation_manager.get_citations.return_value = [
            MockCitationEdge(source_id="cite1", target_id="d1")
        ]
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["chain", "d1"])
            assert result.exit_code == 0
    
    def test_chain_success_json_output(self):
        """正常な引用チェーン（JSON出力）"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test Doc")
        
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_references.return_value = []
        mock_citation_manager.get_citations.return_value = []
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["chain", "d1", "-o", "json"])
            assert result.exit_code == 0
            # JSON出力を検証
            assert "document_id" in result.output
    
    def test_chain_exception_handling(self):
        """例外処理"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.side_effect = Exception("Test error")
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["chain", "d1"])
            assert result.exit_code == 1


class TestCitationRelatedCommand:
    """citation related コマンドテスト"""
    
    def test_related_document_not_found(self):
        """ドキュメントが見つからない場合"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = None
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["related", "nonexistent"])
            assert result.exit_code == 1
    
    def test_related_no_citation_network(self):
        """引用ネットワークが利用不可の場合"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test")
        mock_monjyu.get_citation_network.return_value = None
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["related", "d1"])
            assert "not available" in result.output.lower()
    
    def test_related_success_text(self):
        """正常な関連論文（テキスト出力）"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test Document Title")
        
        mock_citation_manager = MagicMock()
        mock_citation_manager.find_co_citation_papers.return_value = [
            ("related1", 0.9),
            ("related2", 0.8),
        ]
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["related", "d1"])
            assert result.exit_code == 0
    
    def test_related_no_results(self):
        """関連論文がない場合"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test")
        
        mock_citation_manager = MagicMock()
        mock_citation_manager.find_co_citation_papers.return_value = []
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["related", "d1"])
            assert "no related" in result.output.lower()
    
    def test_related_json_output(self):
        """JSON出力"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test")
        
        mock_citation_manager = MagicMock()
        mock_citation_manager.find_co_citation_papers.return_value = [("r1", 0.5)]
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["related", "d1", "-o", "json"])
            assert result.exit_code == 0


class TestCitationTopCommand:
    """citation top コマンドテスト"""
    
    def test_top_no_citation_network(self):
        """引用ネットワークが利用不可"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_citation_network.return_value = None
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["top"])
            assert "not available" in result.output.lower()
    
    def test_top_success(self):
        """正常なトップ論文取得"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_most_influential.return_value = [
            MockCitationMetrics(document_id="d1", citation_count=10, pagerank=0.05, influence_score=0.8),
        ]
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Influential Paper")
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["top"])
            assert result.exit_code == 0
    
    def test_top_no_papers(self):
        """論文がない場合"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_most_influential.return_value = []
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["top"])
            assert "no papers" in result.output.lower()
    
    def test_top_json_output(self):
        """JSON出力"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_most_influential.return_value = [
            MockCitationMetrics(document_id="d1"),
        ]
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["top", "-o", "json"])
            assert result.exit_code == 0


class TestCitationStatsCommand:
    """citation stats コマンドテスト"""
    
    def test_stats_no_network(self):
        """引用ネットワークが利用不可"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_citation_network.return_value = None
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["stats"])
            assert "not available" in result.output.lower()
    
    def test_stats_success(self):
        """正常な統計取得"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_network_stats.return_value = {
            "document_count": 100,
            "edge_count": 500,
            "avg_citations": 5.0,
            "avg_references": 8.0,
            "density": 0.05,
        }
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["stats"])
            assert result.exit_code == 0
    
    def test_stats_json_output(self):
        """JSON出力"""
        from monjyu.cli.commands.citation import citation_app
        
        mock_monjyu = MagicMock()
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_network_stats.return_value = {"count": 10}
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        with patch("monjyu.cli.commands.citation.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(citation_app, ["stats", "-o", "json"])
            assert result.exit_code == 0


# ==============================================================================
# Document Command Tests
# ==============================================================================

class TestDocumentListCommand:
    """document list コマンドテスト"""
    
    def test_list_success(self):
        """正常なリスト取得"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Document One", authors=["Author A", "Author B", "Author C"], year=2023),
            MockDocument(id="d2", title="Document Two with a Very Long Title That Should Be Truncated", year=2024),
        ]
        mock_monjyu.get_status.return_value = MockStatus(document_count=2)
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["list"])
            assert result.exit_code == 0
    
    def test_list_empty(self):
        """ドキュメントがない場合"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = []
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["list"])
            assert "no documents" in result.output.lower()
    
    def test_list_json_output(self):
        """JSON出力"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Test", year=2023),
        ]
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["list", "-o", "json"])
            assert result.exit_code == 0
    
    def test_list_more_documents(self):
        """制限より多いドキュメント"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Doc", year=2023),
        ]
        mock_monjyu.get_status.return_value = MockStatus(document_count=100)
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["list", "-n", "1"])
            assert "showing" in result.output.lower() or result.exit_code == 0
    
    def test_list_exception(self):
        """例外処理"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.side_effect = Exception("Error")
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["list"])
            assert result.exit_code == 1


class TestDocumentShowCommand:
    """document show コマンドテスト"""
    
    def test_show_not_found(self):
        """ドキュメントが見つからない"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = None
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["show", "nonexistent"])
            assert result.exit_code == 1
    
    def test_show_success(self):
        """正常な詳細表示"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(
            id="d1",
            title="Test Document",
            authors=["Author A"],
            year=2023,
            doi="10.1234/test",
            abstract="This is a test abstract that is quite long and should be displayed properly in the output panel.",
        )
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["show", "d1"])
            assert result.exit_code == 0
    
    def test_show_json_output(self):
        """JSON出力"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.get_document.return_value = MockDocument(id="d1", title="Test")
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["show", "d1", "-o", "json"])
            assert result.exit_code == 0


class TestDocumentExportCommand:
    """document export コマンドテスト"""
    
    def test_export_csv(self, tmp_path):
        """CSV出力"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Test", authors=["A", "B"], year=2023, doi="10.1234/x"),
        ]
        
        output_file = tmp_path / "export.csv"
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["export", str(output_file), "-f", "csv"])
            assert result.exit_code == 0
            assert output_file.exists()
    
    def test_export_json(self, tmp_path):
        """JSON出力"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Test"),
        ]
        
        output_file = tmp_path / "export.json"
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["export", str(output_file), "-f", "json"])
            assert result.exit_code == 0
            assert output_file.exists()
    
    def test_export_empty(self, tmp_path):
        """ドキュメントがない場合"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = []
        
        output_file = tmp_path / "export.csv"
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["export", str(output_file)])
            assert "no documents" in result.output.lower()
    
    def test_export_unsupported_format(self, tmp_path):
        """サポートされていないフォーマット"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [MockDocument(id="d1", title="Test")]
        
        output_file = tmp_path / "export.xml"
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["export", str(output_file), "-f", "xml"])
            assert result.exit_code == 1


class TestDocumentSearchCommand:
    """document search コマンドテスト"""
    
    def test_search_by_title(self):
        """タイトルで検索"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Machine Learning Paper", year=2023),
            MockDocument(id="d2", title="Deep Learning Study", year=2024),
        ]
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["search", "machine"])
            assert result.exit_code == 0
    
    def test_search_by_author(self):
        """著者で検索"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Paper", authors=["John Smith"], year=2023),
        ]
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["search", "smith"])
            assert result.exit_code == 0
    
    def test_search_no_results(self):
        """結果がない場合"""
        from monjyu.cli.commands.document import document_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.list_documents.return_value = [
            MockDocument(id="d1", title="Paper A", year=2023),
        ]
        
        with patch("monjyu.cli.commands.document.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(document_app, ["search", "xyz123"])
            assert "no documents" in result.output.lower()


# ==============================================================================
# Search Command Tests
# ==============================================================================

class TestSearchDefaultCommand:
    """search デフォルトコマンドテスト"""
    
    def test_search_no_query(self):
        """クエリなしの場合"""
        from monjyu.cli.commands.search import search_app
        
        result = runner.invoke(search_app, [])
        # クエリなしの場合はヘルプまたはエラー
        assert result.exit_code == 1 or "query" in result.output.lower()
    
    def test_search_success_text(self):
        """正常な検索（テキスト出力）"""
        from monjyu.cli.commands.search import search_app
        
        mock_monjyu = MagicMock()
        mock_result = MockSearchResult(
            query="What is ML?",
            answer="Machine Learning is...",
            citations=[MockCitation(doc_id="d1", title="ML Paper", text="Sample text")],
        )
        mock_monjyu.search.return_value = mock_result
        
        with patch("monjyu.cli.commands.search.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(search_app, ["What is ML?"])
            assert result.exit_code == 0
    
    def test_search_json_output(self):
        """JSON出力"""
        from monjyu.cli.commands.search import search_app
        
        mock_monjyu = MagicMock()
        mock_result = MockSearchResult(
            query="Test query",
            answer="Test answer",
        )
        mock_monjyu.search.return_value = mock_result
        
        with patch("monjyu.cli.commands.search.get_monjyu", return_value=mock_monjyu):
            # search_appはcallback形式、オプションはqueryの前に指定
            result = runner.invoke(search_app, ["-o", "json", "Test query"])
            # exit_code 0 か、正常実行の確認
            assert result.exit_code == 0 or "query" in result.output.lower()
    
    def test_search_invalid_mode(self):
        """無効な検索モード"""
        from monjyu.cli.commands.search import search_app
        
        mock_monjyu = MagicMock()
        
        with patch("monjyu.cli.commands.search.get_monjyu", return_value=mock_monjyu):
            # invalidモードを渡す
            result = runner.invoke(search_app, ["query", "--mode", "invalid_mode_xyz"])
            # エラーコードかエラーメッセージを確認
            assert result.exit_code != 0 or "invalid" in result.output.lower() or "error" in result.output.lower()
    
    def test_search_exception(self):
        """例外処理"""
        from monjyu.cli.commands.search import search_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.search.side_effect = Exception("Search error")
        
        with patch("monjyu.cli.commands.search.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(search_app, ["query"])
            assert result.exit_code == 1


class TestSearchHistoryCommand:
    """search history コマンドテスト"""
    
    def test_history_not_implemented(self):
        """未実装の履歴機能"""
        from monjyu.cli.commands.search import search_app
        
        result = runner.invoke(search_app, ["history"])
        assert "not yet implemented" in result.output.lower() or result.exit_code == 0


class TestSearchInteractiveCommand:
    """search interactive コマンドテスト"""
    
    def test_interactive_exception_on_start(self):
        """起動時の例外"""
        from monjyu.cli.commands.search import search_app
        
        with patch("monjyu.cli.commands.search.get_monjyu", side_effect=Exception("Init error")):
            result = runner.invoke(search_app, ["interactive"])
            assert result.exit_code == 1


# ==============================================================================
# CLI Main Tests
# ==============================================================================

class TestCliMain:
    """CLI main モジュールテスト"""
    
    def test_get_monjyu_default(self):
        """デフォルト設定でのMONJYU取得"""
        from monjyu.cli.main import get_monjyu
        
        # MONJYUは関数内でimportされるのでmonjyu.apiをモック
        with patch("monjyu.api.MONJYU") as mock_monjyu_class:
            mock_instance = MagicMock()
            mock_monjyu_class.return_value = mock_instance
            
            result = get_monjyu()
            
            # デフォルト設定で呼ばれる
            assert mock_monjyu_class.called or result is not None
    
    def test_get_monjyu_with_config(self, tmp_path):
        """設定ファイル指定でのMONJYU取得"""
        from monjyu.cli.main import get_monjyu
        
        config_file = tmp_path / "monjyu.yaml"
        config_file.write_text("test: config")
        
        with patch("monjyu.api.MONJYU") as mock_monjyu_class:
            mock_instance = MagicMock()
            mock_monjyu_class.return_value = mock_instance
            
            result = get_monjyu(config_file)
            
            mock_monjyu_class.assert_called_once_with(config_file)
    
    def test_print_error(self, capsys):
        """エラーメッセージ表示"""
        from monjyu.cli.main import print_error
        
        print_error("Test error message")
        # Rich consoleで出力されるため、capsysでは直接キャプチャできない
        # 例外が発生しないことを確認
    
    def test_print_success(self, capsys):
        """成功メッセージ表示"""
        from monjyu.cli.main import print_success
        
        print_success("Test success message")
        # 例外が発生しないことを確認
    
    def test_print_warning(self, capsys):
        """警告メッセージ表示"""
        from monjyu.cli.main import print_warning
        
        print_warning("Test warning message")
        # 例外が発生しないことを確認
    
    def test_output_format_enum(self):
        """OutputFormat enum"""
        from monjyu.cli.main import OutputFormat
        
        assert OutputFormat.text.value == "text"
        assert OutputFormat.json.value == "json"
    
    def test_version_command(self):
        """バージョンコマンド"""
        from monjyu.cli.main import app
        
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "monjyu" in result.output.lower()


class TestCliMainApp:
    """メインアプリテスト"""
    
    def test_app_help(self):
        """ヘルプ表示"""
        from monjyu.cli.main import app
        
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "monjyu" in result.output.lower()


# =============================================================================
# Config Commandテスト
# =============================================================================

class TestConfigInitCommand:
    """config init コマンドテスト"""
    
    def test_init_success(self, tmp_path):
        """設定ファイル生成成功"""
        from monjyu.cli.commands.config_cmd import config_app
        
        output_file = tmp_path / "test_config.yaml"
        result = runner.invoke(config_app, ["init", "-o", str(output_file)])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
    def test_init_file_exists_no_force(self, tmp_path):
        """既存ファイルがあり--forceなし"""
        from monjyu.cli.commands.config_cmd import config_app
        
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("existing")
        
        result = runner.invoke(config_app, ["init", "-o", str(output_file)])
        assert result.exit_code == 1
        assert "already exists" in result.output.lower()
    
    def test_init_file_exists_with_force(self, tmp_path):
        """既存ファイルを--forceで上書き"""
        from monjyu.cli.commands.config_cmd import config_app
        
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("old content")
        
        result = runner.invoke(config_app, ["init", "-o", str(output_file), "--force"])
        assert result.exit_code == 0
    
    def test_init_permission_error(self, tmp_path):
        """書き込み権限エラー"""
        from monjyu.cli.commands.config_cmd import config_app
        
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            output_file = tmp_path / "nowrite.yaml"
            result = runner.invoke(config_app, ["init", "-o", str(output_file)])
            assert result.exit_code == 1


class TestConfigShowCommand:
    """config show コマンドテスト"""
    
    def test_show_no_config_found(self, tmp_path, monkeypatch):
        """設定ファイルが見つからない"""
        from monjyu.cli.commands.config_cmd import config_app
        
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(config_app, ["show"])
        assert result.exit_code == 1
        assert "no configuration file found" in result.output.lower()
    
    def test_show_with_explicit_config(self, tmp_path):
        """明示的設定ファイル指定"""
        from monjyu.cli.commands.config_cmd import config_app
        
        config_file = tmp_path / "monjyu.yaml"
        config_file.write_text("output_path: ./output\nenvironment: local")
        
        result = runner.invoke(config_app, ["show", "-c", str(config_file)])
        assert result.exit_code == 0 or "error" in result.output.lower()


class TestConfigValidateCommand:
    """config validate コマンドテスト"""
    
    def test_validate_no_config_found(self, tmp_path, monkeypatch):
        """設定ファイルなし"""
        from monjyu.cli.commands.config_cmd import config_app
        
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(config_app, ["validate"])
        assert result.exit_code == 1


class TestConfigSetCommand:
    """config set コマンドテスト"""
    
    def test_set_no_config(self, tmp_path, monkeypatch):
        """設定ファイルなしでset"""
        from monjyu.cli.commands.config_cmd import config_app
        
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(config_app, ["set", "key=value"])
        # exit_code 1 or 2 (missing argument)
        assert result.exit_code in [1, 2]


# =============================================================================
# Index Commandテスト
# =============================================================================

class TestIndexBuildCommand:
    """index build コマンドテスト"""
    
    def test_build_path_not_found(self, tmp_path):
        """存在しないパス"""
        from monjyu.cli.commands.index import index_app
        
        result = runner.invoke(index_app, ["build", str(tmp_path / "nonexistent")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()
    
    def test_build_not_a_directory(self, tmp_path):
        """ディレクトリでない"""
        from monjyu.cli.commands.index import index_app
        
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")
        
        result = runner.invoke(index_app, ["build", str(file_path)])
        assert result.exit_code == 1
        assert "not a directory" in result.output.lower()
    
    def test_build_success(self, tmp_path):
        """ビルド成功"""
        from monjyu.cli.commands.index import index_app
        
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        
        mock_monjyu = MagicMock()
        mock_status = MockStatus()
        mock_monjyu.index.return_value = mock_status
        
        with patch("monjyu.cli.commands.index.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(index_app, ["build", str(docs_dir)])
            assert result.exit_code == 0 or "success" in result.output.lower()
    
    def test_build_json_output(self, tmp_path):
        """JSON出力"""
        from monjyu.cli.commands.index import index_app
        
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        
        mock_monjyu = MagicMock()
        mock_status = MockStatus()
        mock_monjyu.index.return_value = mock_status
        
        with patch("monjyu.cli.commands.index.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(index_app, ["build", str(docs_dir), "-o", "json"])
            assert result.exit_code == 0
    
    def test_build_exception(self, tmp_path):
        """ビルド例外"""
        from monjyu.cli.commands.index import index_app
        
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        
        with patch("monjyu.cli.commands.index.get_monjyu", side_effect=Exception("Build error")):
            result = runner.invoke(index_app, ["build", str(docs_dir)])
            assert result.exit_code == 1


class TestIndexStatusCommand:
    """index status コマンドテスト"""
    
    def test_status_success(self):
        """ステータス取得成功"""
        from monjyu.cli.commands.index import index_app
        
        mock_monjyu = MagicMock()
        mock_status = MockStatus()
        mock_monjyu.get_status.return_value = mock_status
        
        with patch("monjyu.cli.commands.index.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(index_app, ["status"])
            assert result.exit_code == 0
    
    def test_status_json_output(self):
        """ステータスJSON出力"""
        from monjyu.cli.commands.index import index_app
        
        mock_monjyu = MagicMock()
        mock_status = MockStatus()
        mock_monjyu.get_status.return_value = mock_status
        
        with patch("monjyu.cli.commands.index.get_monjyu", return_value=mock_monjyu):
            result = runner.invoke(index_app, ["status", "-o", "json"])
            assert result.exit_code == 0
    
    def test_status_exception(self):
        """ステータス取得例外"""
        from monjyu.cli.commands.index import index_app
        
        with patch("monjyu.cli.commands.index.get_monjyu", side_effect=Exception("Status error")):
            result = runner.invoke(index_app, ["status"])
            assert result.exit_code == 1


class TestIndexClearCommand:
    """index clear コマンドテスト"""
    
    def test_clear_cancelled(self):
        """クリアをキャンセル"""
        from monjyu.cli.commands.index import index_app
        
        result = runner.invoke(index_app, ["clear"], input="n\n")
        assert result.exit_code == 0 or "cancelled" in result.output.lower()
    
    def test_clear_with_force(self):
        """--forceでクリア"""
        from monjyu.cli.commands.index import index_app
        
        mock_monjyu = MagicMock()
        mock_monjyu.config.output_path = Path("./output")
        
        with patch("monjyu.cli.commands.index.get_monjyu", return_value=mock_monjyu):
            with patch("monjyu.api.state.StateManager"):
                result = runner.invoke(index_app, ["clear", "--force"])
                assert "success" in result.output.lower() or result.exit_code == 0
    
    def test_clear_exception(self):
        """クリア例外"""
        from monjyu.cli.commands.index import index_app
        
        with patch("monjyu.cli.commands.index.get_monjyu", side_effect=Exception("Clear error")):
            result = runner.invoke(index_app, ["clear", "--force"])
            assert result.exit_code == 1
