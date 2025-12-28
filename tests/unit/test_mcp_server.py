# MONJYU MCP Server Unit Tests
"""
FEAT-009: MCP Server - 単体テスト
FastMCPベースの新しい実装に対応
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import List


# ========== Mock Classes ==========


@dataclass
class MockCitation:
    doc_id: str = "doc_001"
    title: str = "Test Paper"
    text: str = "Some citation text"
    relevance_score: float = 0.95


@dataclass
class MockSearchResult:
    query: str = "test query"
    answer: str = "This is the answer"
    citations: list = None
    search_mode: Mock = None
    search_level: int = 1
    total_time_ms: float = 100.5
    llm_calls: int = 2
    citation_count: int = 1
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = [MockCitation()]
        if self.search_mode is None:
            self.search_mode = Mock(value="lazy")


@dataclass
class MockDocumentInfo:
    id: str = "doc_001"
    title: str = "Test Paper"
    authors: list = None
    year: int = 2023
    doi: str = "10.1234/test"
    abstract: str = "Test abstract"
    chunk_count: int = 10
    citation_count: int = 5
    reference_count: int = 3
    influence_score: float = 0.75
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = ["Author A", "Author B"]


@dataclass
class MockStatus:
    index_status: Mock = None
    is_ready: bool = True
    index_levels_built: list = None
    document_count: int = 100
    text_unit_count: int = 1000
    noun_phrase_count: int = 500
    community_count: int = 10
    citation_edge_count: int = 200
    last_error: str = None
    
    def __post_init__(self):
        if self.index_status is None:
            self.index_status = Mock(value="ready")
        if self.index_levels_built is None:
            self.index_levels_built = [Mock(value=0), Mock(value=1)]


@dataclass  
class MockCitationEdge:
    source_id: str = "doc_001"
    target_id: str = "doc_002"


@dataclass
class MockMetrics:
    document_id: str = "doc_001"
    citation_count: int = 5
    reference_count: int = 3
    pagerank: float = 0.001234
    hub_score: float = 0.5
    authority_score: float = 0.6
    influence_score: float = 0.75


def create_mock_monjyu():
    """MONJYUのモックを作成"""
    mock = Mock()
    mock.search.return_value = MockSearchResult()
    mock.get_document.return_value = MockDocumentInfo()
    mock.list_documents.return_value = [MockDocumentInfo(), MockDocumentInfo(id="doc_002")]
    mock.get_status.return_value = MockStatus()
    mock.get_citation_network.return_value = None
    return mock


# ========== Legacy Tool Definition Tests (for backward compatibility) ==========


class TestLegacyToolDefinitions:
    """レガシーツール定義のテスト（後方互換性）"""

    def test_tools_list(self):
        from monjyu.mcp_server.tools import MONJYU_TOOLS

        assert len(MONJYU_TOOLS) == 7

    def test_tool_names(self):
        from monjyu.mcp_server.tools import MONJYU_TOOLS

        tool_names = [t.name for t in MONJYU_TOOLS]
        
        assert "monjyu_search" in tool_names
        assert "monjyu_get_document" in tool_names
        assert "monjyu_list_documents" in tool_names
        assert "monjyu_citation_chain" in tool_names
        assert "monjyu_find_related" in tool_names
        assert "monjyu_status" in tool_names
        assert "monjyu_get_metrics" in tool_names

    def test_get_tool_by_name(self):
        from monjyu.mcp_server.tools import get_tool_by_name

        tool = get_tool_by_name("monjyu_search")
        assert tool is not None
        assert tool.name == "monjyu_search"

        unknown = get_tool_by_name("unknown_tool")
        assert unknown is None

    def test_search_tool_schema(self):
        from monjyu.mcp_server.tools import get_tool_by_name

        tool = get_tool_by_name("monjyu_search")
        schema = tool.inputSchema
        
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "mode" in schema["properties"]
        assert "top_k" in schema["properties"]
        assert "query" in schema["required"]


# ========== Legacy Handler Tests (for backward compatibility) ==========


class TestLegacyHandlers:
    """レガシーハンドラーのテスト（後方互換性）"""

    @pytest.mark.asyncio
    async def test_handle_search(self):
        from monjyu.mcp_server.handlers import handle_search

        mock_monjyu = create_mock_monjyu()
        args = {"query": "What is Transformer?", "mode": "lazy"}

        result = await handle_search(mock_monjyu, args)

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["query"] == "test query"
        assert response["answer"] == "This is the answer"
        assert "citations" in response
        assert "search_info" in response

    @pytest.mark.asyncio
    async def test_handle_search_missing_query(self):
        from monjyu.mcp_server.handlers import handle_search

        mock_monjyu = create_mock_monjyu()
        args = {}

        result = await handle_search(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert "error" in response

    @pytest.mark.asyncio
    async def test_handle_get_document(self):
        from monjyu.mcp_server.handlers import handle_get_document

        mock_monjyu = create_mock_monjyu()
        args = {"document_id": "doc_001"}

        result = await handle_get_document(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert response["id"] == "doc_001"
        assert response["title"] == "Test Paper"
        assert "authors" in response
        assert "citation_metrics" in response

    @pytest.mark.asyncio
    async def test_handle_get_document_not_found(self):
        from monjyu.mcp_server.handlers import handle_get_document

        mock_monjyu = create_mock_monjyu()
        mock_monjyu.get_document.return_value = None
        args = {"document_id": "nonexistent"}

        result = await handle_get_document(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert "error" in response
        assert "not found" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_list_documents(self):
        from monjyu.mcp_server.handlers import handle_list_documents

        mock_monjyu = create_mock_monjyu()
        args = {"limit": 10}

        result = await handle_list_documents(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert response["count"] == 2
        assert len(response["documents"]) == 2

    @pytest.mark.asyncio
    async def test_handle_citation_chain(self):
        from monjyu.mcp_server.handlers import handle_citation_chain

        mock_monjyu = create_mock_monjyu()
        args = {"document_id": "doc_001", "depth": 2}

        result = await handle_citation_chain(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert response["document"]["id"] == "doc_001"
        assert "references" in response
        assert "cited_by" in response

    @pytest.mark.asyncio
    async def test_handle_find_related(self):
        from monjyu.mcp_server.handlers import handle_find_related

        mock_monjyu = create_mock_monjyu()
        args = {"document_id": "doc_001", "top_k": 5}

        result = await handle_find_related(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert response["source_document"]["id"] == "doc_001"
        assert "related_papers" in response

    @pytest.mark.asyncio
    async def test_handle_status(self):
        from monjyu.mcp_server.handlers import handle_status

        mock_monjyu = create_mock_monjyu()
        args = {}

        result = await handle_status(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert response["index_status"] == "ready"
        assert response["is_ready"] is True
        assert "statistics" in response
        assert response["statistics"]["documents"] == 100

    @pytest.mark.asyncio
    async def test_handle_get_metrics(self):
        from monjyu.mcp_server.handlers import handle_get_metrics

        mock_monjyu = create_mock_monjyu()
        args = {"document_id": "doc_001"}

        result = await handle_get_metrics(mock_monjyu, args)

        response = json.loads(result[0].text)
        assert response["document"]["id"] == "doc_001"
        assert "metrics" in response


# ========== Dispatcher Tests ==========


class TestDispatcher:
    """ディスパッチャーのテスト"""

    @pytest.mark.asyncio
    async def test_dispatch_search(self):
        from monjyu.mcp_server.handlers import dispatch_tool

        mock_monjyu = create_mock_monjyu()

        result = await dispatch_tool("monjyu_search", {"query": "test"}, mock_monjyu)

        response = json.loads(result[0].text)
        assert "query" in response

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        from monjyu.mcp_server.handlers import dispatch_tool

        mock_monjyu = create_mock_monjyu()

        result = await dispatch_tool("unknown_tool", {}, mock_monjyu)

        response = json.loads(result[0].text)
        assert "error" in response
        assert "Unknown tool" in response["error"]


# ========== FastMCP Server Tests ==========


class TestFastMCPServer:
    """FastMCPベースのサーバーテスト"""

    def test_mcp_instance_import(self):
        """MCPインスタンスのインポートテスト"""
        from monjyu.mcp_server import mcp
        assert mcp is not None

    def test_server_functions_import(self):
        """サーバー関数のインポートテスト"""
        from monjyu.mcp_server import main, run, get_monjyu, reset_monjyu, set_monjyu
        assert callable(main)
        assert callable(run)
        assert callable(get_monjyu)
        assert callable(reset_monjyu)
        assert callable(set_monjyu)

    def test_get_monjyu(self):
        """MONJYUインスタンス取得テスト"""
        from monjyu.mcp_server.server import get_monjyu, reset_monjyu

        reset_monjyu()
        monjyu = get_monjyu()
        assert monjyu is not None

    def test_set_monjyu(self):
        """MONJYUインスタンス設定テスト"""
        from monjyu.mcp_server.server import set_monjyu, get_monjyu, reset_monjyu

        mock = Mock()
        set_monjyu(mock)
        
        result = get_monjyu()
        assert result is mock
        
        reset_monjyu()

    def test_json_format(self):
        """JSON形式変換テスト"""
        from monjyu.mcp_server.server import json_format

        data = {"key": "value", "number": 42}
        result = json_format(data)

        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_error_format(self):
        """エラー形式変換テスト"""
        from monjyu.mcp_server.server import error_format

        result = error_format("Test error message")

        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error"] == "Test error message"


# ========== FastMCP Tool Tests ==========


class TestFastMCPTools:
    """FastMCPツール関数のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_monjyu_search_basic(self):
        """検索ツールの基本テスト"""
        from monjyu.mcp_server.server import monjyu_search

        result = await monjyu_search("What is Transformer?", mode="lazy", top_k=10)

        response = json.loads(result)
        assert response["query"] == "test query"
        assert response["answer"] == "This is the answer"
        assert "citations" in response
        assert "search_info" in response

    @pytest.mark.asyncio
    async def test_monjyu_search_empty_query(self):
        """空クエリでエラーを返す"""
        from monjyu.mcp_server.server import monjyu_search

        result = await monjyu_search("", mode="lazy", top_k=10)

        response = json.loads(result)
        assert "error" in response
        assert "Query is required" in response["error"]

    @pytest.mark.asyncio
    async def test_monjyu_get_document_success(self):
        """ドキュメント取得成功テスト"""
        from monjyu.mcp_server.server import monjyu_get_document

        result = await monjyu_get_document("doc_001")

        response = json.loads(result)
        assert response["id"] == "doc_001"
        assert response["title"] == "Test Paper"
        assert "authors" in response
        assert "citation_metrics" in response

    @pytest.mark.asyncio
    async def test_monjyu_get_document_not_found(self):
        """ドキュメント未発見テスト"""
        from monjyu.mcp_server.server import monjyu_get_document

        self.mock_monjyu.get_document.return_value = None

        result = await monjyu_get_document("nonexistent")

        response = json.loads(result)
        assert "error" in response
        assert "not found" in response["error"]

    @pytest.mark.asyncio
    async def test_monjyu_get_document_empty_id(self):
        """空IDでエラーを返す"""
        from monjyu.mcp_server.server import monjyu_get_document

        result = await monjyu_get_document("")

        response = json.loads(result)
        assert "error" in response

    @pytest.mark.asyncio
    async def test_monjyu_list_documents(self):
        """ドキュメント一覧取得テスト"""
        from monjyu.mcp_server.server import monjyu_list_documents

        result = await monjyu_list_documents(limit=20, offset=0)

        response = json.loads(result)
        assert "documents" in response
        assert response["count"] == 2
        assert response["offset"] == 0
        assert response["limit"] == 20

    @pytest.mark.asyncio
    async def test_monjyu_list_documents_with_offset(self):
        """オフセット付き一覧取得テスト"""
        from monjyu.mcp_server.server import monjyu_list_documents

        result = await monjyu_list_documents(limit=10, offset=1)

        response = json.loads(result)
        assert response["offset"] == 1
        # オフセット1で2件中1件が返る
        assert response["count"] == 1

    @pytest.mark.asyncio
    async def test_monjyu_list_documents_limit_clamping(self):
        """limit の範囲制限テスト"""
        from monjyu.mcp_server.server import monjyu_list_documents

        result = await monjyu_list_documents(limit=200, offset=0)

        response = json.loads(result)
        # 100にクランプされる
        assert response["limit"] == 100

    @pytest.mark.asyncio
    async def test_monjyu_status(self):
        """ステータス取得テスト"""
        from monjyu.mcp_server.server import monjyu_status

        result = await monjyu_status()

        response = json.loads(result)
        assert response["index_status"] == "ready"
        assert response["is_ready"] is True
        assert "levels_built" in response
        assert "statistics" in response
        assert response["statistics"]["documents"] == 100

    @pytest.mark.asyncio
    async def test_monjyu_citation_chain(self):
        """引用チェーン取得テスト"""
        from monjyu.mcp_server.server import monjyu_citation_chain

        result = await monjyu_citation_chain("doc_001", depth=1)

        response = json.loads(result)
        assert "document" in response
        assert response["document"]["id"] == "doc_001"
        assert "references" in response
        assert "cited_by" in response
        assert response["depth"] == 1

    @pytest.mark.asyncio
    async def test_monjyu_citation_chain_empty_id(self):
        """空IDでエラーを返す"""
        from monjyu.mcp_server.server import monjyu_citation_chain

        result = await monjyu_citation_chain("", depth=1)

        response = json.loads(result)
        assert "error" in response

    @pytest.mark.asyncio
    async def test_monjyu_find_related(self):
        """関連論文検索テスト"""
        from monjyu.mcp_server.server import monjyu_find_related

        result = await monjyu_find_related("doc_001", top_k=10)

        response = json.loads(result)
        assert "source_document" in response
        assert "related_papers" in response
        assert response["count"] >= 0

    @pytest.mark.asyncio
    async def test_monjyu_get_metrics(self):
        """メトリクス取得テスト"""
        from monjyu.mcp_server.server import monjyu_get_metrics

        result = await monjyu_get_metrics("doc_001")

        response = json.loads(result)
        assert "document" in response
        assert "metrics" in response
        assert "citation_count" in response["metrics"]
        assert "pagerank" in response["metrics"]

    @pytest.mark.asyncio
    async def test_monjyu_get_metrics_not_found(self):
        """ドキュメント未発見テスト"""
        from monjyu.mcp_server.server import monjyu_get_metrics

        self.mock_monjyu.get_document.return_value = None

        result = await monjyu_get_metrics("nonexistent")

        response = json.loads(result)
        assert "error" in response


# ========== Module Export Tests ==========


class TestModuleExports:
    """モジュールエクスポートのテスト"""

    def test_exports_from_init(self):
        """__init__.py からのエクスポートテスト"""
        from monjyu.mcp_server import mcp, main, run, get_monjyu, reset_monjyu, set_monjyu

        assert mcp is not None
        assert callable(main)
        assert callable(run)
        assert callable(get_monjyu)
        assert callable(reset_monjyu)
        assert callable(set_monjyu)

    def test_main_http_export(self):
        """main_http関数のエクスポートテスト"""
        from monjyu.mcp_server import main_http

        assert callable(main_http)


# ========== HTTP Transport Tests ==========


class TestHTTPTransport:
    """HTTPトランスポートのテスト"""

    def test_main_http_function_exists(self):
        """main_http関数が存在することを確認"""
        from monjyu.mcp_server.server import main_http

        assert callable(main_http)

    def test_main_http_signature(self):
        """main_http関数のシグネチャを確認"""
        import inspect
        from monjyu.mcp_server.server import main_http

        sig = inspect.signature(main_http)
        params = list(sig.parameters.keys())

        # host と port パラメータがあること
        assert "host" in params
        assert "port" in params

    def test_main_http_defaults(self):
        """main_http関数のデフォルト値を確認"""
        import inspect
        from monjyu.mcp_server.server import main_http

        sig = inspect.signature(main_http)
        params = sig.parameters

        # デフォルト値の確認
        assert params["host"].default == "127.0.0.1"
        assert params["port"].default == 8080


class TestCLIArguments:
    """CLIアーギュメントのテスト"""

    def test_help_includes_http_option(self, capsys):
        """--helpに--httpオプションが含まれることを確認"""
        import sys
        from monjyu.mcp_server.server import run

        old_argv = sys.argv
        try:
            sys.argv = ["monjyu-mcp", "--help"]
            run()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        assert "--http" in captured.out

    def test_help_includes_host_option(self, capsys):
        """--helpに--hostオプションが含まれることを確認"""
        import sys
        from monjyu.mcp_server.server import run

        old_argv = sys.argv
        try:
            sys.argv = ["monjyu-mcp", "--help"]
            run()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        assert "--host" in captured.out

    def test_help_includes_port_option(self, capsys):
        """--helpに--portオプションが含まれることを確認"""
        import sys
        from monjyu.mcp_server.server import run

        old_argv = sys.argv
        try:
            sys.argv = ["monjyu-mcp", "--help"]
            run()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        assert "--port" in captured.out

    def test_version_displays(self, capsys):
        """--versionでバージョンが表示されることを確認"""
        import sys
        from monjyu.mcp_server.server import run

        old_argv = sys.argv
        try:
            sys.argv = ["monjyu-mcp", "--version"]
            run()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        # バージョン番号を含む
        assert "0." in captured.out


# ========== FastMCP Resource Tests ==========


class TestFastMCPResources:
    """FastMCPリソース関数のテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        # index_levels 属性を追加
        self.mock_monjyu.get_status.return_value.index_levels = ["level0", "level1"]
        self.mock_monjyu.get_status.return_value.entity_count = 500
        self.mock_monjyu.get_status.return_value.relationship_count = 2000
        self.mock_monjyu.get_status.return_value.last_updated = None
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_resource_index_status(self):
        """インデックスステータスリソースのテスト"""
        from monjyu.mcp_server.server import resource_index_status

        result = await resource_index_status()

        response = json.loads(result)
        assert response["type"] == "index_status"
        assert response["is_ready"] is True
        assert response["document_count"] == 100

    @pytest.mark.asyncio
    async def test_resource_documents_list(self):
        """ドキュメント一覧リソースのテスト"""
        from monjyu.mcp_server.server import resource_documents_list

        result = await resource_documents_list()

        response = json.loads(result)
        assert "documents" in response
        assert response["count"] == 2

    @pytest.mark.asyncio
    async def test_resource_document_detail(self):
        """ドキュメント詳細リソースのテスト"""
        from monjyu.mcp_server.server import resource_document_detail

        result = await resource_document_detail("doc_001")

        response = json.loads(result)
        assert response["id"] == "doc_001"
        assert response["title"] == "Test Paper"

    @pytest.mark.asyncio
    async def test_resource_document_detail_not_found(self):
        """ドキュメント未発見リソースのテスト"""
        from monjyu.mcp_server.server import resource_document_detail

        self.mock_monjyu.get_document.return_value = None

        result = await resource_document_detail("nonexistent")

        response = json.loads(result)
        assert "error" in response

    @pytest.mark.asyncio
    async def test_resource_document_content(self):
        """ドキュメントコンテンツリソースのテスト"""
        from monjyu.mcp_server.server import resource_document_content

        # テキストユニットのモック
        mock_tu = Mock()
        mock_tu.id = "tu_001"
        mock_tu.text = "Test content"
        mock_tu.chunk_index = 0
        self.mock_monjyu.get_text_units.return_value = [mock_tu]

        result = await resource_document_content("doc_001")

        response = json.loads(result)
        assert response["type"] == "document_content"
        assert "text_units" in response

    @pytest.mark.asyncio
    async def test_resource_document_citations(self):
        """引用リソースのテスト"""
        from monjyu.mcp_server.server import resource_document_citations

        result = await resource_document_citations("doc_001")

        response = json.loads(result)
        assert response["document_id"] == "doc_001"
        assert "references" in response or "error" in response

    @pytest.mark.asyncio
    async def test_resource_citation_network(self):
        """ネットワークリソースのテスト"""
        from monjyu.mcp_server.server import resource_citation_network

        result = await resource_citation_network()

        response = json.loads(result)
        # ネットワークが存在しない場合でもエラーにならない
        assert "type" in response or "error" in response


# ========== FastMCP Prompt Tests ==========


class TestFastMCPPrompts:
    """FastMCPプロンプト関数のテスト"""

    @pytest.mark.asyncio
    async def test_prompt_literature_review(self):
        """文献レビュープロンプトのテスト"""
        from monjyu.mcp_server.server import literature_review

        result = await literature_review("deep learning")

        assert isinstance(result, str)
        assert "deep learning" in result
        # プロンプトに必要な要素が含まれていることを確認
        assert "monjyu_search" in result

    @pytest.mark.asyncio
    async def test_prompt_literature_review_with_focus(self):
        """フォーカス指定の文献レビュープロンプトのテスト"""
        from monjyu.mcp_server.server import literature_review

        result = await literature_review("transformer", focus_area="applications")

        assert "transformer" in result
        assert "applications" in result

    @pytest.mark.asyncio
    async def test_prompt_paper_summary(self):
        """論文要約プロンプトのテスト"""
        from monjyu.mcp_server.server import paper_summary

        result = await paper_summary(document_id="doc_001")

        assert isinstance(result, str)
        assert "doc_001" in result

    @pytest.mark.asyncio
    async def test_prompt_paper_summary_with_title(self):
        """タイトル指定の論文要約プロンプトのテスト"""
        from monjyu.mcp_server.server import paper_summary

        result = await paper_summary(title="Attention Is All You Need")

        assert "Attention Is All You Need" in result

    @pytest.mark.asyncio
    async def test_prompt_compare_papers(self):
        """論文比較プロンプトのテスト"""
        from monjyu.mcp_server.server import compare_papers

        result = await compare_papers(paper_ids="doc_001, doc_002")

        assert isinstance(result, str)
        assert "doc_001" in result
        assert "doc_002" in result

    @pytest.mark.asyncio
    async def test_prompt_compare_papers_with_topic(self):
        """トピック指定の論文比較プロンプトのテスト"""
        from monjyu.mcp_server.server import compare_papers

        result = await compare_papers(topic="transformer architecture")

        assert "transformer architecture" in result

    @pytest.mark.asyncio
    async def test_prompt_research_question(self):
        """研究質問プロンプトのテスト"""
        from monjyu.mcp_server.server import research_question

        result = await research_question("artificial intelligence")

        assert isinstance(result, str)
        assert "artificial intelligence" in result

    @pytest.mark.asyncio
    async def test_prompt_research_question_with_interest(self):
        """関心指定の研究質問プロンプトのテスト"""
        from monjyu.mcp_server.server import research_question

        result = await research_question(
            "machine learning",
            current_interest="explainability"
        )

        assert "machine learning" in result
        assert "explainability" in result

    @pytest.mark.asyncio
    async def test_prompt_citation_analysis(self):
        """引用分析プロンプトのテスト"""
        from monjyu.mcp_server.server import citation_analysis

        result = await citation_analysis(document_id="doc_001")

        assert isinstance(result, str)
        assert "doc_001" in result

    @pytest.mark.asyncio
    async def test_prompt_citation_analysis_full_network(self):
        """ネットワーク全体の引用分析プロンプトのテスト"""
        from monjyu.mcp_server.server import citation_analysis

        result = await citation_analysis()

        assert isinstance(result, str)
        # ネットワーク全体分析の指示が含まれる
        assert "monjyu://citation-network" in result or "monjyu_status" in result


# ========== Integration Tests ==========


class TestToolResourceConsistency:
    """ツールとリソースの一貫性テスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        # index_levels 属性を追加
        self.mock_monjyu.get_status.return_value.index_levels = ["level0", "level1"]
        self.mock_monjyu.get_status.return_value.entity_count = 500
        self.mock_monjyu.get_status.return_value.relationship_count = 2000
        self.mock_monjyu.get_status.return_value.last_updated = None
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_status_tool_and_resource_consistency(self):
        """ステータスのツールとリソースが一貫していることを確認"""
        from monjyu.mcp_server.server import monjyu_status, resource_index_status

        tool_result = await monjyu_status()
        resource_result = await resource_index_status()

        tool_data = json.loads(tool_result)
        resource_data = json.loads(resource_result)

        # 同じドキュメント数
        assert tool_data["statistics"]["documents"] == resource_data["document_count"]
        # 同じステータス
        assert tool_data["is_ready"] == resource_data["is_ready"]

    @pytest.mark.asyncio
    async def test_document_tool_and_resource_consistency(self):
        """ドキュメント取得のツールとリソースが一貫していることを確認"""
        from monjyu.mcp_server.server import monjyu_get_document, resource_document_detail

        tool_result = await monjyu_get_document("doc_001")
        resource_result = await resource_document_detail("doc_001")

        tool_data = json.loads(tool_result)
        resource_data = json.loads(resource_result)

        # 同じID
        assert tool_data["id"] == resource_data["id"]
        # 同じタイトル
        assert tool_data["title"] == resource_data["title"]


# ========== Error Handling Tests ==========


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """各テストの前にモックをセットアップ"""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_mock_monjyu()
        # index_levels 属性を追加
        self.mock_monjyu.get_status.return_value.index_levels = ["level0", "level1"]
        self.mock_monjyu.get_status.return_value.entity_count = 500
        self.mock_monjyu.get_status.return_value.relationship_count = 2000
        self.mock_monjyu.get_status.return_value.last_updated = None
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()

    @pytest.mark.asyncio
    async def test_search_exception_handling(self):
        """検索中の例外ハンドリング"""
        from monjyu.mcp_server.server import monjyu_search

        self.mock_monjyu.search.side_effect = Exception("Search failed")

        result = await monjyu_search("test query")
        response = json.loads(result)

        assert "error" in response

    @pytest.mark.asyncio
    async def test_document_exception_handling(self):
        """ドキュメント取得中の例外ハンドリング"""
        from monjyu.mcp_server.server import monjyu_get_document

        self.mock_monjyu.get_document.side_effect = Exception("Database error")

        result = await monjyu_get_document("doc_001")
        response = json.loads(result)

        assert "error" in response

    @pytest.mark.asyncio
    async def test_resource_exception_handling(self):
        """リソースアクセス中の例外ハンドリング"""
        from monjyu.mcp_server.server import resource_index_status

        self.mock_monjyu.get_status.side_effect = Exception("Status unavailable")

        result = await resource_index_status()
        response = json.loads(result)

        assert "error" in response
