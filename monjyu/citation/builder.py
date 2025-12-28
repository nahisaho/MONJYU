# Citation Graph Builder
"""
monjyu.citation.builder - 引用グラフ構築

FEAT-006: Citation Network
- NetworkX DiGraph を使用
- 内部/外部エッジの区別
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monjyu.document.base import Document

from monjyu.citation.base import (
    ReferenceMatchStatus,
    CitationEdge,
    CitationGraph,
    CitationNetworkConfig,
)
from monjyu.citation.resolver import (
    ReferenceResolver,
    DefaultReferenceResolver,
    generate_external_ref_key,
)


class CitationGraphBuilder(ABC):
    """引用グラフ構築器の抽象基底クラス"""

    @abstractmethod
    def build(self, documents: list[Document]) -> CitationGraph:
        """文書から引用グラフを構築"""
        pass

    @abstractmethod
    def add_document(self, graph: CitationGraph, document: Document) -> CitationGraph:
        """既存グラフに文書を追加"""
        pass


class DefaultCitationGraphBuilder(CitationGraphBuilder):
    """デフォルト引用グラフ構築器"""

    def __init__(
        self,
        resolver: ReferenceResolver | None = None,
        config: CitationNetworkConfig | None = None,
    ):
        self.config = config or CitationNetworkConfig()
        self.resolver = resolver or DefaultReferenceResolver(self.config)

    def build(self, documents: list[Document]) -> CitationGraph:
        """文書から引用グラフを構築"""
        # 参照解決器のインデックスを構築
        self.resolver.build_index(documents)

        # グラフを初期化
        graph = CitationGraph()

        # 文書ノードを追加
        for doc in documents:
            metadata = {
                "title": doc.metadata.get("title", "") if doc.metadata else "",
                "doi": doc.metadata.get("doi") if doc.metadata else None,
            }
            graph.add_document(doc.doc_id, metadata)

        # 引用エッジを追加
        for doc in documents:
            self._process_document_references(graph, doc)

        return graph

    def add_document(self, graph: CitationGraph, document: Document) -> CitationGraph:
        """既存グラフに文書を追加"""
        # 文書ノードを追加
        metadata = {
            "title": document.metadata.get("title", "") if document.metadata else "",
            "doi": document.metadata.get("doi") if document.metadata else None,
        }
        graph.add_document(document.doc_id, metadata)

        # 引用エッジを処理
        self._process_document_references(graph, document)

        return graph

    def _process_document_references(
        self,
        graph: CitationGraph,
        document: Document,
    ) -> None:
        """文書の参照を処理してエッジを追加"""
        # メタデータから参照リストを取得
        references: list[str] = []
        if document.metadata:
            refs = document.metadata.get("references", [])
            if isinstance(refs, list):
                references = refs

        if not references:
            return

        # 参照を解決
        resolved = self.resolver.resolve_batch(document.doc_id, references)

        # エッジを追加
        for ref in resolved:
            if ref.status == ReferenceMatchStatus.UNRESOLVED:
                # 外部参照
                ext_key = generate_external_ref_key(ref.raw_reference)
                if ext_key not in graph.external_refs:
                    graph.add_external_reference(ext_key, ref.raw_reference)

                edge = CitationEdge(
                    source_id=document.doc_id,
                    target_id=ext_key,
                    is_internal=False,
                    confidence=0.0,
                    reference_text=ref.raw_reference,
                    match_status=ref.status,
                )
            else:
                # 内部参照
                if ref.target_doc_id is None:
                    continue

                edge = CitationEdge(
                    source_id=document.doc_id,
                    target_id=ref.target_doc_id,
                    is_internal=True,
                    confidence=ref.confidence,
                    reference_text=ref.raw_reference,
                    match_status=ref.status,
                )

            graph.add_citation_edge(edge)


class MockCitationGraphBuilder(CitationGraphBuilder):
    """テスト用のモック引用グラフ構築器"""

    def __init__(self, mock_graph: CitationGraph | None = None):
        self._mock_graph = mock_graph

    def build(self, documents: list[Document]) -> CitationGraph:
        """モックグラフを返す"""
        if self._mock_graph:
            return self._mock_graph

        # 文書のみのグラフを作成（エッジなし）
        graph = CitationGraph()
        for doc in documents:
            graph.add_document(doc.doc_id)
        return graph

    def add_document(self, graph: CitationGraph, document: Document) -> CitationGraph:
        """モックでは単にノードを追加"""
        graph.add_document(document.doc_id)
        return graph
