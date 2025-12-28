# Citation Analyzer
"""
monjyu.citation.analyzer - 引用分析

FEAT-006: Citation Network
- パス探索（BFS）
- 共引用分析 (Co-citation)
- 書誌結合分析 (Bibliographic Coupling)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator

import networkx as nx

from monjyu.citation.base import (
    CitationGraph,
    CitationPath,
    RelatedPaper,
    CitationNetworkConfig,
)


class CitationAnalyzer(ABC):
    """引用分析器の抽象基底クラス"""

    @abstractmethod
    def find_citation_paths(
        self,
        graph: CitationGraph,
        source_id: str,
        target_id: str,
        max_length: int | None = None,
    ) -> list[CitationPath]:
        """2文書間の引用パスを探索"""
        pass

    @abstractmethod
    def find_co_citations(
        self,
        graph: CitationGraph,
        doc_id: str,
        min_count: int | None = None,
    ) -> list[RelatedPaper]:
        """共引用される論文を発見"""
        pass

    @abstractmethod
    def find_bibliographic_coupling(
        self,
        graph: CitationGraph,
        doc_id: str,
        min_count: int | None = None,
    ) -> list[RelatedPaper]:
        """書誌結合関係にある論文を発見"""
        pass

    @abstractmethod
    def get_citation_context(
        self,
        graph: CitationGraph,
        doc_id: str,
        depth: int = 1,
    ) -> CitationGraph:
        """指定文書の引用コンテキスト（周辺グラフ）を取得"""
        pass


class DefaultCitationAnalyzer(CitationAnalyzer):
    """デフォルト引用分析器"""

    def __init__(self, config: CitationNetworkConfig | None = None):
        self.config = config or CitationNetworkConfig()

    def find_citation_paths(
        self,
        graph: CitationGraph,
        source_id: str,
        target_id: str,
        max_length: int | None = None,
    ) -> list[CitationPath]:
        """2文書間の引用パスを探索（BFS）"""
        max_len = max_length or self.config.max_path_length
        g = graph.graph

        if source_id not in g or target_id not in g:
            return []

        paths: list[CitationPath] = []

        try:
            # 全パスを探索（短いものから）
            for path in nx.all_simple_paths(g, source_id, target_id, cutoff=max_len):
                paths.append(CitationPath(
                    path=tuple(path),
                    length=len(path) - 1,  # エッジ数
                ))
        except nx.NetworkXNoPath:
            pass

        # 長さでソート
        paths.sort(key=lambda p: p.length)
        return paths

    def find_shortest_path(
        self,
        graph: CitationGraph,
        source_id: str,
        target_id: str,
    ) -> CitationPath | None:
        """最短引用パスを探索"""
        g = graph.graph

        if source_id not in g or target_id not in g:
            return None

        try:
            path = nx.shortest_path(g, source_id, target_id)
            return CitationPath(
                path=tuple(path),
                length=len(path) - 1,
            )
        except nx.NetworkXNoPath:
            return None

    def find_co_citations(
        self,
        graph: CitationGraph,
        doc_id: str,
        min_count: int | None = None,
    ) -> list[RelatedPaper]:
        """
        共引用分析: doc_id と一緒に引用されることが多い論文を発見

        共引用 = 同じ論文から引用されている（同じ親を持つ）
        """
        min_cnt = min_count or self.config.min_co_citation_count
        g = graph.graph

        if doc_id not in g:
            return []

        # doc_id を引用している論文（親）
        citing_papers = set(g.predecessors(doc_id))

        if not citing_papers:
            return []

        # 各親が引用している他の論文をカウント
        co_citation_count: dict[str, set[str]] = defaultdict(set)

        for citing_paper in citing_papers:
            # citing_paper が引用している論文
            cited_by_same = set(g.successors(citing_paper))
            cited_by_same.discard(doc_id)  # 自分自身を除外

            for other_doc in cited_by_same:
                co_citation_count[other_doc].add(citing_paper)

        # 結果を構築
        related: list[RelatedPaper] = []
        for other_id, shared in co_citation_count.items():
            if len(shared) >= min_cnt:
                # スコア = 共通引用元数 / doc_idの被引用数
                score = len(shared) / len(citing_papers) if citing_papers else 0.0
                related.append(RelatedPaper(
                    doc_id=other_id,
                    relation_type="co_citation",
                    similarity_score=score,
                    shared_papers=frozenset(shared),
                ))

        # スコアでソート
        related.sort(key=lambda r: r.similarity_score, reverse=True)
        return related

    def find_bibliographic_coupling(
        self,
        graph: CitationGraph,
        doc_id: str,
        min_count: int | None = None,
    ) -> list[RelatedPaper]:
        """
        書誌結合分析: doc_id と同じ論文を引用している論文を発見

        書誌結合 = 同じ論文を引用している（同じ子を持つ）
        """
        min_cnt = min_count or self.config.min_coupling_count
        g = graph.graph

        if doc_id not in g:
            return []

        # doc_id が引用している論文（子）
        referenced_papers = set(g.successors(doc_id))

        if not referenced_papers:
            return []

        # 各子を引用している他の論文をカウント
        coupling_count: dict[str, set[str]] = defaultdict(set)

        for ref_paper in referenced_papers:
            # ref_paper を引用している論文
            citing_same = set(g.predecessors(ref_paper))
            citing_same.discard(doc_id)  # 自分自身を除外

            for other_doc in citing_same:
                if other_doc in graph.internal_doc_ids:  # 内部文書のみ
                    coupling_count[other_doc].add(ref_paper)

        # 結果を構築
        related: list[RelatedPaper] = []
        for other_id, shared in coupling_count.items():
            if len(shared) >= min_cnt:
                # スコア = 共通引用数 / doc_idの引用数
                score = len(shared) / len(referenced_papers) if referenced_papers else 0.0
                related.append(RelatedPaper(
                    doc_id=other_id,
                    relation_type="bibliographic_coupling",
                    similarity_score=score,
                    shared_papers=frozenset(shared),
                ))

        # スコアでソート
        related.sort(key=lambda r: r.similarity_score, reverse=True)
        return related

    def get_citation_context(
        self,
        graph: CitationGraph,
        doc_id: str,
        depth: int = 1,
    ) -> CitationGraph:
        """指定文書の引用コンテキスト（周辺グラフ）を取得"""
        g = graph.graph

        if doc_id not in g:
            return CitationGraph()

        # BFSで周辺ノードを収集
        nodes_to_include: set[str] = {doc_id}
        current_level = {doc_id}

        for _ in range(depth):
            next_level: set[str] = set()
            for node in current_level:
                # 引用先
                next_level.update(g.successors(node))
                # 被引用元
                next_level.update(g.predecessors(node))
            nodes_to_include.update(next_level)
            current_level = next_level

        # サブグラフを作成
        subgraph = g.subgraph(nodes_to_include).copy()

        # CitationGraph として返す
        context = CitationGraph()
        context.graph = subgraph
        context.internal_doc_ids = nodes_to_include & graph.internal_doc_ids
        context.external_refs = {
            k: v for k, v in graph.external_refs.items()
            if k in nodes_to_include
        }

        return context

    def get_connected_components(
        self,
        graph: CitationGraph,
    ) -> list[set[str]]:
        """連結成分を取得"""
        # 無向グラフとして連結成分を計算
        undirected = graph.graph.to_undirected()
        return [set(c) for c in nx.connected_components(undirected)]

    def get_strongly_connected_components(
        self,
        graph: CitationGraph,
    ) -> list[set[str]]:
        """強連結成分を取得"""
        return [set(c) for c in nx.strongly_connected_components(graph.graph)]


class MockCitationAnalyzer(CitationAnalyzer):
    """テスト用のモック引用分析器"""

    def __init__(
        self,
        mock_paths: dict[tuple[str, str], list[CitationPath]] | None = None,
        mock_co_citations: dict[str, list[RelatedPaper]] | None = None,
        mock_couplings: dict[str, list[RelatedPaper]] | None = None,
    ):
        self._mock_paths = mock_paths or {}
        self._mock_co_citations = mock_co_citations or {}
        self._mock_couplings = mock_couplings or {}

    def find_citation_paths(
        self,
        graph: CitationGraph,
        source_id: str,
        target_id: str,
        max_length: int | None = None,
    ) -> list[CitationPath]:
        """モックパスを返す"""
        key = (source_id, target_id)
        return self._mock_paths.get(key, [])

    def find_co_citations(
        self,
        graph: CitationGraph,
        doc_id: str,
        min_count: int | None = None,
    ) -> list[RelatedPaper]:
        """モック共引用を返す"""
        return self._mock_co_citations.get(doc_id, [])

    def find_bibliographic_coupling(
        self,
        graph: CitationGraph,
        doc_id: str,
        min_count: int | None = None,
    ) -> list[RelatedPaper]:
        """モック書誌結合を返す"""
        return self._mock_couplings.get(doc_id, [])

    def get_citation_context(
        self,
        graph: CitationGraph,
        doc_id: str,
        depth: int = 1,
    ) -> CitationGraph:
        """空のコンテキストを返す"""
        return CitationGraph()
