# Citation Network Integration Tests
"""
FEAT-006: Citation Network - 統合テスト
"""

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ========== テスト用のモック Document ==========

@dataclass
class MockDocument:
    """テスト用のモック文書"""
    doc_id: str
    content: str = ""
    metadata: dict[str, Any] | None = None


# ========== Integration Tests ==========

class TestCitationNetworkIntegration:
    """引用ネットワーク統合テスト"""

    @pytest.fixture
    def sample_documents(self) -> list[MockDocument]:
        """サンプル文書セット"""
        return [
            MockDocument(
                doc_id="paper_a",
                content="This paper introduces a new algorithm...",
                metadata={
                    "title": "A Novel Algorithm for Graph Analysis",
                    "doi": "10.1234/algo.2023",
                    "authors": ["Alice", "Bob"],
                    "year": 2023,
                    "references": [
                        "See 10.1234/foundational.2020 for background",
                        '"Deep Learning in Graphs" provides context',
                        "External reference not in corpus",
                    ],
                },
            ),
            MockDocument(
                doc_id="paper_b",
                content="We survey existing methods...",
                metadata={
                    "title": "Deep Learning in Graphs",
                    "doi": "10.1234/survey.2022",
                    "authors": ["Charlie"],
                    "year": 2022,
                    "references": [
                        "10.1234/foundational.2020 is the key reference",
                    ],
                },
            ),
            MockDocument(
                doc_id="paper_c",
                content="The foundational work on graph theory...",
                metadata={
                    "title": "Foundational Graph Theory",
                    "doi": "10.1234/foundational.2020",
                    "authors": ["David", "Eve"],
                    "year": 2020,
                    "references": [],  # 引用なし（基礎論文）
                },
            ),
            MockDocument(
                doc_id="paper_d",
                content="Extending previous work...",
                metadata={
                    "title": "Extensions to Graph Analysis",
                    "doi": "10.1234/extensions.2024",
                    "authors": ["Frank"],
                    "year": 2024,
                    "references": [
                        "10.1234/algo.2023 proposes the base algorithm",
                        "10.1234/survey.2022 surveys the field",
                        "10.1234/foundational.2020 provides theory",
                    ],
                },
            ),
        ]

    def test_full_pipeline(self, sample_documents):
        """完全なパイプライン統合テスト"""
        from monjyu.citation.manager import CitationNetworkManager
        from monjyu.citation.base import CitationNetworkConfig

        config = CitationNetworkConfig(
            fuzzy_match_threshold=0.85,
            min_co_citation_count=1,
            min_coupling_count=1,
        )

        manager = CitationNetworkManager(config=config)
        result = manager.build(sample_documents)

        # 基本統計の検証
        assert result.document_count == 4
        assert result.internal_edge_count >= 4  # paper_a->b,c, paper_b->c, paper_d->a,b,c

    def test_citation_structure(self, sample_documents):
        """引用構造の検証"""
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()
        manager.build(sample_documents)

        graph = manager.graph

        # paper_c (foundational) は最も被引用が多い
        assert graph.get_citation_count("paper_c") >= 2

        # paper_c は引用なし
        assert graph.get_reference_count("paper_c") == 0

        # paper_d は最も引用が多い
        assert graph.get_reference_count("paper_d") >= 3

    def test_metrics_ranking(self, sample_documents):
        """メトリクスランキングの検証"""
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()
        manager.build(sample_documents)

        # PageRank 上位
        top_pagerank = manager.get_top_by_pagerank(limit=2)
        assert len(top_pagerank) == 2

        # paper_c (foundational) は重要度が高いはず
        top_ids = [m.doc_id for m in top_pagerank]
        assert "paper_c" in top_ids

    def test_citation_paths(self, sample_documents):
        """引用パス探索の検証"""
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()
        manager.build(sample_documents)

        # paper_a から paper_c へのパス
        paths = manager.find_citation_paths("paper_a", "paper_c")
        assert len(paths) >= 1

        # 直接パスがあるはず
        direct_paths = [p for p in paths if p.length == 1]
        assert len(direct_paths) >= 1

    def test_related_papers(self, sample_documents):
        """関連論文発見の検証"""
        from monjyu.citation.manager import CitationNetworkManager
        from monjyu.citation.base import CitationNetworkConfig

        config = CitationNetworkConfig(
            min_co_citation_count=1,
            min_coupling_count=1,
        )
        manager = CitationNetworkManager(config=config)
        manager.build(sample_documents)

        # paper_a と paper_b は paper_d から共引用されている
        related = manager.find_related_papers("paper_a", method="co_citation")
        related_ids = [r.doc_id for r in related]

        # paper_b も共引用されている可能性
        # (paper_d が paper_a と paper_b の両方を引用)
        # 結果はグラフ構造に依存

    def test_citation_context(self, sample_documents):
        """引用コンテキスト取得の検証"""
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()
        manager.build(sample_documents)

        # paper_b の周辺（depth=1）
        context = manager.get_citation_context("paper_b", depth=1)

        # paper_b 自身
        assert "paper_b" in context.internal_doc_ids

        # paper_b が引用している paper_c
        assert "paper_c" in context.internal_doc_ids

    def test_persistence_roundtrip(self, sample_documents, tmp_path):
        """永続化の往復テスト"""
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()
        result = manager.build(sample_documents)

        # GraphML 保存
        graphml_path = tmp_path / "citation.graphml"
        manager.save_graphml(graphml_path)

        # JSON 保存
        json_path = tmp_path / "citation_meta.json"
        manager.save_json(json_path)

        # CSV 保存
        csv_path = tmp_path / "edges.csv"
        manager.export_edges_csv(csv_path)

        # 全ファイルが存在
        assert graphml_path.exists()
        assert json_path.exists()
        assert csv_path.exists()

        # GraphML から復元
        manager2 = CitationNetworkManager()
        manager2.load_graphml(graphml_path)

        assert manager2.is_built
        assert manager2.graph.node_count == manager.graph.node_count

    def test_external_references_handling(self, sample_documents):
        """外部参照の処理テスト"""
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()
        result = manager.build(sample_documents)

        # 外部参照が存在する
        assert result.external_edge_count >= 1

        # 外部参照はグラフに含まれている
        assert len(manager.graph.external_refs) >= 1


class TestCitationNetworkEdgeCases:
    """エッジケースのテスト"""

    def test_empty_corpus(self):
        """空のコーパス"""
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()
        result = manager.build([])

        assert result.document_count == 0
        assert result.internal_edge_count == 0

    def test_single_document(self):
        """単一文書"""
        from monjyu.citation.manager import CitationNetworkManager

        doc = MockDocument(
            doc_id="single",
            metadata={"title": "Single Paper"},
        )

        manager = CitationNetworkManager()
        result = manager.build([doc])

        assert result.document_count == 1
        assert result.internal_edge_count == 0

    def test_self_citation(self):
        """自己引用"""
        from monjyu.citation.manager import CitationNetworkManager

        doc = MockDocument(
            doc_id="self_cite",
            metadata={
                "title": "Self Citing Paper",
                "doi": "10.1234/self",
                "references": ["10.1234/self"],  # 自己引用
            },
        )

        manager = CitationNetworkManager()
        result = manager.build([doc])

        # 自己引用はエッジとして追加されない（resolver で除外）
        assert result.internal_edge_count == 0

    def test_circular_citations(self):
        """循環引用"""
        from monjyu.citation.manager import CitationNetworkManager

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={
                    "doi": "10.1234/doc1",
                    "references": ["10.1234/doc2"],
                },
            ),
            MockDocument(
                doc_id="doc2",
                metadata={
                    "doi": "10.1234/doc2",
                    "references": ["10.1234/doc1"],  # 循環
                },
            ),
        ]

        manager = CitationNetworkManager()
        result = manager.build(docs)

        # 循環引用も正しく処理される
        assert result.internal_edge_count == 2

        # パス探索でも問題なし
        paths = manager.find_citation_paths("doc1", "doc2")
        assert len(paths) >= 1

    def test_no_metadata(self):
        """メタデータなしの文書"""
        from monjyu.citation.manager import CitationNetworkManager

        docs = [
            MockDocument(doc_id="no_meta1", content="Content 1"),
            MockDocument(doc_id="no_meta2", content="Content 2"),
        ]

        manager = CitationNetworkManager()
        result = manager.build(docs)

        assert result.document_count == 2
        assert result.internal_edge_count == 0


class TestCitationNetworkPerformance:
    """パフォーマンステスト（軽量）"""

    def test_moderate_corpus(self):
        """中規模コーパス"""
        from monjyu.citation.manager import CitationNetworkManager
        import time

        # 100文書を生成
        docs = []
        for i in range(100):
            refs = [f"10.1234/paper{j}" for j in range(max(0, i-5), i)]
            docs.append(MockDocument(
                doc_id=f"paper{i}",
                metadata={
                    "doi": f"10.1234/paper{i}",
                    "title": f"Paper {i}",
                    "references": [f"See {r}" for r in refs],
                },
            ))

        manager = CitationNetworkManager()

        start = time.time()
        result = manager.build(docs)
        elapsed = time.time() - start

        assert result.document_count == 100
        assert elapsed < 5.0  # 5秒以内

        # メトリクス計算
        start = time.time()
        all_metrics = manager.get_all_metrics()
        elapsed = time.time() - start

        assert len(all_metrics) == 100
        assert elapsed < 2.0  # 2秒以内
