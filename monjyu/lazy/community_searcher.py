# Community Searcher
"""
コミュニティ検索 - Level 1インデックスからコミュニティを検索

TASK-005-04: CommunitySearcher 実装
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

from monjyu.lazy.base import SearchCandidate, SearchLevel

if TYPE_CHECKING:
    from monjyu.graph.base import Community, NounPhraseNode
    from monjyu.index.level1 import Level1Index


class CommunitySearcher:
    """コミュニティ検索"""

    def __init__(
        self,
        level1_index: "Level1Index | None" = None,
        embedding_client: Any = None,
        level0_dir: str | Path | None = None,
    ):
        """
        Args:
            level1_index: Level 1インデックス
            embedding_client: 埋め込みクライアント（省略時はキーワードマッチング）
            level0_dir: Level 0インデックスのディレクトリ
        """
        self.index = level1_index
        self.embedding_client = embedding_client
        self.level0_dir = Path(level0_dir) if level0_dir else None

        # コミュニティ埋め込みのキャッシュ
        self._community_embeddings: dict[str, list[float]] | None = None

        # ノードIDから情報を取得するためのマップ
        self._node_map: dict[str, "NounPhraseNode"] | None = None
        self._community_map: dict[str, "Community"] | None = None

    def _ensure_maps(self) -> None:
        """マップを構築（必要に応じて）"""
        if self._node_map is None and self.index is not None:
            self._node_map = {node.id: node for node in self.index.nodes}
        if self._community_map is None and self.index is not None:
            self._community_map = {comm.id: comm for comm in self.index.communities}

    def search(self, query: str, top_k: int = 5) -> list[SearchCandidate]:
        """
        クエリに関連するコミュニティを検索

        Args:
            query: クエリ
            top_k: 返す候補数

        Returns:
            検索候補リスト
        """
        if self.index is None or not self.index.communities:
            return []

        self._ensure_maps()

        if self.embedding_client is not None:
            return self._search_by_embedding(query, top_k)
        else:
            return self._search_by_keyword(query, top_k)

    def _search_by_embedding(self, query: str, top_k: int) -> list[SearchCandidate]:
        """埋め込みベースの検索"""
        # コミュニティ埋め込みを構築
        if self._community_embeddings is None:
            self._build_community_embeddings()

        if not self._community_embeddings:
            return []

        # クエリ埋め込み
        query_embedding = self.embedding_client.embed(query)

        # 類似度計算
        scores: list[tuple[str, float]] = []
        for comm_id, emb in self._community_embeddings.items():
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((comm_id, score))

        # Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_communities = scores[:top_k]

        candidates = []
        for comm_id, score in top_communities:
            comm = self._community_map.get(comm_id)
            if comm:
                candidates.append(
                    SearchCandidate(
                        id=comm_id,
                        source="community",
                        priority=score,
                        level=SearchLevel.LEVEL_1,
                        text=", ".join(comm.representative_phrases[:5]),
                        metadata={
                            "size": comm.size,
                            "node_ids": comm.node_ids,
                            "level": comm.level,
                        },
                    )
                )

        return candidates

    def _search_by_keyword(self, query: str, top_k: int) -> list[SearchCandidate]:
        """キーワードベースの検索"""
        query_terms = set(query.lower().split())

        scores: list[tuple[str, float]] = []
        for comm in self.index.communities:
            # 代表フレーズとのマッチングスコア
            comm_terms = set()
            for phrase in comm.representative_phrases:
                comm_terms.update(phrase.lower().split())

            # Jaccard類似度的なスコア
            intersection = len(query_terms & comm_terms)
            union = len(query_terms | comm_terms)
            score = intersection / union if union > 0 else 0.0

            if score > 0:
                scores.append((comm.id, score))

        # Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_communities = scores[:top_k]

        candidates = []
        for comm_id, score in top_communities:
            comm = self._community_map.get(comm_id)
            if comm:
                candidates.append(
                    SearchCandidate(
                        id=comm_id,
                        source="community",
                        priority=score,
                        level=SearchLevel.LEVEL_1,
                        text=", ".join(comm.representative_phrases[:5]),
                        metadata={
                            "size": comm.size,
                            "node_ids": comm.node_ids,
                            "level": comm.level,
                        },
                    )
                )

        return candidates

    def get_text_units(self, community_id: str) -> list[tuple[str, str, str]]:
        """
        コミュニティに属するTextUnitを取得

        Args:
            community_id: コミュニティID

        Returns:
            (text_unit_id, document_id, text)のリスト
        """
        self._ensure_maps()

        if self._community_map is None:
            return []

        comm = self._community_map.get(community_id)
        if not comm:
            return []

        # ノードからTextUnit IDを収集
        text_unit_ids: set[str] = set()
        for node_id in comm.node_ids:
            node = self._node_map.get(node_id)
            if node:
                text_unit_ids.update(node.text_unit_ids)

        if not text_unit_ids:
            return []

        # Level 0からTextUnitを読み込み
        return self._load_text_units(list(text_unit_ids))

    def _load_text_units(
        self, text_unit_ids: list[str]
    ) -> list[tuple[str, str, str]]:
        """TextUnitを読み込み"""
        if self.level0_dir is None:
            return []

        text_units_path = self.level0_dir / "text_units.parquet"
        if not text_units_path.exists():
            return []

        try:
            table = pq.read_table(text_units_path)
            df = table.to_pandas()

            filtered = df[df["id"].isin(text_unit_ids)]

            return [
                (row["id"], row.get("document_id", ""), row.get("text", ""))
                for _, row in filtered.iterrows()
            ]
        except Exception:
            return []

    def _build_community_embeddings(self) -> None:
        """コミュニティ埋め込みを構築"""
        self._community_embeddings = {}

        if self.index is None or self.embedding_client is None:
            return

        for comm in self.index.communities:
            # 代表フレーズを結合して埋め込み
            text = " ".join(comm.representative_phrases[:10])
            if text:
                try:
                    emb = self.embedding_client.embed(text)
                    self._community_embeddings[comm.id] = emb
                except Exception:
                    pass

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """コサイン類似度"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class MockCommunitySearcher:
    """テスト用モックコミュニティ検索器"""

    def __init__(
        self,
        mock_communities: list[dict[str, Any]] | None = None,
        mock_text_units: list[tuple[str, str, str]] | None = None,
    ):
        """
        Args:
            mock_communities: モックコミュニティデータ
            mock_text_units: モックTextUnitデータ
        """
        self.mock_communities = mock_communities or [
            {
                "id": "comm_1",
                "phrases": ["machine learning", "neural network"],
                "size": 10,
            },
            {
                "id": "comm_2",
                "phrases": ["natural language", "text processing"],
                "size": 8,
            },
        ]
        self.mock_text_units = mock_text_units or [
            ("tu_1", "doc_1", "Machine learning is a subset of AI."),
            ("tu_2", "doc_1", "Neural networks are inspired by the brain."),
            ("tu_3", "doc_2", "Natural language processing enables text understanding."),
        ]
        self.search_call_count = 0

    def search(self, query: str, top_k: int = 5) -> list[SearchCandidate]:
        """クエリに関連するコミュニティを検索"""
        self.search_call_count += 1

        candidates = []
        for i, comm in enumerate(self.mock_communities[:top_k]):
            candidates.append(
                SearchCandidate(
                    id=comm["id"],
                    source="community",
                    priority=0.9 - i * 0.1,
                    level=SearchLevel.LEVEL_1,
                    text=", ".join(comm["phrases"]),
                    metadata={"size": comm["size"]},
                )
            )

        return candidates

    def get_text_units(self, community_id: str) -> list[tuple[str, str, str]]:
        """コミュニティに属するTextUnitを取得"""
        # モックでは全TextUnitを返す
        return self.mock_text_units
