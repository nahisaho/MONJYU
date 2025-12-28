# Lazy Search Engine
"""
Lazy Search エンジン - LazyGraphRAG のメインファサード

TASK-005-06: LazySearchEngine 実装
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from monjyu.lazy.base import (
    Claim,
    LazySearchConfig,
    LazySearchResult,
    LazySearchState,
    RelevanceScore,
    SearchCandidate,
    SearchLevel,
)
from monjyu.lazy.claim_extractor import ClaimExtractor
from monjyu.lazy.community_searcher import CommunitySearcher
from monjyu.lazy.iterative_deepener import IterativeDeepener
from monjyu.lazy.relevance_tester import RelevanceTester
from monjyu.search.base import Citation

if TYPE_CHECKING:
    from monjyu.index.level1 import Level1Index
    from monjyu.search.answer_synthesizer import LLMClientProtocol
    from monjyu.search.engine import VectorSearchEngine
    from monjyu.search.query_encoder import EmbeddingClientProtocol


class LazySearchEngine:
    """Lazy Search エンジン（Facade）"""

    SYNTHESIS_PROMPT = """以下の情報に基づいて、質問に回答してください。

## 収集された情報
{context}

## 質問
{query}

## 指示
- 情報を統合して簡潔に回答してください
- 使用した情報源は [1], [2] のように引用してください
- 情報が不十分な場合はその旨を明示してください
"""

    def __init__(
        self,
        vector_search_engine: "VectorSearchEngine",
        llm_client: "LLMClientProtocol",
        level1_index: "Level1Index | None" = None,
        embedding_client: "EmbeddingClientProtocol | None" = None,
        config: LazySearchConfig | None = None,
        level0_dir: str | None = None,
    ):
        """
        Args:
            vector_search_engine: ベクトル検索エンジン
            level1_index: Level 1インデックス
            embedding_client: 埋め込みクライアント
            llm_client: LLMクライアント
            config: 設定
            level0_dir: Level 0インデックスのディレクトリ
        """
        self.vector_engine = vector_search_engine
        self.llm_client = llm_client
        self.config = config or LazySearchConfig()

        # コンポーネント初期化
        self.relevance_tester = RelevanceTester(
            llm_client, max_workers=self.config.max_workers
        )
        self.claim_extractor = ClaimExtractor(
            llm_client, max_claims_per_text=self.config.max_claims_per_text
        )

        # コミュニティ検索（Level 1が利用可能な場合）
        if level1_index is not None:
            self.community_searcher = CommunitySearcher(
                level1_index=level1_index,
                embedding_client=embedding_client,
                level0_dir=level0_dir,
            )
        else:
            self.community_searcher = None

        # 動的深化器
        self.deepener = IterativeDeepener(
            llm_client=llm_client,
            community_searcher=self.community_searcher,
            max_iterations=self.config.max_iterations,
            max_llm_calls=self.config.max_llm_calls,
        )

    def search(
        self,
        query: str,
        max_level: SearchLevel = SearchLevel.LEVEL_1,
    ) -> LazySearchResult:
        """
        Lazy Search を実行

        Args:
            query: 検索クエリ
            max_level: 最大検索レベル

        Returns:
            検索結果
        """
        start_time = time.time()

        # 状態初期化
        state = LazySearchState(query=query)

        # Step 1: Level 0 初期検索
        initial_candidates = self._initial_search(query, state)

        # Step 2: 関連性テスト
        relevant_candidates = self._test_relevance(query, initial_candidates, state)

        # Step 3: クレーム抽出
        if relevant_candidates:
            self._extract_claims(query, relevant_candidates, state)

        # Step 4: 動的深化（必要に応じて）
        if max_level.value >= SearchLevel.LEVEL_1.value:
            self._iterate_deepening(state)

        # Step 5: 最終回答生成
        answer, citations = self._synthesize_answer(query, state)

        total_time = (time.time() - start_time) * 1000

        return LazySearchResult(
            query=query,
            answer=answer,
            claims=state.claims,
            citations=citations,
            search_level_reached=state.current_level,
            llm_calls=state.llm_calls,
            tokens_used=state.tokens_used,
            total_time_ms=total_time,
            final_state=state if self.config.include_debug_state else None,
        )

    def search_level0_only(self, query: str) -> LazySearchResult:
        """
        Level 0 のみで検索（Baseline RAG相当）

        Args:
            query: 検索クエリ

        Returns:
            検索結果
        """
        return self.search(query, max_level=SearchLevel.LEVEL_0)

    def _initial_search(
        self, query: str, state: LazySearchState
    ) -> list[SearchCandidate]:
        """Level 0 初期検索"""
        # ベクトル検索
        vector_results = self.vector_engine.search(
            query,
            top_k=self.config.initial_top_k,
            synthesize=False,
        )

        candidates = []
        for hit in vector_results.search_results.hits:
            candidate = SearchCandidate(
                id=hit.text_unit_id,
                source="vector",
                priority=hit.score,
                level=SearchLevel.LEVEL_0,
                text=hit.text,
                metadata={
                    "document_id": hit.document_id,
                    "document_title": hit.document_title,
                },
            )
            candidates.append(candidate)
            state.add_candidate(candidate)

        # コミュニティ検索（Level 1が有効な場合）
        if self.config.include_communities and self.community_searcher is not None:
            community_candidates = self.community_searcher.search(
                query, top_k=self.config.community_top_k
            )
            for candidate in community_candidates:
                state.add_candidate(candidate)

        state.current_level = SearchLevel.LEVEL_0
        return candidates

    def _test_relevance(
        self, query: str, candidates: list[SearchCandidate], state: LazySearchState
    ) -> list[SearchCandidate]:
        """関連性テスト"""
        if not candidates:
            return []

        results = self.relevance_tester.filter_relevant(
            query,
            candidates,
            min_relevance=self.config.min_relevance,
            parallel=self.config.parallel_relevance_test,
        )

        state.llm_calls += len(candidates)

        relevant = []
        for candidate, score in results:
            state.mark_visited(candidate)
            state.context.append(candidate.text)
            relevant.append(candidate)

        return relevant

    def _extract_claims(
        self, query: str, candidates: list[SearchCandidate], state: LazySearchState
    ) -> None:
        """クレーム抽出"""
        claims = self.claim_extractor.extract_batch(query, candidates)

        if self.config.merge_duplicates:
            # 既存クレームとマージ
            existing_texts = {c.text.lower().strip() for c in state.claims}
            for claim in claims:
                normalized = claim.text.lower().strip()
                if normalized not in existing_texts:
                    state.claims.append(claim)
                    existing_texts.add(normalized)
        else:
            state.claims.extend(claims)

        state.llm_calls += len(candidates)

    def _iterate_deepening(self, state: LazySearchState) -> None:
        """動的深化イテレーション"""
        while (
            state.iterations < self.config.max_iterations
            and self.deepener.should_deepen(state)
        ):
            self._deepen_one_iteration(state)
            state.iterations += 1

    def _deepen_one_iteration(self, state: LazySearchState) -> None:
        """1回の深化イテレーション"""
        # 次の候補を取得
        next_candidates = self.deepener.get_next_candidates(
            state, batch_size=self.config.batch_size
        )

        if not next_candidates:
            return

        # コミュニティ展開
        text_candidates = []
        for candidate in next_candidates:
            if candidate.source == "community":
                # コミュニティからTextUnitを展開
                expanded = self.deepener.expand_from_community(candidate.id, state)
                for exp_candidate in expanded:
                    state.add_candidate(exp_candidate)
                state.mark_visited(candidate)
            else:
                text_candidates.append(candidate)

        # テキスト候補の関連性テスト
        if text_candidates:
            relevant = self._test_relevance(state.query, text_candidates, state)
            if relevant:
                self._extract_claims(state.query, relevant, state)

        state.current_level = SearchLevel.LEVEL_1

    def _synthesize_answer(
        self, query: str, state: LazySearchState
    ) -> tuple[str, list[Citation]]:
        """最終回答を合成"""
        if not state.claims:
            # クレームがない場合はコンテキストから直接回答
            if state.context:
                context_text = "\n".join(
                    f"[{i + 1}] {text[:500]}" for i, text in enumerate(state.context[:10])
                )
            else:
                return "関連する情報が見つかりませんでした。", []

            prompt = self.SYNTHESIS_PROMPT.format(context=context_text, query=query)
            answer = self.llm_client.generate(prompt)
            state.llm_calls += 1
            return answer, []

        # クレームからコンテキスト構築
        context_parts = []
        source_map: dict[int, Claim] = {}

        max_claims = min(len(state.claims), self.config.max_claims_in_answer)
        for i, claim in enumerate(state.claims[:max_claims]):
            context_parts.append(f"[{i + 1}] {claim.text}")
            source_map[i + 1] = claim

        context_text = "\n".join(context_parts)

        # 回答生成
        prompt = self.SYNTHESIS_PROMPT.format(context=context_text, query=query)
        answer = self.llm_client.generate(prompt)
        state.llm_calls += 1

        # 引用抽出
        citations = self._extract_citations(answer, source_map)

        return answer, citations

    def _extract_citations(
        self, answer: str, source_map: dict[int, Claim]
    ) -> list[Citation]:
        """回答から引用を抽出"""
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))

        citations = []
        for idx in sorted(cited_indices):
            if idx in source_map:
                claim = source_map[idx]
                citations.append(
                    Citation(
                        text_unit_id=claim.source_text_unit_id,
                        document_id=claim.source_document_id,
                        document_title="",
                        text_snippet=claim.text,
                        relevance_score=claim.confidence,
                    )
                )

        return citations


def create_local_lazy_engine(
    vector_search_engine: "VectorSearchEngine",
    llm_client: "LLMClientProtocol",
    level1_index: "Level1Index | None" = None,
    embedding_client: "EmbeddingClientProtocol | None" = None,
    level0_dir: str | None = None,
    config: LazySearchConfig | None = None,
) -> LazySearchEngine:
    """
    ローカル環境用の Lazy Search エンジンを作成

    Args:
        vector_search_engine: ベクトル検索エンジン
        llm_client: LLMクライアント
        level1_index: Level 1インデックス
        embedding_client: 埋め込みクライアント
        level0_dir: Level 0インデックスのディレクトリ
        config: 設定

    Returns:
        設定済みの LazySearchEngine
    """
    return LazySearchEngine(
        vector_search_engine=vector_search_engine,
        level1_index=level1_index,
        embedding_client=embedding_client,
        llm_client=llm_client,
        config=config or LazySearchConfig(),
        level0_dir=level0_dir,
    )


class MockLazySearchEngine:
    """テスト用モック Lazy Search エンジン"""

    def __init__(
        self,
        default_answer: str = "This is a mock answer.",
        default_claims: list[dict[str, Any]] | None = None,
    ):
        """
        Args:
            default_answer: デフォルト回答
            default_claims: デフォルトクレームリスト
        """
        self.default_answer = default_answer
        self.default_claims = default_claims or [
            {
                "text": "Mock claim 1",
                "source_text_unit_id": "tu_1",
                "source_document_id": "doc_1",
            },
            {
                "text": "Mock claim 2",
                "source_text_unit_id": "tu_2",
                "source_document_id": "doc_1",
            },
        ]
        self.search_call_count = 0

    def search(
        self, query: str, max_level: SearchLevel = SearchLevel.LEVEL_1
    ) -> LazySearchResult:
        """Lazy Search を実行"""
        self.search_call_count += 1

        claims = [
            Claim(
                text=c["text"],
                source_text_unit_id=c["source_text_unit_id"],
                source_document_id=c["source_document_id"],
            )
            for c in self.default_claims
        ]

        citations = [
            Citation(
                text_unit_id=c.source_text_unit_id,
                document_id=c.source_document_id,
                document_title="Mock Document",
                text_snippet=c.text,
                relevance_score=1.0,
            )
            for c in claims
        ]

        return LazySearchResult(
            query=query,
            answer=self.default_answer,
            claims=claims,
            citations=citations,
            search_level_reached=max_level,
            llm_calls=5,
            tokens_used=500,
            total_time_ms=100.0,
        )

    def search_level0_only(self, query: str) -> LazySearchResult:
        """Level 0 のみで検索"""
        return self.search(query, max_level=SearchLevel.LEVEL_0)
