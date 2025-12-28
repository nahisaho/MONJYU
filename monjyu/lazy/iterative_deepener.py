# Iterative Deepener
"""
動的深化 - 情報の十分性を判定し、必要に応じて深化

TASK-005-03: IterativeDeepener 実装
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monjyu.lazy.base import LazySearchState, SearchCandidate, SearchLevel

if TYPE_CHECKING:
    from monjyu.lazy.community_searcher import CommunitySearcher
    from monjyu.search.answer_synthesizer import LLMClientProtocol


class IterativeDeepener:
    """動的深化"""

    SUFFICIENCY_PROMPT = """以下の情報で、クエリに十分に回答できますか？

クエリ: {query}

収集された情報:
{claims}

判定:
- SUFFICIENT: 十分な情報がある（質問に対して具体的で完全な回答が可能）
- INSUFFICIENT: もっと情報が必要（回答が不完全または曖昧になる）

回答は SUFFICIENT または INSUFFICIENT のみを出力してください。"""

    def __init__(
        self,
        llm_client: "LLMClientProtocol",
        community_searcher: "CommunitySearcher | None" = None,
        max_iterations: int = 5,
        max_llm_calls: int = 20,
        min_claims_for_check: int = 3,
    ):
        """
        Args:
            llm_client: LLMクライアント
            community_searcher: コミュニティ検索器
            max_iterations: 最大イテレーション数
            max_llm_calls: 最大LLMコール数
            min_claims_for_check: 十分性チェック開始のための最小クレーム数
        """
        self.llm_client = llm_client
        self.community_searcher = community_searcher
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.min_claims_for_check = min_claims_for_check

    def should_deepen(self, state: LazySearchState) -> bool:
        """
        深化すべきか判定

        Args:
            state: 現在の検索状態

        Returns:
            深化すべきならTrue
        """
        # 停止条件1: LLMコール上限
        if state.llm_calls >= self.max_llm_calls:
            return False

        # 停止条件2: イテレーション上限
        if state.iterations >= self.max_iterations:
            return False

        # 停止条件3: キューが空
        if not state.priority_queue:
            return False

        # クレームが少なすぎる場合は深化続行
        if state.claim_count < self.min_claims_for_check:
            return True

        # LLMで十分性を判定
        claims_text = "\n".join(f"- {c.text}" for c in state.claims[:20])
        prompt = self.SUFFICIENCY_PROMPT.format(
            query=state.query,
            claims=claims_text,
        )

        try:
            response = self.llm_client.generate(prompt, max_tokens=20)
            state.llm_calls += 1
            response = response.strip().upper()

            # 「INSUFFICIENT」が含まれていれば深化続行
            return "INSUFFICIENT" in response
        except Exception:
            # エラー時は安全のため深化停止
            return False

    def get_next_candidates(
        self, state: LazySearchState, batch_size: int = 5
    ) -> list[SearchCandidate]:
        """
        次の候補を取得

        Args:
            state: 現在の検索状態
            batch_size: 取得するバッチサイズ

        Returns:
            次の検索候補リスト
        """
        candidates = []

        for _ in range(batch_size * 2):  # 訪問済みスキップを考慮して多めに取得
            candidate = state.pop_candidate()
            if candidate is None:
                break

            # 訪問済みスキップ
            if state.is_visited(candidate):
                continue

            candidates.append(candidate)

            if len(candidates) >= batch_size:
                break

        return candidates

    def expand_from_community(
        self, community_id: str, state: LazySearchState
    ) -> list[SearchCandidate]:
        """
        コミュニティから展開

        Args:
            community_id: コミュニティID
            state: 現在の検索状態

        Returns:
            展開された検索候補リスト
        """
        if self.community_searcher is None:
            return []

        # コミュニティ内のTextUnitを取得
        text_units = self.community_searcher.get_text_units(community_id)

        candidates = []
        for tu_id, doc_id, text in text_units:
            if tu_id not in state.visited_text_units:
                candidates.append(
                    SearchCandidate(
                        id=tu_id,
                        source="community",
                        priority=0.5,  # コミュニティ展開は中程度の優先度
                        level=SearchLevel.LEVEL_1,
                        text=text,
                        metadata={"document_id": doc_id},
                    )
                )

        return candidates


class MockIterativeDeepener:
    """テスト用モック動的深化器"""

    def __init__(
        self,
        should_deepen_result: bool = True,
        max_deepening_count: int = 2,
    ):
        """
        Args:
            should_deepen_result: should_deepenのデフォルト結果
            max_deepening_count: 深化を許可する最大回数
        """
        self.should_deepen_result = should_deepen_result
        self.max_deepening_count = max_deepening_count
        self.deepen_call_count = 0

    def should_deepen(self, state: LazySearchState) -> bool:
        """深化すべきか判定"""
        # イテレーション上限
        if state.iterations >= self.max_deepening_count:
            return False

        # キューが空
        if not state.priority_queue:
            return False

        self.deepen_call_count += 1
        return self.should_deepen_result

    def get_next_candidates(
        self, state: LazySearchState, batch_size: int = 5
    ) -> list[SearchCandidate]:
        """次の候補を取得"""
        candidates = []

        for _ in range(batch_size):
            candidate = state.pop_candidate()
            if candidate is None:
                break
            if not state.is_visited(candidate):
                candidates.append(candidate)

        return candidates

    def expand_from_community(
        self, community_id: str, state: LazySearchState
    ) -> list[SearchCandidate]:
        """コミュニティから展開"""
        # モックでは空リストを返す
        return []
