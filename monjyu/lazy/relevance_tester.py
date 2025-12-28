# Relevance Tester
"""
関連性テスト - クエリとテキストの関連性をLLMで判定

TASK-005-01: RelevanceTester 実装
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from monjyu.lazy.base import RelevanceScore, SearchCandidate

if TYPE_CHECKING:
    from monjyu.search.answer_synthesizer import LLMClientProtocol


class RelevanceTester:
    """関連性テスト"""

    RELEVANCE_PROMPT = """以下のテキストが、クエリに対してどの程度関連しているか判定してください。

クエリ: {query}

テキスト:
{text}

判定基準:
- HIGH: クエリの回答に直接役立つ情報を含む
- MEDIUM: 関連するが、直接的な回答ではない
- LOW: ほとんど関連がない

回答は HIGH, MEDIUM, LOW のいずれかのみを出力してください。"""

    def __init__(
        self,
        llm_client: "LLMClientProtocol",
        max_text_length: int = 1000,
        max_workers: int = 5,
    ):
        """
        Args:
            llm_client: LLMクライアント
            max_text_length: テキストの最大長
            max_workers: 並列処理の最大ワーカー数
        """
        self.llm_client = llm_client
        self.max_text_length = max_text_length
        self.max_workers = max_workers

    def test(self, query: str, text: str) -> RelevanceScore:
        """
        単一テキストの関連性をテスト

        Args:
            query: クエリ
            text: テスト対象テキスト

        Returns:
            関連性スコア
        """
        # テキストを切り詰め
        truncated_text = text[: self.max_text_length]
        if len(text) > self.max_text_length:
            truncated_text += "..."

        prompt = self.RELEVANCE_PROMPT.format(query=query, text=truncated_text)

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            response = response.strip().upper()

            if "HIGH" in response:
                return RelevanceScore.HIGH
            elif "MEDIUM" in response:
                return RelevanceScore.MEDIUM
            else:
                return RelevanceScore.LOW
        except Exception:
            # エラー時はLOWとして扱う
            return RelevanceScore.LOW

    def test_batch(
        self, query: str, texts: list[str], parallel: bool = True
    ) -> list[RelevanceScore]:
        """
        バッチで関連性をテスト

        Args:
            query: クエリ
            texts: テスト対象テキストのリスト
            parallel: 並列処理を使用するか

        Returns:
            関連性スコアのリスト
        """
        if not texts:
            return []

        if parallel and len(texts) > 1:
            # 並列処理（APIレート制限に注意）
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [executor.submit(self.test, query, text) for text in texts]
                results = [f.result() for f in futures]
            return results
        else:
            return [self.test(query, text) for text in texts]

    def filter_relevant(
        self,
        query: str,
        candidates: list[SearchCandidate],
        min_relevance: RelevanceScore = RelevanceScore.MEDIUM,
        parallel: bool = True,
    ) -> list[tuple[SearchCandidate, RelevanceScore]]:
        """
        関連性でフィルタリング

        Args:
            query: クエリ
            candidates: 検索候補のリスト
            min_relevance: 最低関連性スコア
            parallel: 並列処理を使用するか

        Returns:
            (候補, スコア)のリスト（フィルタ済み）
        """
        if not candidates:
            return []

        texts = [c.text for c in candidates]
        scores = self.test_batch(query, texts, parallel=parallel)

        results = []
        for candidate, score in zip(candidates, scores):
            if score.value >= min_relevance.value:
                results.append((candidate, score))

        return results

    def partition_by_relevance(
        self,
        query: str,
        candidates: list[SearchCandidate],
        parallel: bool = True,
    ) -> dict[RelevanceScore, list[SearchCandidate]]:
        """
        関連性でパーティション分け

        Args:
            query: クエリ
            candidates: 検索候補のリスト
            parallel: 並列処理を使用するか

        Returns:
            関連性スコアごとの候補リスト
        """
        if not candidates:
            return {
                RelevanceScore.HIGH: [],
                RelevanceScore.MEDIUM: [],
                RelevanceScore.LOW: [],
            }

        texts = [c.text for c in candidates]
        scores = self.test_batch(query, texts, parallel=parallel)

        result: dict[RelevanceScore, list[SearchCandidate]] = {
            RelevanceScore.HIGH: [],
            RelevanceScore.MEDIUM: [],
            RelevanceScore.LOW: [],
        }

        for candidate, score in zip(candidates, scores):
            result[score].append(candidate)

        return result


class MockRelevanceTester:
    """テスト用モック関連性テスター"""

    def __init__(self, default_score: RelevanceScore = RelevanceScore.MEDIUM):
        """
        Args:
            default_score: デフォルトの関連性スコア
        """
        self.default_score = default_score
        self.call_count = 0
        self._keyword_scores: dict[str, RelevanceScore] = {}

    def set_keyword_score(self, keyword: str, score: RelevanceScore) -> None:
        """キーワードに対するスコアを設定"""
        self._keyword_scores[keyword] = score

    def test(self, query: str, text: str) -> RelevanceScore:
        """単一テキストの関連性をテスト"""
        self.call_count += 1

        # キーワードベースでスコアを決定
        for keyword, score in self._keyword_scores.items():
            if keyword.lower() in text.lower():
                return score

        return self.default_score

    def test_batch(
        self, query: str, texts: list[str], parallel: bool = True
    ) -> list[RelevanceScore]:
        """バッチで関連性をテスト"""
        return [self.test(query, text) for text in texts]

    def filter_relevant(
        self,
        query: str,
        candidates: list[SearchCandidate],
        min_relevance: RelevanceScore = RelevanceScore.MEDIUM,
        parallel: bool = True,
    ) -> list[tuple[SearchCandidate, RelevanceScore]]:
        """関連性でフィルタリング"""
        if not candidates:
            return []

        results = []
        for candidate in candidates:
            score = self.test(query, candidate.text)
            if score.value >= min_relevance.value:
                results.append((candidate, score))
        return results
