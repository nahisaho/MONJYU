"""GlobalSearch implementation with map-reduce pattern."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

from .prompts import get_map_prompt, get_reduce_prompt
from .types import (
    CommunityInfo,
    GlobalSearchConfig,
    GlobalSearchResult,
    MapResult,
)


class LLMClientProtocol(Protocol):
    """LLMクライアントのプロトコル"""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """プロンプトから応答を生成"""
        ...

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        ...


class CommunityStoreProtocol(Protocol):
    """コミュニティストアのプロトコル"""

    def get_communities_by_level(self, level: int) -> List[CommunityInfo]:
        """指定レベルのコミュニティを取得"""
        ...

    def get_top_communities(
        self, level: int, top_k: int
    ) -> List[CommunityInfo]:
        """上位k個のコミュニティを取得"""
        ...


@dataclass
class GlobalSearch:
    """Map-Reduceパターンによるグローバル検索
    
    コミュニティレポートを使用して広範な質問に回答します。
    
    Attributes:
        llm_client: LLMクライアント
        community_store: コミュニティストア
        config: 検索設定
    """

    llm_client: LLMClientProtocol
    community_store: CommunityStoreProtocol
    config: GlobalSearchConfig = field(default_factory=GlobalSearchConfig)

    def search(
        self,
        query: str,
        level: Optional[int] = None,
        config: Optional[GlobalSearchConfig] = None,
    ) -> GlobalSearchResult:
        """グローバル検索を実行
        
        Args:
            query: 検索クエリ
            level: コミュニティレベル（省略時はconfig値）
            config: 一時的な設定（省略時はインスタンス設定）
            
        Returns:
            GlobalSearchResult: 検索結果
        """
        start_time = time.time()
        effective_config = config or self.config
        effective_level = level if level is not None else effective_config.community_level

        # コミュニティを取得
        communities = self.community_store.get_top_communities(
            level=effective_level,
            top_k=effective_config.top_k_communities,
        )

        if not communities:
            return GlobalSearchResult(
                query=query,
                answer="No community data available for the specified level.",
                communities_used=[],
                map_results=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                tokens_used=0,
                community_level=effective_level,
            )

        # Map-Reduce実行
        if effective_config.map_reduce_enabled:
            map_results = self._map_phase(query, communities, effective_config)
            answer, reduce_tokens = self._reduce_phase(
                query, map_results, effective_config
            )
            total_tokens = sum(r.tokens_used for r in map_results) + reduce_tokens
        else:
            # 直接回答（コンテキスト連結）
            map_results = []
            answer, total_tokens = self._direct_answer(
                query, communities, effective_config
            )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return GlobalSearchResult(
            query=query,
            answer=answer,
            communities_used=communities,
            map_results=map_results,
            processing_time_ms=processing_time_ms,
            tokens_used=total_tokens,
            community_level=effective_level,
        )

    def _map_phase(
        self,
        query: str,
        communities: List[CommunityInfo],
        config: GlobalSearchConfig,
    ) -> List[MapResult]:
        """Mapフェーズ: 各コミュニティから部分回答を生成
        
        Args:
            query: 検索クエリ
            communities: コミュニティリスト
            config: 検索設定
            
        Returns:
            部分回答リスト
        """
        map_results: List[MapResult] = []
        prompt_template = get_map_prompt(config.response_language)

        for community in communities:
            # コンテキストトークン数チェック
            context_text = self._format_community_context(community)
            context_tokens = self.llm_client.count_tokens(context_text)
            
            if context_tokens > config.max_context_tokens:
                # トークン制限を超える場合はスキップまたは切り詰め
                continue

            # プロンプト生成
            prompt = prompt_template.format(
                title=community.title,
                summary=community.summary,
                findings="\n".join(f"- {f}" for f in community.findings),
                entities=", ".join(community.key_entities[:10]),
                query=query,
            )

            # LLM呼び出し
            partial_answer = self.llm_client.generate(
                prompt,
                temperature=config.temperature,
            )

            # 関連性スコアを計算（簡易実装）
            relevance_score = self._calculate_relevance(partial_answer)

            tokens_used = self.llm_client.count_tokens(prompt + partial_answer)

            map_results.append(
                MapResult(
                    community_id=community.community_id,
                    community_title=community.title,
                    partial_answer=partial_answer,
                    relevance_score=relevance_score,
                    tokens_used=tokens_used,
                )
            )

        # 関連性スコアでソート
        map_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return map_results

    def _reduce_phase(
        self,
        query: str,
        map_results: List[MapResult],
        config: GlobalSearchConfig,
    ) -> tuple[str, int]:
        """Reduceフェーズ: 部分回答を統合
        
        Args:
            query: 検索クエリ
            map_results: 部分回答リスト
            config: 検索設定
            
        Returns:
            (統合回答, 使用トークン数)
        """
        if not map_results:
            return "No relevant information found in the community data.", 0

        # 関連性のある回答のみフィルタ
        relevant_results = [
            r for r in map_results
            if r.relevance_score > 0.1 and "no relevant information" not in r.partial_answer.lower()
        ]

        if not relevant_results:
            return "No relevant information found across the analyzed communities.", 0

        # 部分回答を整形
        partial_answers_text = "\n\n".join(
            f"[{r.community_title}]\n{r.partial_answer}"
            for r in relevant_results
        )

        prompt_template = get_reduce_prompt(config.response_language)
        prompt = prompt_template.format(
            query=query,
            partial_answers=partial_answers_text,
        )

        # LLM呼び出し
        answer = self.llm_client.generate(
            prompt,
            temperature=config.temperature,
        )

        tokens_used = self.llm_client.count_tokens(prompt + answer)

        return answer, tokens_used

    def _direct_answer(
        self,
        query: str,
        communities: List[CommunityInfo],
        config: GlobalSearchConfig,
    ) -> tuple[str, int]:
        """Map-Reduceなしの直接回答
        
        Args:
            query: 検索クエリ
            communities: コミュニティリスト
            config: 検索設定
            
        Returns:
            (回答, 使用トークン数)
        """
        # コンテキストを連結
        context_parts = []
        total_tokens = 0

        for community in communities:
            context = self._format_community_context(community)
            context_tokens = self.llm_client.count_tokens(context)
            
            if total_tokens + context_tokens > config.max_context_tokens:
                break
                
            context_parts.append(context)
            total_tokens += context_tokens

        if not context_parts:
            return "No community data available within token limits.", 0

        context_text = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Based on the following research community data, answer the question.

{context_text}

Question: {query}

Answer:"""

        answer = self.llm_client.generate(
            prompt,
            temperature=config.temperature,
        )

        tokens_used = self.llm_client.count_tokens(prompt + answer)

        return answer, tokens_used

    def _format_community_context(self, community: CommunityInfo) -> str:
        """コミュニティ情報をテキストに整形"""
        findings_text = "\n".join(f"- {f}" for f in community.findings)
        entities_text = ", ".join(community.key_entities[:10])
        
        return f"""Title: {community.title}
Summary: {community.summary}
Key Findings:
{findings_text}
Key Entities: {entities_text}"""

    def _calculate_relevance(self, answer: str) -> float:
        """回答の関連性スコアを計算（簡易実装）
        
        Args:
            answer: 部分回答
            
        Returns:
            関連性スコア (0.0-1.0)
        """
        # 「関連情報なし」系の回答は低スコア
        no_info_phrases = [
            "no relevant information",
            "関連情報なし",
            "not found",
            "cannot answer",
            "回答できません",
        ]
        
        answer_lower = answer.lower()
        for phrase in no_info_phrases:
            if phrase in answer_lower:
                return 0.0

        # 長さベースの簡易スコア（より長い回答 = より詳細 = より関連性が高い可能性）
        # 実際のシステムではより洗練されたスコアリングを使用
        length = len(answer)
        if length < 50:
            return 0.2
        elif length < 150:
            return 0.5
        elif length < 300:
            return 0.7
        else:
            return 0.9


# シンプルなインメモリ実装（テスト用）
@dataclass
class InMemoryCommunityStore:
    """テスト・開発用のインメモリコミュニティストア"""

    communities: List[CommunityInfo] = field(default_factory=list)

    def add_community(self, community: CommunityInfo) -> None:
        """コミュニティを追加"""
        self.communities.append(community)

    def get_communities_by_level(self, level: int) -> List[CommunityInfo]:
        """指定レベルのコミュニティを取得"""
        return [c for c in self.communities if c.level == level]

    def get_top_communities(
        self, level: int, top_k: int
    ) -> List[CommunityInfo]:
        """上位k個のコミュニティを取得（サイズ順）"""
        level_communities = self.get_communities_by_level(level)
        sorted_communities = sorted(
            level_communities,
            key=lambda c: c.size,
            reverse=True,
        )
        return sorted_communities[:top_k]

    def clear(self) -> None:
        """全コミュニティを削除"""
        self.communities.clear()

    def count(self) -> int:
        """コミュニティ数を取得"""
        return len(self.communities)


# テスト用のモックLLMクライアント
@dataclass
class MockLLMClient:
    """テスト用のモックLLMクライアント"""

    responses: dict = field(default_factory=dict)
    default_response: str = "This is a mock response."
    tokens_per_char: float = 0.25  # 簡易トークン計算

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """モック応答を生成"""
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        return self.default_response

    def count_tokens(self, text: str) -> int:
        """簡易トークンカウント"""
        return int(len(text) * self.tokens_per_char)
