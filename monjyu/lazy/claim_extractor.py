# Claim Extractor
"""
クレーム抽出 - テキストからクエリに関連する主張を抽出

TASK-005-02: ClaimExtractor 実装
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monjyu.lazy.base import Claim, RelevanceScore, SearchCandidate

if TYPE_CHECKING:
    from monjyu.search.answer_synthesizer import LLMClientProtocol


class ClaimExtractor:
    """クレーム抽出"""

    EXTRACTION_PROMPT = """以下のテキストから、クエリに関連する主要な主張（claim）を抽出してください。

クエリ: {query}

テキスト:
{text}

出力形式:
各主張を1行ずつ、「- 」で始めて記述してください。
主張は事実に基づく簡潔な文にしてください。
最大{max_claims}個まで抽出してください。

例:
- Transformerは自己注意機構を使用する
- BERTは双方向のコンテキストを学習する"""

    def __init__(
        self,
        llm_client: "LLMClientProtocol",
        max_text_length: int = 2000,
        max_claims_per_text: int = 5,
    ):
        """
        Args:
            llm_client: LLMクライアント
            max_text_length: テキストの最大長
            max_claims_per_text: 1テキストから抽出するクレームの最大数
        """
        self.llm_client = llm_client
        self.max_text_length = max_text_length
        self.max_claims_per_text = max_claims_per_text

    def extract(
        self,
        query: str,
        text: str,
        source_text_unit_id: str = "",
        source_document_id: str = "",
    ) -> list[Claim]:
        """
        テキストからクレームを抽出

        Args:
            query: クエリ
            text: テキスト
            source_text_unit_id: ソースのTextUnit ID
            source_document_id: ソースのDocument ID

        Returns:
            クレームのリスト
        """
        # テキストを切り詰め
        truncated_text = text[: self.max_text_length]
        if len(text) > self.max_text_length:
            truncated_text += "..."

        prompt = self.EXTRACTION_PROMPT.format(
            query=query,
            text=truncated_text,
            max_claims=self.max_claims_per_text,
        )

        try:
            response = self.llm_client.generate(prompt, max_tokens=500)
            return self._parse_claims(
                response,
                source_text_unit_id,
                source_document_id,
            )
        except Exception:
            # エラー時は空リスト
            return []

    def _parse_claims(
        self,
        response: str,
        source_text_unit_id: str,
        source_document_id: str,
    ) -> list[Claim]:
        """LLMレスポンスからクレームをパース"""
        claims = []

        for line in response.strip().split("\n"):
            line = line.strip()
            # 「- 」または「・」で始まる行をクレームとして認識
            if line.startswith("- "):
                claim_text = line[2:].strip()
            elif line.startswith("・"):
                claim_text = line[1:].strip()
            elif line.startswith("* "):
                claim_text = line[2:].strip()
            else:
                continue

            if claim_text and len(claim_text) >= 5:
                claims.append(
                    Claim(
                        text=claim_text,
                        source_text_unit_id=source_text_unit_id,
                        source_document_id=source_document_id,
                    )
                )

                if len(claims) >= self.max_claims_per_text:
                    break

        return claims

    def extract_batch(
        self, query: str, candidates: list[SearchCandidate]
    ) -> list[Claim]:
        """
        バッチでクレーム抽出

        Args:
            query: クエリ
            candidates: 検索候補のリスト

        Returns:
            クレームのリスト（重複除去済み）
        """
        all_claims = []

        for candidate in candidates:
            claims = self.extract(
                query,
                candidate.text,
                source_text_unit_id=candidate.id,
                source_document_id=candidate.metadata.get("document_id", ""),
            )
            all_claims.extend(claims)

        return self._merge_duplicates(all_claims)

    def _merge_duplicates(self, claims: list[Claim]) -> list[Claim]:
        """
        重複クレームをマージ

        簡易的な重複判定（将来的にはembedding類似度を使用）
        """
        seen: set[str] = set()
        unique: list[Claim] = []

        for claim in claims:
            # 正規化してチェック
            normalized = self._normalize_text(claim.text)
            if normalized not in seen:
                seen.add(normalized)
                unique.append(claim)

        return unique

    def _normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        # 小文字化、余分な空白を削除
        normalized = text.lower().strip()
        # 連続する空白を1つに
        import re

        normalized = re.sub(r"\s+", " ", normalized)
        return normalized


class MockClaimExtractor:
    """テスト用モッククレーム抽出器"""

    def __init__(self, default_claims: list[str] | None = None):
        """
        Args:
            default_claims: デフォルトで返すクレームのテキストリスト
        """
        self.default_claims = default_claims or ["Default claim 1", "Default claim 2"]
        self.call_count = 0
        self._custom_responses: dict[str, list[str]] = {}

    def set_custom_claims(self, text_contains: str, claims: list[str]) -> None:
        """特定のテキストに対するカスタムクレームを設定"""
        self._custom_responses[text_contains] = claims

    def extract(
        self,
        query: str,
        text: str,
        source_text_unit_id: str = "",
        source_document_id: str = "",
    ) -> list[Claim]:
        """テキストからクレームを抽出"""
        self.call_count += 1

        # カスタムレスポンスをチェック
        claim_texts = self.default_claims
        for key, claims in self._custom_responses.items():
            if key.lower() in text.lower():
                claim_texts = claims
                break

        return [
            Claim(
                text=claim_text,
                source_text_unit_id=source_text_unit_id,
                source_document_id=source_document_id,
                relevance_score=RelevanceScore.HIGH,
            )
            for claim_text in claim_texts
        ]

    def extract_batch(
        self, query: str, candidates: list[SearchCandidate]
    ) -> list[Claim]:
        """バッチでクレーム抽出"""
        all_claims = []
        for candidate in candidates:
            claims = self.extract(
                query,
                candidate.text,
                source_text_unit_id=candidate.id,
                source_document_id=candidate.metadata.get("document_id", ""),
            )
            all_claims.extend(claims)
        return all_claims
