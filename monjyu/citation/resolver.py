# Reference Resolver
"""
monjyu.citation.resolver - 参照解決

FEAT-006: Citation Network
- DOI マッチング
- タイトルマッチング（完全一致 + ファジーマッチング）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import TYPE_CHECKING
import re
import hashlib

if TYPE_CHECKING:
    from monjyu.document.base import Document

from monjyu.citation.base import (
    ReferenceMatchStatus,
    ResolvedReference,
    CitationNetworkConfig,
)


@dataclass
class DocumentIndex:
    """文書インデックス（検索用）"""

    doc_id: str
    doi: str | None
    title: str
    normalized_title: str  # 正規化タイトル


class ReferenceResolver(ABC):
    """参照解決器の抽象基底クラス"""

    @abstractmethod
    def build_index(self, documents: list[Document]) -> None:
        """文書インデックスを構築"""
        pass

    @abstractmethod
    def resolve(
        self,
        source_doc_id: str,
        reference_text: str,
    ) -> ResolvedReference:
        """参照を解決"""
        pass

    @abstractmethod
    def resolve_batch(
        self,
        source_doc_id: str,
        references: list[str],
    ) -> list[ResolvedReference]:
        """複数参照を一括解決"""
        pass


class DefaultReferenceResolver(ReferenceResolver):
    """デフォルト参照解決器"""

    # DOIパターン
    DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s]+")

    def __init__(self, config: CitationNetworkConfig | None = None):
        self.config = config or CitationNetworkConfig()
        self._doc_index: dict[str, DocumentIndex] = {}
        self._doi_index: dict[str, str] = {}  # DOI -> doc_id
        self._title_index: dict[str, str] = {}  # normalized_title -> doc_id

    @staticmethod
    def normalize_title(title: str) -> str:
        """タイトルを正規化"""
        # 小文字化、記号除去、空白正規化
        normalized = title.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def extract_doi(text: str) -> str | None:
        """テキストからDOIを抽出"""
        match = DefaultReferenceResolver.DOI_PATTERN.search(text)
        if match:
            doi = match.group()
            # 末尾の句読点を除去
            doi = doi.rstrip(".,;:)")
            return doi
        return None

    @staticmethod
    def extract_title(text: str) -> str | None:
        """テキストからタイトルを抽出（ヒューリスティック）"""
        # 引用符で囲まれたテキスト
        quoted = re.search(r'"([^"]+)"', text)
        if quoted:
            return quoted.group(1)

        # 最初の文（.で終わるまで）でタイトルらしきもの
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) > 10:
            return first_sentence

        return None

    def build_index(self, documents: list[Document]) -> None:
        """文書インデックスを構築"""
        self._doc_index.clear()
        self._doi_index.clear()
        self._title_index.clear()

        for doc in documents:
            # DOIを取得
            doi = doc.metadata.get("doi") if doc.metadata else None

            # タイトルを取得
            title = ""
            if doc.metadata:
                title = doc.metadata.get("title", "")
            if not title:
                # メタデータにない場合は最初の行をタイトルとして使用
                title = doc.content.split("\n")[0][:200] if doc.content else ""

            normalized_title = self.normalize_title(title)

            # インデックスに追加
            doc_index = DocumentIndex(
                doc_id=doc.doc_id,
                doi=doi,
                title=title,
                normalized_title=normalized_title,
            )
            self._doc_index[doc.doc_id] = doc_index

            if doi and self.config.enable_doi_matching:
                self._doi_index[doi.lower()] = doc.doc_id

            if normalized_title and self.config.enable_title_matching:
                self._title_index[normalized_title] = doc.doc_id

    def resolve(
        self,
        source_doc_id: str,
        reference_text: str,
    ) -> ResolvedReference:
        """参照を解決"""
        # 1. DOIマッチング
        if self.config.enable_doi_matching:
            doi = self.extract_doi(reference_text)
            if doi:
                doi_lower = doi.lower()
                if doi_lower in self._doi_index:
                    target_id = self._doi_index[doi_lower]
                    if target_id != source_doc_id:  # 自己参照を除外
                        return ResolvedReference(
                            source_doc_id=source_doc_id,
                            target_doc_id=target_id,
                            status=ReferenceMatchStatus.MATCHED_DOI,
                            confidence=1.0,
                            raw_reference=reference_text,
                            matched_doi=doi,
                        )

        # 2. タイトルマッチング
        if self.config.enable_title_matching:
            title = self.extract_title(reference_text)
            if title:
                normalized = self.normalize_title(title)

                # 完全一致
                if normalized in self._title_index:
                    target_id = self._title_index[normalized]
                    if target_id != source_doc_id:
                        target_title = self._doc_index[target_id].title
                        return ResolvedReference(
                            source_doc_id=source_doc_id,
                            target_doc_id=target_id,
                            status=ReferenceMatchStatus.MATCHED_TITLE_EXACT,
                            confidence=1.0,
                            raw_reference=reference_text,
                            matched_title=target_title,
                        )

                # ファジーマッチング
                best_match = self._fuzzy_match_title(normalized, source_doc_id)
                if best_match:
                    return replace(
                        best_match,
                        source_doc_id=source_doc_id,
                        raw_reference=reference_text,
                    )

        # 未解決（外部参照）
        return ResolvedReference(
            source_doc_id=source_doc_id,
            target_doc_id=None,
            status=ReferenceMatchStatus.UNRESOLVED,
            confidence=0.0,
            raw_reference=reference_text,
        )

    def _fuzzy_match_title(
        self,
        query_title: str,
        source_doc_id: str,
    ) -> ResolvedReference | None:
        """ファジータイトルマッチング"""
        best_score = 0.0
        best_match: DocumentIndex | None = None

        for doc_index in self._doc_index.values():
            if doc_index.doc_id == source_doc_id:
                continue

            score = SequenceMatcher(
                None, query_title, doc_index.normalized_title
            ).ratio()

            if score > best_score and score >= self.config.fuzzy_match_threshold:
                best_score = score
                best_match = doc_index

        if best_match:
            return ResolvedReference(
                source_doc_id="",  # 呼び出し側で設定
                target_doc_id=best_match.doc_id,
                status=ReferenceMatchStatus.MATCHED_TITLE_FUZZY,
                confidence=best_score,
                raw_reference="",  # 呼び出し側で設定
                matched_title=best_match.title,
            )

        return None

    def resolve_batch(
        self,
        source_doc_id: str,
        references: list[str],
    ) -> list[ResolvedReference]:
        """複数参照を一括解決"""
        return [self.resolve(source_doc_id, ref) for ref in references]


def generate_external_ref_key(reference_text: str) -> str:
    """外部参照用のキーを生成"""
    # ハッシュベースのキー生成
    hash_val = hashlib.md5(reference_text.encode()).hexdigest()[:8]
    return f"ext_{hash_val}"


class MockReferenceResolver(ReferenceResolver):
    """テスト用のモック参照解決器"""

    def __init__(
        self,
        mock_resolutions: dict[tuple[str, str], ResolvedReference] | None = None,
    ):
        """
        Args:
            mock_resolutions: (source_doc_id, reference_text) -> ResolvedReference のマップ
        """
        self._mock_resolutions = mock_resolutions or {}
        self._indexed = False

    def build_index(self, documents: list[Document]) -> None:
        """モックでは何もしない"""
        self._indexed = True

    def resolve(
        self,
        source_doc_id: str,
        reference_text: str,
    ) -> ResolvedReference:
        """モック解決"""
        key = (source_doc_id, reference_text)
        if key in self._mock_resolutions:
            return self._mock_resolutions[key]

        # デフォルトは未解決
        return ResolvedReference(
            source_doc_id=source_doc_id,
            target_doc_id=None,
            status=ReferenceMatchStatus.UNRESOLVED,
            confidence=0.0,
            raw_reference=reference_text,
        )

    def resolve_batch(
        self,
        source_doc_id: str,
        references: list[str],
    ) -> list[ResolvedReference]:
        """モック一括解決"""
        return [self.resolve(source_doc_id, ref) for ref in references]
