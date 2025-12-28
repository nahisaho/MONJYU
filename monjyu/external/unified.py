# Unified Metadata Client - NFR-INT-001
"""
monjyu.external.unified - 統合メタデータクライアント

複数の外部APIを統合して論文メタデータを取得。
フォールバックとマージ機能を提供。

Priority:
1. Semantic Scholar（引用ネットワークが豊富）
2. CrossRef（DOIの正式なソース）
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import (
    Author,
    Citation,
    ExternalAPIConfig,
    PaperMetadata,
)
from .semantic_scholar import SemanticScholarClient, SemanticScholarConfig
from .crossref import CrossRefClient, CrossRefConfig

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMetadataConfig:
    """統合メタデータクライアント設定
    
    Attributes:
        semantic_scholar_api_key: Semantic Scholar API Key
        crossref_mailto: CrossRef Polite Pool用メールアドレス
        prefer_semantic_scholar: Semantic Scholarを優先するか
        merge_results: 結果をマージするか
        timeout: タイムアウト（秒）
    """
    semantic_scholar_api_key: Optional[str] = None
    crossref_mailto: Optional[str] = None
    prefer_semantic_scholar: bool = True
    merge_results: bool = True
    timeout: float = 30.0


class UnifiedMetadataClient:
    """統合メタデータクライアント
    
    Semantic ScholarとCrossRefを統合して論文メタデータを取得。
    
    Examples:
        >>> async with UnifiedMetadataClient() as client:
        ...     paper = await client.get_paper_by_doi("10.18653/v1/N19-1423")
        ...     print(f"Title: {paper.title}")
        ...     print(f"Citations: {paper.citation_count}")
        ...     print(f"Source: {paper.source}")
        
        >>> config = UnifiedMetadataConfig(
        ...     semantic_scholar_api_key="your-key",
        ...     crossref_mailto="your@email.com",
        ... )
        >>> async with UnifiedMetadataClient(config) as client:
        ...     papers = await client.search_papers("transformer attention")
    """
    
    def __init__(self, config: Optional[UnifiedMetadataConfig] = None):
        self.config = config or UnifiedMetadataConfig()
        
        # Semantic Scholarクライアント
        ss_config = SemanticScholarConfig(
            api_key=self.config.semantic_scholar_api_key,
            timeout=self.config.timeout,
        )
        self._semantic_scholar = SemanticScholarClient(ss_config)
        
        # CrossRefクライアント
        cr_config = CrossRefConfig(
            mailto=self.config.crossref_mailto,
            timeout=self.config.timeout,
        )
        self._crossref = CrossRefClient(cr_config)
    
    async def close(self) -> None:
        """クライアントをクローズ"""
        await asyncio.gather(
            self._semantic_scholar.close(),
            self._crossref.close(),
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """DOIで論文を取得
        
        両方のAPIから取得を試み、結果をマージ（設定に応じて）。
        
        Args:
            doi: DOI
            
        Returns:
            論文メタデータ（見つからない場合はNone）
        """
        results: List[Optional[PaperMetadata]] = []
        
        # 並行して両方のAPIを呼び出し
        try:
            ss_result, cr_result = await asyncio.gather(
                self._semantic_scholar.get_paper_by_doi(doi),
                self._crossref.get_paper_by_doi(doi),
                return_exceptions=True,
            )
            
            if isinstance(ss_result, PaperMetadata):
                results.append(ss_result)
            elif isinstance(ss_result, Exception):
                logger.debug(f"Semantic Scholar error for {doi}: {ss_result}")
            
            if isinstance(cr_result, PaperMetadata):
                results.append(cr_result)
            elif isinstance(cr_result, Exception):
                logger.debug(f"CrossRef error for {doi}: {cr_result}")
                
        except Exception as e:
            logger.warning(f"Failed to get paper by DOI {doi}: {e}")
            return None
        
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # マージ
        if self.config.merge_results:
            return self._merge_papers(results)
        
        # 優先APIの結果を返す
        if self.config.prefer_semantic_scholar:
            ss_paper = next((p for p in results if p and p.source == "semantic_scholar"), None)
            return ss_paper or results[0]
        else:
            cr_paper = next((p for p in results if p and p.source == "crossref"), None)
            return cr_paper or results[0]
    
    async def get_paper_by_arxiv(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """arXiv IDで論文を取得
        
        Args:
            arxiv_id: arXiv ID
            
        Returns:
            論文メタデータ
        """
        return await self._semantic_scholar.get_paper_by_arxiv(arxiv_id)
    
    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        source: Optional[str] = None,
    ) -> List[PaperMetadata]:
        """論文を検索
        
        Args:
            query: 検索クエリ
            limit: 取得件数
            source: 使用するソース（"semantic_scholar", "crossref", None=両方）
            
        Returns:
            論文メタデータのリスト
        """
        if source == "semantic_scholar":
            return await self._semantic_scholar.search_papers(query, limit)
        elif source == "crossref":
            return await self._crossref.search_papers(query, limit)
        
        # 両方のAPIから取得
        try:
            ss_papers, cr_papers = await asyncio.gather(
                self._semantic_scholar.search_papers(query, limit),
                self._crossref.search_papers(query, limit),
                return_exceptions=True,
            )
            
            all_papers: List[PaperMetadata] = []
            
            if isinstance(ss_papers, list):
                all_papers.extend(ss_papers)
            
            if isinstance(cr_papers, list):
                all_papers.extend(cr_papers)
            
            # 重複を除去（DOIベース）
            unique_papers = self._deduplicate_papers(all_papers)
            
            return unique_papers[:limit]
            
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return []
    
    async def get_citations(
        self,
        doi: str,
        limit: int = 100,
    ) -> List[Citation]:
        """論文の引用を取得
        
        Semantic Scholarから取得（より豊富な引用データ）。
        
        Args:
            doi: DOI
            limit: 取得件数
            
        Returns:
            引用リスト
        """
        paper = await self._semantic_scholar.get_paper_by_doi(doi)
        if paper:
            return await self._semantic_scholar.get_citations(paper.paper_id, limit)
        return []
    
    async def get_references(
        self,
        doi: str,
        limit: int = 100,
    ) -> List[Citation]:
        """論文の参照を取得
        
        Args:
            doi: DOI
            limit: 取得件数
            
        Returns:
            参照リスト
        """
        # 両方から取得してマージ
        try:
            ss_refs, cr_refs = await asyncio.gather(
                self._get_ss_references(doi, limit),
                self._crossref.get_references(doi, limit),
                return_exceptions=True,
            )
            
            all_refs: List[Citation] = []
            
            if isinstance(ss_refs, list):
                all_refs.extend(ss_refs)
            
            if isinstance(cr_refs, list):
                all_refs.extend(cr_refs)
            
            # 重複を除去
            unique_refs = self._deduplicate_citations(all_refs)
            
            return unique_refs[:limit]
            
        except Exception as e:
            logger.warning(f"Failed to get references for {doi}: {e}")
            return []
    
    async def _get_ss_references(self, doi: str, limit: int) -> List[Citation]:
        """Semantic Scholarから参照を取得"""
        paper = await self._semantic_scholar.get_paper_by_doi(doi)
        if paper:
            return await self._semantic_scholar.get_references(paper.paper_id, limit)
        return []
    
    async def enrich_paper(self, paper: PaperMetadata) -> PaperMetadata:
        """論文メタデータを他のソースから補完
        
        Args:
            paper: 既存の論文メタデータ
            
        Returns:
            補完された論文メタデータ
        """
        if not paper.doi:
            return paper
        
        # 別のソースから情報を取得
        other_source = await self.get_paper_by_doi(paper.doi)
        
        if not other_source:
            return paper
        
        return self._merge_papers([paper, other_source])
    
    def _merge_papers(self, papers: List[Optional[PaperMetadata]]) -> Optional[PaperMetadata]:
        """複数の論文メタデータをマージ"""
        valid_papers = [p for p in papers if p is not None]
        
        if not valid_papers:
            return None
        
        if len(valid_papers) == 1:
            return valid_papers[0]
        
        # ベースを選択
        base = valid_papers[0]
        other = valid_papers[1]
        
        # 各フィールドをマージ（空でない方を優先）
        merged = PaperMetadata(
            paper_id=base.paper_id or other.paper_id,
            title=base.title or other.title,
            abstract=base.abstract or other.abstract,
            authors=base.authors if base.authors else other.authors,
            year=base.year or other.year,
            venue=base.venue or other.venue,
            doi=base.doi or other.doi,
            arxiv_id=base.arxiv_id or other.arxiv_id,
            citation_count=max(base.citation_count, other.citation_count),
            reference_count=max(base.reference_count, other.reference_count),
            citations=base.citations if base.citations else other.citations,
            references=base.references if base.references else other.references,
            fields_of_study=base.fields_of_study or other.fields_of_study,
            source="unified",
            url=base.url or other.url,
            pdf_url=base.pdf_url or other.pdf_url,
            metadata={
                **other.metadata,
                **base.metadata,
                "merged_from": [base.source, other.source],
            },
        )
        
        return merged
    
    def _deduplicate_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """重複論文を除去（DOIベース）"""
        seen_dois: set[str] = set()
        unique: List[PaperMetadata] = []
        
        for paper in papers:
            key = paper.doi or paper.paper_id
            if key not in seen_dois:
                seen_dois.add(key)
                unique.append(paper)
        
        return unique
    
    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """重複引用を除去"""
        seen_ids: set[str] = set()
        unique: List[Citation] = []
        
        for cit in citations:
            key = cit.doi or cit.paper_id
            if key not in seen_ids:
                seen_ids.add(key)
                unique.append(cit)
        
        return unique


def create_unified_client(
    semantic_scholar_api_key: Optional[str] = None,
    crossref_mailto: Optional[str] = None,
    prefer_semantic_scholar: bool = True,
) -> UnifiedMetadataClient:
    """統合メタデータクライアントを作成するファクトリ関数
    
    Args:
        semantic_scholar_api_key: Semantic Scholar API Key
        crossref_mailto: CrossRef用メールアドレス
        prefer_semantic_scholar: Semantic Scholarを優先するか
        
    Returns:
        設定済みのUnifiedMetadataClient
    """
    config = UnifiedMetadataConfig(
        semantic_scholar_api_key=semantic_scholar_api_key,
        crossref_mailto=crossref_mailto,
        prefer_semantic_scholar=prefer_semantic_scholar,
    )
    return UnifiedMetadataClient(config)
