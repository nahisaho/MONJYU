# Semantic Scholar API Client - NFR-INT-001
"""
monjyu.external.semantic_scholar - Semantic Scholar API クライアント

Semantic Scholar API v1 との統合
- 論文検索
- DOI/arXiv IDによる論文取得
- 引用・参照ネットワーク取得

API Documentation: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .base import (
    Author,
    Citation,
    ExternalAPIClient,
    ExternalAPIConfig,
    PaperMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class SemanticScholarConfig(ExternalAPIConfig):
    """Semantic Scholar API設定
    
    Attributes:
        api_key: API Key（オプション、レート制限緩和用）
        include_citations: 引用を取得するか
        include_references: 参照を取得するか
        fields: 取得するフィールド
    """
    api_key: Optional[str] = None
    include_citations: bool = False
    include_references: bool = False
    fields: str = (
        "paperId,title,abstract,authors,year,venue,"
        "externalIds,citationCount,referenceCount,"
        "fieldsOfStudy,url,openAccessPdf"
    )


class SemanticScholarClient(ExternalAPIClient):
    """Semantic Scholar APIクライアント
    
    Semantic Scholar Graph APIを使用して論文メタデータを取得。
    
    Examples:
        >>> async with SemanticScholarClient() as client:
        ...     paper = await client.get_paper_by_doi("10.18653/v1/N19-1423")
        ...     print(paper.title)
        "BERT: Pre-training of Deep Bidirectional Transformers..."
        
        >>> async with SemanticScholarClient() as client:
        ...     papers = await client.search_papers("transformer attention")
        ...     for p in papers:
        ...         print(f"{p.title} ({p.year})")
    """
    
    def __init__(self, config: Optional[SemanticScholarConfig] = None):
        self._config = config or SemanticScholarConfig()
        super().__init__(self._config)
    
    @property
    def api_name(self) -> str:
        return "SemanticScholar"
    
    @property
    def base_url(self) -> str:
        return "https://api.semanticscholar.org/graph/v1"
    
    def _get_headers(self) -> Dict[str, str]:
        """APIヘッダーを取得"""
        headers = {}
        if self._config.api_key:
            headers["x-api-key"] = self._config.api_key
        return headers
    
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """DOIで論文を取得
        
        Args:
            doi: DOI (例: "10.18653/v1/N19-1423")
            
        Returns:
            論文メタデータ（見つからない場合はNone）
        """
        return await self.get_paper(f"DOI:{doi}")
    
    async def get_paper_by_arxiv(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """arXiv IDで論文を取得
        
        Args:
            arxiv_id: arXiv ID (例: "1706.03762")
            
        Returns:
            論文メタデータ（見つからない場合はNone）
        """
        # arXiv IDからバージョン番号を除去
        clean_id = arxiv_id.replace("arXiv:", "").split("v")[0]
        return await self.get_paper(f"ARXIV:{clean_id}")
    
    async def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """論文IDで論文を取得
        
        Args:
            paper_id: Semantic Scholar論文ID、DOI:xxx、ARXIV:xxx形式
            
        Returns:
            論文メタデータ（見つからない場合はNone）
        """
        try:
            encoded_id = quote(paper_id, safe=":")
            data = await self._request(
                "GET",
                f"/paper/{encoded_id}",
                params={"fields": self._config.fields},
                headers=self._get_headers(),
            )
            return self._parse_paper(data)
        except Exception as e:
            if "404" in str(e):
                logger.debug(f"Paper not found: {paper_id}")
                return None
            logger.warning(f"Failed to get paper {paper_id}: {e}")
            raise
    
    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        year: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[PaperMetadata]:
        """論文を検索
        
        Args:
            query: 検索クエリ
            limit: 取得件数（最大100）
            offset: オフセット
            year: 発行年フィルター（例: "2020", "2018-2022"）
            fields_of_study: 研究分野フィルター
            
        Returns:
            論文メタデータのリスト
        """
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": self._config.fields,
        }
        
        if year:
            params["year"] = year
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        try:
            data = await self._request(
                "GET",
                "/paper/search",
                params=params,
                headers=self._get_headers(),
            )
            
            papers = []
            for item in data.get("data", []):
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return []
    
    async def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Citation]:
        """論文の引用（この論文を引用している論文）を取得
        
        Args:
            paper_id: 論文ID
            limit: 取得件数（最大1000）
            offset: オフセット
            
        Returns:
            引用リスト
        """
        try:
            encoded_id = quote(paper_id, safe=":")
            data = await self._request(
                "GET",
                f"/paper/{encoded_id}/citations",
                params={
                    "limit": min(limit, 1000),
                    "offset": offset,
                    "fields": "paperId,title,year,externalIds,isInfluential",
                },
                headers=self._get_headers(),
            )
            
            citations = []
            for item in data.get("data", []):
                citing_paper = item.get("citingPaper", {})
                if citing_paper:
                    citations.append(self._parse_citation(citing_paper, item.get("isInfluential", False)))
            
            return citations
            
        except Exception as e:
            logger.warning(f"Failed to get citations for {paper_id}: {e}")
            return []
    
    async def get_references(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Citation]:
        """論文の参照（この論文が引用している論文）を取得
        
        Args:
            paper_id: 論文ID
            limit: 取得件数（最大1000）
            offset: オフセット
            
        Returns:
            参照リスト
        """
        try:
            encoded_id = quote(paper_id, safe=":")
            data = await self._request(
                "GET",
                f"/paper/{encoded_id}/references",
                params={
                    "limit": min(limit, 1000),
                    "offset": offset,
                    "fields": "paperId,title,year,externalIds,isInfluential",
                },
                headers=self._get_headers(),
            )
            
            references = []
            for item in data.get("data", []):
                cited_paper = item.get("citedPaper", {})
                if cited_paper:
                    references.append(self._parse_citation(cited_paper, item.get("isInfluential", False)))
            
            return references
            
        except Exception as e:
            logger.warning(f"Failed to get references for {paper_id}: {e}")
            return []
    
    async def get_author(self, author_id: str) -> Optional[Author]:
        """著者情報を取得
        
        Args:
            author_id: Semantic Scholar著者ID
            
        Returns:
            著者情報（見つからない場合はNone）
        """
        try:
            data = await self._request(
                "GET",
                f"/author/{author_id}",
                params={"fields": "authorId,name,affiliations"},
                headers=self._get_headers(),
            )
            
            affiliations = data.get("affiliations", [])
            return Author(
                name=data.get("name", ""),
                author_id=data.get("authorId"),
                affiliation=affiliations[0] if affiliations else None,
            )
            
        except Exception as e:
            logger.warning(f"Failed to get author {author_id}: {e}")
            return None
    
    def _parse_paper(self, data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """APIレスポンスをPaperMetadataに変換"""
        if not data or not data.get("paperId"):
            return None
        
        # 著者をパース
        authors = []
        for author_data in data.get("authors", []):
            authors.append(Author(
                name=author_data.get("name", ""),
                author_id=author_data.get("authorId"),
            ))
        
        # 外部IDを取得
        external_ids = data.get("externalIds", {}) or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        
        # PDF URLを取得
        open_access_pdf = data.get("openAccessPdf", {}) or {}
        pdf_url = open_access_pdf.get("url")
        
        return PaperMetadata(
            paper_id=data["paperId"],
            title=data.get("title", ""),
            abstract=data.get("abstract", "") or "",
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            doi=doi,
            arxiv_id=arxiv_id,
            citation_count=data.get("citationCount", 0) or 0,
            reference_count=data.get("referenceCount", 0) or 0,
            fields_of_study=data.get("fieldsOfStudy", []) or [],
            source="semantic_scholar",
            url=data.get("url"),
            pdf_url=pdf_url,
        )
    
    def _parse_citation(self, data: Dict[str, Any], is_influential: bool = False) -> Citation:
        """引用データをCitationに変換"""
        external_ids = data.get("externalIds", {}) or {}
        
        return Citation(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            doi=external_ids.get("DOI"),
            year=data.get("year"),
            is_influential=is_influential,
        )


def create_semantic_scholar_client(
    api_key: Optional[str] = None,
    include_citations: bool = False,
    include_references: bool = False,
) -> SemanticScholarClient:
    """Semantic Scholarクライアントを作成するファクトリ関数
    
    Args:
        api_key: API Key（オプション）
        include_citations: 引用を含めるか
        include_references: 参照を含めるか
        
    Returns:
        設定済みのSemanticScholarClient
    """
    config = SemanticScholarConfig(
        api_key=api_key,
        include_citations=include_citations,
        include_references=include_references,
    )
    return SemanticScholarClient(config)
