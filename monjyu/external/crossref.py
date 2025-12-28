# CrossRef API Client - NFR-INT-001
"""
monjyu.external.crossref - CrossRef API クライアント

CrossRef REST API との統合
- DOI解決・メタデータ取得
- 論文検索
- 引用関係取得

API Documentation: https://api.crossref.org/swagger-ui/index.html
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
class CrossRefConfig(ExternalAPIConfig):
    """CrossRef API設定
    
    Attributes:
        mailto: Polite Poolアクセス用のメールアドレス
            指定するとレート制限が緩和される
    """
    mailto: Optional[str] = None


class CrossRefClient(ExternalAPIClient):
    """CrossRef APIクライアント
    
    CrossRef REST APIを使用してDOIからメタデータを取得。
    
    Examples:
        >>> async with CrossRefClient() as client:
        ...     paper = await client.get_paper_by_doi("10.18653/v1/N19-1423")
        ...     print(paper.title)
        "BERT: Pre-training of Deep Bidirectional Transformers..."
        
        >>> config = CrossRefConfig(mailto="your@email.com")
        >>> async with CrossRefClient(config) as client:
        ...     papers = await client.search_papers("machine learning")
    """
    
    def __init__(self, config: Optional[CrossRefConfig] = None):
        self._config = config or CrossRefConfig()
        super().__init__(self._config)
    
    @property
    def api_name(self) -> str:
        return "CrossRef"
    
    @property
    def base_url(self) -> str:
        return "https://api.crossref.org"
    
    def _add_mailto_param(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """mailtoパラメータを追加（Polite Pool用）"""
        if self._config.mailto:
            params["mailto"] = self._config.mailto
        return params
    
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """DOIで論文を取得
        
        Args:
            doi: DOI (例: "10.18653/v1/N19-1423")
            
        Returns:
            論文メタデータ（見つからない場合はNone）
        """
        try:
            # DOIをURLエンコード
            encoded_doi = quote(doi, safe="")
            params = self._add_mailto_param({})
            
            data = await self._request(
                "GET",
                f"/works/{encoded_doi}",
                params=params if params else None,
            )
            
            message = data.get("message", {})
            return self._parse_work(message)
            
        except Exception as e:
            if "404" in str(e):
                logger.debug(f"DOI not found: {doi}")
                return None
            logger.warning(f"Failed to get DOI {doi}: {e}")
            raise
    
    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        filter_type: Optional[str] = None,
        from_year: Optional[int] = None,
        until_year: Optional[int] = None,
    ) -> List[PaperMetadata]:
        """論文を検索
        
        Args:
            query: 検索クエリ
            limit: 取得件数（最大100）
            offset: オフセット
            filter_type: タイプフィルター（例: "journal-article"）
            from_year: 開始年
            until_year: 終了年
            
        Returns:
            論文メタデータのリスト
        """
        params: Dict[str, Any] = {
            "query": query,
            "rows": min(limit, 100),
            "offset": offset,
        }
        
        # フィルター構築
        filters = []
        if filter_type:
            filters.append(f"type:{filter_type}")
        if from_year:
            filters.append(f"from-pub-date:{from_year}")
        if until_year:
            filters.append(f"until-pub-date:{until_year}")
        
        if filters:
            params["filter"] = ",".join(filters)
        
        params = self._add_mailto_param(params)
        
        try:
            data = await self._request(
                "GET",
                "/works",
                params=params,
            )
            
            papers = []
            for item in data.get("message", {}).get("items", []):
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return []
    
    async def get_references(
        self,
        doi: str,
        limit: int = 100,
    ) -> List[Citation]:
        """論文の参照（この論文が引用している論文）を取得
        
        Args:
            doi: DOI
            limit: 取得件数
            
        Returns:
            参照リスト
        """
        paper = await self.get_paper_by_doi(doi)
        if paper:
            return paper.references[:limit]
        return []
    
    async def get_journal_info(self, issn: str) -> Optional[Dict[str, Any]]:
        """ISSNでジャーナル情報を取得
        
        Args:
            issn: ISSN
            
        Returns:
            ジャーナル情報
        """
        try:
            params = self._add_mailto_param({})
            data = await self._request(
                "GET",
                f"/journals/{issn}",
                params=params if params else None,
            )
            return data.get("message")
        except Exception as e:
            logger.warning(f"Failed to get journal {issn}: {e}")
            return None
    
    async def get_funder_works(
        self,
        funder_id: str,
        limit: int = 20,
    ) -> List[PaperMetadata]:
        """ファンダーIDで資金提供された論文を取得
        
        Args:
            funder_id: Funder ID（例: "10.13039/100000001" for NSF）
            limit: 取得件数
            
        Returns:
            論文メタデータのリスト
        """
        try:
            encoded_id = quote(funder_id, safe="")
            params = self._add_mailto_param({"rows": min(limit, 100)})
            
            data = await self._request(
                "GET",
                f"/funders/{encoded_id}/works",
                params=params,
            )
            
            papers = []
            for item in data.get("message", {}).get("items", []):
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.warning(f"Failed to get funder works {funder_id}: {e}")
            return []
    
    def _parse_work(self, data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """APIレスポンスをPaperMetadataに変換"""
        if not data:
            return None
        
        doi = data.get("DOI", "")
        
        # タイトルを取得（リスト形式の場合あり）
        title_list = data.get("title", [])
        title = title_list[0] if title_list else ""
        
        # アブストラクトを取得
        abstract = data.get("abstract", "")
        # HTMLタグを除去（簡易的）
        if abstract:
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract)
        
        # 著者をパース
        authors = []
        for author_data in data.get("author", []):
            given = author_data.get("given", "")
            family = author_data.get("family", "")
            name = f"{given} {family}".strip() if given or family else ""
            
            if name:
                affiliations = author_data.get("affiliation", [])
                affiliation = affiliations[0].get("name") if affiliations else None
                
                authors.append(Author(
                    name=name,
                    orcid=author_data.get("ORCID"),
                    affiliation=affiliation,
                ))
        
        # 発行年を取得
        year = None
        published = data.get("published", {})
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        
        # 出版先を取得
        venue = None
        container_title = data.get("container-title", [])
        if container_title:
            venue = container_title[0]
        
        # 参照をパース
        references = []
        for ref in data.get("reference", [])[:100]:  # 最大100件
            ref_doi = ref.get("DOI", "")
            ref_title = ref.get("article-title", "") or ref.get("unstructured", "")
            
            if ref_doi or ref_title:
                references.append(Citation(
                    paper_id=ref_doi or f"ref-{len(references)}",
                    title=ref_title[:200],  # タイトルを200文字に制限
                    doi=ref_doi if ref_doi else None,
                    year=ref.get("year"),
                ))
        
        # URLを構築
        url = f"https://doi.org/{doi}" if doi else None
        
        # PDF URLを取得（ライセンスがオープンの場合）
        pdf_url = None
        for link in data.get("link", []):
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break
        
        return PaperMetadata(
            paper_id=doi or str(hash(title)),
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            citation_count=data.get("is-referenced-by-count", 0),
            reference_count=data.get("references-count", 0),
            references=references,
            source="crossref",
            url=url,
            pdf_url=pdf_url,
            metadata={
                "type": data.get("type"),
                "publisher": data.get("publisher"),
                "issn": data.get("ISSN", []),
                "subject": data.get("subject", []),
            },
        )


def create_crossref_client(
    mailto: Optional[str] = None,
) -> CrossRefClient:
    """CrossRefクライアントを作成するファクトリ関数
    
    Args:
        mailto: Polite Pool用のメールアドレス（推奨）
        
    Returns:
        設定済みのCrossRefClient
    """
    config = CrossRefConfig(mailto=mailto)
    return CrossRefClient(config)
