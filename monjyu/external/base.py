# External API Base Classes - NFR-INT-001
"""
monjyu.external.base - 外部API基底クラス

共通のデータモデルとプロトコル定義
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ExternalAPIError(Exception):
    """外部APIエラーの基底クラス"""
    
    def __init__(self, message: str, api_name: str = "", status_code: int = 0):
        super().__init__(message)
        self.api_name = api_name
        self.status_code = status_code


class RateLimitError(ExternalAPIError):
    """レート制限エラー"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        api_name: str = "",
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, api_name, 429)
        self.retry_after = retry_after


class APIResponseError(ExternalAPIError):
    """APIレスポンスエラー"""
    
    def __init__(
        self,
        message: str,
        api_name: str = "",
        status_code: int = 0,
        response_body: Optional[str] = None,
    ):
        super().__init__(message, api_name, status_code)
        self.response_body = response_body


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Author:
    """著者情報
    
    Attributes:
        name: 著者名
        author_id: 外部APIでの著者ID（オプション）
        affiliation: 所属機関（オプション）
        orcid: ORCID ID（オプション）
    """
    name: str
    author_id: Optional[str] = None
    affiliation: Optional[str] = None
    orcid: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "name": self.name,
            "author_id": self.author_id,
            "affiliation": self.affiliation,
            "orcid": self.orcid,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Author":
        """辞書から生成"""
        return cls(
            name=data.get("name", ""),
            author_id=data.get("author_id"),
            affiliation=data.get("affiliation"),
            orcid=data.get("orcid"),
        )


@dataclass
class Citation:
    """引用情報
    
    Attributes:
        paper_id: 引用論文のID
        title: 引用論文のタイトル
        doi: DOI（オプション）
        year: 発行年（オプション）
        is_influential: 重要な引用かどうか
    """
    paper_id: str
    title: str
    doi: Optional[str] = None
    year: Optional[int] = None
    is_influential: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "doi": self.doi,
            "year": self.year,
            "is_influential": self.is_influential,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        """辞書から生成"""
        return cls(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            doi=data.get("doi"),
            year=data.get("year"),
            is_influential=data.get("is_influential", False),
        )


@dataclass
class PaperMetadata:
    """論文メタデータ
    
    外部APIから取得した論文の統合メタデータ。
    
    Attributes:
        paper_id: 論文ID（API固有またはDOI）
        title: タイトル
        abstract: アブストラクト
        authors: 著者リスト
        year: 発行年
        venue: 発表場所（ジャーナル/会議）
        doi: DOI
        arxiv_id: arXiv ID
        citation_count: 被引用数
        reference_count: 参照数
        citations: 引用リスト
        references: 参照リスト
        fields_of_study: 研究分野
        source: データソース
        url: 論文URL
        pdf_url: PDF URL
        retrieved_at: 取得日時
        metadata: 追加メタデータ
    
    Examples:
        >>> paper = PaperMetadata(
        ...     paper_id="10.1234/example",
        ...     title="Example Paper",
        ...     authors=[Author(name="John Doe")],
        ...     year=2024,
        ... )
    """
    paper_id: str
    title: str
    abstract: str = ""
    authors: List[Author] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    citations: List[Citation] = field(default_factory=list)
    references: List[Citation] = field(default_factory=list)
    fields_of_study: List[str] = field(default_factory=list)
    source: str = ""
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.retrieved_at is None:
            self.retrieved_at = datetime.now()
    
    @property
    def author_names(self) -> List[str]:
        """著者名リストを取得"""
        return [a.name for a in self.authors]
    
    @property
    def has_doi(self) -> bool:
        """DOIがあるかどうか"""
        return self.doi is not None and self.doi != ""
    
    @property
    def has_arxiv(self) -> bool:
        """arXiv IDがあるかどうか"""
        return self.arxiv_id is not None and self.arxiv_id != ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [a.to_dict() for a in self.authors],
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "citations": [c.to_dict() for c in self.citations],
            "references": [r.to_dict() for r in self.references],
            "fields_of_study": self.fields_of_study,
            "source": self.source,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperMetadata":
        """辞書から生成"""
        retrieved_at = None
        if data.get("retrieved_at"):
            try:
                retrieved_at = datetime.fromisoformat(data["retrieved_at"])
            except (ValueError, TypeError):
                pass
        
        return cls(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=[Author.from_dict(a) for a in data.get("authors", [])],
            year=data.get("year"),
            venue=data.get("venue"),
            doi=data.get("doi"),
            arxiv_id=data.get("arxiv_id"),
            citation_count=data.get("citation_count", 0),
            reference_count=data.get("reference_count", 0),
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
            references=[Citation.from_dict(r) for r in data.get("references", [])],
            fields_of_study=data.get("fields_of_study", []),
            source=data.get("source", ""),
            url=data.get("url"),
            pdf_url=data.get("pdf_url"),
            retrieved_at=retrieved_at,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExternalAPIConfig:
    """外部API設定の基底クラス
    
    Attributes:
        timeout: リクエストタイムアウト（秒）
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）
        rate_limit_delay: レート制限時の待機時間（秒）
        user_agent: User-Agentヘッダー
    """
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 60.0
    user_agent: str = "MONJYU/1.0 (https://github.com/monjyu)"


# =============================================================================
# Protocol
# =============================================================================


class ExternalAPIClientProtocol(Protocol):
    """外部APIクライアントプロトコル"""
    
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """DOIで論文を取得"""
        ...
    
    async def search_papers(
        self, query: str, limit: int = 10
    ) -> List[PaperMetadata]:
        """論文を検索"""
        ...
    
    async def get_citations(
        self, paper_id: str, limit: int = 100
    ) -> List[Citation]:
        """引用を取得"""
        ...
    
    async def get_references(
        self, paper_id: str, limit: int = 100
    ) -> List[Citation]:
        """参照を取得"""
        ...


# =============================================================================
# Base Client
# =============================================================================


class ExternalAPIClient(ABC):
    """外部APIクライアントの基底クラス
    
    共通のHTTPリクエスト処理とリトライロジックを提供。
    """
    
    def __init__(self, config: Optional[ExternalAPIConfig] = None):
        self.config = config or ExternalAPIConfig()
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    @abstractmethod
    def api_name(self) -> str:
        """API名"""
        ...
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """ベースURL"""
        ...
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTPセッションを取得"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={"User-Agent": self.config.user_agent},
            )
        return self._session
    
    async def close(self) -> None:
        """セッションをクローズ"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """HTTPリクエストを実行（リトライ付き）
        
        Args:
            method: HTTPメソッド
            endpoint: エンドポイント（ベースURLからの相対パス）
            params: クエリパラメータ
            headers: 追加ヘッダー
            json_data: JSONボディ
            
        Returns:
            レスポンスJSON
            
        Raises:
            RateLimitError: レート制限超過
            APIResponseError: APIエラー
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.config.max_retries):
            try:
                async with session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=json_data,
                ) as response:
                    # レート制限チェック
                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        retry_seconds = int(retry_after) if retry_after else self.config.rate_limit_delay
                        raise RateLimitError(
                            f"Rate limit exceeded for {self.api_name}",
                            api_name=self.api_name,
                            retry_after=retry_seconds,
                        )
                    
                    # エラーレスポンスチェック
                    if response.status >= 400:
                        body = await response.text()
                        raise APIResponseError(
                            f"{self.api_name} returned {response.status}",
                            api_name=self.api_name,
                            status_code=response.status,
                            response_body=body,
                        )
                    
                    return await response.json()
                    
            except RateLimitError:
                raise
            except APIResponseError as e:
                if e.status_code >= 500:
                    # サーバーエラーはリトライ
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                continue
        
        raise ExternalAPIError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}",
            api_name=self.api_name,
        )
    
    @abstractmethod
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """DOIで論文を取得"""
        ...
    
    @abstractmethod
    async def search_papers(
        self, query: str, limit: int = 10
    ) -> List[PaperMetadata]:
        """論文を検索"""
        ...
    
    async def get_citations(
        self, paper_id: str, limit: int = 100
    ) -> List[Citation]:
        """引用を取得（デフォルト実装）"""
        return []
    
    async def get_references(
        self, paper_id: str, limit: int = 100
    ) -> List[Citation]:
        """参照を取得（デフォルト実装）"""
        return []
