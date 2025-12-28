"""InMemory VectorSearch implementation."""

import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .types import (
    EmbedderProtocol,
    IndexedDocument,
    SearchHit,
    VectorSearchConfig,
    VectorSearchProtocol,
    VectorSearchResult,
)


def cosine_similarity(
    query_vector: NDArray[np.float32],
    doc_vectors: NDArray[np.float32],
) -> NDArray[np.float32]:
    """コサイン類似度を計算
    
    Args:
        query_vector: クエリベクトル (dim,)
        doc_vectors: ドキュメントベクトル (n, dim)
        
    Returns:
        類似度スコア (n,)
    """
    # 正規化
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    doc_norms = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-10)
    
    # コサイン類似度
    return np.dot(doc_norms, query_norm)


def keyword_match_score(query: str, content: str) -> float:
    """キーワードマッチスコアを計算
    
    Args:
        query: クエリ文字列
        content: コンテンツ
        
    Returns:
        マッチスコア (0.0-1.0)
    """
    # 単語分割（英語+日本語対応）
    query_words = set(re.findall(r'\w+', query.lower()))
    content_words = set(re.findall(r'\w+', content.lower()))
    
    if not query_words:
        return 0.0
    
    # Jaccard係数風のスコア
    matches = len(query_words & content_words)
    return matches / len(query_words)


class InMemoryVectorSearch(VectorSearchProtocol):
    """インメモリベクトル検索
    
    テスト・開発用のインメモリ実装。
    """
    
    def __init__(
        self,
        embedder: EmbedderProtocol,
        config: Optional[VectorSearchConfig] = None,
    ):
        """初期化
        
        Args:
            embedder: 埋め込みモデル
            config: 設定
        """
        self.embedder = embedder
        self.config = config or VectorSearchConfig()
        self._documents: List[IndexedDocument] = []
        self._vectors: Optional[NDArray[np.float32]] = None
    
    async def add_documents(
        self,
        documents: List[IndexedDocument],
    ) -> int:
        """ドキュメントを追加
        
        Args:
            documents: 追加するドキュメント
            
        Returns:
            追加した件数
        """
        self._documents.extend(documents)
        
        # ベクトルキャッシュを更新
        if self._documents:
            self._vectors = np.array([d.vector for d in self._documents])
        
        return len(documents)
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> int:
        """テキストを追加（埋め込みを自動生成）
        
        Args:
            texts: テキスト一覧
            metadatas: メタデータ一覧
            chunk_ids: チャンクID一覧
            
        Returns:
            追加した件数
        """
        metadatas = metadatas or [{} for _ in texts]
        chunk_ids = chunk_ids or [f"chunk_{i}" for i in range(len(texts))]
        
        # バッチ埋め込み
        vectors = await self.embedder.embed_batch(texts)
        
        documents = [
            IndexedDocument(
                chunk_id=chunk_id,
                content=text,
                vector=vector,
                metadata=metadata,
                paper_id=metadata.get("paper_id"),
                paper_title=metadata.get("paper_title"),
                section_type=metadata.get("section_type"),
            )
            for chunk_id, text, vector, metadata in zip(chunk_ids, texts, vectors, metadatas)
        ]
        
        return await self.add_documents(documents)
    
    def count(self) -> int:
        """インデックス済みドキュメント数"""
        return len(self._documents)
    
    def clear(self) -> None:
        """インデックスをクリア"""
        self._documents = []
        self._vectors = None
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> VectorSearchResult:
        """テキストでベクトル検索
        
        Args:
            query: クエリ文字列
            top_k: 取得件数
            filter: フィルタ条件
            
        Returns:
            検索結果
        """
        start_time = time.time()
        
        if not self._documents:
            return VectorSearchResult(
                hits=[],
                total_count=0,
                processing_time_ms=0.0,
                query=query,
            )
        
        # クエリ埋め込み
        query_vector = await self.embedder.embed(query)
        
        result = await self.search_by_vector(query_vector, top_k, filter)
        result.query = query
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    async def search_by_vector(
        self,
        vector: NDArray[np.float32],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> VectorSearchResult:
        """ベクトルで直接検索
        
        Args:
            vector: クエリベクトル
            top_k: 取得件数
            filter: フィルタ条件
            
        Returns:
            検索結果
        """
        start_time = time.time()
        
        if not self._documents or self._vectors is None:
            return VectorSearchResult(
                hits=[],
                total_count=0,
                processing_time_ms=0.0,
            )
        
        # フィルタ適用
        filtered_indices = self._apply_filter(filter)
        
        if not filtered_indices:
            return VectorSearchResult(
                hits=[],
                total_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        # フィルタ後のベクトル
        filtered_vectors = self._vectors[filtered_indices]
        
        # コサイン類似度計算
        scores = cosine_similarity(vector, filtered_vectors)
        
        # 上位k件取得
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        hits = []
        for idx in top_indices:
            orig_idx = filtered_indices[idx]
            doc = self._documents[orig_idx]
            score = float(scores[idx])
            
            if score >= self.config.min_score:
                hits.append(SearchHit(
                    chunk_id=doc.chunk_id,
                    score=score,
                    content=doc.content,
                    metadata=doc.metadata if self.config.include_metadata else {},
                    paper_id=doc.paper_id,
                    paper_title=doc.paper_title,
                    section_type=doc.section_type,
                ))
        
        return VectorSearchResult(
            hits=hits,
            total_count=len(hits),
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> VectorSearchResult:
        """ハイブリッド検索（ベクトル + キーワード）
        
        Args:
            query: クエリ文字列
            top_k: 取得件数
            alpha: ベクトルスコアの重み（0.0-1.0）
            filter: フィルタ条件
            
        Returns:
            検索結果
        """
        start_time = time.time()
        
        if not self._documents or self._vectors is None:
            return VectorSearchResult(
                hits=[],
                total_count=0,
                processing_time_ms=0.0,
                query=query,
            )
        
        # クエリ埋め込み
        query_vector = await self.embedder.embed(query)
        
        # フィルタ適用
        filtered_indices = self._apply_filter(filter)
        
        if not filtered_indices:
            return VectorSearchResult(
                hits=[],
                total_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                query=query,
            )
        
        # フィルタ後のベクトル
        filtered_vectors = self._vectors[filtered_indices]
        
        # ベクトルスコア
        vector_scores = cosine_similarity(query_vector, filtered_vectors)
        
        # キーワードスコア
        keyword_scores = np.array([
            keyword_match_score(query, self._documents[idx].content)
            for idx in filtered_indices
        ])
        
        # ハイブリッドスコア
        combined_scores = alpha * vector_scores + (1 - alpha) * keyword_scores
        
        # 上位k件取得
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        hits = []
        for idx in top_indices:
            orig_idx = filtered_indices[idx]
            doc = self._documents[orig_idx]
            score = float(combined_scores[idx])
            
            if score >= self.config.min_score:
                hits.append(SearchHit(
                    chunk_id=doc.chunk_id,
                    score=score,
                    content=doc.content,
                    metadata=doc.metadata if self.config.include_metadata else {},
                    paper_id=doc.paper_id,
                    paper_title=doc.paper_title,
                    section_type=doc.section_type,
                ))
        
        return VectorSearchResult(
            hits=hits,
            total_count=len(hits),
            processing_time_ms=(time.time() - start_time) * 1000,
            query=query,
        )
    
    def _apply_filter(
        self,
        filter: Optional[Dict[str, Any]],
    ) -> List[int]:
        """フィルタを適用してインデックスを取得
        
        Args:
            filter: フィルタ条件
            
        Returns:
            フィルタ後のインデックス一覧
        """
        if not filter:
            return list(range(len(self._documents)))
        
        indices = []
        for i, doc in enumerate(self._documents):
            match = True
            for key, value in filter.items():
                # メタデータでフィルタ
                if key == "paper_id":
                    if doc.paper_id != value:
                        match = False
                        break
                elif key == "section_type":
                    if doc.section_type != value:
                        match = False
                        break
                elif key in doc.metadata:
                    if doc.metadata[key] != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                indices.append(i)
        
        return indices


def create_in_memory_search(
    embedder: EmbedderProtocol,
    config: Optional[VectorSearchConfig] = None,
) -> InMemoryVectorSearch:
    """InMemoryVectorSearchを作成
    
    Args:
        embedder: 埋め込みモデル
        config: 設定
        
    Returns:
        InMemoryVectorSearchインスタンス
    """
    return InMemoryVectorSearch(embedder=embedder, config=config)
