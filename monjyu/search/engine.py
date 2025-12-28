# Vector Search Engine
"""
ベクトル検索エンジン - Facade パターン

TASK-004-06: VectorSearchEngine実装
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from monjyu.search.answer_synthesizer import AnswerSynthesizer
from monjyu.search.base import (
    AnswerSynthesizerProtocol,
    QueryEncoderProtocol,
    SearchMode,
    SearchResponse,
    SearchResults,
    SynthesizedAnswer,
    VectorSearcherProtocol,
)
from monjyu.search.query_encoder import QueryEncoder, QueryExpander

if TYPE_CHECKING:
    from monjyu.search.query_encoder import EmbeddingClientProtocol, LLMClientProtocol


@dataclass
class VectorSearchConfig:
    """検索エンジン設定"""

    # 検索パラメータ
    top_k: int = 10
    threshold: float = 0.0
    mode: SearchMode = SearchMode.VECTOR

    # ハイブリッド検索設定
    hybrid_alpha: float = 0.5  # ベクトルスコアの重み

    # クエリ拡張設定
    expand_query: bool = False
    num_expansions: int = 3

    # 回答合成設定
    synthesize: bool = True
    system_prompt: str | None = None

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "top_k": self.top_k,
            "threshold": self.threshold,
            "mode": self.mode.value,
            "hybrid_alpha": self.hybrid_alpha,
            "expand_query": self.expand_query,
            "num_expansions": self.num_expansions,
            "synthesize": self.synthesize,
        }


class VectorSearchEngine:
    """ベクトル検索エンジン（Facade）"""

    def __init__(
        self,
        embedding_client: "EmbeddingClientProtocol",
        vector_searcher: VectorSearcherProtocol,
        llm_client: "LLMClientProtocol",
        config: VectorSearchConfig | None = None,
    ):
        """
        Args:
            embedding_client: 埋め込みクライアント
            vector_searcher: ベクトル検索クライアント
            llm_client: LLMクライアント
            config: 検索設定
        """
        self.encoder = QueryEncoder(embedding_client)
        self.searcher = vector_searcher
        self.synthesizer = AnswerSynthesizer(llm_client)
        self.expander = QueryExpander(llm_client)
        self.config = config or VectorSearchConfig()
        self._llm_client = llm_client
        self._embedding_client = embedding_client

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: SearchMode | None = None,
        synthesize: bool | None = None,
        threshold: float | None = None,
    ) -> SearchResponse:
        """
        検索を実行

        Args:
            query: 検索クエリ
            top_k: 返す結果数（省略時はconfig値）
            mode: 検索モード（省略時はconfig値）
            synthesize: 回答合成を行うか（省略時はconfig値）
            threshold: 類似度閾値（省略時はconfig値）

        Returns:
            検索レスポンス
        """
        total_start = time.time()

        # パラメータ設定
        top_k = top_k if top_k is not None else self.config.top_k
        mode = mode if mode is not None else self.config.mode
        synthesize = synthesize if synthesize is not None else self.config.synthesize
        threshold = threshold if threshold is not None else self.config.threshold

        # 1. クエリエンコード
        query_vector = self.encoder.encode(query)

        # 2. 検索実行
        search_start = time.time()
        search_results = self._execute_search(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            mode=mode,
            threshold=threshold,
        )
        search_time = (time.time() - search_start) * 1000

        # 3. 回答合成（オプション）
        synthesis_start = time.time()
        if synthesize and search_results.hits:
            answer = self.synthesizer.synthesize(
                query=query,
                context=search_results.hits,
                system_prompt=self.config.system_prompt,
            )
        else:
            answer = SynthesizedAnswer(answer="", citations=[])
        synthesis_time = (time.time() - synthesis_start) * 1000

        total_time = (time.time() - total_start) * 1000

        return SearchResponse(
            query=query,
            answer=answer,
            search_results=search_results,
            total_time_ms=total_time,
            search_time_ms=search_time,
            synthesis_time_ms=synthesis_time,
            mode=mode,
            top_k=top_k,
        )

    def _execute_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        mode: SearchMode,
        threshold: float,
    ) -> SearchResults:
        """検索を実行（内部メソッド）"""
        if mode == SearchMode.HYBRID:
            return self.searcher.hybrid_search(
                query_text=query,
                query_vector=query_vector,
                top_k=top_k,
                alpha=self.config.hybrid_alpha,
            )
        elif mode == SearchMode.KEYWORD:
            # キーワード検索はハイブリッドでalpha=0として実装
            return self.searcher.hybrid_search(
                query_text=query,
                query_vector=query_vector,
                top_k=top_k,
                alpha=0.0,
            )
        else:  # VECTOR
            return self.searcher.search(
                query_vector=query_vector,
                top_k=top_k,
                threshold=threshold,
            )

    def search_with_expansion(
        self,
        query: str,
        top_k: int | None = None,
        mode: SearchMode | None = None,
        num_expansions: int | None = None,
    ) -> SearchResponse:
        """
        クエリ拡張を伴う検索

        Args:
            query: 検索クエリ
            top_k: 返す結果数
            mode: 検索モード
            num_expansions: 拡張クエリ数

        Returns:
            検索レスポンス
        """
        total_start = time.time()

        # パラメータ設定
        top_k = top_k if top_k is not None else self.config.top_k
        mode = mode if mode is not None else self.config.mode
        num_expansions = num_expansions or self.config.num_expansions

        # 1. クエリ拡張
        expanded_queries = self.expander.expand(query, num_expansions)

        # 2. 各クエリで検索
        all_hits_map: dict[str, any] = {}
        for expanded_query in expanded_queries:
            query_vector = self.encoder.encode(expanded_query)
            results = self._execute_search(
                query=expanded_query,
                query_vector=query_vector,
                top_k=top_k,
                mode=mode,
                threshold=self.config.threshold,
            )

            # スコアを集約
            for hit in results.hits:
                if hit.text_unit_id not in all_hits_map:
                    all_hits_map[hit.text_unit_id] = hit
                else:
                    # 最高スコアを保持
                    if hit.score > all_hits_map[hit.text_unit_id].score:
                        all_hits_map[hit.text_unit_id] = hit

        # 3. Top-K選択
        all_hits = sorted(all_hits_map.values(), key=lambda x: x.score, reverse=True)
        top_hits = all_hits[:top_k]

        search_results = SearchResults(
            hits=top_hits,
            total_count=len(top_hits),
            search_time_ms=0,  # 後で更新
        )

        # 4. 回答合成
        synthesis_start = time.time()
        if self.config.synthesize and top_hits:
            answer = self.synthesizer.synthesize(
                query=query,
                context=top_hits,
                system_prompt=self.config.system_prompt,
            )
        else:
            answer = SynthesizedAnswer(answer="", citations=[])
        synthesis_time = (time.time() - synthesis_start) * 1000

        total_time = (time.time() - total_start) * 1000
        search_time = total_time - synthesis_time

        return SearchResponse(
            query=query,
            answer=answer,
            search_results=search_results,
            total_time_ms=total_time,
            search_time_ms=search_time,
            synthesis_time_ms=synthesis_time,
            mode=mode,
            top_k=top_k,
        )

    def retrieve_only(
        self,
        query: str,
        top_k: int | None = None,
        mode: SearchMode | None = None,
    ) -> SearchResults:
        """
        検索のみ実行（回答合成なし）

        Args:
            query: 検索クエリ
            top_k: 返す結果数
            mode: 検索モード

        Returns:
            検索結果
        """
        response = self.search(
            query=query,
            top_k=top_k,
            mode=mode,
            synthesize=False,
        )
        return response.search_results

    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            "config": self.config.to_dict(),
            "encoder_cache_size": len(self.encoder._cache),
        }


# === Factory Functions ===


def create_local_search_engine(
    vector_db_path: str = "./output/index/level_0/vector_index",
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
    config: VectorSearchConfig | None = None,
) -> VectorSearchEngine:
    """
    ローカル環境用検索エンジンを作成

    Args:
        vector_db_path: LanceDBパス
        embedding_model: Ollama埋め込みモデル
        llm_model: Ollama LLMモデル
        ollama_host: Ollamaホスト
        config: 検索設定

    Returns:
        VectorSearchEngine
    """
    from monjyu.search.answer_synthesizer import OllamaLLMClient
    from monjyu.search.query_encoder import OllamaEmbeddingClient
    from monjyu.search.vector_searcher import LanceDBVectorSearcher

    embedding_client = OllamaEmbeddingClient(model=embedding_model, host=ollama_host)
    llm_client = OllamaLLMClient(model=llm_model, host=ollama_host)
    vector_searcher = LanceDBVectorSearcher(db_path=vector_db_path)

    return VectorSearchEngine(
        embedding_client=embedding_client,
        vector_searcher=vector_searcher,
        llm_client=llm_client,
        config=config,
    )


def create_azure_search_engine(
    search_endpoint: str,
    search_api_key: str,
    search_index_name: str,
    openai_endpoint: str,
    openai_api_key: str,
    embedding_deployment: str,
    llm_deployment: str,
    config: VectorSearchConfig | None = None,
) -> VectorSearchEngine:
    """
    Azure環境用検索エンジンを作成

    Args:
        search_endpoint: Azure AI Searchエンドポイント
        search_api_key: Azure AI Search APIキー
        search_index_name: インデックス名
        openai_endpoint: Azure OpenAIエンドポイント
        openai_api_key: Azure OpenAI APIキー
        embedding_deployment: 埋め込みデプロイメント名
        llm_deployment: LLMデプロイメント名
        config: 検索設定

    Returns:
        VectorSearchEngine
    """
    from monjyu.search.answer_synthesizer import AzureOpenAILLMClient
    from monjyu.search.query_encoder import AzureOpenAIEmbeddingClient
    from monjyu.search.vector_searcher import AzureAISearchVectorSearcher

    embedding_client = AzureOpenAIEmbeddingClient(
        endpoint=openai_endpoint,
        api_key=openai_api_key,
        deployment_name=embedding_deployment,
    )
    llm_client = AzureOpenAILLMClient(
        endpoint=openai_endpoint,
        api_key=openai_api_key,
        deployment_name=llm_deployment,
    )
    vector_searcher = AzureAISearchVectorSearcher(
        endpoint=search_endpoint,
        api_key=search_api_key,
        index_name=search_index_name,
    )

    return VectorSearchEngine(
        embedding_client=embedding_client,
        vector_searcher=vector_searcher,
        llm_client=llm_client,
        config=config,
    )
