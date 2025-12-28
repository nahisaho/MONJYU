# MONJYU Main Facade
"""
monjyu.api.monjyu - メインFacade API

FEAT-007: Python API (MONJYU Facade)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from monjyu.api.base import (
    MONJYUConfig,
    MONJYUStatus,
    SearchMode,
    IndexLevel,
    IndexStatus,
    SearchResult,
    DocumentInfo,
    Citation,
    IndexBuildResult,
)
from monjyu.api.config import ConfigManager
from monjyu.api.state import StateManager
from monjyu.api.factory import ComponentFactory


class MONJYU:
    """MONJYU メインAPI (Facade)"""

    def __init__(
        self,
        config: str | Path | dict[str, Any] | MONJYUConfig | None = None,
    ):
        """
        MONJYU を初期化

        Args:
            config: 設定ファイルパス、辞書、またはMONJYUConfigオブジェクト
        """
        # 設定読み込み
        if config is None:
            self._config_manager = ConfigManager()
        elif isinstance(config, (str, Path)):
            self._config_manager = ConfigManager.from_yaml(config)
        elif isinstance(config, dict):
            self._config_manager = ConfigManager.from_dict(config)
        elif isinstance(config, MONJYUConfig):
            self._config_manager = ConfigManager.from_config(config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        # コンポーネント初期化
        self._factory = ComponentFactory(self.config)
        self._state_manager = StateManager(self.config.output_path)

    @property
    def config(self) -> MONJYUConfig:
        """設定を取得"""
        return self._config_manager.config

    def get_status(self) -> MONJYUStatus:
        """ステータスを取得"""
        return self._state_manager.status

    # ========== インデックス構築 ==========

    def index(
        self,
        path: str | Path,
        levels: list[IndexLevel] | None = None,
        rebuild: bool = False,
        show_progress: bool = True,
    ) -> MONJYUStatus:
        """
        ドキュメントをインデックス化

        Args:
            path: ドキュメントのパス（ファイルまたはディレクトリ）
            levels: 構築するインデックスレベル（デフォルト: config設定）
            rebuild: 既存インデックスを再構築するか
            show_progress: 進捗を表示するか

        Returns:
            MONJYUStatus: 更新後のステータス
        """
        try:
            self._state_manager.update(index_status=IndexStatus.BUILDING)

            levels = levels or self.config.index_levels
            path = Path(path)

            # パスの検証
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

            start_time = time.time()

            # 1. ドキュメント処理
            documents = self._process_documents(path, show_progress)

            # 2. Level 0 インデックス構築
            if IndexLevel.LEVEL_0 in levels:
                self._build_level0_index(documents, rebuild, show_progress)

            # 3. Level 1 インデックス構築
            if IndexLevel.LEVEL_1 in levels:
                self._build_level1_index(documents, rebuild, show_progress)

            # 4. 引用ネットワーク構築
            self._build_citation_network(documents, show_progress)

            # 完了
            elapsed_ms = (time.time() - start_time) * 1000
            self._state_manager.update(
                index_status=IndexStatus.READY,
                index_levels_built=levels,
            )

            if show_progress:
                print(f"Index built successfully in {elapsed_ms:.0f}ms")

            return self.get_status()

        except Exception as e:
            self._state_manager.update(
                index_status=IndexStatus.ERROR,
                last_error=str(e),
            )
            raise

    def _process_documents(
        self,
        path: Path,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """ドキュメントを処理"""
        documents: list[dict[str, Any]] = []

        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob("*.md")) + list(path.rglob("*.txt"))

        if show_progress:
            print(f"Processing {len(files)} files...")

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                doc = {
                    "id": file_path.stem,
                    "title": file_path.stem,
                    "content": content,
                    "path": str(file_path),
                }
                documents.append(doc)
            except Exception as e:
                if show_progress:
                    print(f"Warning: Failed to process {file_path}: {e}")

        self._state_manager.update(document_count=len(documents))

        return documents

    def _build_level0_index(
        self,
        documents: list[dict[str, Any]],
        rebuild: bool = False,
        show_progress: bool = True,
    ) -> IndexBuildResult:
        """Level 0 インデックスを構築"""
        if show_progress:
            print("Building Level 0 index (Vector)...")

        start_time = time.time()

        # チャンク分割
        text_units: list[dict[str, Any]] = []
        for doc in documents:
            chunks = self._chunk_text(
                doc["content"],
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            for i, chunk in enumerate(chunks):
                text_units.append({
                    "id": f"{doc['id']}_chunk_{i}",
                    "doc_id": doc["id"],
                    "text": chunk,
                    "chunk_index": i,
                })

        self._state_manager.update(text_unit_count=len(text_units))

        elapsed_ms = (time.time() - start_time) * 1000

        return IndexBuildResult(
            success=True,
            level=IndexLevel.LEVEL_0,
            duration_ms=elapsed_ms,
            items_processed=len(documents),
            items_indexed=len(text_units),
        )

    def _build_level1_index(
        self,
        documents: list[dict[str, Any]],
        rebuild: bool = False,
        show_progress: bool = True,
    ) -> IndexBuildResult:
        """Level 1 インデックスを構築"""
        if show_progress:
            print("Building Level 1 index (NLP Graph)...")

        start_time = time.time()

        # 簡易的なNLP処理（名詞句抽出のシミュレーション）
        noun_phrases: set[str] = set()
        for doc in documents:
            # 単語を抽出（簡易実装）
            words = doc["content"].split()
            for word in words:
                if len(word) > 3:
                    noun_phrases.add(word.lower())

        # コミュニティ検出のシミュレーション
        community_count = max(1, len(noun_phrases) // 10)

        self._state_manager.update(
            noun_phrase_count=len(noun_phrases),
            community_count=community_count,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return IndexBuildResult(
            success=True,
            level=IndexLevel.LEVEL_1,
            duration_ms=elapsed_ms,
            items_processed=len(documents),
            items_indexed=len(noun_phrases),
        )

    def _build_citation_network(
        self,
        documents: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> None:
        """引用ネットワークを構築"""
        if show_progress:
            print("Building citation network...")

        # 引用ネットワークマネージャーを使用
        # 現時点ではモック実装
        self._state_manager.update(citation_edge_count=0)

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 1200,
        overlap: int = 100,
    ) -> list[str]:
        """テキストをチャンクに分割"""
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks

    # ========== 検索 ==========

    def search(
        self,
        query: str,
        mode: SearchMode | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """
        検索を実行

        Args:
            query: 検索クエリ
            mode: 検索モード（デフォルト: config設定）
            top_k: 取得件数
            **kwargs: モード固有のパラメータ

        Returns:
            SearchResult: 検索結果
        """
        mode = mode or self.config.default_search_mode
        top_k = top_k or self.config.default_top_k

        start_time = time.time()

        if mode == SearchMode.AUTO:
            mode = self._select_search_mode(query)

        if mode == SearchMode.VECTOR:
            result = self._search_vector(query, top_k, **kwargs)
        elif mode == SearchMode.LAZY:
            result = self._search_lazy(query, top_k, **kwargs)
        else:
            raise ValueError(f"Unsupported search mode: {mode}")

        result.total_time_ms = (time.time() - start_time) * 1000
        return result

    def _select_search_mode(self, query: str) -> SearchMode:
        """検索モードを自動選択"""
        # 複雑なクエリの判定
        complex_indicators = [
            "why",
            "how",
            "explain",
            "compare",
            "difference",
            "relationship",
        ]
        words = query.lower().split()

        if len(words) > 10 or any(ind in words for ind in complex_indicators):
            return SearchMode.LAZY
        else:
            return SearchMode.VECTOR

    def _search_vector(
        self,
        query: str,
        top_k: int,
        **kwargs: Any,
    ) -> SearchResult:
        """ベクトル検索"""
        # モック実装
        embedding_client = self._factory.get_embedding_client()
        llm_client = self._factory.get_llm_client()

        # 埋め込み生成
        query_vector = embedding_client.embed(query)

        # LLMで回答生成
        answer = llm_client.generate(f"Answer this query: {query}")

        return SearchResult(
            query=query,
            answer=answer,
            citations=[],
            search_mode=SearchMode.VECTOR,
            search_level=0,
            llm_calls=1,
        )

    def _search_lazy(
        self,
        query: str,
        top_k: int,
        **kwargs: Any,
    ) -> SearchResult:
        """Lazy検索（LazyGraphRAG）"""
        max_level = kwargs.get("max_level", 1)

        # モック実装
        embedding_client = self._factory.get_embedding_client()
        llm_client = self._factory.get_llm_client()

        # 埋め込み生成
        query_vector = embedding_client.embed(query)

        # LLMで回答生成
        answer = llm_client.generate(f"[LazyGraphRAG] Answer: {query}")

        return SearchResult(
            query=query,
            answer=answer,
            citations=[],
            search_mode=SearchMode.LAZY,
            search_level=max_level,
            llm_calls=2,  # Lazy検索は複数回LLMを呼ぶ
        )

    # ========== ドキュメント操作 ==========

    def get_document(self, document_id: str) -> DocumentInfo | None:
        """ドキュメント情報を取得"""
        # モック実装
        return None

    def list_documents(self, limit: int = 100) -> list[DocumentInfo]:
        """ドキュメント一覧を取得"""
        # モック実装
        return []

    # ========== 引用ネットワーク ==========

    def get_citation_network(self) -> Any:
        """引用ネットワークマネージャーを取得"""
        return self._factory.get_citation_network_manager()

    def find_related_papers(
        self,
        document_id: str,
        top_k: int = 10,
    ) -> list[Any]:
        """関連論文を検索"""
        manager = self.get_citation_network()
        return manager.find_related_papers(document_id, method="both")[:top_k]


def create_monjyu(
    config: str | Path | dict[str, Any] | MONJYUConfig | None = None,
) -> MONJYU:
    """MONJYUインスタンスを作成するファクトリ関数"""
    return MONJYU(config)
