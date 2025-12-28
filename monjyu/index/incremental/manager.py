"""Incremental Index Manager.

差分インデックス更新を管理する。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from monjyu.index.incremental.tracker import (
    DocumentTracker,
    FileTracker,
    compute_document_hash,
)
from monjyu.index.incremental.types import (
    ChangeType,
    DocumentChange,
    DocumentRecord,
    IncrementalIndexConfig,
    IncrementalIndexState,
    IndexChangeSet,
)

if TYPE_CHECKING:
    from monjyu.document.models import AcademicPaperDocument, TextUnit
    from monjyu.index.level0.builder import Level0Index, Level0IndexBuilder

logger = logging.getLogger(__name__)


class IncrementalIndexManager:
    """インクリメンタルインデックスマネージャ
    
    ドキュメントの差分を検出し、効率的にインデックスを更新する。
    
    Features:
        - コンテンツハッシュによる変更検出
        - 追加・更新・削除の差分処理
        - バッチ処理による効率化
        - 変更履歴の管理
        - ドライランモード
    
    Example:
        >>> config = IncrementalIndexConfig(
        ...     output_dir="./output/index",
        ...     batch_size=50,
        ... )
        >>> manager = IncrementalIndexManager(config)
        >>> 
        >>> # 変更を検出
        >>> change_set = await manager.detect_changes(documents, text_units)
        >>> print(f"Changes: {change_set.total_changes}")
        >>> 
        >>> # 変更を適用
        >>> result = await manager.update(
        ...     documents=documents,
        ...     text_units=text_units,
        ...     builder=level0_builder,
        ... )
    """
    
    def __init__(
        self,
        config: IncrementalIndexConfig | None = None,
    ) -> None:
        """初期化
        
        Args:
            config: インクリメンタルインデックス設定
        """
        self.config = config or IncrementalIndexConfig()
        self._state: IncrementalIndexState | None = None
        self._tracker: DocumentTracker | None = None
        self._file_tracker: FileTracker | None = None
        
        # 出力ディレクトリを確保
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 状態ファイルパス
        self.state_path = self.output_dir / self.config.state_file
    
    @property
    def state(self) -> IncrementalIndexState:
        """状態を取得（遅延読み込み）"""
        if self._state is None:
            self._state = self._load_or_create_state()
        return self._state
    
    @property
    def tracker(self) -> DocumentTracker:
        """ドキュメント追跡器を取得"""
        if self._tracker is None:
            self._tracker = DocumentTracker(self.state)
        return self._tracker
    
    @property
    def file_tracker(self) -> FileTracker:
        """ファイル追跡器を取得"""
        if self._file_tracker is None:
            self._file_tracker = FileTracker(self.state)
        return self._file_tracker
    
    def _load_or_create_state(self) -> IncrementalIndexState:
        """状態を読み込みまたは新規作成"""
        if self.state_path.exists():
            try:
                state = IncrementalIndexState.load(self.state_path)
                logger.info(f"Loaded incremental state: {len(state.documents)} documents")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, creating new state")
        
        return IncrementalIndexState(version=self.config.state_version)
    
    def save_state(self) -> None:
        """状態を保存"""
        self.state.save(self.state_path)
        logger.info(f"Saved incremental state to {self.state_path}")
    
    def detect_changes(
        self,
        documents: List["AcademicPaperDocument"],
        text_units: List["TextUnit"],
        check_deleted: bool = True,
    ) -> IndexChangeSet:
        """変更を検出
        
        Args:
            documents: ドキュメントリスト
            text_units: TextUnitリスト
            check_deleted: 削除も検出するか
            
        Returns:
            変更セット
        """
        return self.tracker.detect_changes(
            documents=documents,
            text_units=text_units,
            check_deleted=check_deleted,
        )
    
    async def update(
        self,
        documents: List["AcademicPaperDocument"],
        text_units: List["TextUnit"],
        builder: "Level0IndexBuilder",
        on_progress: Callable[[int, int], None] | None = None,
    ) -> "Level0Index":
        """インデックスを更新
        
        Args:
            documents: ドキュメントリスト
            text_units: TextUnitリスト
            builder: Level0インデックスビルダー
            on_progress: 進捗コールバック
            
        Returns:
            更新後のLevel0Index
        """
        # 変更を検出
        change_set = self.detect_changes(documents, text_units)
        
        if change_set.total_changes == 0:
            logger.info("No changes detected, skipping update")
            return await builder.build(documents, text_units)
        
        logger.info(
            f"Updating index: "
            f"added={change_set.added_count}, "
            f"modified={change_set.modified_count}, "
            f"deleted={change_set.deleted_count}"
        )
        
        # ドライランモードの場合
        if self.config.dry_run:
            logger.info("Dry run mode: skipping actual update")
            return await builder.build(documents, text_units)
        
        # 追加・更新するTextUnitを取得
        units_to_add = self.tracker.get_text_units_to_add(change_set, text_units)
        
        # 削除するTextUnit IDを取得
        ids_to_remove = self.tracker.get_text_unit_ids_to_remove(change_set)
        
        # 初回ビルドまたは全更新が必要な場合
        if not self.state.documents or change_set.deleted_count > 0:
            # 削除がある場合は再構築
            logger.info("Rebuilding index due to deletions or empty state")
            
            # 削除されていないドキュメントのTextUnitのみ使用
            deleted_doc_ids = {
                c.document_id for c in change_set.changes 
                if c.change_type == ChangeType.DELETED
            }
            filtered_units = [
                u for u in text_units 
                if u.document_id not in deleted_doc_ids
            ]
            filtered_docs = [
                d for d in documents 
                if d.id not in deleted_doc_ids
            ]
            
            index = await builder.build(filtered_docs, filtered_units)
        else:
            # 差分追加
            logger.info(f"Adding {len(units_to_add)} text units to index")
            
            # 追加するドキュメントを取得
            docs_to_add = [
                d for d in documents 
                if d.id in self.tracker.get_documents_to_reindex(change_set)
            ]
            
            index = await builder.add(docs_to_add, units_to_add)
        
        # 状態を更新
        self.tracker.apply_changes(change_set, documents, text_units)
        
        # 状態を保存
        self.save_state()
        
        return index
    
    async def update_from_directory(
        self,
        directory: Path,
        document_loader: Callable[[Path], "AcademicPaperDocument"],
        text_unit_generator: Callable[["AcademicPaperDocument"], List["TextUnit"]],
        builder: "Level0IndexBuilder",
        patterns: List[str] | None = None,
    ) -> "Level0Index":
        """ディレクトリからインデックスを更新
        
        Args:
            directory: 対象ディレクトリ
            document_loader: ドキュメントローダー関数
            text_unit_generator: TextUnit生成関数
            builder: Level0インデックスビルダー
            patterns: ファイルパターン
            
        Returns:
            更新後のLevel0Index
        """
        # ファイル変更を検出
        file_changes = self.file_tracker.detect_file_changes(directory, patterns)
        
        # 変更があるファイルのみ処理
        changed_files = [
            path for path, change_type in file_changes.items()
            if change_type in (ChangeType.ADDED, ChangeType.MODIFIED)
        ]
        
        if not changed_files and not any(
            ct == ChangeType.DELETED for ct in file_changes.values()
        ):
            logger.info("No file changes detected")
            # 既存のインデックスを返す
            documents = []
            text_units = []
            for record in self.state.documents.values():
                if record.file_path:
                    try:
                        doc = document_loader(Path(record.file_path))
                        documents.append(doc)
                        text_units.extend(text_unit_generator(doc))
                    except Exception as e:
                        logger.warning(f"Failed to load {record.file_path}: {e}")
            
            return await builder.build(documents, text_units)
        
        logger.info(f"Processing {len(changed_files)} changed files")
        
        # ドキュメントを読み込み
        documents = []
        text_units = []
        
        for path in changed_files:
            try:
                doc = document_loader(Path(path))
                documents.append(doc)
                units = text_unit_generator(doc)
                text_units.extend(units)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        
        # 削除されていないファイルの既存ドキュメントも追加
        unchanged_files = [
            path for path, change_type in file_changes.items()
            if change_type == ChangeType.UNCHANGED
        ]
        
        for path in unchanged_files:
            try:
                doc = document_loader(Path(path))
                documents.append(doc)
                units = text_unit_generator(doc)
                text_units.extend(units)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        
        # インデックスを更新
        return await self.update(
            documents=documents,
            text_units=text_units,
            builder=builder,
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """インデックス状態のサマリーを取得
        
        Returns:
            サマリー情報
        """
        return {
            "total_documents": self.state.total_documents,
            "total_text_units": self.state.total_text_units,
            "last_update": self.state.last_update,
            "version": self.state.version,
            "change_history_count": len(self.state.change_history),
        }
    
    def reset(self) -> None:
        """状態をリセット"""
        self._state = IncrementalIndexState(version=self.config.state_version)
        self._tracker = None
        self._file_tracker = None
        
        # 状態ファイルを削除
        if self.state_path.exists():
            self.state_path.unlink()
        
        logger.info("Reset incremental state")
