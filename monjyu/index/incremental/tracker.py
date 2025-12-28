"""Document Tracker for Incremental Index.

ドキュメントの変更を追跡し、差分を検出する。
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from monjyu.index.incremental.types import (
    ChangeType,
    DocumentChange,
    DocumentRecord,
    IndexChangeSet,
    IncrementalIndexState,
)

if TYPE_CHECKING:
    from monjyu.document.models import AcademicPaperDocument, TextUnit

logger = logging.getLogger(__name__)


def compute_content_hash(content: str) -> str:
    """コンテンツハッシュを計算
    
    Args:
        content: コンテンツ文字列
        
    Returns:
        SHA-256ハッシュ（最初の16文字）
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def compute_document_hash(document: "AcademicPaperDocument") -> str:
    """ドキュメントハッシュを計算
    
    Args:
        document: ドキュメント
        
    Returns:
        Content Hash
    """
    # タイトル + 本文 + メタデータからハッシュ生成
    content_parts = [
        document.title or "",
        document.abstract or "",
    ]
    
    # セクションがある場合は追加
    if hasattr(document, "sections") and document.sections:
        for section in document.sections:
            if hasattr(section, "text"):
                content_parts.append(section.text or "")
    
    # 全テキストがある場合は追加
    if hasattr(document, "text") and document.text:
        content_parts.append(document.text)
    
    combined = "\n".join(content_parts)
    return compute_content_hash(combined)


def compute_text_unit_hash(text_unit: "TextUnit") -> str:
    """TextUnitハッシュを計算
    
    Args:
        text_unit: TextUnit
        
    Returns:
        Content Hash
    """
    return compute_content_hash(text_unit.text)


class DocumentTracker:
    """ドキュメント追跡器
    
    ドキュメントの追加・変更・削除を追跡し、
    差分インデックス更新のための変更セットを生成する。
    
    Example:
        >>> tracker = DocumentTracker(state)
        >>> 
        >>> # 変更検出
        >>> change_set = tracker.detect_changes(
        ...     documents=new_documents,
        ...     text_units=new_text_units,
        ... )
        >>> 
        >>> print(f"Added: {change_set.added_count}")
        >>> print(f"Modified: {change_set.modified_count}")
        >>> print(f"Deleted: {change_set.deleted_count}")
    """
    
    def __init__(self, state: IncrementalIndexState) -> None:
        """初期化
        
        Args:
            state: インクリメンタルインデックス状態
        """
        self.state = state
    
    def detect_changes(
        self,
        documents: List["AcademicPaperDocument"],
        text_units: List["TextUnit"],
        check_deleted: bool = True,
    ) -> IndexChangeSet:
        """変更を検出
        
        Args:
            documents: 新しいドキュメントリスト
            text_units: 新しいTextUnitリスト
            check_deleted: 削除されたドキュメントも検出するか
            
        Returns:
            変更セット
        """
        import uuid
        
        changes: List[DocumentChange] = []
        
        # TextUnitをドキュメントIDでグループ化
        doc_text_units: Dict[str, List["TextUnit"]] = {}
        for unit in text_units:
            doc_id = unit.document_id or ""
            if doc_id not in doc_text_units:
                doc_text_units[doc_id] = []
            doc_text_units[doc_id].append(unit)
        
        # 新しいドキュメントIDのセット
        new_doc_ids = {doc.id for doc in documents}
        
        # 各ドキュメントの変更を検出
        for document in documents:
            units = doc_text_units.get(document.id, [])
            change = self._detect_document_change(document, units)
            changes.append(change)
        
        # 削除されたドキュメントを検出
        if check_deleted:
            for doc_id in self.state.documents:
                if doc_id not in new_doc_ids:
                    existing = self.state.get_document(doc_id)
                    if existing:
                        changes.append(DocumentChange(
                            document_id=doc_id,
                            change_type=ChangeType.DELETED,
                            old_hash=existing.content_hash,
                            old_text_unit_ids=existing.text_unit_ids.copy(),
                        ))
        
        # 変更セットを作成
        change_set = IndexChangeSet(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            changes=changes,
        )
        
        logger.info(
            f"Detected changes: "
            f"added={change_set.added_count}, "
            f"modified={change_set.modified_count}, "
            f"deleted={change_set.deleted_count}"
        )
        
        return change_set
    
    def _detect_document_change(
        self,
        document: "AcademicPaperDocument",
        text_units: List["TextUnit"],
    ) -> DocumentChange:
        """単一ドキュメントの変更を検出
        
        Args:
            document: ドキュメント
            text_units: 関連するTextUnitリスト
            
        Returns:
            ドキュメント変更情報
        """
        new_hash = compute_document_hash(document)
        new_unit_ids = [unit.id for unit in text_units]
        
        existing = self.state.get_document(document.id)
        
        if existing is None:
            # 新規追加
            return DocumentChange(
                document_id=document.id,
                change_type=ChangeType.ADDED,
                new_hash=new_hash,
                new_text_unit_ids=new_unit_ids,
            )
        
        # 既存ドキュメント
        if existing.content_hash == new_hash:
            # 変更なし
            return DocumentChange(
                document_id=document.id,
                change_type=ChangeType.UNCHANGED,
                old_hash=existing.content_hash,
                new_hash=new_hash,
                old_text_unit_ids=existing.text_unit_ids,
                new_text_unit_ids=new_unit_ids,
            )
        
        # 内容変更
        return DocumentChange(
            document_id=document.id,
            change_type=ChangeType.MODIFIED,
            old_hash=existing.content_hash,
            new_hash=new_hash,
            old_text_unit_ids=existing.text_unit_ids,
            new_text_unit_ids=new_unit_ids,
        )
    
    def apply_changes(
        self,
        change_set: IndexChangeSet,
        documents: List["AcademicPaperDocument"],
        text_units: List["TextUnit"],
    ) -> None:
        """変更を状態に適用
        
        Args:
            change_set: 変更セット
            documents: ドキュメントリスト
            text_units: TextUnitリスト
        """
        # ドキュメントをIDでマップ
        doc_map = {doc.id: doc for doc in documents}
        
        # TextUnitをドキュメントIDでグループ化
        doc_text_units: Dict[str, List[str]] = {}
        for unit in text_units:
            doc_id = unit.document_id or ""
            if doc_id not in doc_text_units:
                doc_text_units[doc_id] = []
            doc_text_units[doc_id].append(unit.id)
        
        for change in change_set.changes:
            if change.change_type == ChangeType.ADDED:
                # 新規追加
                doc = doc_map.get(change.document_id)
                if doc:
                    record = DocumentRecord(
                        document_id=change.document_id,
                        content_hash=change.new_hash or "",
                        text_unit_ids=change.new_text_unit_ids,
                        file_path=getattr(doc, "file_path", None),
                        metadata={
                            "title": doc.title,
                        },
                    )
                    self.state.add_document(record)
            
            elif change.change_type == ChangeType.MODIFIED:
                # 更新
                existing = self.state.get_document(change.document_id)
                if existing:
                    existing.content_hash = change.new_hash or ""
                    existing.text_unit_ids = change.new_text_unit_ids
                    existing.indexed_at = datetime.now(timezone.utc).isoformat()
                    self.state.last_update = datetime.now(timezone.utc).isoformat()
            
            elif change.change_type == ChangeType.DELETED:
                # 削除
                self.state.remove_document(change.document_id)
        
        # 変更セットを適用済みにマーク
        change_set.applied = True
        change_set.applied_at = datetime.now(timezone.utc).isoformat()
        
        # 履歴に追加
        self.state.add_change_set(change_set)
        
        logger.info(
            f"Applied changes: "
            f"added={change_set.added_count}, "
            f"modified={change_set.modified_count}, "
            f"deleted={change_set.deleted_count}"
        )
    
    def get_documents_to_reindex(
        self,
        change_set: IndexChangeSet,
    ) -> Set[str]:
        """再インデックスが必要なドキュメントIDを取得
        
        Args:
            change_set: 変更セット
            
        Returns:
            再インデックスが必要なドキュメントIDのセット
        """
        result = set()
        
        for change in change_set.changes:
            if change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
                result.add(change.document_id)
        
        return result
    
    def get_text_units_to_add(
        self,
        change_set: IndexChangeSet,
        text_units: List["TextUnit"],
    ) -> List["TextUnit"]:
        """追加するTextUnitを取得
        
        Args:
            change_set: 変更セット
            text_units: TextUnitリスト
            
        Returns:
            追加するTextUnitリスト
        """
        # 追加するTextUnit IDのセット
        ids_to_add = set(change_set.all_text_units_to_add)
        
        # 変更されたドキュメントの全TextUnitも追加
        for change in change_set.changes:
            if change.change_type == ChangeType.MODIFIED:
                ids_to_add.update(change.new_text_unit_ids)
        
        return [unit for unit in text_units if unit.id in ids_to_add]
    
    def get_text_unit_ids_to_remove(
        self,
        change_set: IndexChangeSet,
    ) -> List[str]:
        """削除するTextUnit IDを取得
        
        Args:
            change_set: 変更セット
            
        Returns:
            削除するTextUnit IDリスト
        """
        ids_to_remove = set(change_set.all_text_units_to_remove)
        
        # 変更されたドキュメントの古いTextUnitも削除
        for change in change_set.changes:
            if change.change_type == ChangeType.MODIFIED:
                ids_to_remove.update(change.old_text_unit_ids)
        
        return list(ids_to_remove)


class FileTracker:
    """ファイルベースの追跡器
    
    ファイルシステムの変更を追跡する。
    """
    
    def __init__(self, state: IncrementalIndexState) -> None:
        """初期化
        
        Args:
            state: インクリメンタルインデックス状態
        """
        self.state = state
    
    def scan_directory(
        self,
        directory: Path,
        patterns: List[str] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """ディレクトリをスキャン
        
        Args:
            directory: スキャン対象ディレクトリ
            patterns: ファイルパターン（例: ["*.pdf", "*.txt"]）
            
        Returns:
            ファイルパス -> ファイル情報のマップ
        """
        if patterns is None:
            patterns = ["*.pdf", "*.txt", "*.md"]
        
        files = {}
        
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    files[str(file_path)] = {
                        "path": str(file_path),
                        "modified_time": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                        "size": stat.st_size,
                    }
        
        return files
    
    def detect_file_changes(
        self,
        directory: Path,
        patterns: List[str] | None = None,
    ) -> Dict[str, ChangeType]:
        """ファイル変更を検出
        
        Args:
            directory: スキャン対象ディレクトリ
            patterns: ファイルパターン
            
        Returns:
            ファイルパス -> 変更タイプのマップ
        """
        current_files = self.scan_directory(directory, patterns)
        
        # 既存のファイルパスを取得
        existing_paths = {
            record.file_path
            for record in self.state.documents.values()
            if record.file_path
        }
        
        changes = {}
        
        # 新規・変更ファイルを検出
        for path, info in current_files.items():
            if path not in existing_paths:
                changes[path] = ChangeType.ADDED
            else:
                # 変更時刻で変更を検出
                for record in self.state.documents.values():
                    if record.file_path == path:
                        if record.file_modified_time != info["modified_time"]:
                            changes[path] = ChangeType.MODIFIED
                        else:
                            changes[path] = ChangeType.UNCHANGED
                        break
        
        # 削除ファイルを検出
        current_paths = set(current_files.keys())
        for path in existing_paths:
            if path not in current_paths:
                changes[path] = ChangeType.DELETED
        
        return changes
