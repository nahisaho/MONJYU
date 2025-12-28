"""Incremental Index Types.

差分インデックス更新で使用する型定義。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ChangeType(Enum):
    """変更タイプ
    
    Attributes:
        ADDED: 新規追加
        MODIFIED: 内容変更
        DELETED: 削除
        UNCHANGED: 変更なし
    """
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class DocumentChange:
    """ドキュメント変更情報
    
    Attributes:
        document_id: ドキュメントID
        change_type: 変更タイプ
        old_hash: 変更前のContent Hash
        new_hash: 変更後のContent Hash
        old_text_unit_ids: 変更前のTextUnit ID リスト
        new_text_unit_ids: 変更後のTextUnit ID リスト
        detected_at: 検出日時
    """
    document_id: str
    change_type: ChangeType
    old_hash: str | None = None
    new_hash: str | None = None
    old_text_unit_ids: List[str] = field(default_factory=list)
    new_text_unit_ids: List[str] = field(default_factory=list)
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    @property
    def is_modified(self) -> bool:
        """内容が変更されたか（ADDEDはfalse）"""
        return self.change_type == ChangeType.MODIFIED
    
    @property
    def text_units_to_add(self) -> List[str]:
        """追加するTextUnit ID"""
        if self.change_type == ChangeType.ADDED:
            return self.new_text_unit_ids
        elif self.change_type == ChangeType.MODIFIED:
            return [
                id_ for id_ in self.new_text_unit_ids
                if id_ not in self.old_text_unit_ids
            ]
        return []
    
    @property
    def text_units_to_remove(self) -> List[str]:
        """削除するTextUnit ID"""
        if self.change_type == ChangeType.DELETED:
            return self.old_text_unit_ids
        elif self.change_type == ChangeType.MODIFIED:
            return [
                id_ for id_ in self.old_text_unit_ids
                if id_ not in self.new_text_unit_ids
            ]
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "document_id": self.document_id,
            "change_type": self.change_type.name,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "old_text_unit_ids": self.old_text_unit_ids,
            "new_text_unit_ids": self.new_text_unit_ids,
            "detected_at": self.detected_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentChange:
        """辞書から作成"""
        return cls(
            document_id=data["document_id"],
            change_type=ChangeType[data["change_type"]],
            old_hash=data.get("old_hash"),
            new_hash=data.get("new_hash"),
            old_text_unit_ids=data.get("old_text_unit_ids", []),
            new_text_unit_ids=data.get("new_text_unit_ids", []),
            detected_at=data.get("detected_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class IndexChangeSet:
    """インデックス変更セット
    
    Attributes:
        id: 変更セットID
        created_at: 作成日時
        changes: 変更リスト
        applied: 適用済みフラグ
        applied_at: 適用日時
    """
    id: str
    created_at: str
    changes: List[DocumentChange] = field(default_factory=list)
    applied: bool = False
    applied_at: str | None = None
    
    @property
    def added_count(self) -> int:
        """追加ドキュメント数"""
        return sum(1 for c in self.changes if c.change_type == ChangeType.ADDED)
    
    @property
    def modified_count(self) -> int:
        """変更ドキュメント数"""
        return sum(1 for c in self.changes if c.change_type == ChangeType.MODIFIED)
    
    @property
    def deleted_count(self) -> int:
        """削除ドキュメント数"""
        return sum(1 for c in self.changes if c.change_type == ChangeType.DELETED)
    
    @property
    def total_changes(self) -> int:
        """変更総数（UNCHANGED除く）"""
        return sum(
            1 for c in self.changes
            if c.change_type != ChangeType.UNCHANGED
        )
    
    @property
    def all_text_units_to_add(self) -> List[str]:
        """追加するすべてのTextUnit ID"""
        result = []
        for change in self.changes:
            result.extend(change.text_units_to_add)
        return result
    
    @property
    def all_text_units_to_remove(self) -> List[str]:
        """削除するすべてのTextUnit ID"""
        result = []
        for change in self.changes:
            result.extend(change.text_units_to_remove)
        return result
    
    def get_changes_by_type(self, change_type: ChangeType) -> List[DocumentChange]:
        """タイプ別の変更を取得"""
        return [c for c in self.changes if c.change_type == change_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "changes": [c.to_dict() for c in self.changes],
            "applied": self.applied,
            "applied_at": self.applied_at,
            "summary": {
                "added": self.added_count,
                "modified": self.modified_count,
                "deleted": self.deleted_count,
                "total": self.total_changes,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IndexChangeSet:
        """辞書から作成"""
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            changes=[DocumentChange.from_dict(c) for c in data.get("changes", [])],
            applied=data.get("applied", False),
            applied_at=data.get("applied_at"),
        )


@dataclass
class DocumentRecord:
    """ドキュメントレコード（追跡用）
    
    Attributes:
        document_id: ドキュメントID
        content_hash: Content Hash
        text_unit_ids: 関連するTextUnit IDリスト
        file_path: ファイルパス（オプション）
        file_modified_time: ファイル更新日時
        indexed_at: インデックス日時
        metadata: メタデータ
    """
    document_id: str
    content_hash: str
    text_unit_ids: List[str] = field(default_factory=list)
    file_path: str | None = None
    file_modified_time: str | None = None
    indexed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "text_unit_ids": self.text_unit_ids,
            "file_path": self.file_path,
            "file_modified_time": self.file_modified_time,
            "indexed_at": self.indexed_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentRecord:
        """辞書から作成"""
        return cls(
            document_id=data["document_id"],
            content_hash=data["content_hash"],
            text_unit_ids=data.get("text_unit_ids", []),
            file_path=data.get("file_path"),
            file_modified_time=data.get("file_modified_time"),
            indexed_at=data.get("indexed_at", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IncrementalIndexState:
    """インクリメンタルインデックス状態
    
    Attributes:
        id: 状態ID
        documents: ドキュメントレコードのマップ
        change_history: 変更履歴
        total_indexed: インデックス済み総数
        last_update: 最終更新日時
        created_at: 作成日時
        version: バージョン
    """
    id: str = field(default_factory=lambda: str(__import__("uuid").uuid4()))
    documents: Dict[str, DocumentRecord] = field(default_factory=dict)
    change_history: List[IndexChangeSet] = field(default_factory=list)
    total_indexed: int = 0
    last_update: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: str = "1.0.0"
    
    @property
    def total_documents(self) -> int:
        """ドキュメント数"""
        return len(self.documents)
    
    @property
    def document_count(self) -> int:
        """ドキュメント数（エイリアス）"""
        return len(self.documents)
    
    @property
    def total_text_units(self) -> int:
        """TextUnit総数"""
        return sum(len(doc.text_unit_ids) for doc in self.documents.values())
    
    @property
    def text_unit_count(self) -> int:
        """TextUnit総数（エイリアス）"""
        return self.total_text_units
    
    def get_document(self, document_id: str) -> DocumentRecord | None:
        """ドキュメントレコードを取得"""
        return self.documents.get(document_id)
    
    def add_document(self, record: DocumentRecord) -> None:
        """ドキュメントを追加"""
        self.documents[record.document_id] = record
        self.total_indexed += 1
        self.last_update = datetime.now(timezone.utc).isoformat()
    
    def remove_document(self, document_id: str) -> DocumentRecord | None:
        """ドキュメントを削除"""
        record = self.documents.pop(document_id, None)
        if record:
            self.last_update = datetime.now(timezone.utc).isoformat()
        return record
    
    def add_change_set(self, change_set: IndexChangeSet) -> None:
        """変更セットを追加"""
        self.change_history.append(change_set)
        self.last_update = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "documents": {
                doc_id: record.to_dict()
                for doc_id, record in self.documents.items()
            },
            "change_history": [cs.to_dict() for cs in self.change_history],
            "total_indexed": self.total_indexed,
            "last_update": self.last_update,
            "created_at": self.created_at,
            "version": self.version,
            "stats": {
                "document_count": self.document_count,
                "text_unit_count": self.text_unit_count,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IncrementalIndexState:
        """辞書から作成"""
        return cls(
            id=data.get("id", str(__import__("uuid").uuid4())),
            documents={
                doc_id: DocumentRecord.from_dict(record)
                for doc_id, record in data.get("documents", {}).items()
            },
            change_history=[
                IndexChangeSet.from_dict(cs)
                for cs in data.get("change_history", [])
            ],
            total_indexed=data.get("total_indexed", 0),
            last_update=data.get("last_update"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            version=data.get("version", "1.0.0"),
        )
    
    def save(self, path: Path) -> None:
        """状態を保存"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> IncrementalIndexState:
        """状態を読み込み"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class IncrementalIndexConfig:
    """インクリメンタルインデックス設定
    
    Attributes:
        output_dir: 出力ディレクトリ
        state_file: 状態ファイル名
        state_version: 状態バージョン
        
        # 変更検出設定
        use_content_hash: Content Hashで変更検出
        use_file_modified_time: ファイル更新日時で変更検出
        
        # 処理設定
        batch_size: バッチサイズ
        parallel_workers: 並列ワーカー数
        
        # 履歴設定
        max_history: 保持する変更履歴の最大数
        
        # オプション
        show_progress: 進捗表示
        dry_run: ドライラン（実際には更新しない）
    """
    output_dir: str | Path = "./output/incremental_index"
    state_file: str = "incremental_state.json"
    state_version: str = "1.0.0"
    
    # 変更検出
    use_content_hash: bool = True
    use_file_modified_time: bool = True
    
    # 処理設定
    batch_size: int = 50
    parallel_workers: int = 4
    
    # 履歴設定
    max_history: int = 100
    
    # オプション
    show_progress: bool = True
    dry_run: bool = False
    
    @property
    def state_path(self) -> Path:
        """状態ファイルパス"""
        return Path(self.output_dir) / self.state_file
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "output_dir": str(self.output_dir),
            "state_file": self.state_file,
            "use_content_hash": self.use_content_hash,
            "use_file_modified_time": self.use_file_modified_time,
            "batch_size": self.batch_size,
            "parallel_workers": self.parallel_workers,
            "max_history": self.max_history,
            "show_progress": self.show_progress,
            "dry_run": self.dry_run,
        }
