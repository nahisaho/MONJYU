"""Incremental Index Module.

差分インデックス更新を実現するモジュール。

Features:
    - ドキュメントの追加・更新・削除の検出
    - 差分のみのインデックス更新
    - Content Hash によるドキュメント変更検出
    - 状態管理とロールバック
"""

from monjyu.index.incremental.types import (
    ChangeType,
    DocumentChange,
    DocumentRecord,
    IncrementalIndexConfig,
    IncrementalIndexState,
    IndexChangeSet,
)
from monjyu.index.incremental.tracker import (
    DocumentTracker,
    FileTracker,
    compute_content_hash,
    compute_document_hash,
    compute_text_unit_hash,
)
from monjyu.index.incremental.manager import (
    IncrementalIndexManager,
)

__all__ = [
    # Types
    "ChangeType",
    "DocumentChange",
    "DocumentRecord",
    "IncrementalIndexConfig",
    "IncrementalIndexState",
    "IndexChangeSet",
    # Tracker
    "DocumentTracker",
    "FileTracker",
    "compute_content_hash",
    "compute_document_hash",
    "compute_text_unit_hash",
    # Manager
    "IncrementalIndexManager",
]
