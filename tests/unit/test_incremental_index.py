"""Incremental Index Unit Tests.

インクリメンタルインデックスのユニットテスト。
"""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monjyu.index.incremental.types import (
    ChangeType,
    DocumentChange,
    DocumentRecord,
    IndexChangeSet,
    IncrementalIndexConfig,
    IncrementalIndexState,
)
from monjyu.index.incremental.tracker import (
    DocumentTracker,
    FileTracker,
    compute_content_hash,
    compute_document_hash,
)
from monjyu.index.incremental.manager import IncrementalIndexManager


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """一時ディレクトリを作成"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_document():
    """モックドキュメントを作成"""
    def _create(doc_id: str = None, title: str = "Test", abstract: str = "Abstract"):
        doc = MagicMock()
        doc.id = doc_id or str(uuid.uuid4())
        doc.title = title
        doc.abstract = abstract
        doc.sections = []
        doc.text = f"{title}\n{abstract}"
        return doc
    return _create


@pytest.fixture
def mock_text_unit():
    """モックTextUnitを作成"""
    def _create(unit_id: str = None, doc_id: str = None, text: str = "Test text"):
        unit = MagicMock()
        unit.id = unit_id or str(uuid.uuid4())
        unit.document_id = doc_id or str(uuid.uuid4())
        unit.text = text
        return unit
    return _create


@pytest.fixture
def sample_state():
    """サンプル状態を作成"""
    state = IncrementalIndexState()
    
    # サンプルドキュメントを追加
    state.add_document(DocumentRecord(
        document_id="doc-1",
        content_hash="hash-1",
        text_unit_ids=["unit-1", "unit-2"],
    ))
    state.add_document(DocumentRecord(
        document_id="doc-2",
        content_hash="hash-2",
        text_unit_ids=["unit-3"],
    ))
    
    return state


# =============================================================================
# Test: compute_content_hash
# =============================================================================

class TestComputeContentHash:
    """compute_content_hash のテスト"""
    
    def test_returns_string(self):
        """文字列を返すことを確認"""
        result = compute_content_hash("test content")
        assert isinstance(result, str)
    
    def test_returns_16_chars(self):
        """16文字を返すことを確認"""
        result = compute_content_hash("test content")
        assert len(result) == 16
    
    def test_same_input_same_output(self):
        """同じ入力は同じ出力を返す"""
        result1 = compute_content_hash("test content")
        result2 = compute_content_hash("test content")
        assert result1 == result2
    
    def test_different_input_different_output(self):
        """異なる入力は異なる出力を返す"""
        result1 = compute_content_hash("content 1")
        result2 = compute_content_hash("content 2")
        assert result1 != result2
    
    def test_unicode_content(self):
        """Unicode コンテンツを処理できる"""
        result = compute_content_hash("日本語テスト 🚀")
        assert isinstance(result, str)
        assert len(result) == 16


# =============================================================================
# Test: ChangeType
# =============================================================================

class TestChangeType:
    """ChangeType のテスト"""
    
    def test_enum_values(self):
        """列挙値が存在することを確認"""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"
        assert ChangeType.UNCHANGED.value == "unchanged"


# =============================================================================
# Test: DocumentChange
# =============================================================================

class TestDocumentChange:
    """DocumentChange のテスト"""
    
    def test_create_added(self):
        """追加変更を作成できる"""
        change = DocumentChange(
            document_id="doc-1",
            change_type=ChangeType.ADDED,
            new_hash="new-hash",
            new_text_unit_ids=["unit-1"],
        )
        assert change.document_id == "doc-1"
        assert change.change_type == ChangeType.ADDED
        assert change.is_modified is False
    
    def test_create_modified(self):
        """変更を作成できる"""
        change = DocumentChange(
            document_id="doc-1",
            change_type=ChangeType.MODIFIED,
            old_hash="old-hash",
            new_hash="new-hash",
            old_text_unit_ids=["old-unit"],
            new_text_unit_ids=["new-unit"],
        )
        assert change.is_modified is True
    
    def test_text_units_to_add(self):
        """追加するTextUnitを取得できる"""
        change = DocumentChange(
            document_id="doc-1",
            change_type=ChangeType.MODIFIED,
            old_text_unit_ids=["unit-1", "unit-2"],
            new_text_unit_ids=["unit-2", "unit-3"],
        )
        assert set(change.text_units_to_add) == {"unit-3"}
    
    def test_text_units_to_remove(self):
        """削除するTextUnitを取得できる"""
        change = DocumentChange(
            document_id="doc-1",
            change_type=ChangeType.MODIFIED,
            old_text_unit_ids=["unit-1", "unit-2"],
            new_text_unit_ids=["unit-2", "unit-3"],
        )
        assert set(change.text_units_to_remove) == {"unit-1"}
    
    def test_to_dict_from_dict(self):
        """シリアライズ・デシリアライズできる"""
        change = DocumentChange(
            document_id="doc-1",
            change_type=ChangeType.MODIFIED,
            old_hash="old",
            new_hash="new",
        )
        data = change.to_dict()
        restored = DocumentChange.from_dict(data)
        
        assert restored.document_id == change.document_id
        assert restored.change_type == change.change_type
        assert restored.old_hash == change.old_hash
        assert restored.new_hash == change.new_hash


# =============================================================================
# Test: IndexChangeSet
# =============================================================================

class TestIndexChangeSet:
    """IndexChangeSet のテスト"""
    
    def test_count_properties(self):
        """カウントプロパティが正しく動作する"""
        changes = [
            DocumentChange("doc-1", ChangeType.ADDED),
            DocumentChange("doc-2", ChangeType.ADDED),
            DocumentChange("doc-3", ChangeType.MODIFIED),
            DocumentChange("doc-4", ChangeType.DELETED),
        ]
        change_set = IndexChangeSet(
            id="set-1",
            created_at=datetime.now(timezone.utc).isoformat(),
            changes=changes,
        )
        
        assert change_set.added_count == 2
        assert change_set.modified_count == 1
        assert change_set.deleted_count == 1
        assert change_set.total_changes == 4
    
    def test_get_changes_by_type(self):
        """タイプ別に変更を取得できる"""
        changes = [
            DocumentChange("doc-1", ChangeType.ADDED),
            DocumentChange("doc-2", ChangeType.MODIFIED),
        ]
        change_set = IndexChangeSet(
            id="set-1",
            created_at=datetime.now(timezone.utc).isoformat(),
            changes=changes,
        )
        
        added = change_set.get_changes_by_type(ChangeType.ADDED)
        assert len(added) == 1
        assert added[0].document_id == "doc-1"
    
    def test_to_dict_from_dict(self):
        """シリアライズ・デシリアライズできる"""
        changes = [DocumentChange("doc-1", ChangeType.ADDED)]
        change_set = IndexChangeSet(
            id="set-1",
            created_at="2024-01-01T00:00:00Z",
            changes=changes,
        )
        
        data = change_set.to_dict()
        restored = IndexChangeSet.from_dict(data)
        
        assert restored.id == change_set.id
        assert len(restored.changes) == 1


# =============================================================================
# Test: DocumentRecord
# =============================================================================

class TestDocumentRecord:
    """DocumentRecord のテスト"""
    
    def test_create_minimal(self):
        """最小限のパラメータで作成できる"""
        record = DocumentRecord(
            document_id="doc-1",
            content_hash="hash-1",
            text_unit_ids=["unit-1"],
        )
        assert record.document_id == "doc-1"
        assert record.content_hash == "hash-1"
    
    def test_default_values(self):
        """デフォルト値が設定される"""
        record = DocumentRecord(
            document_id="doc-1",
            content_hash="hash-1",
            text_unit_ids=[],
        )
        assert record.indexed_at is not None
        assert record.metadata == {}


# =============================================================================
# Test: IncrementalIndexState
# =============================================================================

class TestIncrementalIndexState:
    """IncrementalIndexState のテスト"""
    
    def test_add_document(self):
        """ドキュメントを追加できる"""
        state = IncrementalIndexState()
        record = DocumentRecord("doc-1", "hash-1", ["unit-1"])
        
        state.add_document(record)
        
        assert state.total_documents == 1
        assert "doc-1" in state.documents
    
    def test_remove_document(self):
        """ドキュメントを削除できる"""
        state = IncrementalIndexState()
        record = DocumentRecord("doc-1", "hash-1", ["unit-1"])
        state.add_document(record)
        
        state.remove_document("doc-1")
        
        assert state.total_documents == 0
        assert "doc-1" not in state.documents
    
    def test_get_document(self):
        """ドキュメントを取得できる"""
        state = IncrementalIndexState()
        record = DocumentRecord("doc-1", "hash-1", ["unit-1"])
        state.add_document(record)
        
        result = state.get_document("doc-1")
        
        assert result is not None
        assert result.content_hash == "hash-1"
    
    def test_total_text_units(self):
        """TextUnit総数を取得できる"""
        state = IncrementalIndexState()
        state.add_document(DocumentRecord("doc-1", "hash-1", ["unit-1", "unit-2"]))
        state.add_document(DocumentRecord("doc-2", "hash-2", ["unit-3"]))
        
        assert state.total_text_units == 3
    
    def test_save_load(self, temp_dir):
        """保存・読み込みできる"""
        state = IncrementalIndexState()
        state.add_document(DocumentRecord("doc-1", "hash-1", ["unit-1"]))
        
        path = temp_dir / "state.json"
        state.save(path)
        
        loaded = IncrementalIndexState.load(path)
        
        assert loaded.total_documents == 1
        assert "doc-1" in loaded.documents


# =============================================================================
# Test: IncrementalIndexConfig
# =============================================================================

class TestIncrementalIndexConfig:
    """IncrementalIndexConfig のテスト"""
    
    def test_default_values(self):
        """デフォルト値が設定される"""
        config = IncrementalIndexConfig()
        
        assert config.batch_size == 50
        assert config.use_content_hash is True
        assert config.dry_run is False
    
    def test_custom_values(self):
        """カスタム値を設定できる"""
        config = IncrementalIndexConfig(
            output_dir="./custom",
            batch_size=100,
            dry_run=True,
        )
        
        assert str(config.output_dir) == "./custom"
        assert config.batch_size == 100
        assert config.dry_run is True


# =============================================================================
# Test: DocumentTracker
# =============================================================================

class TestDocumentTracker:
    """DocumentTracker のテスト"""
    
    def test_detect_added_document(self, mock_document, mock_text_unit, sample_state):
        """追加ドキュメントを検出できる"""
        tracker = DocumentTracker(IncrementalIndexState())
        
        doc = mock_document(doc_id="new-doc")
        unit = mock_text_unit(doc_id="new-doc")
        
        change_set = tracker.detect_changes(
            documents=[doc],
            text_units=[unit],
            check_deleted=False,
        )
        
        assert change_set.added_count == 1
        assert change_set.changes[0].change_type == ChangeType.ADDED
    
    def test_detect_unchanged_document(self, mock_document, mock_text_unit):
        """変更なしドキュメントを検出できる"""
        # 状態を作成
        state = IncrementalIndexState()
        doc = mock_document(doc_id="doc-1")
        doc_hash = compute_document_hash(doc)
        state.add_document(DocumentRecord("doc-1", doc_hash, ["unit-1"]))
        
        tracker = DocumentTracker(state)
        unit = mock_text_unit(doc_id="doc-1")
        
        change_set = tracker.detect_changes(
            documents=[doc],
            text_units=[unit],
            check_deleted=False,
        )
        
        assert change_set.added_count == 0
        assert change_set.modified_count == 0
    
    def test_detect_modified_document(self, mock_document, mock_text_unit):
        """変更ドキュメントを検出できる"""
        state = IncrementalIndexState()
        state.add_document(DocumentRecord("doc-1", "old-hash", ["unit-1"]))
        
        tracker = DocumentTracker(state)
        
        doc = mock_document(doc_id="doc-1", title="Updated Title")
        unit = mock_text_unit(doc_id="doc-1")
        
        change_set = tracker.detect_changes(
            documents=[doc],
            text_units=[unit],
            check_deleted=False,
        )
        
        assert change_set.modified_count == 1
        assert change_set.changes[0].change_type == ChangeType.MODIFIED
    
    def test_detect_deleted_document(self, sample_state):
        """削除ドキュメントを検出できる"""
        tracker = DocumentTracker(sample_state)
        
        change_set = tracker.detect_changes(
            documents=[],
            text_units=[],
            check_deleted=True,
        )
        
        assert change_set.deleted_count == 2
    
    def test_apply_changes_adds_document(self, mock_document, mock_text_unit):
        """変更適用で追加ドキュメントが追加される"""
        state = IncrementalIndexState()
        tracker = DocumentTracker(state)
        
        doc = mock_document(doc_id="new-doc")
        unit = mock_text_unit(unit_id="new-unit", doc_id="new-doc")
        
        change_set = tracker.detect_changes([doc], [unit], check_deleted=False)
        tracker.apply_changes(change_set, [doc], [unit])
        
        assert state.total_documents == 1
        assert "new-doc" in state.documents
    
    def test_get_documents_to_reindex(self):
        """再インデックス対象ドキュメントを取得できる"""
        tracker = DocumentTracker(IncrementalIndexState())
        
        changes = [
            DocumentChange("doc-1", ChangeType.ADDED),
            DocumentChange("doc-2", ChangeType.MODIFIED),
            DocumentChange("doc-3", ChangeType.DELETED),
            DocumentChange("doc-4", ChangeType.UNCHANGED),
        ]
        change_set = IndexChangeSet(
            id="set-1",
            created_at=datetime.now(timezone.utc).isoformat(),
            changes=changes,
        )
        
        docs_to_reindex = tracker.get_documents_to_reindex(change_set)
        
        assert docs_to_reindex == {"doc-1", "doc-2"}


# =============================================================================
# Test: IncrementalIndexManager
# =============================================================================

class TestIncrementalIndexManager:
    """IncrementalIndexManager のテスト"""
    
    def test_create_with_default_config(self, temp_dir):
        """デフォルト設定で作成できる"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        
        assert manager.config == config
    
    def test_state_lazy_loading(self, temp_dir):
        """状態が遅延読み込みされる"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        
        # _state は None
        assert manager._state is None
        
        # state プロパティアクセスで読み込み
        state = manager.state
        
        assert state is not None
        assert manager._state is not None
    
    def test_save_and_load_state(self, temp_dir):
        """状態を保存・読み込みできる"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        
        # ドキュメントを追加
        manager.state.add_document(DocumentRecord("doc-1", "hash-1", ["unit-1"]))
        manager.save_state()
        
        # 新しいマネージャーで読み込み
        manager2 = IncrementalIndexManager(config)
        
        assert manager2.state.total_documents == 1
    
    def test_detect_changes(self, temp_dir, mock_document, mock_text_unit):
        """変更を検出できる"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        
        doc = mock_document(doc_id="doc-1")
        unit = mock_text_unit(doc_id="doc-1")
        
        change_set = manager.detect_changes([doc], [unit])
        
        assert change_set.added_count == 1
    
    def test_get_summary(self, temp_dir):
        """サマリーを取得できる"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        manager.state.add_document(DocumentRecord("doc-1", "hash-1", ["unit-1"]))
        
        summary = manager.get_summary()
        
        assert summary["total_documents"] == 1
        assert summary["total_text_units"] == 1
    
    def test_reset(self, temp_dir):
        """リセットできる"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        
        manager.state.add_document(DocumentRecord("doc-1", "hash-1", ["unit-1"]))
        manager.save_state()
        
        manager.reset()
        
        assert manager.state.total_documents == 0
        assert not manager.state_path.exists()
    
    @pytest.mark.asyncio
    async def test_update_no_changes(self, temp_dir, mock_document, mock_text_unit):
        """変更なしの場合はスキップされる"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        
        doc = mock_document(doc_id="doc-1")
        doc_hash = compute_document_hash(doc)
        manager.state.add_document(DocumentRecord("doc-1", doc_hash, ["unit-1"]))
        
        unit = mock_text_unit(doc_id="doc-1")
        
        builder = AsyncMock()
        builder.build = AsyncMock(return_value=MagicMock())
        
        await manager.update([doc], [unit], builder)
        
        # build が呼ばれる（変更なしでも）
        builder.build.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_dry_run(self, temp_dir, mock_document, mock_text_unit):
        """ドライランモードで実際の更新をスキップ"""
        config = IncrementalIndexConfig(output_dir=temp_dir, dry_run=True)
        manager = IncrementalIndexManager(config)
        
        doc = mock_document(doc_id="doc-1")
        unit = mock_text_unit(doc_id="doc-1")
        
        builder = AsyncMock()
        builder.build = AsyncMock(return_value=MagicMock())
        
        await manager.update([doc], [unit], builder)
        
        # ドライランでも build は呼ばれる
        builder.build.assert_called_once()
        # 状態は更新されない
        assert manager.state.total_documents == 0


# =============================================================================
# Test: FileTracker
# =============================================================================

class TestFileTracker:
    """FileTracker のテスト"""
    
    def test_scan_directory(self, temp_dir):
        """ディレクトリをスキャンできる"""
        # テストファイルを作成
        (temp_dir / "test.txt").write_text("test content")
        (temp_dir / "sub").mkdir()
        (temp_dir / "sub" / "nested.txt").write_text("nested content")
        
        tracker = FileTracker(IncrementalIndexState())
        files = tracker.scan_directory(temp_dir, patterns=["*.txt"])
        
        assert len(files) == 2
    
    def test_detect_added_file(self, temp_dir):
        """追加ファイルを検出できる"""
        (temp_dir / "new.txt").write_text("new content")
        
        tracker = FileTracker(IncrementalIndexState())
        changes = tracker.detect_file_changes(temp_dir, patterns=["*.txt"])
        
        new_file_path = str(temp_dir / "new.txt")
        assert new_file_path in changes
        assert changes[new_file_path] == ChangeType.ADDED
    
    def test_detect_deleted_file(self, temp_dir):
        """削除ファイルを検出できる"""
        state = IncrementalIndexState()
        state.add_document(DocumentRecord(
            document_id="doc-1",
            content_hash="hash-1",
            text_unit_ids=["unit-1"],
            file_path=str(temp_dir / "deleted.txt"),
        ))
        
        tracker = FileTracker(state)
        changes = tracker.detect_file_changes(temp_dir, patterns=["*.txt"])
        
        deleted_path = str(temp_dir / "deleted.txt")
        assert deleted_path in changes
        assert changes[deleted_path] == ChangeType.DELETED


# =============================================================================
# Integration Tests
# =============================================================================

class TestIncrementalIndexIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, temp_dir, mock_document, mock_text_unit):
        """完全なワークフローをテスト"""
        config = IncrementalIndexConfig(output_dir=temp_dir)
        manager = IncrementalIndexManager(config)
        
        # Step 1: 初回インデックス
        doc1 = mock_document(doc_id="doc-1", title="Document 1")
        doc1.file_path = None  # file_pathをNoneに設定
        unit1 = mock_text_unit(unit_id="unit-1", doc_id="doc-1")
        
        builder = AsyncMock()
        builder.build = AsyncMock(return_value=MagicMock())
        builder.add = AsyncMock(return_value=MagicMock())
        
        # 状態をチェックのみ（保存しない）
        change_set = manager.detect_changes([doc1], [unit1])
        assert change_set.added_count == 1
        
        # tracker を使って適用
        manager.tracker.apply_changes(change_set, [doc1], [unit1])
        assert manager.state.total_documents == 1
        
        # Step 2: 新しいドキュメントを追加
        doc2 = mock_document(doc_id="doc-2", title="Document 2")
        doc2.file_path = None
        unit2 = mock_text_unit(unit_id="unit-2", doc_id="doc-2")
        
        # doc1のハッシュを更新（変更なし扱いにするため）
        doc1_hash = compute_document_hash(doc1)
        manager.state.documents["doc-1"].content_hash = doc1_hash
        
        change_set2 = manager.detect_changes([doc1, doc2], [unit1, unit2], check_deleted=False)
        manager.tracker.apply_changes(change_set2, [doc1, doc2], [unit1, unit2])
        
        assert manager.state.total_documents == 2
        
        # Step 3: ドキュメントを削除
        # doc1 のみ残す
        manager.state.documents["doc-1"].content_hash = compute_document_hash(doc1)
        
        change_set3 = manager.detect_changes([doc1], [unit1], check_deleted=True)
        manager.tracker.apply_changes(change_set3, [doc1], [unit1])
        
        # 削除後は1ドキュメント
        assert manager.state.total_documents == 1
        assert "doc-1" in manager.state.documents
        assert "doc-2" not in manager.state.documents
