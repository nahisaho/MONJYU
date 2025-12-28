"""Incremental Index Example.

差分インデックス更新の具体的な使用例。
"""

import asyncio
import tempfile
from pathlib import Path

from monjyu.index.incremental import (
    IncrementalIndexManager,
    IncrementalIndexConfig,
    DocumentTracker,
    FileTracker,
    ChangeType,
)


async def example_document_tracking():
    """ドキュメント変更追跡の例"""
    print("=" * 60)
    print("Document Tracking Example")
    print("=" * 60)
    
    # Mock documents (実際のプロジェクトではDocumentPipelineから取得)
    class MockDocument:
        def __init__(self, id: str, title: str, content: str):
            self.id = id
            self.title = title
            self.content = content
    
    # 初回: 3つのドキュメント
    initial_docs = [
        MockDocument("doc1", "Document 1", "Content of document 1"),
        MockDocument("doc2", "Document 2", "Content of document 2"),
        MockDocument("doc3", "Document 3", "Content of document 3"),
    ]
    
    tracker = DocumentTracker()
    
    # 初回トラッキング
    changes = tracker.detect_changes(initial_docs)
    print(f"\nInitial indexing:")
    print(f"  Added: {changes.added_count}")
    print(f"  Modified: {changes.modified_count}")
    print(f"  Deleted: {changes.deleted_count}")
    
    # 変更をステートに適用
    tracker.apply_changes(changes)
    
    # 更新: doc2を変更、doc3を削除、doc4を追加
    updated_docs = [
        MockDocument("doc1", "Document 1", "Content of document 1"),  # 変更なし
        MockDocument("doc2", "Document 2", "UPDATED content of document 2"),  # 変更
        MockDocument("doc4", "Document 4", "New document 4"),  # 新規
        # doc3は削除
    ]
    
    changes = tracker.detect_changes(updated_docs)
    print(f"\nAfter updates:")
    print(f"  Added: {changes.added_count}")
    print(f"  Modified: {changes.modified_count}")
    print(f"  Deleted: {changes.deleted_count}")
    
    # 各変更の詳細
    for change in changes.changes:
        if change.change_type != ChangeType.UNCHANGED:
            print(f"  - {change.document_id}: {change.change_type.value}")


async def example_file_tracking():
    """ファイルシステム変更追跡の例"""
    print("\n" + "=" * 60)
    print("File Tracking Example")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # 初期ファイル作成
        (tmppath / "file1.txt").write_text("Content 1")
        (tmppath / "file2.txt").write_text("Content 2")
        (tmppath / "file3.md").write_text("# Markdown")
        
        tracker = FileTracker()
        
        # 初回スキャン
        state = tracker.scan_directory(tmppath, patterns=["*.txt", "*.md"])
        print(f"\nInitial scan:")
        print(f"  Files tracked: {len(state)}")
        
        # 変更を適用
        changes = tracker.detect_file_changes(tmppath, patterns=["*.txt", "*.md"])
        print(f"  Changes detected: {len(changes.changes)}")
        
        # ファイルを変更
        (tmppath / "file1.txt").write_text("Updated Content 1")
        (tmppath / "file4.txt").write_text("New file 4")
        (tmppath / "file2.txt").unlink()  # 削除
        
        # 変更検出
        changes = tracker.detect_file_changes(tmppath, patterns=["*.txt", "*.md"])
        print(f"\nAfter file system changes:")
        print(f"  Added: {changes.added_count}")
        print(f"  Modified: {changes.modified_count}")
        print(f"  Deleted: {changes.deleted_count}")


async def example_incremental_manager():
    """IncrementalIndexManager の統合例"""
    print("\n" + "=" * 60)
    print("Incremental Index Manager Example")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = IncrementalIndexConfig(
            output_dir=tmpdir,
            state_version="1.0",
            batch_size=100,
            dry_run=False,  # Trueにすると実際の更新をスキップ
        )
        
        manager = IncrementalIndexManager(config)
        
        # サマリー表示
        summary = manager.get_summary()
        print(f"\nInitial state:")
        print(f"  Tracked documents: {summary.get('tracked_documents', 0)}")
        
        # Mock documents
        class MockDoc:
            def __init__(self, id: str, content: str):
                self.id = id
                self.content = content
        
        class MockTextUnit:
            def __init__(self, id: str, text: str, document_id: str):
                self.id = id
                self.text = text
                self.document_id = document_id
        
        docs = [MockDoc(f"doc{i}", f"Content {i}") for i in range(5)]
        text_units = [
            MockTextUnit(f"tu{i}", f"Text unit {i}", f"doc{i % 5}")
            for i in range(15)
        ]
        
        # 変更検出 (dry_run mode)
        manager.config.dry_run = True
        change_set = manager.detect_changes(docs, text_units)
        
        print(f"\nDry-run change detection:")
        print(f"  Would add: {change_set.added_count}")
        print(f"  Would modify: {change_set.modified_count}")
        print(f"  Would delete: {change_set.deleted_count}")
        
        # 実際の更新
        manager.config.dry_run = False
        await manager.update(docs, text_units)
        
        # ステート保存
        manager.save_state()
        print(f"\nState saved to: {tmpdir}")
        
        # サマリー確認
        summary = manager.get_summary()
        print(f"\nFinal state:")
        print(f"  Tracked documents: {summary.get('tracked_documents', 0)}")


async def main():
    """メイン関数"""
    await example_document_tracking()
    await example_file_tracking()
    await example_incremental_manager()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
