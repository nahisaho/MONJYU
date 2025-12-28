# Parquet Storage
"""
Parquet-based storage for MONJYU data persistence.

Provides efficient columnar storage for:
- Documents metadata
- Text units
- Embeddings
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_type_hints

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from monjyu.document.models import AcademicPaperDocument, TextUnit
    from monjyu.embedding.base import EmbeddingRecord


T = TypeVar("T")


class ParquetStorage:
    """Parquet ストレージ
    
    ドキュメント、TextUnit、埋め込みをParquet形式で永続化する。
    列指向フォーマットで効率的な読み書きとクエリをサポート。
    
    Example:
        >>> storage = ParquetStorage(output_dir="./output/index/level_0")
        >>> storage.write_text_units(text_units)
        >>> storage.write_embeddings(embeddings)
        >>> 
        >>> # 読み込み
        >>> units = storage.read_text_units()
    
    Attributes:
        output_dir: 出力ディレクトリ
    """
    
    # ファイル名定義
    DOCUMENTS_FILE = "documents.parquet"
    TEXT_UNITS_FILE = "text_units.parquet"
    EMBEDDINGS_FILE = "embeddings.parquet"
    
    def __init__(self, output_dir: str | Path = "./output/index/level_0") -> None:
        """初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Documents ===
    
    def write_documents(
        self,
        documents: list["AcademicPaperDocument"],
        append: bool = False,
    ) -> Path:
        """ドキュメントを保存
        
        Args:
            documents: ドキュメントのリスト
            append: 追記モード
            
        Returns:
            保存先パス
        """
        file_path = self.output_dir / self.DOCUMENTS_FILE
        
        # データを辞書形式に変換
        records = []
        for doc in documents:
            record = {
                "file_name": doc.file_name,
                "file_type": doc.file_type,
                "title": doc.title,
                "doi": doc.doi,
                "arxiv_id": doc.arxiv_id,
                "abstract": doc.abstract,
                "language": doc.language,
                "page_count": doc.page_count,
                "authors": [a.name for a in doc.authors],
                "keywords": doc.keywords,
                "section_count": len(doc.sections),
            }
            records.append(record)
        
        # PyArrow Tableに変換
        table = pa.Table.from_pylist(records)
        
        # 保存
        if append and file_path.exists():
            existing = pq.read_table(file_path)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, file_path)
        return file_path
    
    def read_documents(self) -> list[dict[str, Any]]:
        """ドキュメントを読み込み
        
        Returns:
            ドキュメントデータのリスト
        """
        file_path = self.output_dir / self.DOCUMENTS_FILE
        if not file_path.exists():
            return []
        
        table = pq.read_table(file_path)
        return table.to_pylist()
    
    # === Text Units ===
    
    def write_text_units(
        self,
        text_units: list["TextUnit"],
        append: bool = False,
    ) -> Path:
        """TextUnitを保存
        
        Args:
            text_units: TextUnitのリスト
            append: 追記モード
            
        Returns:
            保存先パス
        """
        file_path = self.output_dir / self.TEXT_UNITS_FILE
        
        # データを辞書形式に変換
        records = []
        for unit in text_units:
            record = {
                "id": unit.id,
                "text": unit.text,
                "n_tokens": unit.n_tokens,
                "document_id": unit.document_id,
                "section_type": unit.section_type,
                "chunk_index": unit.chunk_index,
            }
            # メタデータをフラット化
            if unit.metadata:
                for key, value in unit.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        record[f"meta_{key}"] = value
            records.append(record)
        
        # PyArrow Tableに変換
        table = pa.Table.from_pylist(records)
        
        # 保存
        if append and file_path.exists():
            existing = pq.read_table(file_path)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, file_path)
        return file_path
    
    def read_text_units(
        self,
        columns: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """TextUnitを読み込み
        
        Args:
            columns: 取得するカラム（省略時は全カラム）
            filter_expr: フィルター式
            
        Returns:
            TextUnitデータのリスト
        """
        file_path = self.output_dir / self.TEXT_UNITS_FILE
        if not file_path.exists():
            return []
        
        table = pq.read_table(file_path, columns=columns)
        return table.to_pylist()
    
    def read_text_units_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """IDでTextUnitを読み込み
        
        Args:
            ids: TextUnit IDのリスト
            
        Returns:
            TextUnitデータのリスト
        """
        all_units = self.read_text_units()
        id_set = set(ids)
        return [u for u in all_units if u["id"] in id_set]
    
    # === Embeddings ===
    
    def write_embeddings(
        self,
        embeddings: list["EmbeddingRecord"],
        append: bool = False,
    ) -> Path:
        """埋め込みを保存
        
        Args:
            embeddings: EmbeddingRecordのリスト
            append: 追記モード
            
        Returns:
            保存先パス
        """
        file_path = self.output_dir / self.EMBEDDINGS_FILE
        
        # データを辞書形式に変換
        records = []
        for emb in embeddings:
            record = {
                "id": emb.id,
                "text_unit_id": emb.text_unit_id,
                "vector": emb.vector,
                "model": emb.model,
                "dimensions": emb.dimensions,
            }
            records.append(record)
        
        # PyArrow Tableに変換（ベクトルは固定長リスト型）
        table = pa.Table.from_pylist(records)
        
        # 保存
        if append and file_path.exists():
            existing = pq.read_table(file_path)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, file_path)
        return file_path
    
    def read_embeddings(
        self,
        include_vectors: bool = True,
    ) -> list[dict[str, Any]]:
        """埋め込みを読み込み
        
        Args:
            include_vectors: ベクトルを含めるか
            
        Returns:
            埋め込みデータのリスト
        """
        file_path = self.output_dir / self.EMBEDDINGS_FILE
        if not file_path.exists():
            return []
        
        columns = None
        if not include_vectors:
            # ベクトル以外のカラムを取得
            schema = pq.read_schema(file_path)
            columns = [f.name for f in schema if f.name != "vector"]
        
        table = pq.read_table(file_path, columns=columns)
        return table.to_pylist()
    
    def read_embeddings_by_text_unit_ids(
        self,
        text_unit_ids: list[str],
    ) -> list[dict[str, Any]]:
        """TextUnit IDで埋め込みを読み込み
        
        Args:
            text_unit_ids: TextUnit IDのリスト
            
        Returns:
            埋め込みデータのリスト
        """
        all_embeddings = self.read_embeddings()
        id_set = set(text_unit_ids)
        return [e for e in all_embeddings if e["text_unit_id"] in id_set]
    
    # === Utilities ===
    
    def exists(self, file_type: str = "text_units") -> bool:
        """ファイルが存在するかチェック
        
        Args:
            file_type: ファイルタイプ ("documents", "text_units", "embeddings")
            
        Returns:
            存在する場合True
        """
        file_map = {
            "documents": self.DOCUMENTS_FILE,
            "text_units": self.TEXT_UNITS_FILE,
            "embeddings": self.EMBEDDINGS_FILE,
        }
        file_name = file_map.get(file_type)
        if not file_name:
            msg = f"未知のファイルタイプ: {file_type}"
            raise ValueError(msg)
        
        return (self.output_dir / file_name).exists()
    
    def get_stats(self) -> dict[str, Any]:
        """ストレージの統計情報を取得
        
        Returns:
            統計情報
        """
        stats = {
            "output_dir": str(self.output_dir),
            "documents_count": 0,
            "text_units_count": 0,
            "embeddings_count": 0,
            "total_size_bytes": 0,
        }
        
        for file_type, file_name in [
            ("documents", self.DOCUMENTS_FILE),
            ("text_units", self.TEXT_UNITS_FILE),
            ("embeddings", self.EMBEDDINGS_FILE),
        ]:
            file_path = self.output_dir / file_name
            if file_path.exists():
                table = pq.read_table(file_path)
                stats[f"{file_type}_count"] = table.num_rows
                stats["total_size_bytes"] += file_path.stat().st_size
        
        return stats
    
    def clear(self, file_type: str | None = None) -> None:
        """ストレージをクリア
        
        Args:
            file_type: クリアするファイルタイプ（省略時は全て）
        """
        if file_type:
            file_map = {
                "documents": self.DOCUMENTS_FILE,
                "text_units": self.TEXT_UNITS_FILE,
                "embeddings": self.EMBEDDINGS_FILE,
            }
            file_name = file_map.get(file_type)
            if file_name:
                file_path = self.output_dir / file_name
                if file_path.exists():
                    file_path.unlink()
        else:
            for file_name in [self.DOCUMENTS_FILE, self.TEXT_UNITS_FILE, self.EMBEDDINGS_FILE]:
                file_path = self.output_dir / file_name
                if file_path.exists():
                    file_path.unlink()
