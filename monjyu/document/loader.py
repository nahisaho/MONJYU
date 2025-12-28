# File Loader
"""
File loading and format detection for MONJYU.

Supports multiple document formats including:
- Text files (.txt, .md)
- Structured documents (.json, .html, .xml)
- Office documents (.docx, .pptx, .xlsx)
- PDF documents (.pdf)
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterator, Protocol


class FileLoaderProtocol(Protocol):
    """ファイルローダープロトコル"""
    
    def detect_format(self, path: Path) -> str:
        """ファイル形式を検出"""
        ...
    
    def load(self, path: Path) -> bytes:
        """ファイルを読み込み"""
        ...
    
    def load_batch(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> list[tuple[Path, bytes]]:
        """ディレクトリからバッチ読み込み"""
        ...


class FileLoader:
    """ファイルローダー
    
    サポートされるファイル形式の検出と読み込みを行う。
    
    Example:
        >>> loader = FileLoader()
        >>> file_type = loader.detect_format(Path("paper.pdf"))
        >>> content = loader.load(Path("paper.pdf"))
    """
    
    SUPPORTED_FORMATS: dict[str, str] = {
        # Text formats
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".rst": "text/x-rst",
        # Structured formats
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
        ".xml": "application/xml",
        ".csv": "text/csv",
        # Document formats
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".doc": "application/msword",
        ".rtf": "application/rtf",
        # Academic formats
        ".tex": "application/x-tex",
        ".bib": "application/x-bibtex",
    }
    
    def __init__(self, additional_formats: dict[str, str] | None = None) -> None:
        """初期化
        
        Args:
            additional_formats: 追加でサポートするファイル形式
        """
        self._formats = self.SUPPORTED_FORMATS.copy()
        if additional_formats:
            self._formats.update(additional_formats)
    
    @property
    def supported_extensions(self) -> list[str]:
        """サポートする拡張子一覧"""
        return list(self._formats.keys())
    
    def detect_format(self, path: Path) -> str:
        """ファイル形式を検出
        
        Args:
            path: ファイルパス
            
        Returns:
            ファイル拡張子（例: ".pdf"）
            
        Raises:
            ValueError: サポートされていないファイル形式
        """
        suffix = path.suffix.lower()
        
        # 拡張子で判定
        if suffix in self._formats:
            return suffix
        
        # MIMEタイプからフォールバック
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            for ext, mt in self._formats.items():
                if mt == mime_type:
                    return ext
        
        raise ValueError(
            f"Unsupported file format: {path.suffix} "
            f"(supported: {', '.join(self.supported_extensions)})"
        )
    
    def is_supported(self, path: Path) -> bool:
        """ファイル形式がサポートされているか確認
        
        Args:
            path: ファイルパス
            
        Returns:
            サポートされている場合True
        """
        try:
            self.detect_format(path)
            return True
        except ValueError:
            return False
    
    def load(self, path: Path) -> bytes:
        """ファイルを読み込み
        
        Args:
            path: ファイルパス
            
        Returns:
            ファイル内容（バイト列）
            
        Raises:
            FileNotFoundError: ファイルが存在しない
            PermissionError: 読み取り権限がない
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        return path.read_bytes()
    
    def load_text(self, path: Path, encoding: str = "utf-8") -> str:
        """テキストファイルを読み込み
        
        Args:
            path: ファイルパス
            encoding: 文字エンコーディング
            
        Returns:
            ファイル内容（文字列）
        """
        content = self.load(path)
        return content.decode(encoding)
    
    def load_batch(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> list[tuple[Path, bytes]]:
        """ディレクトリからバッチ読み込み
        
        Args:
            directory: ディレクトリパス
            pattern: ファイルパターン（glob形式）
            recursive: 再帰的に検索するか
            
        Returns:
            (パス, 内容)のリスト
        """
        return list(self.iter_batch(directory, pattern, recursive))
    
    def iter_batch(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> Iterator[tuple[Path, bytes]]:
        """ディレクトリからイテレータで読み込み
        
        Args:
            directory: ディレクトリパス
            pattern: ファイルパターン（glob形式）
            recursive: 再帰的に検索するか
            
        Yields:
            (パス, 内容)のタプル
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        glob_method = directory.rglob if recursive else directory.glob
        
        for path in sorted(glob_method(pattern)):
            if path.is_file() and self.is_supported(path):
                try:
                    yield (path, self.load(path))
                except (PermissionError, OSError) as e:
                    # ログ出力して継続
                    import logging
                    logging.warning(f"Failed to load {path}: {e}")
                    continue
    
    def list_files(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> list[Path]:
        """ディレクトリ内のサポートファイル一覧
        
        Args:
            directory: ディレクトリパス
            pattern: ファイルパターン（glob形式）
            recursive: 再帰的に検索するか
            
        Returns:
            ファイルパスのリスト
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        glob_method = directory.rglob if recursive else directory.glob
        
        return [
            path for path in sorted(glob_method(pattern))
            if path.is_file() and self.is_supported(path)
        ]
    
    def get_mime_type(self, path: Path) -> str:
        """MIMEタイプを取得
        
        Args:
            path: ファイルパス
            
        Returns:
            MIMEタイプ文字列
        """
        file_type = self.detect_format(path)
        return self._formats.get(file_type, "application/octet-stream")
