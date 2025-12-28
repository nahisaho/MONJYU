# Unit Tests for FileLoader
"""Tests for monjyu.document.loader module."""

from __future__ import annotations

from pathlib import Path

import pytest

from monjyu.document.loader import FileLoader


class TestFileLoader:
    """FileLoader の単体テスト"""
    
    @pytest.fixture
    def loader(self) -> FileLoader:
        """FileLoaderインスタンス"""
        return FileLoader()
    
    # --- detect_format tests ---
    
    def test_detect_format_txt(self, loader: FileLoader) -> None:
        """TXTファイルのフォーマット検出"""
        assert loader.detect_format(Path("test.txt")) == ".txt"
    
    def test_detect_format_md(self, loader: FileLoader) -> None:
        """Markdownファイルのフォーマット検出"""
        assert loader.detect_format(Path("readme.md")) == ".md"
    
    def test_detect_format_pdf(self, loader: FileLoader) -> None:
        """PDFファイルのフォーマット検出"""
        assert loader.detect_format(Path("paper.pdf")) == ".pdf"
    
    def test_detect_format_json(self, loader: FileLoader) -> None:
        """JSONファイルのフォーマット検出"""
        assert loader.detect_format(Path("data.json")) == ".json"
    
    def test_detect_format_html(self, loader: FileLoader) -> None:
        """HTMLファイルのフォーマット検出"""
        assert loader.detect_format(Path("page.html")) == ".html"
        assert loader.detect_format(Path("page.htm")) == ".htm"
    
    def test_detect_format_docx(self, loader: FileLoader) -> None:
        """DOCXファイルのフォーマット検出"""
        assert loader.detect_format(Path("document.docx")) == ".docx"
    
    def test_detect_format_latex(self, loader: FileLoader) -> None:
        """LaTeXファイルのフォーマット検出"""
        assert loader.detect_format(Path("paper.tex")) == ".tex"
    
    def test_detect_format_case_insensitive(self, loader: FileLoader) -> None:
        """拡張子の大文字小文字を無視"""
        assert loader.detect_format(Path("file.PDF")) == ".pdf"
        assert loader.detect_format(Path("file.TXT")) == ".txt"
    
    def test_detect_format_unknown(self, loader: FileLoader) -> None:
        """未知の拡張子でエラー"""
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.detect_format(Path("file.xyz"))
    
    # --- load tests ---
    
    def test_load_text_file(self, loader: FileLoader, tmp_path: Path) -> None:
        """テキストファイルの読み込み（bytes）"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")
        
        content = loader.load(test_file)
        assert isinstance(content, bytes)
        assert b"Hello, World!" in content
    
    def test_load_md_file(self, loader: FileLoader, tmp_path: Path) -> None:
        """Markdownファイルの読み込み"""
        test_file = tmp_path / "readme.md"
        test_file.write_text("# Title\n\nContent", encoding="utf-8")
        
        content = loader.load(test_file)
        assert isinstance(content, bytes)
        assert b"# Title" in content
    
    def test_load_json_file(self, loader: FileLoader, tmp_path: Path) -> None:
        """JSONファイルの読み込み"""
        test_file = tmp_path / "data.json"
        test_file.write_text('{"key": "value"}', encoding="utf-8")
        
        content = loader.load(test_file)
        assert isinstance(content, bytes)
        assert b"key" in content
    
    def test_load_pdf_returns_bytes(self, loader: FileLoader, tmp_path: Path) -> None:
        """PDFファイルはバイトを返す"""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")
        
        content = loader.load(test_file)
        assert isinstance(content, bytes)
        assert b"%PDF" in content
    
    def test_load_nonexistent_file(self, loader: FileLoader) -> None:
        """存在しないファイルでエラー"""
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.txt"))
    
    # --- load_batch tests ---
    
    def test_load_batch(self, loader: FileLoader, tmp_path: Path) -> None:
        """バッチ読み込み（ディレクトリから）"""
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content {i}")
        
        results = loader.load_batch(tmp_path)
        assert len(results) == 3
        assert all(isinstance(r[1], bytes) for r in results)
    
    def test_load_batch_with_pattern(self, loader: FileLoader, tmp_path: Path) -> None:
        """パターン指定でバッチ読み込み"""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.md").write_text("c")
        
        results = loader.load_batch(tmp_path, pattern="*.txt")
        assert len(results) == 2
    
    # --- list_files tests ---
    
    def test_list_files(self, loader: FileLoader, tmp_path: Path) -> None:
        """ディレクトリ内のファイル一覧"""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.pdf").write_bytes(b"%PDF")
        
        files = loader.list_files(tmp_path)
        assert len(files) == 3
    
    def test_list_files_with_pattern(self, loader: FileLoader, tmp_path: Path) -> None:
        """パターンでフィルタ"""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.pdf").write_bytes(b"%PDF")
        
        txt_files = loader.list_files(tmp_path, pattern="*.txt")
        assert len(txt_files) == 2
        assert all(f.suffix == ".txt" for f in txt_files)
    
    def test_list_files_recursive(self, loader: FileLoader, tmp_path: Path) -> None:
        """再帰的な検索"""
        (tmp_path / "a.txt").write_text("a")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "b.txt").write_text("b")
        
        # 再帰
        files_recursive = loader.list_files(tmp_path, recursive=True)
        assert len(files_recursive) == 2
        
        # 非再帰
        files_non_recursive = loader.list_files(tmp_path, recursive=False)
        assert len(files_non_recursive) == 1
    
    def test_list_files_nonexistent_dir(self, loader: FileLoader) -> None:
        """存在しないディレクトリでエラー"""
        with pytest.raises(FileNotFoundError):
            loader.list_files(Path("/nonexistent/dir"))
    
    def test_list_files_filters_unsupported(self, loader: FileLoader, tmp_path: Path) -> None:
        """サポート外のファイルはフィルタされる"""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.xyz").write_text("b")  # サポート外
        
        files = loader.list_files(tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".txt"
    
    # --- SUPPORTED_FORMATS tests ---
    
    def test_supported_formats(self, loader: FileLoader) -> None:
        """サポートされているフォーマット一覧"""
        assert ".txt" in loader.SUPPORTED_FORMATS
        assert ".pdf" in loader.SUPPORTED_FORMATS
        assert ".md" in loader.SUPPORTED_FORMATS
        assert ".json" in loader.SUPPORTED_FORMATS
        assert ".html" in loader.SUPPORTED_FORMATS
    
    # --- is_supported tests ---
    
    def test_is_supported_true(self, loader: FileLoader) -> None:
        """サポートされている形式"""
        assert loader.is_supported(Path("test.txt"))
        assert loader.is_supported(Path("test.pdf"))
    
    def test_is_supported_false(self, loader: FileLoader) -> None:
        """サポートされていない形式"""
        assert not loader.is_supported(Path("test.xyz"))
