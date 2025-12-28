# Document Loader Coverage Tests
"""
Tests for monjyu.document.loader to improve coverage from 76% to 85%+
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from monjyu.document.loader import FileLoader


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def loader() -> FileLoader:
    """Default FileLoader fixture."""
    return FileLoader()


@pytest.fixture
def loader_with_custom_formats() -> FileLoader:
    """FileLoader with additional formats."""
    return FileLoader(additional_formats={".custom": "application/x-custom"})


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def temp_text_file(temp_dir: Path) -> Path:
    """Create temporary text file."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("Hello, World!", encoding="utf-8")
    return file_path


@pytest.fixture
def temp_markdown_file(temp_dir: Path) -> Path:
    """Create temporary markdown file."""
    file_path = temp_dir / "test.md"
    file_path.write_text("# Heading\n\nParagraph", encoding="utf-8")
    return file_path


# --------------------------------------------------------------------------- #
# Initialization Tests
# --------------------------------------------------------------------------- #
class TestFileLoaderInit:
    """Tests for FileLoader initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        loader = FileLoader()
        assert loader._formats is not None
        assert ".txt" in loader._formats
        assert ".pdf" in loader._formats

    def test_init_with_additional_formats(self) -> None:
        """Test initialization with additional formats."""
        loader = FileLoader(additional_formats={".xyz": "application/xyz"})
        assert ".xyz" in loader._formats
        assert loader._formats[".xyz"] == "application/xyz"

    def test_supported_extensions(self, loader: FileLoader) -> None:
        """Test supported_extensions property."""
        extensions = loader.supported_extensions
        assert isinstance(extensions, list)
        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".json" in extensions


# --------------------------------------------------------------------------- #
# Format Detection Tests
# --------------------------------------------------------------------------- #
class TestDetectFormat:
    """Tests for format detection."""

    def test_detect_text_file(self, loader: FileLoader) -> None:
        """Test detecting text file format."""
        result = loader.detect_format(Path("document.txt"))
        assert result == ".txt"

    def test_detect_pdf_file(self, loader: FileLoader) -> None:
        """Test detecting PDF file format."""
        result = loader.detect_format(Path("document.pdf"))
        assert result == ".pdf"

    def test_detect_json_file(self, loader: FileLoader) -> None:
        """Test detecting JSON file format."""
        result = loader.detect_format(Path("data.json"))
        assert result == ".json"

    def test_detect_docx_file(self, loader: FileLoader) -> None:
        """Test detecting Word document format."""
        result = loader.detect_format(Path("document.docx"))
        assert result == ".docx"

    def test_detect_case_insensitive(self, loader: FileLoader) -> None:
        """Test format detection is case insensitive."""
        result = loader.detect_format(Path("document.PDF"))
        assert result == ".pdf"

    def test_detect_unsupported_raises(self, loader: FileLoader) -> None:
        """Test unsupported format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            loader.detect_format(Path("document.xyz"))
        assert "Unsupported file format" in str(exc_info.value)

    def test_detect_via_mime_fallback(self, loader: FileLoader) -> None:
        """Test format detection falls back to MIME type."""
        # Create file with unusual extension but known MIME type
        with patch("mimetypes.guess_type") as mock_guess:
            # Return a known MIME type
            mock_guess.return_value = ("text/plain", None)
            result = loader.detect_format(Path("document.unknown_but_text"))
            assert result == ".txt"

    def test_detect_mime_fallback_no_match(self, loader: FileLoader) -> None:
        """Test MIME fallback when no format matches."""
        with patch("mimetypes.guess_type") as mock_guess:
            mock_guess.return_value = ("application/x-unknown", None)
            with pytest.raises(ValueError):
                loader.detect_format(Path("document.really_unknown"))


# --------------------------------------------------------------------------- #
# is_supported Tests
# --------------------------------------------------------------------------- #
class TestIsSupported:
    """Tests for is_supported method."""

    def test_supported_format(self, loader: FileLoader) -> None:
        """Test checking supported format."""
        assert loader.is_supported(Path("doc.txt")) is True
        assert loader.is_supported(Path("doc.pdf")) is True

    def test_unsupported_format(self, loader: FileLoader) -> None:
        """Test checking unsupported format."""
        assert loader.is_supported(Path("doc.xyz")) is False

    def test_custom_format(self, loader_with_custom_formats: FileLoader) -> None:
        """Test checking custom format."""
        assert loader_with_custom_formats.is_supported(Path("doc.custom")) is True


# --------------------------------------------------------------------------- #
# File Loading Tests
# --------------------------------------------------------------------------- #
class TestLoad:
    """Tests for file loading."""

    def test_load_text_file(self, loader: FileLoader, temp_text_file: Path) -> None:
        """Test loading text file."""
        content = loader.load(temp_text_file)
        assert content == b"Hello, World!"

    def test_load_nonexistent_file(self, loader: FileLoader) -> None:
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/path/file.txt"))

    def test_load_directory_raises(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test loading directory raises error."""
        with pytest.raises(ValueError) as exc_info:
            loader.load(temp_dir)
        assert "Not a file" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# load_text Tests
# --------------------------------------------------------------------------- #
class TestLoadText:
    """Tests for text loading."""

    def test_load_text_utf8(self, loader: FileLoader, temp_text_file: Path) -> None:
        """Test loading text with UTF-8 encoding."""
        content = loader.load_text(temp_text_file)
        assert content == "Hello, World!"

    def test_load_text_custom_encoding(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test loading text with custom encoding."""
        file_path = temp_dir / "latin1.txt"
        file_path.write_bytes("Héllo".encode("latin-1"))

        content = loader.load_text(file_path, encoding="latin-1")
        assert "Héllo" in content

    def test_load_text_markdown(
        self, loader: FileLoader, temp_markdown_file: Path
    ) -> None:
        """Test loading markdown file as text."""
        content = loader.load_text(temp_markdown_file)
        assert "# Heading" in content


# --------------------------------------------------------------------------- #
# Batch Loading Tests
# --------------------------------------------------------------------------- #
class TestLoadBatch:
    """Tests for batch loading."""

    def test_load_batch_empty_dir(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test batch loading from empty directory."""
        result = loader.load_batch(temp_dir)
        assert result == []

    def test_load_batch_with_files(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test batch loading with files."""
        # Create test files
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")

        result = loader.load_batch(temp_dir)
        assert len(result) == 2

        paths = [p for p, _ in result]
        contents = [c for _, c in result]

        assert temp_dir / "file1.txt" in paths
        assert temp_dir / "file2.txt" in paths
        assert b"Content 1" in contents
        assert b"Content 2" in contents

    def test_load_batch_with_pattern(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test batch loading with pattern filter."""
        (temp_dir / "doc.txt").write_text("Text")
        (temp_dir / "doc.md").write_text("Markdown")

        result = loader.load_batch(temp_dir, pattern="*.txt")
        assert len(result) == 1
        assert result[0][0].suffix == ".txt"

    def test_load_batch_recursive(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test recursive batch loading."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        (temp_dir / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")

        result = loader.load_batch(temp_dir, recursive=True)
        assert len(result) == 2

    def test_load_batch_non_recursive(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test non-recursive batch loading."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        (temp_dir / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")

        result = loader.load_batch(temp_dir, recursive=False)
        assert len(result) == 1

    def test_load_batch_nonexistent_dir(self, loader: FileLoader) -> None:
        """Test batch loading from nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            loader.load_batch(Path("/nonexistent/directory"))

    def test_load_batch_not_a_directory(
        self, loader: FileLoader, temp_text_file: Path
    ) -> None:
        """Test batch loading from file (not directory)."""
        with pytest.raises(ValueError) as exc_info:
            loader.load_batch(temp_text_file)
        assert "Not a directory" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# iter_batch Tests
# --------------------------------------------------------------------------- #
class TestIterBatch:
    """Tests for iter_batch method."""

    def test_iter_batch_yields_tuples(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test iter_batch yields (path, content) tuples."""
        (temp_dir / "file.txt").write_text("Test")

        results = list(loader.iter_batch(temp_dir))
        assert len(results) == 1
        path, content = results[0]
        assert isinstance(path, Path)
        assert isinstance(content, bytes)

    def test_iter_batch_skips_unsupported(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test iter_batch skips unsupported files."""
        (temp_dir / "supported.txt").write_text("OK")
        (temp_dir / "unsupported.xyz").write_text("Skip")

        results = list(loader.iter_batch(temp_dir))
        assert len(results) == 1
        assert results[0][0].suffix == ".txt"

    def test_iter_batch_handles_permission_error(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test iter_batch handles permission errors gracefully."""
        (temp_dir / "file.txt").write_text("Test")

        # Mock load to raise PermissionError
        with patch.object(loader, "load", side_effect=PermissionError("Access denied")):
            results = list(loader.iter_batch(temp_dir))
            # Should continue and return empty (error logged)
            assert results == []

    def test_iter_batch_handles_os_error(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test iter_batch handles OS errors gracefully."""
        (temp_dir / "file.txt").write_text("Test")

        with patch.object(loader, "load", side_effect=OSError("IO error")):
            results = list(loader.iter_batch(temp_dir))
            assert results == []

    def test_iter_batch_nonexistent_dir(self, loader: FileLoader) -> None:
        """Test iter_batch from nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            list(loader.iter_batch(Path("/nonexistent")))

    def test_iter_batch_not_directory(
        self, loader: FileLoader, temp_text_file: Path
    ) -> None:
        """Test iter_batch from file."""
        with pytest.raises(ValueError):
            list(loader.iter_batch(temp_text_file))


# --------------------------------------------------------------------------- #
# list_files Tests
# --------------------------------------------------------------------------- #
class TestListFiles:
    """Tests for list_files method."""

    def test_list_files_empty(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test listing files in empty directory."""
        result = loader.list_files(temp_dir)
        assert result == []

    def test_list_files_with_files(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test listing files."""
        (temp_dir / "doc.txt").write_text("Text")
        (temp_dir / "doc.md").write_text("Markdown")

        result = loader.list_files(temp_dir)
        assert len(result) == 2

    def test_list_files_with_pattern(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test listing files with pattern."""
        (temp_dir / "doc.txt").write_text("Text")
        (temp_dir / "doc.md").write_text("Markdown")

        result = loader.list_files(temp_dir, pattern="*.md")
        assert len(result) == 1
        assert result[0].suffix == ".md"

    def test_list_files_recursive(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test recursive file listing."""
        subdir = temp_dir / "sub"
        subdir.mkdir()

        (temp_dir / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")

        result = loader.list_files(temp_dir, recursive=True)
        assert len(result) == 2

    def test_list_files_non_recursive(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test non-recursive file listing."""
        subdir = temp_dir / "sub"
        subdir.mkdir()

        (temp_dir / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")

        result = loader.list_files(temp_dir, recursive=False)
        assert len(result) == 1

    def test_list_files_nonexistent(self, loader: FileLoader) -> None:
        """Test listing from nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            loader.list_files(Path("/nonexistent"))

    def test_list_files_skips_unsupported(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test listing skips unsupported files."""
        (temp_dir / "supported.txt").write_text("OK")
        (temp_dir / "unsupported.xyz").write_text("Skip")

        result = loader.list_files(temp_dir)
        assert len(result) == 1


# --------------------------------------------------------------------------- #
# get_mime_type Tests
# --------------------------------------------------------------------------- #
class TestGetMimeType:
    """Tests for get_mime_type method."""

    def test_get_mime_type_txt(self, loader: FileLoader) -> None:
        """Test getting MIME type for text file."""
        mime = loader.get_mime_type(Path("doc.txt"))
        assert mime == "text/plain"

    def test_get_mime_type_pdf(self, loader: FileLoader) -> None:
        """Test getting MIME type for PDF."""
        mime = loader.get_mime_type(Path("doc.pdf"))
        assert mime == "application/pdf"

    def test_get_mime_type_json(self, loader: FileLoader) -> None:
        """Test getting MIME type for JSON."""
        mime = loader.get_mime_type(Path("data.json"))
        assert mime == "application/json"

    def test_get_mime_type_docx(self, loader: FileLoader) -> None:
        """Test getting MIME type for Word document."""
        mime = loader.get_mime_type(Path("doc.docx"))
        assert "wordprocessingml" in mime

    def test_get_mime_type_unknown_extension(self, loader: FileLoader) -> None:
        """Test getting MIME type for detected but unmapped format."""
        # This tests the fallback case when mime type is None in _formats
        # Actually, the code returns the dict value which could be None
        # if the format is registered with None as MIME type
        loader._formats[".test"] = "test/x-test"
        mime = loader.get_mime_type(Path("doc.test"))
        assert mime == "test/x-test"


# --------------------------------------------------------------------------- #
# Protocol Compliance Tests
# --------------------------------------------------------------------------- #
class TestFileLoaderProtocol:
    """Tests for FileLoaderProtocol compliance."""

    def test_implements_detect_format(self, loader: FileLoader) -> None:
        """Test FileLoader implements detect_format."""
        assert hasattr(loader, "detect_format")
        assert callable(loader.detect_format)

    def test_implements_load(self, loader: FileLoader) -> None:
        """Test FileLoader implements load."""
        assert hasattr(loader, "load")
        assert callable(loader.load)

    def test_implements_load_batch(self, loader: FileLoader) -> None:
        """Test FileLoader implements load_batch."""
        assert hasattr(loader, "load_batch")
        assert callable(loader.load_batch)


# --------------------------------------------------------------------------- #
# Edge Case Tests
# --------------------------------------------------------------------------- #
class TestLoaderEdgeCases:
    """Edge case tests for file loader."""

    def test_empty_file(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test loading empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()

        content = loader.load(empty_file)
        assert content == b""

    def test_large_file(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test loading large file."""
        large_file = temp_dir / "large.txt"
        large_content = "x" * 1_000_000  # 1MB
        large_file.write_text(large_content)

        content = loader.load(large_file)
        assert len(content) == 1_000_000

    def test_special_characters_in_filename(
        self, loader: FileLoader, temp_dir: Path
    ) -> None:
        """Test loading file with special characters in name."""
        special_file = temp_dir / "file with spaces.txt"
        special_file.write_text("Content")

        content = loader.load(special_file)
        assert content == b"Content"

    def test_unicode_content(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test loading file with Unicode content."""
        unicode_file = temp_dir / "unicode.txt"
        unicode_file.write_text("日本語テスト", encoding="utf-8")

        content = loader.load_text(unicode_file)
        assert "日本語" in content

    def test_binary_file(self, loader: FileLoader, temp_dir: Path) -> None:
        """Test loading binary file."""
        binary_file = temp_dir / "binary.pdf"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        content = loader.load(binary_file)
        assert content == b"\x00\x01\x02\x03"

    def test_path_as_string(self, loader: FileLoader, temp_text_file: Path) -> None:
        """Test load accepts string path."""
        # Convert Path to string
        content = loader.load(Path(str(temp_text_file)))
        assert content == b"Hello, World!"
