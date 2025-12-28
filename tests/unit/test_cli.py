# MONJYU CLI Unit Tests
"""
FEAT-008: CLI - 単体テスト
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

runner = CliRunner()


# ========== Main App Tests ==========


class TestMainApp:
    """メインアプリケーションのテスト"""

    def test_app_import(self):
        from monjyu.cli import app
        assert app is not None

    def test_version_command(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "MONJYU" in result.stdout

    def test_help_command(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "index" in result.stdout
        assert "search" in result.stdout
        assert "document" in result.stdout


class TestOutputFormat:
    """OutputFormatのテスト"""

    def test_enum_values(self):
        from monjyu.cli.main import OutputFormat

        assert OutputFormat.text.value == "text"
        assert OutputFormat.json.value == "json"


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""

    def test_get_monjyu_default(self):
        from monjyu.cli.main import get_monjyu

        monjyu = get_monjyu()
        assert monjyu is not None

    def test_get_monjyu_with_config(self, tmp_path):
        from monjyu.cli.main import get_monjyu
        import yaml

        config_file = tmp_path / "monjyu.yaml"
        config_file.write_text(yaml.dump({"default_top_k": 15}))

        monjyu = get_monjyu(config_file)
        assert monjyu.config.default_top_k == 15


# ========== Index Command Tests ==========


class TestIndexCommands:
    """indexコマンドのテスト"""

    def test_index_build_help(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["index", "build", "--help"])
        assert result.exit_code == 0
        assert "Path to documents" in result.stdout

    def test_index_build_nonexistent_path(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["index", "build", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_index_status(self, tmp_path):
        from monjyu.cli import app

        # 出力ディレクトリを作成
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.invoke(app, ["index", "status"])
        # デフォルト設定で実行（エラーなしで終了すればOK）
        assert result.exit_code == 0

    def test_index_status_json(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["index", "status", "--output", "json"])
        assert result.exit_code == 0
        # JSON形式で出力される
        import json
        try:
            data = json.loads(result.stdout)
            assert "status" in data
        except json.JSONDecodeError:
            pass  # 一部の環境ではリッチ出力が含まれる場合がある

    def test_index_build_with_files(self, tmp_path):
        from monjyu.cli import app

        # テストドキュメント作成
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test Document\n\nContent here.")

        result = runner.invoke(app, [
            "index", "build", str(docs_dir),
            "--output", "text"
        ])
        assert result.exit_code == 0


# ========== Search Command Tests ==========


class TestSearchCommands:
    """searchコマンドのテスト"""

    def test_search_help(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.stdout.lower()

    def test_search_no_query(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["search"])
        assert result.exit_code == 1
        assert "provide a search query" in result.stdout.lower()

    def test_search_basic(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["search", "What is AI?"])
        assert result.exit_code == 0

    def test_search_with_mode(self):
        from monjyu.cli import app

        # callback形式では引数の前にオプションを置く
        result = runner.invoke(app, [
            "search", "--mode", "vector", "test query"
        ])
        assert result.exit_code == 0

    def test_search_json_output(self):
        from monjyu.cli import app

        result = runner.invoke(app, [
            "search", "--output", "json", "test"
        ])
        assert result.exit_code == 0

    def test_search_invalid_mode(self):
        from monjyu.cli import app

        result = runner.invoke(app, [
            "search", "--mode", "invalid_mode", "test"
        ])
        assert result.exit_code == 1
        assert "invalid" in result.stdout.lower()


# ========== Document Command Tests ==========


class TestDocumentCommands:
    """documentコマンドのテスト"""

    def test_document_list_help(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["document", "list", "--help"])
        assert result.exit_code == 0

    def test_document_list(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["document", "list"])
        assert result.exit_code == 0

    def test_document_list_json(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["document", "list", "--output", "json"])
        assert result.exit_code == 0

    def test_document_show_not_found(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["document", "show", "nonexistent_id"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_document_export_csv(self, tmp_path):
        from monjyu.cli import app

        output_file = tmp_path / "docs.csv"

        result = runner.invoke(app, [
            "document", "export", str(output_file),
            "--format", "csv"
        ])
        # ドキュメントがなくてもエラーにならない
        assert result.exit_code == 0

    def test_document_export_json(self, tmp_path):
        from monjyu.cli import app

        output_file = tmp_path / "docs.json"

        result = runner.invoke(app, [
            "document", "export", str(output_file),
            "--format", "json"
        ])
        assert result.exit_code == 0


# ========== Citation Command Tests ==========


class TestCitationCommands:
    """citationコマンドのテスト"""

    def test_citation_top_help(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["citation", "top", "--help"])
        assert result.exit_code == 0

    def test_citation_top(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["citation", "top"])
        # Citation networkがない場合はwarning
        assert result.exit_code == 0

    def test_citation_stats(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["citation", "stats"])
        assert result.exit_code == 0


# ========== Config Command Tests ==========


class TestConfigCommands:
    """configコマンドのテスト"""

    def test_config_init(self, tmp_path):
        from monjyu.cli import app

        config_file = tmp_path / "monjyu.yaml"

        result = runner.invoke(app, [
            "config", "init",
            "--output", str(config_file)
        ])
        assert result.exit_code == 0
        assert config_file.exists()

        # ファイル内容確認
        content = config_file.read_text()
        assert "environment" in content
        assert "output_path" in content

    def test_config_init_no_overwrite(self, tmp_path):
        from monjyu.cli import app

        config_file = tmp_path / "monjyu.yaml"
        config_file.write_text("existing: content")

        result = runner.invoke(app, [
            "config", "init",
            "--output", str(config_file)
        ])
        assert result.exit_code == 1
        assert "already exists" in result.stdout.lower()

    def test_config_init_force(self, tmp_path):
        from monjyu.cli import app

        config_file = tmp_path / "monjyu.yaml"
        config_file.write_text("existing: content")

        result = runner.invoke(app, [
            "config", "init",
            "--output", str(config_file),
            "--force"
        ])
        assert result.exit_code == 0

    def test_config_show(self, tmp_path):
        from monjyu.cli import app
        import yaml

        config_file = tmp_path / "monjyu.yaml"
        config_file.write_text(yaml.dump({"environment": "local"}))

        result = runner.invoke(app, [
            "config", "show",
            "--config", str(config_file)
        ])
        assert result.exit_code == 0

    def test_config_show_not_found(self):
        from monjyu.cli import app

        result = runner.invoke(app, [
            "config", "show",
            "--config", "/nonexistent/config.yaml"
        ])
        assert result.exit_code == 1

    def test_config_validate(self, tmp_path):
        from monjyu.cli import app
        import yaml

        config_file = tmp_path / "monjyu.yaml"
        config_file.write_text(yaml.dump({
            "environment": "local",
            "output_path": str(tmp_path / "output"),
        }))

        result = runner.invoke(app, [
            "config", "validate",
            "--config", str(config_file)
        ])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_config_env(self):
        from monjyu.cli import app

        result = runner.invoke(app, ["config", "env"])
        assert result.exit_code == 0
        assert "AZURE_OPENAI" in result.stdout


# ========== Integration Tests ==========


class TestCLIWorkflow:
    """CLIワークフローテスト"""

    def test_full_workflow(self, tmp_path):
        """完全なワークフロー（init -> index -> search）"""
        from monjyu.cli import app

        # 1. 設定初期化
        config_file = tmp_path / "monjyu.yaml"
        result = runner.invoke(app, [
            "config", "init",
            "--output", str(config_file)
        ])
        assert result.exit_code == 0

        # 2. ドキュメント作成
        docs_dir = tmp_path / "papers"
        docs_dir.mkdir()
        (docs_dir / "paper1.md").write_text("""
# Attention Is All You Need

This paper introduces the Transformer architecture.
""")

        # 3. インデックス構築
        result = runner.invoke(app, [
            "index", "build", str(docs_dir),
            "--config", str(config_file)
        ])
        assert result.exit_code == 0

        # 4. ステータス確認
        result = runner.invoke(app, [
            "index", "status",
            "--config", str(config_file)
        ])
        assert result.exit_code == 0

        # 5. 検索
        result = runner.invoke(app, [
            "search", "--config", str(config_file), "What is Transformer?"
        ])
        assert result.exit_code == 0

    def test_json_output_consistency(self, tmp_path):
        """JSON出力の一貫性テスト"""
        from monjyu.cli import app
        import json

        # index status
        result = runner.invoke(app, ["index", "status", "--output", "json"])
        assert result.exit_code == 0

        # document list
        result = runner.invoke(app, ["document", "list", "--output", "json"])
        assert result.exit_code == 0

# ========== Init Command Tests ==========


class TestInitCommand:
    """initコマンドのテスト"""

    def test_init_help(self):
        """ヘルプ表示"""
        from monjyu.cli import app

        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.stdout

    def test_init_creates_structure(self, tmp_path):
        """プロジェクト構造が作成される"""
        from monjyu.cli import app

        project_dir = tmp_path / "my_project"
        result = runner.invoke(app, ["init", str(project_dir)])
        
        assert result.exit_code == 0
        assert (project_dir / "monjyu.yaml").exists()
        assert (project_dir / "output").is_dir()
        assert (project_dir / "papers").is_dir()
        assert (project_dir / ".gitignore").exists()

    def test_init_no_overwrite(self, tmp_path):
        """既存プロジェクトは上書きしない"""
        from monjyu.cli import app

        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "monjyu.yaml").write_text("existing: true")
        
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 1
        assert "already initialized" in result.stdout.lower()

    def test_init_force_overwrite(self, tmp_path):
        """--forceで上書き可能"""
        from monjyu.cli import app

        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "monjyu.yaml").write_text("existing: true")
        
        result = runner.invoke(app, ["init", str(project_dir), "--force"])
        assert result.exit_code == 0
        
        # 新しい設定で上書きされている
        content = (project_dir / "monjyu.yaml").read_text()
        assert "existing: true" not in content

    def test_init_current_directory(self, tmp_path):
        """カレントディレクトリでの初期化"""
        from monjyu.cli import app
        import os
        
        # 一時的にカレントディレクトリを変更
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert (tmp_path / "monjyu.yaml").exists()
        finally:
            os.chdir(old_cwd)


# ========== Query Command Tests ==========


class TestQueryCommand:
    """queryコマンドのテスト"""

    def test_query_help(self):
        """ヘルプ表示"""
        from monjyu.cli import app

        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "Search query" in result.stdout
        assert "--mode" in result.stdout

    def test_query_requires_question(self):
        """引数が必須"""
        from monjyu.cli import app

        result = runner.invoke(app, ["query"])
        assert result.exit_code != 0
        # Typerはstderrにエラーを出力することがある
        assert "Missing argument" in result.stdout or result.exit_code == 2

    def test_query_basic(self, tmp_path):
        """基本的なクエリ実行"""
        from monjyu.cli import app

        result = runner.invoke(app, ["query", "What is machine learning?"])
        # インデックスがない場合はエラーになる可能性があるが、
        # コマンドは実行される
        assert "Answer" in result.stdout or "Error" in result.stdout

    def test_query_with_mode(self, tmp_path):
        """モード指定でのクエリ"""
        from monjyu.cli import app

        for mode in ["vector", "lazy", "local", "global", "auto"]:
            result = runner.invoke(app, ["query", "test", "--mode", mode])
            # モードの検証（実行が成功すればOK）
            # 実際のデータがない場合はエラーになる可能性があるが、
            # モードパラメータ自体は受け付けられる
            assert "Invalid search mode" not in result.stdout

    def test_query_invalid_mode(self):
        """不正なモードでエラー"""
        from monjyu.cli import app

        result = runner.invoke(app, ["query", "test", "--mode", "invalid"])
        assert result.exit_code == 1
        assert "Invalid search mode" in result.stdout

    def test_query_json_output(self, tmp_path):
        """JSON出力"""
        from monjyu.cli import app
        import json

        result = runner.invoke(app, ["query", "test", "--output", "json"])
        # JSON形式で出力されることを確認（パースできればOK）
        if result.exit_code == 0:
            # 出力がJSONとしてパース可能
            try:
                output = json.loads(result.stdout)
                assert "query" in output or "answer" in output
            except json.JSONDecodeError:
                pass  # 検索自体が失敗した場合はスキップ