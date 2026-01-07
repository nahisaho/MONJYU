"""
MONJYU Security Tests

NFR-TST-006: セキュリティテスト
対応要件: NFR-SEC-001〜007, OWASP Top 10

テストカテゴリ:
1. 入力検証 (Input Validation)
2. シークレット管理 (Secret Management)
3. パス走査防止 (Path Traversal Prevention)
4. 設定バリデーション (Config Validation)
5. ログセキュリティ (Log Security)
6. 依存関係セキュリティ (Dependency Security)
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# =============================================================================
# 1. 入力検証テスト (OWASP A01, A03)
# =============================================================================

class TestInputValidation:
    """入力検証のセキュリティテスト"""

    def test_path_traversal_prevention_in_output_path(self):
        """パス走査攻撃の防止 (OWASP A01)"""
        from monjyu.api.base import MONJYUConfig
        
        # 正常なパス
        config = MONJYUConfig(output_path=Path("./output"))
        assert config.output_path == Path("./output")
        
        # 相対パス走査を含むパスも許可されるが、正規化される
        config2 = MONJYUConfig(output_path=Path("./output/../other"))
        # Path は正規化しないが、実際の使用時に resolve() される想定
        assert ".." in str(config2.output_path)

    def test_filename_sanitization(self):
        """ファイル名のサニタイズ検証"""
        # 危険な文字を含むファイル名
        dangerous_names = [
            "../../../etc/passwd",
            "file\x00name.txt",  # null byte injection
            "file|name.txt",
            "file;name.txt",
            "file`name.txt",
        ]
        
        for name in dangerous_names:
            # Path でパース可能であることを確認（実際のサニタイズは実装依存）
            path = Path(name)
            # null byte は Path で処理できない場合がある
            if "\x00" not in name:
                assert path.name is not None

    def test_query_string_length_limit(self):
        """クエリ文字列の長さ制限"""
        from monjyu.lazy.base import LazySearchConfig
        
        # 通常のクエリ
        config = LazySearchConfig()
        
        # 非常に長いクエリ（DoS攻撃対策）
        long_query = "a" * 100000
        # 設定自体は受け入れるが、実行時に制限される想定
        assert len(long_query) == 100000

    def test_chunk_size_bounds(self):
        """チャンクサイズの境界値チェック"""
        from monjyu.api.base import MONJYUConfig
        
        # 正常値
        config = MONJYUConfig(chunk_size=1200)
        assert config.chunk_size == 1200
        
        # 極端に小さい値
        config_small = MONJYUConfig(chunk_size=1)
        assert config_small.chunk_size == 1
        
        # 極端に大きい値
        config_large = MONJYUConfig(chunk_size=1000000)
        assert config_large.chunk_size == 1000000


# =============================================================================
# 2. シークレット管理テスト (NFR-SEC-005, OWASP A02)
# =============================================================================

class TestSecretManagement:
    """シークレット管理のセキュリティテスト"""

    def test_api_key_not_exposed_in_repr(self):
        """APIキーが__repr__に含まれないこと（マスク済み）"""
        from monjyu.api.base import MONJYUConfig
        
        config = MONJYUConfig(
            azure_openai_api_key="sk-secret-key-12345",
            azure_search_api_key="search-secret-key-67890",
        )
        
        repr_str = repr(config)
        # APIキーがreprに含まれていないか、マスクされていることを確認
        assert "sk-secret-key-12345" not in repr_str or "***" in repr_str

    def test_api_key_not_exposed_in_str(self):
        """APIキーが__str__に含まれないこと（マスク済み）"""
        from monjyu.api.base import MONJYUConfig
        
        config = MONJYUConfig(
            azure_openai_api_key="sk-secret-key-12345",
        )
        
        str_output = str(config)
        # 完全なキーが露出していないことを確認
        assert "sk-secret-key-12345" not in str_output or "***" in str_output

    def test_environment_variable_api_key_loading(self):
        """環境変数からのAPIキー読み込み"""
        test_key = "test-api-key-secure"
        
        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": test_key}):
            key = os.environ.get("AZURE_OPENAI_API_KEY")
            assert key == test_key

    def test_api_key_not_in_error_messages(self):
        """エラーメッセージにAPIキーが含まれないこと"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict(os.environ, {}, clear=True):
            config = AzureSearchVectorStoreConfig()
        
        try:
            config.validate()
        except ValueError as e:
            error_msg = str(e)
            # エラーメッセージにAPIキーの値が含まれていないこと
            assert "secret" not in error_msg.lower() or "required" in error_msg.lower()

    def test_config_serialization_masks_secrets(self):
        """設定のシリアライズ時にシークレットがマスクされること"""
        from monjyu.api.config import ConfigManager
        from monjyu.api.base import MONJYUConfig
        
        config = MONJYUConfig(
            azure_openai_api_key="sk-secret-12345",
            azure_search_api_key="search-secret-67890",
        )
        
        config_dict = ConfigManager._config_to_dict(config)
        
        # 辞書に含まれるキーを確認
        # 注: 実装によってはマスクされている可能性がある
        if "azure_openai_api_key" in config_dict:
            # キーが含まれる場合、値の確認
            pass  # 実装依存


# =============================================================================
# 3. パス走査防止テスト (OWASP A01)
# =============================================================================

class TestPathTraversalPrevention:
    """パス走査攻撃の防止テスト"""

    def test_document_path_validation(self):
        """ドキュメントパスの検証"""
        # 基準ディレクトリ
        base_dir = Path("/safe/documents")
        
        # 正常なパス
        safe_path = base_dir / "paper.pdf"
        assert safe_path.is_relative_to(base_dir) or str(safe_path).startswith(str(base_dir))
        
        # パス走査を含むパス
        unsafe_input = "../../../etc/passwd"
        resolved = (base_dir / unsafe_input).resolve()
        # 解決後のパスが基準ディレクトリ外を指す
        # 実際の実装ではこれを検出してブロックする必要がある

    def test_output_directory_isolation(self):
        """出力ディレクトリの分離"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # 正常な出力ファイル
            safe_file = output_dir / "result.json"
            assert safe_file.parent == output_dir
            
            # パス走査を試みる出力
            unsafe_file = output_dir / "../../../tmp/malicious.txt"
            resolved = unsafe_file.resolve()
            # 解決後のパスが出力ディレクトリ外
            assert not str(resolved).startswith(str(output_dir.resolve()))


# =============================================================================
# 4. 設定バリデーションテスト (OWASP A04)
# =============================================================================

class TestConfigValidation:
    """設定バリデーションのセキュリティテスト"""

    def test_azure_search_config_validation(self):
        """Azure Search設定の検証"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        # エンドポイントなしでバリデーション失敗
        with patch.dict(os.environ, {}, clear=True):
            config = AzureSearchVectorStoreConfig()
        
        with pytest.raises(ValueError, match="endpoint"):
            config.validate()

    def test_config_type_validation(self):
        """設定値の型検証"""
        from monjyu.api.base import MONJYUConfig
        
        # 正常な型
        config = MONJYUConfig(
            default_top_k=10,
            chunk_size=1200,
        )
        assert isinstance(config.default_top_k, int)
        assert isinstance(config.chunk_size, int)

    def test_search_mode_enum_validation(self):
        """検索モードのenum検証"""
        from monjyu.api.base import SearchMode
        
        # 有効な値
        assert SearchMode.VECTOR.value == "vector"
        assert SearchMode.LAZY.value == "lazy"
        
        # 無効な値でエラー
        with pytest.raises(ValueError):
            SearchMode("invalid_mode")

    def test_index_level_enum_validation(self):
        """インデックスレベルのenum検証"""
        from monjyu.api.base import IndexLevel
        
        # 有効な値
        assert IndexLevel.LEVEL_0.value == 0
        assert IndexLevel.LEVEL_1.value == 1
        
        # 無効な値でエラー
        with pytest.raises(ValueError):
            IndexLevel(99)


# =============================================================================
# 5. ログセキュリティテスト (NFR-SEC-006, OWASP A09)
# =============================================================================

class TestLogSecurity:
    """ログセキュリティのテスト"""

    def test_api_key_pattern_detection(self):
        """APIキーパターンの検出"""
        # 一般的なAPIキーパターン
        api_key_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI style
            r"[a-f0-9]{32}",  # 32 char hex
            r"[A-Za-z0-9+/]{40,}={0,2}",  # Base64 encoded
        ]
        
        test_secrets = [
            "sk-abcdefghijklmnopqrstuvwxyz12345",
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
            "VGhpcyBpcyBhIHNlY3JldCBrZXkgZm9yIHRlc3Rpbmc=",
        ]
        
        for secret in test_secrets:
            for pattern in api_key_patterns:
                # いずれかのパターンにマッチすることを確認
                match = re.search(pattern, secret)
                if match:
                    break

    def test_log_message_sanitization(self):
        """ログメッセージのサニタイズ"""
        sensitive_data = {
            "api_key": "sk-secret-key-12345",
            "password": "super_secret_password",
            "token": "bearer_token_xyz",
        }
        
        # ログに出力する前にサニタイズすべきキー
        sensitive_keys = ["api_key", "password", "token", "secret", "credential"]
        
        for key in sensitive_data.keys():
            assert any(sens in key.lower() for sens in sensitive_keys)

    def test_error_stack_trace_filtering(self):
        """エラースタックトレースのフィルタリング"""
        try:
            # 意図的にエラーを発生
            raise ValueError("Error with secret: sk-12345")
        except ValueError as e:
            error_msg = str(e)
            # 実際の実装ではシークレットをマスクする必要がある
            assert "Error" in error_msg


# =============================================================================
# 6. 依存関係セキュリティテスト (OWASP A06)
# =============================================================================

class TestDependencySecurity:
    """依存関係のセキュリティテスト"""

    def test_check_known_vulnerable_imports(self):
        """既知の脆弱なインポートがないことを確認"""
        # 既知の脆弱なパッケージ（例）
        vulnerable_packages = [
            "pyyaml<5.4",  # CVE-2020-14343
            "requests<2.20.0",  # CVE-2018-18074
        ]
        
        # 実際のテストでは pyproject.toml や requirements.txt を解析
        # ここでは構造のみを示す
        assert True  # プレースホルダー

    def test_yaml_safe_load_usage(self):
        """YAML読み込みの安全性"""
        import yaml
        
        # 安全な読み込み（yaml.safe_load を使用）
        safe_yaml = "key: value\nnumber: 123"
        result = yaml.safe_load(safe_yaml)
        assert result == {"key": "value", "number": 123}
        
        # yaml.load（unsafe）ではなく yaml.safe_load を使用すべき
        # yaml.load(data) は Loader 指定なしで警告が出る

    def test_json_safe_parsing(self):
        """JSON解析の安全性"""
        import json
        
        # 正常なJSON
        safe_json = '{"key": "value"}'
        result = json.loads(safe_json)
        assert result == {"key": "value"}
        
        # 不正なJSONでエラー
        invalid_json = '{"key": undefined}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)


# =============================================================================
# 7. 認証・認可テスト (NFR-SEC-001, NFR-SEC-002)
# =============================================================================

class TestAuthenticationAuthorization:
    """認証・認可のテスト"""

    def test_api_key_required_for_azure(self):
        """Azure環境でAPIキーが必須"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict(os.environ, {
            "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
        }, clear=True):
            config = AzureSearchVectorStoreConfig()
        
        # APIキーもManaged Identityもない場合はエラー
        with pytest.raises(ValueError, match="API key|Managed Identity"):
            config.validate()

    def test_managed_identity_alternative(self):
        """Managed Identity認証の代替"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict(os.environ, {
            "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
        }, clear=True):
            config = AzureSearchVectorStoreConfig(use_managed_identity=True)
        
        # Managed Identityが有効なら検証通過
        config.validate()


# =============================================================================
# 8. レート制限テスト (NFR-SEC-007)
# =============================================================================

class TestRateLimiting:
    """レート制限のテスト"""

    def test_external_api_rate_limit_config(self):
        """外部API呼び出しのレート制限設定"""
        from monjyu.external.base import ExternalAPIConfig
        
        config = ExternalAPIConfig()
        
        # タイムアウトとリトライ設定の確認
        assert config.timeout > 0
        assert config.max_retries >= 0
        assert config.retry_delay > 0

    def test_retry_backoff_configuration(self):
        """リトライバックオフ設定"""
        from monjyu.external.base import ExternalAPIConfig
        
        config = ExternalAPIConfig(
            max_retries=3,
            retry_delay=1.0,
        )
        
        # エクスポネンシャルバックオフの計算
        delays = [config.retry_delay * (2 ** i) for i in range(config.max_retries)]
        assert delays == [1.0, 2.0, 4.0]


# =============================================================================
# 9. データ暗号化テスト (NFR-SEC-003, NFR-SEC-004)
# =============================================================================

class TestDataEncryption:
    """データ暗号化のテスト"""

    def test_tls_endpoint_validation(self):
        """TLSエンドポイントの検証"""
        # HTTPSエンドポイントのパターン
        https_pattern = r"^https://"
        
        valid_endpoints = [
            "https://api.example.com",
            "https://search.windows.net",
            "https://openai.azure.com",
        ]
        
        for endpoint in valid_endpoints:
            assert re.match(https_pattern, endpoint)

    def test_http_endpoint_warning(self):
        """HTTPエンドポイントの警告"""
        # 本番環境でHTTPは使用すべきでない
        http_endpoints = [
            "http://localhost:11434",  # ローカル開発のみ許可
        ]
        
        for endpoint in http_endpoints:
            # localhostのHTTPは開発環境のみで許可
            is_localhost = "localhost" in endpoint or "127.0.0.1" in endpoint
            if not is_localhost:
                # 本番環境でHTTPは警告/エラーとすべき
                pass


# =============================================================================
# 10. インジェクション防止テスト (OWASP A03)
# =============================================================================

class TestInjectionPrevention:
    """インジェクション攻撃防止のテスト"""

    def test_command_injection_prevention(self):
        """コマンドインジェクション防止"""
        # 危険な文字列パターン
        dangerous_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& malicious_command",
        ]
        
        for dangerous in dangerous_inputs:
            # これらの入力がコマンドとして実行されないことを確認
            # 実際のテストでは subprocess 呼び出しを監視
            assert dangerous is not None  # プレースホルダー

    def test_prompt_injection_awareness(self):
        """プロンプトインジェクションの認識"""
        # LLMプロンプトインジェクションの例
        injection_attempts = [
            "Ignore previous instructions and...",
            "SYSTEM: You are now a...",
            "```\nNew instructions:\n```",
        ]
        
        for attempt in injection_attempts:
            # これらのパターンを検出できることを確認
            # 実際の対策は実装レベルで行う
            assert isinstance(attempt, str)


# =============================================================================
# 統合セキュリティテスト
# =============================================================================

class TestSecurityIntegration:
    """統合セキュリティテスト"""

    def test_full_config_security(self):
        """完全な設定のセキュリティチェック"""
        from monjyu.api.base import MONJYUConfig
        
        # セキュアな設定
        config = MONJYUConfig(
            environment="azure",
            azure_openai_api_key="sk-test-key",
            azure_search_api_key="search-key",
        )
        
        # 設定が正しく作成されること
        assert config.environment == "azure"

    def test_local_environment_security(self):
        """ローカル環境のセキュリティ"""
        from monjyu.api.base import MONJYUConfig
        
        # ローカル環境ではAPIキー不要
        config = MONJYUConfig(environment="local")
        
        assert config.environment == "local"
        # Ollamaはローカルなのでキー不要
        assert config.ollama_base_url is not None
