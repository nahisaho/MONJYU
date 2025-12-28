# Contributing to MONJYU

MONJYU への貢献を歓迎します！このガイドでは、プロジェクトへの貢献方法を説明します。

## 🚀 Quick Start

```bash
# 1. Fork & Clone
git clone https://github.com/YOUR-USERNAME/MONJYU.git
cd MONJYU

# 2. Setup environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Run tests
pytest tests/
```

## 📁 Project Structure

```
MONJYU/
├── monjyu/                    # メインパッケージ
│   ├── api/                   # MONJYU Facade API
│   ├── cli/                   # CLI (Typer)
│   ├── controller/            # Unified/Progressive/Hybrid
│   ├── document/              # ドキュメント処理
│   ├── embedding/             # Embedding (Azure OpenAI, Ollama)
│   ├── index/                 # Level0/Level1, Extractors
│   ├── query/                 # Vector/Global/Local/Hybrid/Router
│   ├── lazy/                  # LazySearch Engine
│   ├── search/                # Search Engine (Hybrid)
│   ├── citation/              # Citation Network
│   ├── mcp_server/            # MCP Server
│   └── storage/               # Parquet, Cache
├── tests/
│   ├── unit/                  # 単体テスト
│   ├── integration/           # 統合テスト
│   └── e2e/                   # E2Eテスト
├── specs/                     # 仕様書
└── steering/                  # プロジェクトドキュメント
```

## 🧪 Testing

### テストの実行

```bash
# 全テスト
pytest tests/

# 単体テストのみ
pytest tests/unit/

# 統合テストのみ
pytest tests/integration/

# カバレッジ付き
pytest --cov=monjyu --cov-report=html

# 特定のテストファイル
pytest tests/unit/test_hybrid_search.py -v
```

### テストの書き方

```python
# tests/unit/test_example.py
import pytest
from monjyu.module import MyClass

class TestMyClass:
    """MyClassのテスト"""
    
    def test_basic_functionality(self):
        """基本機能のテスト"""
        obj = MyClass()
        result = obj.method()
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_async_method(self):
        """非同期メソッドのテスト"""
        obj = MyClass()
        result = await obj.async_method()
        assert result is not None
```

## 📝 Coding Standards

### Style Guide

- **Python**: PEP 8 準拠
- **Type hints**: 全ての関数に型ヒントを付与
- **Docstrings**: Google style
- **Line length**: 100文字以下

### 例

```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """検索結果.
    
    Attributes:
        query: 検索クエリ
        hits: ヒットしたドキュメント
        total_time_ms: 処理時間 (ミリ秒)
    """
    query: str
    hits: List[Dict[str, Any]]
    total_time_ms: float


def search(
    query: str,
    top_k: int = 10,
    min_score: Optional[float] = None,
) -> SearchResult:
    """検索を実行.
    
    Args:
        query: 検索クエリ
        top_k: 返す結果数
        min_score: 最小スコア閾値
        
    Returns:
        検索結果
        
    Raises:
        ValueError: クエリが空の場合
    """
    if not query:
        raise ValueError("Query cannot be empty")
    
    # 実装...
    return SearchResult(query=query, hits=[], total_time_ms=0.0)
```

## 🔄 Pull Request Process

### 1. Issue の作成

大きな変更を行う前に、まず Issue を作成して議論してください。

### 2. Branch の作成

```bash
# Feature
git checkout -b feature/add-new-search-mode

# Bugfix
git checkout -b fix/search-timeout-issue

# Documentation
git checkout -b docs/update-readme
```

### 3. Commit Message

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: フォーマット変更
- `refactor`: リファクタリング
- `test`: テスト追加/修正
- `chore`: ビルド/ツール変更

**Example:**
```
feat(search): add hybrid search with RRF fusion

- Implement HybridSearch class with multiple fusion methods
- Add support for VECTOR, LAZY, GLOBAL, LOCAL search methods
- Include RRF, WEIGHTED, MAX, COMBSUM, COMBMNZ fusion algorithms

Closes #123
```

### 4. Pull Request

1. テストが全て通ることを確認
2. ドキュメントを更新
3. CHANGELOG.md に変更を追記
4. PR を作成し、レビューを依頼

## 📚 Documentation

### Docstring の書き方

```python
def complex_function(
    param1: str,
    param2: int,
    param3: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """関数の説明（1行）.
    
    より詳細な説明をここに書く。
    複数行に渡っても良い。
    
    Args:
        param1: パラメータ1の説明
        param2: パラメータ2の説明
        param3: パラメータ3の説明（省略可能）
    
    Returns:
        戻り値の説明。辞書の場合はキーと値の説明も含める:
        - key1: 説明
        - key2: 説明
    
    Raises:
        ValueError: param1が空の場合
        TypeError: param2が負の場合
    
    Examples:
        >>> result = complex_function("test", 10)
        >>> print(result["key1"])
        value1
    """
    pass
```

## 🐛 Bug Reports

バグ報告には以下を含めてください：

1. **環境情報**
   - Python バージョン
   - OS
   - MONJYU バージョン

2. **再現手順**
   - 具体的なコード例
   - 入力データ

3. **期待される動作**

4. **実際の動作**

5. **エラーメッセージ/ログ**

## 📞 Contact

- GitHub Issues: バグ報告、機能要望
- Discussions: 質問、議論

## 📜 License

MIT License - 貢献されたコードは MIT License の下で公開されます。

---

Thank you for contributing to MONJYU! 🙏
