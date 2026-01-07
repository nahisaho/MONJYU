# MONJYU クイックスタートガイド

**Version**: 3.5.1  
**Last Updated**: 2026-01-07

このガイドでは、MONJYUを素早く始めるための手順を説明します。

---

## 📋 目次

1. [インストール](#1-インストール)
2. [基本的な使い方](#2-基本的な使い方)
3. [検索モード](#3-検索モード)
4. [CLI の使い方](#4-cli-の使い方)
5. [設定](#5-設定)
6. [MCP Server (Claude Desktop)](#6-mcp-server-claude-desktop)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. インストール

### 要件

- Python 3.10以上
- pip

### 基本インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-org/MONJYU.git
cd MONJYU

# 仮想環境を作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# インストール
pip install -e .
```

### 開発用インストール

```bash
pip install -e ".[dev,docs]"
```

---

## 2. 基本的な使い方

### Python API

```python
from monjyu import MONJYU

# 初期化
monjyu = MONJYU()

# ドキュメントをインデックス化
await monjyu.index("/path/to/papers/")

# 検索
result = await monjyu.search(
    query="What is GraphRAG?",
    mode="auto",  # 自動選択
)

# 結果を表示
print(result.answer)

# 引用元を確認
for doc in result.citations:
    print(f"- {doc.title} (score: {doc.relevance_score:.3f})")
```

### 同期版 API

```python
from monjyu import MONJYU

monjyu = MONJYU()

# 同期版（asyncioなし）
result = monjyu.search_sync(
    query="What is GraphRAG?",
    mode="lazy",
)
```

---

## 3. 検索モード

MONJYUは複数の検索モードを提供します：

| モード | 説明 | 速度 | 品質 | 用途 |
|--------|------|------|------|------|
| `auto` | 自動選択 | ⚡⚡ | ★★★ | 一般用途 |
| `vector` | ベクトル検索 | ⚡⚡⚡ | ★★☆ | 高速検索 |
| `lazy` | LazyGraphRAG | ⚡⚡ | ★★★ | 包括的回答 |
| `hybrid` | RRF融合 | ⚡⚡ | ★★★★ | 最高品質 |
| `global` | コミュニティベース | ⚡ | ★★★ | 広範なトピック |
| `local` | エンティティベース | ⚡⚡ | ★★★ | 特定エンティティ |

### 使用例

```python
# 高速検索が必要な場合
result = await monjyu.search("transformer", mode="vector")

# 詳細な回答が必要な場合
result = await monjyu.search("Compare BERT and GPT", mode="lazy")

# 最高品質の回答が必要な場合
result = await monjyu.search("Latest NLP techniques", mode="hybrid")
```

---

## 4. CLI の使い方

### プロジェクト初期化

```bash
# 新しいプロジェクトを初期化
monjyu init my_project

# 既存ディレクトリで初期化
cd existing_project
monjyu init .
```

### インデックス構築

```bash
# ドキュメントをインデックス化
monjyu index build papers/

# 特定のレベルでインデックス化
monjyu index build papers/ --level 1

# インデックス状態を確認
monjyu index status
```

### 検索

```bash
# 基本的な検索
monjyu query "What is GraphRAG?"

# モードを指定
monjyu query "transformer architecture" --mode vector

# JSON出力
monjyu query "NLP techniques" --output json

# 結果数を指定
monjyu query "machine learning" --top-k 20
```

### その他のコマンド

```bash
# バージョン確認
monjyu version

# ヘルプ
monjyu --help
monjyu index --help
monjyu query --help
```

---

## 5. 設定

### 設定ファイル (monjyu.yaml)

```yaml
# 基本設定
output_path: "./output"
environment: "local"  # "local" or "azure"

# 検索設定
default_search_mode: "lazy"
default_top_k: 10

# チャンク設定
chunk_size: 1200
chunk_overlap: 100

# ローカル LLM (Ollama)
llm_model: "llama3:8b-instruct-q4_K_M"
embedding_model: "nomic-embed-text"
ollama_base_url: "http://localhost:11434"

# Azure設定（オプション）
# azure_openai_endpoint: "https://your-endpoint.openai.azure.com/"
# azure_openai_api_key: "your-key"  # または環境変数で設定
```

### 環境変数

```bash
# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

# Azure AI Search（オプション）
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_API_KEY="your-key"

# Ollama（ローカル）
export OLLAMA_HOST="http://localhost:11434"
```

### Python での設定

```python
from monjyu.api import MONJYU, MONJYUConfig, SearchMode

config = MONJYUConfig(
    output_path="./my_output",
    environment="local",
    default_search_mode=SearchMode.LAZY,
    default_top_k=15,
    chunk_size=1000,
)

monjyu = MONJYU(config=config)
```

---

## 6. MCP Server (Claude Desktop)

### 起動方法

```bash
# stdio モード（デフォルト）
monjyu-mcp

# HTTP モード
monjyu-mcp --http --port 8080
```

### Claude Desktop 設定

`claude_desktop_config.json` に追加：

```json
{
  "mcpServers": {
    "monjyu": {
      "command": "monjyu-mcp"
    }
  }
}
```

### 利用可能なツール

| ツール | 説明 |
|--------|------|
| `monjyu_search` | 学術論文を検索 |
| `monjyu_get_document` | ドキュメント詳細を取得 |
| `monjyu_list_documents` | ドキュメント一覧を取得 |
| `monjyu_citation_chain` | 引用チェーンを取得 |
| `monjyu_find_related` | 関連論文を検索 |
| `monjyu_status` | インデックス状態を確認 |
| `monjyu_get_metrics` | 引用メトリクスを取得 |

---

## 7. トラブルシューティング

### よくある問題

#### インポートエラー

```
ModuleNotFoundError: No module named 'monjyu'
```

**解決策**: インストールを確認
```bash
pip install -e .
```

#### Ollama 接続エラー

```
ConnectionError: Unable to connect to Ollama
```

**解決策**: Ollama が起動しているか確認
```bash
# Ollama を起動
ollama serve

# モデルをプル
ollama pull llama3:8b-instruct-q4_K_M
ollama pull nomic-embed-text
```

#### Azure 認証エラー

```
AuthenticationError: Invalid API key
```

**解決策**: 環境変数を確認
```bash
echo $AZURE_OPENAI_API_KEY
```

### ログの確認

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from monjyu import MONJYU
monjyu = MONJYU()
```

### サポート

- **Issues**: [GitHub Issues](https://github.com/your-org/MONJYU/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

---

## 次のステップ

- [API Reference](API_REFERENCE.md) - 詳細なAPI仕様
- [Architecture](../specs/02_architecture_v3.md) - アーキテクチャ設計
- [Examples](../examples/) - サンプルコード

---

**MONJYU v3.5.1** | 2026-01-07
