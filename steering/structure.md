# MONJYU - Project Structure

## ディレクトリ構成

```
MONJYU/
├── monjyu/                    # メインパッケージ (136 files)
│   ├── api/                   # MONJYU Facade API
│   ├── cli/                   # CLI (Typer)
│   ├── controller/            # Unified/Progressive/Hybrid
│   ├── document/              # ドキュメント処理 (PDF, Word, Excel, PPT)
│   ├── embedding/             # Embedding (Azure OpenAI, Ollama)
│   ├── errors/                # Error Handling (Circuit Breaker, Retry)
│   ├── index/                 # Level0/Level1, Extractors, Incremental
│   ├── query/                 # Vector/Global/Local/Hybrid/Router
│   ├── lazy/                  # LazySearch Engine
│   ├── citation/              # Citation Network
│   ├── mcp_server/            # MCP Server
│   ├── observability/         # Metrics, Tracing
│   ├── graph/                 # グラフ構築・操作
│   ├── nlp/                   # NLP処理
│   ├── search/                # 検索エンジン
│   ├── storage/               # Parquet, Cache
│   └── external/              # 外部連携
├── tests/                     # 2417 tests (80+ files)
│   ├── unit/                  # Unit Tests (2200+)
│   ├── integration/           # Integration Tests (165)
│   └── e2e/                   # E2E Tests (24)
├── docs/                      # ドキュメント
├── config/                    # 設定ファイル
├── examples/                  # 使用例
├── specs/                     # 仕様書 (v3.1/v3.2)
├── steering/                  # プロジェクトメモリ
│   ├── rules/                 # 憲法・ルール
│   ├── product.md             # プロダクトコンテキスト
│   ├── tech.md                # 技術スタック
│   └── structure.md           # 構造定義
├── storage/                   # データストレージ
│   ├── specs/                 # 仕様書
│   ├── archive/               # アーカイブ
│   └── changes/               # 変更履歴
├── pyproject.toml             # Python設定
└── musubix.config.json        # MUSUBIX設定
```

---

**更新日**: 2026-01-07
