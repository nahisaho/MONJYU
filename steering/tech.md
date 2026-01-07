# MONJYU - Technology Stack

## 言語

- Python 3.11+

## フレームワーク・ライブラリ

### Document Processing
- unstructured (PDF, DOCX)
- python-docx, python-pptx
- tiktoken, langdetect

### NLP
- spaCy
- rake-nltk

### Graph
- NetworkX
- leidenalg (コミュニティ検出)
- python-igraph

### Vector Store
- LanceDB
- Azure AI Search (オプション)
- PyArrow

### Embedding & LLM
- Ollama (ローカル)
- Azure OpenAI (オプション)

### CLI & MCP
- Typer (CLI)
- Rich (TUI)
- MCP (Model Context Protocol)

### Web & API
- httpx, aiohttp (async HTTP)
- Pydantic (バリデーション)
- tenacity (リトライ)

## ツール

- pytest, pytest-asyncio, pytest-cov (テスト)
- ruff (リント・フォーマット)
- mypy (型チェック)

## テスト状況

- **2417 tests** (80+ files)
- **83% coverage**
- Unit: 2200+, Integration: 165, E2E: 24

---

**更新日**: 2026-01-07
