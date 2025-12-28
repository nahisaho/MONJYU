# MONJYU Phase 1 (MVP) タスク分解

**ドキュメントID**: MONJYU-TASKS-001  
**作成日**: 2025-12-25  
**ステータス**: Draft

---

## 1. サマリー

### 1.1 フィーチャー別工数

| # | フィーチャー | タスク数 | 見積もり | 累計 |
|---|------------|---------|---------|------|
| FEAT-001 | Document Processing | 8 | 22h | 22h |
| FEAT-002 | Index Level 0 | 10 | 22h | 44h |
| FEAT-003 | Index Level 1 | 9 | 20h | 64h |
| FEAT-004 | Vector Search | 8 | 16h | 80h |
| FEAT-005 | Lazy Search | 9 | 20h | 100h |
| FEAT-006 | Citation Network | 8 | 17h | 117h |
| FEAT-007 | Python API | 8 | 18h | 135h |
| FEAT-008 | CLI | 9 | 14h | 149h |
| FEAT-009 | MCP Server | 10 | 13h | 162h |
| **合計** | | **79** | **162h** | |

### 1.2 人日換算

- **総工数**: 162時間
- **1日8時間換算**: 約20人日
- **1日4時間換算**: 約40人日

---

## 2. 依存関係グラフ

```
                    ┌────────────────────────────────────────────────────────┐
                    │                     Phase 1 MVP                         │
                    └────────────────────────────────────────────────────────┘
                                              │
         ┌────────────────────────────────────┴─────────────────────────────┐
         │                                                                   │
         ▼                                                                   │
┌─────────────────┐                                                         │
│   FEAT-001      │                                                         │
│   Document      │                                                         │
│   Processing    │                                                         │
│     (22h)       │                                                         │
└────────┬────────┘                                                         │
         │                                                                   │
         ├─────────────────────────────────────────────────────────────┐    │
         │                                                              │    │
         ▼                                                              ▼    │
┌─────────────────┐                                            ┌─────────────────┐
│   FEAT-002      │                                            │   FEAT-006      │
│   Index Level 0 │                                            │   Citation      │
│     (22h)       │                                            │   Network       │
└────────┬────────┘                                            │     (17h)       │
         │                                                      └────────┬────────┘
         ├─────────────────┐                                             │
         │                 │                                             │
         ▼                 ▼                                             │
┌─────────────────┐ ┌─────────────────┐                                 │
│   FEAT-003      │ │   FEAT-004      │                                 │
│   Index Level 1 │ │   Vector        │                                 │
│     (20h)       │ │   Search        │                                 │
└────────┬────────┘ │     (16h)       │                                 │
         │          └────────┬────────┘                                 │
         │                   │                                           │
         └───────────────────┼───────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   FEAT-005      │
                    │   Lazy Search   │
                    │     (20h)       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   FEAT-007      │
                    │   Python API    │
                    │     (18h)       │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │   FEAT-008      │           │   FEAT-009      │
     │   CLI           │           │   MCP Server    │
     │     (14h)       │           │     (13h)       │
     └─────────────────┘           └─────────────────┘
```

---

## 3. 実装順序（クリティカルパス）

### 3.1 推奨実装順序

```
Phase 1-A: Foundation (44h)
├── FEAT-001: Document Processing (22h)
└── FEAT-002: Index Level 0 (22h)

Phase 1-B: Index & Search (53h) 
├── FEAT-003: Index Level 1 (20h)  ← FEAT-002 完了後
├── FEAT-004: Vector Search (16h)  ← FEAT-002 完了後（並行可）
└── FEAT-006: Citation Network (17h) ← FEAT-001 完了後（並行可）

Phase 1-C: Core Engine (20h)
└── FEAT-005: Lazy Search (20h)    ← FEAT-003, FEAT-004 完了後

Phase 1-D: Integration (45h)
├── FEAT-007: Python API (18h)     ← FEAT-005 完了後
├── FEAT-008: CLI (14h)            ← FEAT-007 完了後（並行可）
└── FEAT-009: MCP Server (13h)     ← FEAT-007 完了後（並行可）
```

### 3.2 並列実装可能なグループ

| グループ | フィーチャー | 条件 |
|---------|------------|------|
| Group A | FEAT-001 | 開始可能 |
| Group B | FEAT-002, FEAT-006 | FEAT-001 完了後 |
| Group C | FEAT-003, FEAT-004 | FEAT-002 完了後 |
| Group D | FEAT-005 | FEAT-003, FEAT-004 完了後 |
| Group E | FEAT-007 | FEAT-005, FEAT-006 完了後 |
| Group F | FEAT-008, FEAT-009 | FEAT-007 完了後（並行可） |

---

## 4. 詳細タスク一覧

### 4.1 FEAT-001: Document Processing (22h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-001-01 | FileLoader 実装 | 2h | - | ⬜ |
| TASK-001-02 | TextChunker 実装 | 2h | - | ⬜ |
| TASK-001-03 | DocumentParser 実装 | 3h | TASK-001-01 | ⬜ |
| TASK-001-04 | UnstructuredPDFProcessor 実装 | 4h | TASK-001-03 | ⬜ |
| TASK-001-05 | AzureDocIntelPDFProcessor 実装 | 4h | TASK-001-03 | ⬜ |
| TASK-001-06 | DocumentProcessingPipeline 実装 | 2h | TASK-001-01~05 | ⬜ |
| TASK-001-07 | 単体テスト作成 | 3h | TASK-001-01~06 | ⬜ |
| TASK-001-08 | 統合テスト作成 | 2h | TASK-001-07 | ⬜ |

**主要成果物**:
- `monjyu/document/loader.py`
- `monjyu/document/parser.py`
- `monjyu/document/chunker.py`
- `monjyu/document/pdf/unstructured_processor.py`
- `monjyu/document/pdf/azure_processor.py`

---

### 4.2 FEAT-002: Index Level 0 (22h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-002-01 | EmbeddingClient Protocol 定義 | 1h | - | ⬜ |
| TASK-002-02 | OllamaEmbeddingClient 実装 | 2h | TASK-002-01 | ⬜ |
| TASK-002-03 | AzureOpenAIEmbeddingClient 実装 | 2h | TASK-002-01 | ⬜ |
| TASK-002-04 | VectorIndexer Protocol 定義 | 1h | - | ⬜ |
| TASK-002-05 | LanceDBIndexer 実装 | 3h | TASK-002-04 | ⬜ |
| TASK-002-06 | AzureAISearchIndexer 実装 | 3h | TASK-002-04 | ⬜ |
| TASK-002-07 | ParquetStorage 実装 | 2h | - | ⬜ |
| TASK-002-08 | Level0IndexBuilder 実装 | 3h | TASK-002-01~07 | ⬜ |
| TASK-002-09 | 単体テスト作成 | 3h | TASK-002-01~08 | ⬜ |
| TASK-002-10 | 統合テスト作成 | 2h | TASK-002-09 | ⬜ |

**主要成果物**:
- `monjyu/embedding/client.py`
- `monjyu/embedding/ollama.py`
- `monjyu/embedding/azure_openai.py`
- `monjyu/index/level0/builder.py`
- `monjyu/storage/parquet.py`
- `monjyu/storage/lancedb.py`

---

### 4.3 FEAT-003: Index Level 1 (20h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-003-01 | NLPProcessor Protocol 定義 | 1h | - | ⬜ |
| TASK-003-02 | spaCy NLPProcessor 実装 | 3h | TASK-003-01 | ⬜ |
| TASK-003-03 | RAKE キーフレーズ抽出 実装 | 2h | TASK-003-02 | ⬜ |
| TASK-003-04 | NounPhraseGraphBuilder 実装 | 3h | TASK-003-03 | ⬜ |
| TASK-003-05 | LeidenCommunityDetector 実装 | 3h | TASK-003-04 | ⬜ |
| TASK-003-06 | Level1IndexBuilder 実装 | 3h | TASK-003-01~05 | ⬜ |
| TASK-003-07 | 単体テスト作成 | 3h | TASK-003-01~06 | ⬜ |
| TASK-003-08 | 統合テスト作成 | 2h | TASK-003-07 | ⬜ |

**主要成果物**:
- `monjyu/nlp/processor.py`
- `monjyu/nlp/keyword_extractor.py`
- `monjyu/graph/noun_phrase_graph.py`
- `monjyu/graph/community_detector.py`
- `monjyu/index/level1/builder.py`

---

### 4.4 FEAT-004: Vector Search (16h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-004-01 | QueryEncoder 実装 | 2h | FEAT-002 | ⬜ |
| TASK-004-02 | LanceDBVectorSearcher 実装 | 2h | FEAT-002 | ⬜ |
| TASK-004-03 | AzureAISearchVectorSearcher 実装 | 2h | FEAT-002 | ⬜ |
| TASK-004-04 | HybridSearcher 実装 (BM25 + Vector) | 2h | TASK-004-02,03 | ⬜ |
| TASK-004-05 | AnswerSynthesizer 実装 | 3h | - | ⬜ |
| TASK-004-06 | VectorSearchEngine 実装 | 2h | TASK-004-01~05 | ⬜ |
| TASK-004-07 | 単体テスト作成 | 2h | TASK-004-01~06 | ⬜ |
| TASK-004-08 | 統合テスト作成 | 1h | TASK-004-07 | ⬜ |

**主要成果物**:
- `monjyu/search/query_encoder.py`
- `monjyu/search/vector_searcher.py`
- `monjyu/search/hybrid_searcher.py`
- `monjyu/search/answer_synthesizer.py`
- `monjyu/search/vector_search_engine.py`

---

### 4.5 FEAT-005: Lazy Search (20h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-005-01 | RelevanceTester 実装 | 3h | - | ⬜ |
| TASK-005-02 | ClaimExtractor 実装 | 3h | - | ⬜ |
| TASK-005-03 | IterativeDeepener 実装 | 3h | TASK-005-01,02 | ⬜ |
| TASK-005-04 | CommunitySearcher 実装 | 2h | FEAT-003 | ⬜ |
| TASK-005-05 | LazySearchState 実装 | 1h | - | ⬜ |
| TASK-005-06 | LazySearchEngine 実装 | 3h | TASK-005-01~05 | ⬜ |
| TASK-005-07 | 単体テスト作成 | 3h | TASK-005-01~06 | ⬜ |
| TASK-005-08 | 統合テスト作成 | 2h | TASK-005-07 | ⬜ |

**主要成果物**:
- `monjyu/lazy_search/relevance_tester.py`
- `monjyu/lazy_search/claim_extractor.py`
- `monjyu/lazy_search/iterative_deepener.py`
- `monjyu/lazy_search/community_searcher.py`
- `monjyu/lazy_search/engine.py`

---

### 4.6 FEAT-006: Citation Network (17h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-006-01 | ReferenceResolver 実装 | 2h | FEAT-001 | ⬜ |
| TASK-006-02 | CitationGraphBuilder 実装 | 3h | TASK-006-01 | ⬜ |
| TASK-006-03 | MetricsCalculator 実装 (PageRank, HITS) | 3h | TASK-006-02 | ⬜ |
| TASK-006-04 | CitationAnalyzer 実装 | 2h | TASK-006-03 | ⬜ |
| TASK-006-05 | CitationNetworkManager 実装 | 2h | TASK-006-01~04 | ⬜ |
| TASK-006-06 | GraphML エクスポート 実装 | 1h | TASK-006-02 | ⬜ |
| TASK-006-07 | 単体テスト作成 | 2h | TASK-006-01~06 | ⬜ |
| TASK-006-08 | 統合テスト作成 | 2h | TASK-006-07 | ⬜ |

**主要成果物**:
- `monjyu/citation/resolver.py`
- `monjyu/citation/graph_builder.py`
- `monjyu/citation/metrics.py`
- `monjyu/citation/analyzer.py`
- `monjyu/citation/manager.py`

---

### 4.7 FEAT-007: Python API (18h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-007-01 | ConfigManager 実装 | 2h | - | ⬜ |
| TASK-007-02 | StateManager 実装 | 2h | - | ⬜ |
| TASK-007-03 | ComponentFactory 実装 | 3h | FEAT-001~006 | ⬜ |
| TASK-007-04 | MONJYU Facade 実装 | 4h | TASK-007-01~03 | ⬜ |
| TASK-007-05 | AsyncMONJYU 実装 | 2h | TASK-007-04 | ⬜ |
| TASK-007-06 | ユーティリティ関数 実装 | 1h | - | ⬜ |
| TASK-007-07 | 単体テスト作成 | 2h | TASK-007-01~06 | ⬜ |
| TASK-007-08 | 統合テスト・使用例作成 | 2h | TASK-007-07 | ⬜ |

**主要成果物**:
- `monjyu/config.py`
- `monjyu/state.py`
- `monjyu/factory.py`
- `monjyu/monjyu.py` (Facade)
- `monjyu/__init__.py`

---

### 4.8 FEAT-008: CLI (14h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-008-01 | メインアプリ構造 (typer) | 1h | - | ⬜ |
| TASK-008-02 | index コマンド群 | 2h | FEAT-007 | ⬜ |
| TASK-008-03 | search コマンド群 | 2h | FEAT-007 | ⬜ |
| TASK-008-04 | document コマンド群 | 2h | FEAT-007 | ⬜ |
| TASK-008-05 | citation コマンド群 | 2h | FEAT-007 | ⬜ |
| TASK-008-06 | config コマンド群 | 1h | - | ⬜ |
| TASK-008-07 | エラーハンドリング | 1h | TASK-008-01~06 | ⬜ |
| TASK-008-08 | テスト作成 | 2h | TASK-008-01~07 | ⬜ |
| TASK-008-09 | ドキュメント作成 | 1h | TASK-008-01~07 | ⬜ |

**主要成果物**:
- `monjyu/cli/__init__.py`
- `monjyu/cli/main.py`
- `monjyu/cli/commands/index.py`
- `monjyu/cli/commands/search.py`
- `monjyu/cli/commands/document.py`
- `monjyu/cli/commands/citation.py`
- `monjyu/cli/commands/config.py`

---

### 4.9 FEAT-009: MCP Server (13h)

| タスクID | タスク | 見積もり | 依存 | 状態 |
|----------|--------|---------|------|------|
| TASK-009-01 | MCPサーバー基盤 | 2h | - | ⬜ |
| TASK-009-02 | monjyu_search 実装 | 2h | FEAT-007 | ⬜ |
| TASK-009-03 | monjyu_get_document 実装 | 1h | FEAT-007 | ⬜ |
| TASK-009-04 | monjyu_list_documents 実装 | 1h | FEAT-007 | ⬜ |
| TASK-009-05 | monjyu_citation_chain 実装 | 1h | FEAT-007 | ⬜ |
| TASK-009-06 | monjyu_find_related 実装 | 1h | FEAT-007 | ⬜ |
| TASK-009-07 | monjyu_status/metrics 実装 | 1h | FEAT-007 | ⬜ |
| TASK-009-08 | エラーハンドリング | 1h | TASK-009-01~07 | ⬜ |
| TASK-009-09 | テスト作成 | 2h | TASK-009-01~08 | ⬜ |
| TASK-009-10 | ドキュメント・設定例 | 1h | TASK-009-01~08 | ⬜ |

**主要成果物**:
- `monjyu/mcp_server/__init__.py`
- `monjyu/mcp_server/server.py`
- `monjyu/mcp_server/handlers.py`

---

## 5. スプリント計画案

### 5.1 2週間スプリント（1日4時間）

| スプリント | 期間 | フィーチャー | 工数 |
|-----------|------|------------|------|
| Sprint 1 | Week 1-2 | FEAT-001 Document Processing | 22h |
| Sprint 2 | Week 3-4 | FEAT-002 Index Level 0 | 22h |
| Sprint 3 | Week 5-6 | FEAT-003 Index Level 1 + FEAT-006 Citation | 37h |
| Sprint 4 | Week 7-8 | FEAT-004 Vector Search + FEAT-005 Lazy Search | 36h |
| Sprint 5 | Week 9-10 | FEAT-007 Python API + FEAT-008 CLI | 32h |
| Sprint 6 | Week 11 | FEAT-009 MCP Server + 統合テスト | 13h |

**総期間**: 約11週間（2.5ヶ月）

### 5.2 1週間スプリント（フルタイム）

| スプリント | フィーチャー | 工数 |
|-----------|------------|------|
| Sprint 1 | FEAT-001 + FEAT-002 (並行) | 44h |
| Sprint 2 | FEAT-003 + FEAT-004 + FEAT-006 (並行) | 53h |
| Sprint 3 | FEAT-005 + FEAT-007 | 38h |
| Sprint 4 | FEAT-008 + FEAT-009 + 統合 | 27h |

**総期間**: 約4週間（1ヶ月）

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| PDF解析精度不足 | 中 | unstructuredの設定調整、Azure Doc Intelligenceへのフォールバック |
| Ollama応答遅延 | 低 | バッチサイズ調整、非同期処理 |
| Leidenアルゴリズム収束 | 低 | パラメータチューニング、タイムアウト設定 |
| LanceDB容量制限 | 低 | シャーディング検討、Azure AI Search切り替え |
| MCPプロトコル互換性 | 中 | mcp-python-sdk最新版追従 |

---

## 7. 完了定義 (Definition of Done)

各タスク完了時に確認する項目：

- [ ] コードが実装されている
- [ ] 単体テストが作成・通過している
- [ ] docstringが記載されている
- [ ] 型ヒントが付与されている
- [ ] linter (ruff) エラーがない
- [ ] formatter (black) 適用済み
- [ ] フィーチャー仕様の受入基準を満たしている

各フィーチャー完了時：
- [ ] 統合テストが通過している
- [ ] ドキュメントが更新されている
- [ ] PR レビュー完了

---

## 8. 次のアクション

1. **開発環境セットアップ**
   - [ ] `pyproject.toml` 作成
   - [ ] 依存パッケージ定義
   - [ ] pytest / ruff / black 設定

2. **FEAT-001 実装開始**
   - [ ] TASK-001-01: FileLoader 実装
   - [ ] TASK-001-02: TextChunker 実装

実装を開始する場合は `#sdd-implement FEAT-001` をお知らせください。
