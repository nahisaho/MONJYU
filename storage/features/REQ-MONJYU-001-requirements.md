# MONJYU EARS要件定義書 v2.0

**文書番号**: MONJYU-REQ-002  
**バージョン**: 2.0.0  
**作成日**: 2025-12-24  
**ステータス**: Draft  
**準拠**: MUSUBI SDD Constitutional Article IV (EARS Format)

---

## 1. 概要

### 1.1 プロジェクトビジョン

**MONJYU** (文殊) - 「三人寄れば文殊の知恵」

LazyGraphRAGの完全独立実装として、Microsoft GraphRAGと同等以上の機能を約1/100のコストで実現する次世代RAGシステム。

### 1.2 システム範囲

GraphRAGと同等以上の機能を持つ独立パッケージ：

| 機能カテゴリ | GraphRAG対応機能 | MONJYU実装 |
|-------------|-----------------|-----------|
| インデックス | TextUnit生成、Entity/Relationship抽出、Community検出、Community Report生成 | LazyIndex（軽量版） |
| クエリ | Global Search, Local Search, DRIFT Search, Basic Search | LazySearch + 各モード対応 |
| データモデル | Document, TextUnit, Entity, Relationship, Community, CommunityReport, Covariate | 同等データモデル |
| CLI | graphrag init/index/query | monjyu init/index/query |
| API | Python API, Streaming | Python API, Async, Streaming |
| プロンプト | Auto Tuning, Manual Tuning | プロンプトチューニング |

---

## 2. 機能要件（EARS形式）

### 2.1 インデックス機能 (Index Domain)

#### REQ-IDX-001: ドキュメントローダー

**EARS Pattern**: Ubiquitous

> The system SHALL load documents from specified input directories supporting txt, md, pdf, html, csv, json formats.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-001 |
| 優先度 | P0 (必須) |
| 入力 | ディレクトリパス、ファイルパターン |
| 出力 | Document[] (id, title, text, metadata) |
| 受入基準 | 100MB以上のドキュメントセットを処理可能 |
| GraphRAG対応 | documents table |
| トレーサビリティ | DESIGN-IDX-001, TEST-IDX-001 |

---

#### REQ-IDX-002: テキストユニット分割

**EARS Pattern**: Ubiquitous

> The system SHALL split documents into configurable TextUnits with chunk_size and chunk_overlap parameters.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-002 |
| 優先度 | P0 (必須) |
| 入力 | Document[], ChunkConfig (size: 300-1200 tokens, overlap: 0-50%) |
| 出力 | TextUnit[] (id, text, n_tokens, document_ids) |
| 受入基準 | デフォルト300トークン、オーバーラップ100トークン |
| GraphRAG対応 | text_units table |
| トレーサビリティ | DESIGN-IDX-002, TEST-IDX-002 |

---

#### REQ-IDX-003: エンティティ抽出

**EARS Pattern**: Ubiquitous

> The system SHALL extract entities (person, organization, geo, event, concept) from TextUnits using LLM.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-003 |
| 優先度 | P0 (必須) |
| 入力 | TextUnit[] |
| 出力 | Entity[] (title, type, description, text_unit_ids) |
| 受入基準 | precision > 80%, recall > 70% |
| GraphRAG対応 | entities table |
| トレーサビリティ | DESIGN-IDX-003, TEST-IDX-003 |

---

#### REQ-IDX-004: リレーションシップ抽出

**EARS Pattern**: Ubiquitous

> The system SHALL extract relationships between entities from TextUnits using LLM.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-004 |
| 優先度 | P0 (必須) |
| 入力 | TextUnit[], Entity[] |
| 出力 | Relationship[] (source, target, description, weight, text_unit_ids) |
| 受入基準 | エンティティ間の有意な関係を抽出 |
| GraphRAG対応 | relationships table |
| トレーサビリティ | DESIGN-IDX-004, TEST-IDX-004 |

---

#### REQ-IDX-005: エンティティ/リレーションシップ要約

**EARS Pattern**: Event-driven

> WHEN multiple descriptions exist for the same entity or relationship, the system SHALL summarize them into a single coherent description using LLM.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-005 |
| 優先度 | P0 (必須) |
| 入力 | Entity[] / Relationship[] with multiple descriptions |
| 出力 | Entity[] / Relationship[] with summarized description |
| 受入基準 | 重複排除、情報統合 |
| GraphRAG対応 | Entity & Relationship Summarization |
| トレーサビリティ | DESIGN-IDX-005, TEST-IDX-005 |

---

#### REQ-IDX-006: コミュニティ検出

**EARS Pattern**: Ubiquitous

> The system SHALL detect hierarchical communities from the entity-relationship graph using the Leiden algorithm.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-006 |
| 優先度 | P0 (必須) |
| 入力 | Entity[], Relationship[] |
| 出力 | Community[] (id, level, parent, children, entity_ids, relationship_ids) |
| 受入基準 | 階層的クラスタリング、複数レベル |
| GraphRAG対応 | communities table |
| トレーサビリティ | DESIGN-IDX-006, TEST-IDX-006 |

---

#### REQ-IDX-007: コミュニティレポート生成

**EARS Pattern**: Ubiquitous

> The system SHALL generate summary reports for each community describing key entities, relationships, and insights.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-007 |
| 優先度 | P0 (必須) |
| 入力 | Community[], Entity[], Relationship[] |
| 出力 | CommunityReport[] (community_id, title, summary, full_content, findings, rank) |
| 受入基準 | 各コミュニティのエグゼクティブサマリー生成 |
| GraphRAG対応 | community_reports table |
| トレーサビリティ | DESIGN-IDX-007, TEST-IDX-007 |

---

#### REQ-IDX-008: クレーム抽出（オプション）

**EARS Pattern**: Optional Feature

> WHERE claim extraction is enabled, the system SHALL extract factual claims with subject, object, status, and time bounds.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-008 |
| 優先度 | P2 (オプション) |
| 入力 | TextUnit[] |
| 出力 | Covariate[] (subject_id, object_id, type, description, status, start_date, end_date) |
| 受入基準 | プロンプトチューニング必須 |
| GraphRAG対応 | covariates table |
| トレーサビリティ | DESIGN-IDX-008, TEST-IDX-008 |

---

#### REQ-IDX-009: NLPベース軽量インデックス（MONJYU独自）

**EARS Pattern**: Ubiquitous

> The system SHALL extract keywords and named entities using NLP techniques without LLM calls for cost-efficient indexing.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-009 |
| 優先度 | P1 (高) |
| 入力 | TextUnit[] |
| 出力 | NLPFeatures[] (keywords, entities, summary) |
| 受入基準 | LLMコスト0、spaCy/NLTK使用可 |
| GraphRAG対応 | LazyGraphRAG独自機能 |
| トレーサビリティ | DESIGN-IDX-009, TEST-IDX-009 |

---

#### REQ-IDX-010: インクリメンタルインデックス更新

**EARS Pattern**: Event-driven

> WHEN new documents are added, the system SHALL update the index incrementally without full rebuild.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-010 |
| 優先度 | P2 (中) |
| 入力 | 新規Document[] |
| 出力 | 更新されたIndex |
| 受入基準 | 追加分のみ処理、既存データ保持 |
| GraphRAG対応 | Incremental Update |
| トレーサビリティ | DESIGN-IDX-010, TEST-IDX-010 |

---

#### REQ-IDX-011: ベクトルエンベディング

**EARS Pattern**: Ubiquitous

> The system SHALL generate vector embeddings for TextUnits, Entities, and CommunityReports.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-011 |
| 優先度 | P0 (必須) |
| 入力 | TextUnit[] / Entity[] / CommunityReport[] |
| 出力 | Embedding[] (id, vector, model, dimensions) |
| 受入基準 | OpenAI/Azure OpenAI/ローカルモデル対応 |
| GraphRAG対応 | Text Embedding Phase |
| トレーサビリティ | DESIGN-IDX-011, TEST-IDX-011 |

---

#### REQ-IDX-012: グラフ可視化エンベディング

**EARS Pattern**: Optional Feature

> WHERE visualization is enabled, the system SHALL generate Node2Vec embeddings and UMAP coordinates for graph visualization.

| 項目 | 内容 |
|------|------|
| ID | REQ-IDX-012 |
| 優先度 | P2 (オプション) |
| 入力 | Entity[], Relationship[] |
| 出力 | Entity[] with (x, y) coordinates |
| 受入基準 | 2D可視化用座標生成 |
| GraphRAG対応 | Network Visualization Phase |
| トレーサビリティ | DESIGN-IDX-012, TEST-IDX-012 |

---

### 2.2 クエリ機能 (Query Domain)

#### REQ-QRY-001: Global Search

**EARS Pattern**: Event-driven

> WHEN a user submits a global query, the system SHALL generate a response using community reports in a map-reduce pattern.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-001 |
| 優先度 | P0 (必須) |
| 入力 | query: str, community_level: int, max_data_tokens: int |
| 出力 | SearchResult (response, citations, context_data) |
| 受入基準 | データセット全体の要約質問に回答可能 |
| GraphRAG対応 | Global Search |
| トレーサビリティ | DESIGN-QRY-001, TEST-QRY-001 |

---

#### REQ-QRY-002: Local Search

**EARS Pattern**: Event-driven

> WHEN a user submits a local query, the system SHALL generate a response by combining entity information, relationships, and source text chunks.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-002 |
| 優先度 | P0 (必須) |
| 入力 | query: str, top_k_entities: int, context_budget: int |
| 出力 | SearchResult (response, citations, entities, relationships) |
| 受入基準 | 特定エンティティに関する詳細質問に回答可能 |
| GraphRAG対応 | Local Search |
| トレーサビリティ | DESIGN-QRY-002, TEST-QRY-002 |

---

#### REQ-QRY-003: DRIFT Search

**EARS Pattern**: Event-driven

> WHEN a user submits a DRIFT query, the system SHALL combine community context with local search using dynamic follow-up questions.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-003 |
| 優先度 | P0 (必須) |
| 入力 | query: str, primer_config, follow_up_config |
| 出力 | SearchResult with hierarchical Q&A structure |
| 受入基準 | Global + Localの統合検索 |
| GraphRAG対応 | DRIFT Search |
| トレーサビリティ | DESIGN-QRY-003, TEST-QRY-003 |

---

#### REQ-QRY-004: Basic Search (Vector RAG)

**EARS Pattern**: Event-driven

> WHEN a user submits a basic query, the system SHALL perform standard vector similarity search on TextUnits.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-004 |
| 優先度 | P1 (高) |
| 入力 | query: str, top_k: int |
| 出力 | SearchResult (response, source_chunks) |
| 受入基準 | ベースラインRAGとの比較可能 |
| GraphRAG対応 | Basic Search |
| トレーサビリティ | DESIGN-QRY-004, TEST-QRY-004 |

---

#### REQ-QRY-005: LazySearch（MONJYU独自）

**EARS Pattern**: Ubiquitous

> The system SHALL provide budget-controlled lazy search with iterative deepening, query expansion, relevance testing, and claim extraction.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-005 |
| 優先度 | P0 (必須) |
| ステータス | ✅ 実装済み |
| 入力 | query: str, budget: int, context_tokens: int |
| 出力 | SearchResult with claims and context |
| 受入基準 | GraphRAGの1/100コストで同等品質 |
| GraphRAG対応 | LazyGraphRAG独自機能 |
| トレーサビリティ | DESIGN-QRY-005, TEST-QRY-005 |

**サブコンポーネント**:
- REQ-QRY-005a: QueryExpander ✅
- REQ-QRY-005b: RelevanceTester ✅
- REQ-QRY-005c: ClaimExtractor ✅
- REQ-QRY-005d: IterativeDeepener ✅
- REQ-QRY-005e: LazyContextBuilder ✅

---

#### REQ-QRY-006: 質問生成

**EARS Pattern**: Event-driven

> WHEN given an initial query, the system SHALL generate follow-up questions for deeper investigation.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-006 |
| 優先度 | P2 (中) |
| 入力 | query: str, conversation_history: List[dict] |
| 出力 | follow_up_questions: List[str] |
| 受入基準 | 会話継続のための質問提案 |
| GraphRAG対応 | Question Generation |
| トレーサビリティ | DESIGN-QRY-006, TEST-QRY-006 |

---

#### REQ-QRY-007: マルチインデックス検索

**EARS Pattern**: Optional Feature

> WHERE multiple indexes exist, the system SHALL search across multiple data sources simultaneously.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-007 |
| 優先度 | P2 (オプション) |
| 入力 | query: str, index_ids: List[str] |
| 出力 | SearchResult aggregated from multiple indexes |
| 受入基準 | 複数データソースの統合検索 |
| GraphRAG対応 | Multi Index Search |
| トレーサビリティ | DESIGN-QRY-007, TEST-QRY-007 |

---

#### REQ-QRY-008: 会話履歴管理

**EARS Pattern**: State-driven

> WHILE a conversation session is active, the system SHALL maintain conversation history and use it for context-aware responses.

| 項目 | 内容 |
|------|------|
| ID | REQ-QRY-008 |
| 優先度 | P0 (必須) |
| ステータス | ✅ 部分実装済み |
| 入力 | query: str, conversation_history: List[dict] |
| 出力 | SearchResult with conversation context |
| 受入基準 | マルチターン会話対応 |
| GraphRAG対応 | Conversation History |
| トレーサビリティ | DESIGN-QRY-008, TEST-QRY-008 |

---

### 2.3 データモデル要件

#### REQ-DATA-001: GraphRAG互換データモデル

**EARS Pattern**: Ubiquitous

> The system SHALL implement data models compatible with GraphRAG output schemas.

| 項目 | 内容 |
|------|------|
| ID | REQ-DATA-001 |
| 優先度 | P0 (必須) |
| データモデル | Document, TextUnit, Entity, Relationship, Community, CommunityReport, Covariate |
| フォーマット | Parquet (デフォルト), JSON |
| 受入基準 | GraphRAG出力と相互運用可能 |
| トレーサビリティ | DESIGN-DATA-001, TEST-DATA-001 |

---

### 2.4 ストレージ要件

#### REQ-STR-001: ファイルベースストレージ

**EARS Pattern**: Ubiquitous

> The system SHALL persist indexes to file system in Parquet and JSON formats.

| 項目 | 内容 |
|------|------|
| ID | REQ-STR-001 |
| 優先度 | P0 (必須) |
| フォーマット | Parquet (テーブル), JSON (メタデータ) |
| 受入基準 | 読み込み/書き込み < 1秒 (100MB) |
| GraphRAG対応 | File Storage |
| トレーサビリティ | DESIGN-STR-001, TEST-STR-001 |

---

#### REQ-STR-002: ベクトルストア統合

**EARS Pattern**: Optional Feature

> WHERE external vector store is configured, the system SHALL integrate with FAISS, ChromaDB, or LanceDB.

| 項目 | 内容 |
|------|------|
| ID | REQ-STR-002 |
| 優先度 | P1 (高) |
| 対応 | FAISS, ChromaDB, LanceDB |
| 受入基準 | 各ベクトルストアへの読み書き可能 |
| GraphRAG対応 | Vector Store Integration |
| トレーサビリティ | DESIGN-STR-002, TEST-STR-002 |

---

#### REQ-STR-003: LLMキャッシュ

**EARS Pattern**: Event-driven

> WHEN an identical LLM request is made, the system SHALL return cached response to reduce costs.

| 項目 | 内容 |
|------|------|
| ID | REQ-STR-003 |
| 優先度 | P2 (中) |
| 入力 | LLM request hash |
| 出力 | Cached LLM response |
| 受入基準 | キャッシュヒット率 > 80% (同一クエリ) |
| GraphRAG対応 | LLM Cache |
| トレーサビリティ | DESIGN-STR-003, TEST-STR-003 |

---

### 2.5 LLMプロバイダー要件

#### REQ-LLM-001: OpenAI対応

**EARS Pattern**: Ubiquitous

> The system SHALL support OpenAI API with models including GPT-4o, GPT-4o-mini, GPT-3.5-turbo.

| 項目 | 内容 |
|------|------|
| ID | REQ-LLM-001 |
| 優先度 | P0 (必須) |
| モデル | GPT-4o, GPT-4o-mini, GPT-3.5-turbo |
| 受入基準 | Chat Completion, Embeddings API対応 |
| トレーサビリティ | DESIGN-LLM-001, TEST-LLM-001 |

---

#### REQ-LLM-002: Azure OpenAI対応

**EARS Pattern**: Ubiquitous

> The system SHALL support Azure OpenAI Service with deployment-based model configuration.

| 項目 | 内容 |
|------|------|
| ID | REQ-LLM-002 |
| 優先度 | P0 (必須) |
| 設定 | endpoint, api_key, deployment_name, api_version |
| 受入基準 | Azure OpenAI デプロイメント対応 |
| トレーサビリティ | DESIGN-LLM-002, TEST-LLM-002 |

---

#### REQ-LLM-003: ローカルLLM対応

**EARS Pattern**: Optional Feature

> WHERE local LLM is configured, the system SHALL support Ollama and other OpenAI-compatible APIs.

| 項目 | 内容 |
|------|------|
| ID | REQ-LLM-003 |
| 優先度 | P1 (高) |
| 対応 | Ollama, LM Studio, vLLM |
| モデル | Llama, Mistral, Gemma, Qwen |
| 受入基準 | OpenAI互換API対応 |
| トレーサビリティ | DESIGN-LLM-003, TEST-LLM-003 |

---

#### REQ-LLM-004: ストリーミングレスポンス

**EARS Pattern**: Event-driven

> WHEN streaming is enabled, the system SHALL yield response tokens incrementally.

| 項目 | 内容 |
|------|------|
| ID | REQ-LLM-004 |
| 優先度 | P1 (高) |
| 入力 | stream=True |
| 出力 | AsyncIterator[str] |
| 受入基準 | 最初のトークン < 500ms |
| トレーサビリティ | DESIGN-LLM-004, TEST-LLM-004 |

---

### 2.6 CLI/API要件

#### REQ-CLI-001: CLI初期化コマンド

**EARS Pattern**: Event-driven

> WHEN `monjyu init` is executed, the system SHALL create default configuration files and directory structure.

| 項目 | 内容 |
|------|------|
| ID | REQ-CLI-001 |
| 優先度 | P0 (必須) |
| コマンド | `monjyu init [--root PATH]` |
| 出力 | settings.yaml, prompts/, output/ |
| 受入基準 | GraphRAG互換設定構造 |
| GraphRAG対応 | `graphrag init` |
| トレーサビリティ | DESIGN-CLI-001, TEST-CLI-001 |

---

#### REQ-CLI-002: CLIインデックスコマンド

**EARS Pattern**: Event-driven

> WHEN `monjyu index` is executed, the system SHALL build index from input documents.

| 項目 | 内容 |
|------|------|
| ID | REQ-CLI-002 |
| 優先度 | P0 (必須) |
| コマンド | `monjyu index [--root PATH] [--verbose] [--lazy]` |
| 出力 | インデックスファイル (Parquet/JSON) |
| 受入基準 | 進捗表示、エラーハンドリング |
| GraphRAG対応 | `graphrag index` |
| トレーサビリティ | DESIGN-CLI-002, TEST-CLI-002 |

---

#### REQ-CLI-003: CLIクエリコマンド

**EARS Pattern**: Event-driven

> WHEN `monjyu query` is executed, the system SHALL process query and return response.

| 項目 | 内容 |
|------|------|
| ID | REQ-CLI-003 |
| 優先度 | P0 (必須) |
| コマンド | `monjyu query "question" [--method global\|local\|drift\|lazy] [--streaming]` |
| 出力 | 回答テキスト、引用、メタデータ |
| 受入基準 | 各検索モード対応 |
| GraphRAG対応 | `graphrag query` |
| トレーサビリティ | DESIGN-CLI-003, TEST-CLI-003 |

---

#### REQ-API-001: Python Facade API

**EARS Pattern**: Ubiquitous

> The system SHALL provide a unified MONJYU class as the main entry point for all functionality.

| 項目 | 内容 |
|------|------|
| ID | REQ-API-001 |
| 優先度 | P0 (必須) |
| メソッド | `MONJYU.from_config()`, `index()`, `query()`, `configure()` |
| 受入基準 | `from monjyu import MONJYU` で利用可能 |
| トレーサビリティ | DESIGN-API-001, TEST-API-001 |

---

#### REQ-API-002: 非同期API

**EARS Pattern**: Ubiquitous

> The system SHALL provide async/await support for all I/O operations.

| 項目 | 内容 |
|------|------|
| ID | REQ-API-002 |
| 優先度 | P0 (必須) |
| ステータス | ✅ 部分実装済み |
| メソッド | `async query()`, `async index()` |
| 受入基準 | asyncio対応 |
| トレーサビリティ | DESIGN-API-002, TEST-API-002 |

---

### 2.7 プロンプトチューニング要件

#### REQ-PROMPT-001: 自動プロンプトチューニング

**EARS Pattern**: Event-driven

> WHEN `monjyu tune` is executed, the system SHALL generate domain-adapted prompts from sample documents.

| 項目 | 内容 |
|------|------|
| ID | REQ-PROMPT-001 |
| 優先度 | P1 (高) |
| コマンド | `monjyu tune [--root PATH] [--domain DOMAIN]` |
| 出力 | カスタマイズされたプロンプトファイル |
| 受入基準 | ドメイン適応プロンプト生成 |
| GraphRAG対応 | Auto Tuning |
| トレーサビリティ | DESIGN-PROMPT-001, TEST-PROMPT-001 |

---

#### REQ-PROMPT-002: 手動プロンプトカスタマイズ

**EARS Pattern**: Optional Feature

> WHERE custom prompts are configured, the system SHALL use user-provided prompt templates.

| 項目 | 内容 |
|------|------|
| ID | REQ-PROMPT-002 |
| 優先度 | P2 (中) |
| 設定 | prompts/entity_extraction.txt, prompts/summarization.txt |
| 受入基準 | プロンプト上書き可能 |
| GraphRAG対応 | Manual Tuning |
| トレーサビリティ | DESIGN-PROMPT-002, TEST-PROMPT-002 |

---

### 2.8 エラーハンドリング要件

#### REQ-ERR-001: LLMエラー処理

**EARS Pattern**: Unwanted Behavior

> IF LLM API returns an error, THEN the system SHALL retry with exponential backoff and log the error.

| 項目 | 内容 |
|------|------|
| ID | REQ-ERR-001 |
| 優先度 | P0 (必須) |
| 入力 | APIエラー (rate limit, timeout, server error) |
| 出力 | リトライまたはエラーメッセージ |
| 受入基準 | 最大3回リトライ、指数バックオフ |
| トレーサビリティ | DESIGN-ERR-001, TEST-ERR-001 |

---

#### REQ-ERR-002: インデックス検証

**EARS Pattern**: Unwanted Behavior

> IF index is corrupted or incomplete, THEN the system SHALL detect and report the issue before query execution.

| 項目 | 内容 |
|------|------|
| ID | REQ-ERR-002 |
| 優先度 | P1 (高) |
| 入力 | インデックスファイル |
| 出力 | 検証結果、エラーレポート |
| 受入基準 | 破損検出、回復提案 |
| トレーサビリティ | DESIGN-ERR-002, TEST-ERR-002 |

---

## 3. 非機能要件

### 3.1 パフォーマンス

| ID | 要件 | 目標値 | EARS Pattern |
|----|------|--------|--------------|
| NFR-PERF-001 | クエリレイテンシ | 100万チャンクで < 3秒 | The system SHALL respond to queries within 3 seconds for datasets up to 1M chunks. |
| NFR-PERF-002 | インデックス作成速度 | > 1000 チャンク/秒 | The system SHALL index at least 1000 chunks per second on standard hardware. |
| NFR-PERF-003 | メモリ使用量 | 100万チャンクで < 4GB | The system SHALL use less than 4GB RAM for indexes up to 1M chunks. |
| NFR-PERF-004 | LLMコスト効率 | GraphRAGの1/100以下 | The system SHALL achieve comparable quality at less than 1/100 of GraphRAG's LLM costs. |

---

### 3.2 スケーラビリティ

| ID | 要件 | 目標値 |
|----|------|--------|
| NFR-SCALE-001 | 最大ドキュメント数 | 100万ドキュメント |
| NFR-SCALE-002 | 最大チャンク数 | 1000万チャンク |
| NFR-SCALE-003 | 最大エンティティ数 | 100万エンティティ |
| NFR-SCALE-004 | 並列処理 | マルチコア対応 |

---

### 3.3 互換性

| ID | 要件 | 対象 |
|----|------|------|
| NFR-COMPAT-001 | Python バージョン | >= 3.10 |
| NFR-COMPAT-002 | OS | Linux, macOS, Windows |
| NFR-COMPAT-003 | GraphRAG互換 | 出力フォーマット相互運用 |

---

### 3.4 セキュリティ

| ID | 要件 | 対応 |
|----|------|------|
| NFR-SEC-001 | APIキー管理 | 環境変数、設定ファイル暗号化 |
| NFR-SEC-002 | データ保護 | ローカルストレージ、転送時暗号化オプション |

---

## 4. 要件トレーサビリティマトリックス

| 要件ID | 設計 | 実装 | テスト | ステータス |
|--------|------|------|--------|-----------|
| REQ-IDX-001 | DESIGN-IDX-001 | index/loader.py | TEST-IDX-001 | 🔲 未実装 |
| REQ-IDX-002 | DESIGN-IDX-002 | index/chunker.py | TEST-IDX-002 | 🔲 未実装 |
| REQ-IDX-003 | DESIGN-IDX-003 | index/entity_extractor.py | TEST-IDX-003 | 🔲 未実装 |
| REQ-IDX-004 | DESIGN-IDX-004 | index/relationship_extractor.py | TEST-IDX-004 | 🔲 未実装 |
| REQ-IDX-005 | DESIGN-IDX-005 | index/summarizer.py | TEST-IDX-005 | 🔲 未実装 |
| REQ-IDX-006 | DESIGN-IDX-006 | index/community_detector.py | TEST-IDX-006 | 🔲 未実装 |
| REQ-IDX-007 | DESIGN-IDX-007 | index/report_generator.py | TEST-IDX-007 | 🔲 未実装 |
| REQ-IDX-008 | DESIGN-IDX-008 | index/claim_extractor.py | TEST-IDX-008 | 🔲 未実装 |
| REQ-IDX-009 | DESIGN-IDX-009 | index/nlp_extractor.py | TEST-IDX-009 | 🔲 未実装 |
| REQ-IDX-010 | DESIGN-IDX-010 | index/incremental.py | TEST-IDX-010 | 🔲 未実装 |
| REQ-IDX-011 | DESIGN-IDX-011 | index/embedder.py | TEST-IDX-011 | 🔲 未実装 |
| REQ-IDX-012 | DESIGN-IDX-012 | index/visualizer.py | TEST-IDX-012 | 🔲 未実装 |
| REQ-QRY-001 | DESIGN-QRY-001 | query/global_search.py | TEST-QRY-001 | 🔲 未実装 |
| REQ-QRY-002 | DESIGN-QRY-002 | query/local_search.py | TEST-QRY-002 | 🔲 未実装 |
| REQ-QRY-003 | DESIGN-QRY-003 | query/drift_search.py | TEST-QRY-003 | 🔲 未実装 |
| REQ-QRY-004 | DESIGN-QRY-004 | query/basic_search.py | TEST-QRY-004 | 🔲 未実装 |
| REQ-QRY-005 | DESIGN-QRY-005 | lazy_search/*.py | TEST-QRY-005 | ✅ 実装済み |
| REQ-QRY-006 | DESIGN-QRY-006 | query/question_gen.py | TEST-QRY-006 | 🔲 未実装 |
| REQ-QRY-007 | DESIGN-QRY-007 | query/multi_index.py | TEST-QRY-007 | 🔲 未実装 |
| REQ-QRY-008 | DESIGN-QRY-008 | core/conversation.py | TEST-QRY-008 | ✅ 部分実装 |
| REQ-DATA-001 | DESIGN-DATA-001 | data_model/*.py | TEST-DATA-001 | 🔲 未実装 |
| REQ-STR-001 | DESIGN-STR-001 | storage/file_storage.py | TEST-STR-001 | 🔲 未実装 |
| REQ-STR-002 | DESIGN-STR-002 | storage/vector_store/*.py | TEST-STR-002 | 🔲 未実装 |
| REQ-STR-003 | DESIGN-STR-003 | storage/cache.py | TEST-STR-003 | 🔲 未実装 |
| REQ-LLM-001 | DESIGN-LLM-001 | llm/openai_client.py | TEST-LLM-001 | ✅ 部分実装 |
| REQ-LLM-002 | DESIGN-LLM-002 | llm/azure_client.py | TEST-LLM-002 | 🔲 未実装 |
| REQ-LLM-003 | DESIGN-LLM-003 | llm/ollama_client.py | TEST-LLM-003 | 🔲 未実装 |
| REQ-LLM-004 | DESIGN-LLM-004 | llm/*.py (streaming) | TEST-LLM-004 | 🔲 未実装 |
| REQ-CLI-001 | DESIGN-CLI-001 | cli/init_cmd.py | TEST-CLI-001 | 🔲 未実装 |
| REQ-CLI-002 | DESIGN-CLI-002 | cli/index_cmd.py | TEST-CLI-002 | 🔲 未実装 |
| REQ-CLI-003 | DESIGN-CLI-003 | cli/query_cmd.py | TEST-CLI-003 | 🔲 未実装 |
| REQ-API-001 | DESIGN-API-001 | facade.py | TEST-API-001 | 🔲 未実装 |
| REQ-API-002 | DESIGN-API-002 | async API | TEST-API-002 | ✅ 部分実装 |
| REQ-PROMPT-001 | DESIGN-PROMPT-001 | prompt_tune/*.py | TEST-PROMPT-001 | 🔲 未実装 |
| REQ-PROMPT-002 | DESIGN-PROMPT-002 | prompts/*.py | TEST-PROMPT-002 | ✅ 部分実装 |
| REQ-ERR-001 | DESIGN-ERR-001 | core/retry.py | TEST-ERR-001 | 🔲 未実装 |
| REQ-ERR-002 | DESIGN-ERR-002 | index/validator.py | TEST-ERR-002 | 🔲 未実装 |

---

## 5. 優先度サマリー

### Phase 1 (MVP) - P0必須

1. インデックス基盤: REQ-IDX-001〜007, REQ-IDX-011
2. クエリ機能: REQ-QRY-001〜003, REQ-QRY-005
3. CLI/API: REQ-CLI-001〜003, REQ-API-001〜002
4. LLM対応: REQ-LLM-001〜002
5. ストレージ: REQ-STR-001, REQ-DATA-001
6. エラー処理: REQ-ERR-001

### Phase 2 - P1高優先度

1. ローカルLLM: REQ-LLM-003〜004
2. NLP軽量インデックス: REQ-IDX-009
3. ベクトルストア: REQ-STR-002
4. Basic Search: REQ-QRY-004
5. プロンプトチューニング: REQ-PROMPT-001

### Phase 3 - P2オプション

1. クレーム抽出: REQ-IDX-008
2. インクリメンタル更新: REQ-IDX-010
3. 可視化: REQ-IDX-012
4. キャッシュ: REQ-STR-003
5. 質問生成: REQ-QRY-006
6. マルチインデックス: REQ-QRY-007

---

## 6. 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| プロダクトオーナー | | | |
| テックリード | | | |
| QAリード | | | |

---

**Powered by MUSUBI** - Constitutional Article IV (EARS Format) Compliant
