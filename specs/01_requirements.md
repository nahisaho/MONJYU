# MONJYU 要件定義書

**文書番号**: MONJYU-REQ-001  
**バージョン**: 1.0.0  
**作成日**: 2025-12-24  
**ステータス**: Draft

---

## 1. 概要

### 1.1 プロジェクト名
**MONJYU** (文殊) - LazyGraphRAG Standalone Implementation

### 1.2 プロジェクトビジョン
「三人寄れば文殊の知恵」- 少ないリソースで大きな知恵を生み出す。
LazyGraphRAGの完全独立実装として、GraphRAGと同等以上の品質を約1/100のコストで実現する。

### 1.3 スコープ
MONJYUは以下の機能を**単体で**提供する独立パッケージとする：

1. **インデックス作成** - ドキュメントからの軽量インデックス構築
2. **クエリ処理** - 予算制御付きLazy検索
3. **API/CLI** - プログラマティック・コマンドラインアクセス
4. **ストレージ** - インデックスの永続化

---

## 2. 機能要件

### 2.1 インデックス作成機能 (Index)

#### REQ-IDX-000: ドキュメント前処理 (Document Preprocessing)

| 項目 | 内容 |
|------|------|
| **要件** | 各種ファイルフォーマットをテキストに変換し構造化できること |
| **優先度** | 必須 (P0) |
| **参照** | PubSec-Info-Assistant (FileLayoutParsingOther) |

##### 対応ファイルフォーマット

| フォーマット | 処理方式 | ライブラリ |
|-------------|---------|-----------|
| `.txt`, `.json` | テキスト抽出 | unstructured.partition.text |
| `.md` | Markdown解析 | unstructured.partition.md |
| `.html`, `.htm` | HTML解析 | unstructured.partition.html |
| `.csv` | CSV解析 | unstructured.partition.csv |
| `.xml` | XML解析 | unstructured.partition.xml |
| `.docx` | Word解析 | unstructured.partition.docx |
| `.doc` | Word (旧形式) | unstructured.partition.doc |
| `.pptx` | PowerPoint解析 | unstructured.partition.pptx |
| `.ppt` | PowerPoint (旧形式) | unstructured.partition.ppt |
| `.xlsx` | Excel解析 | unstructured.partition.xlsx |
| `.eml`, `.msg` | メール解析 | unstructured.partition.email/msg |
| `.pdf` | PDF解析 | Azure Form Recognizer / unstructured |

##### PDF処理戦略 (REQ-IDX-000-PDF)

| 方式 | 説明 | 用途 |
|------|------|------|
| **Azure Form Recognizer** | Azure AI Document Intelligenceによる高精度解析 | エンタープライズ、複雑なレイアウト |
| **unstructured** | ローカル解析（PyMuPDF/pdfminer） | コスト重視、シンプルなPDF |
| **OCR** | 画像ベースPDFの文字認識 | スキャン文書 |

**Azure Form Recognizer設定**:
```yaml
pdf_processing:
  provider: azure_form_recognizer  # azure_form_recognizer / unstructured
  model: prebuilt-layout           # prebuilt-layout / prebuilt-document
  api_version: "2024-02-29-preview"
  features:
    - tables                       # テーブル抽出
    - figures                      # 図表抽出
    - key_value_pairs              # キーバリューペア
```

##### 前処理パイプライン (REQ-IDX-000-PIPELINE)

```
┌──────────────────────────────────────────────────────────────────┐
│                    Document Preprocessing Pipeline                │
├──────────────────────────────────────────────────────────────────┤
│  1. ファイル検出     │ 拡張子判定、MIME タイプ検出              │
│  2. パーティション   │ unstructured でドキュメント要素抽出      │
│  3. 要素分類        │ Title, SectionHeading, Text, Table等     │
│  4. メタデータ抽出   │ ページ番号、セクション名、タイトル        │
│  5. テーブル変換     │ HTML形式に変換（構造保持）               │
│  6. チャンキング     │ chunk_by_title でセマンティック分割      │
│  7. 出力            │ Chunk (id, text, metadata, page_list)    │
└──────────────────────────────────────────────────────────────────┘
```

##### 出力データ構造

```python
@dataclass
class ProcessedDocument:
    """前処理済みドキュメント"""
    file_name: str
    file_uri: str
    file_type: str
    elements: list[Element]         # unstructured Elements
    metadata: DocumentMetadata
    
@dataclass
class Element:
    """ドキュメント要素"""
    category: str                   # Title, SectionHeading, Text, Table, etc.
    text: str
    metadata: ElementMetadata
    
@dataclass  
class ElementMetadata:
    page_number: int | None
    section: str | None
    title: str | None
    text_as_html: str | None        # テーブルの場合
```

##### unstructured チャンキング設定

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `new_after_n_chars` | 2000 | この文字数後に新チャンク開始 |
| `combine_under_n_chars` | 1000 | この文字数以下は結合 |
| `max_characters` | 2750 | チャンク最大文字数 |
| `multipage_sections` | True | 複数ページにまたがるセクション許可 |

**使用例**:
```python
from unstructured.chunking.title import chunk_by_title

chunks = chunk_by_title(
    elements,
    multipage_sections=True,
    new_after_n_chars=2000,
    combine_text_under_n_chars=1000,
    max_characters=2750
)
```

##### メール固有メタデータ (REQ-IDX-000-EMAIL)

| フィールド | 説明 |
|-----------|------|
| `subject` | 件名 |
| `sent_from` | 送信者 |
| `sent_to` | 受信者リスト |
| `date` | 送信日時 |

---

#### REQ-IDX-001: テキストチャンク分割
| 項目 | 内容 |
|------|------|
| **要件** | ドキュメントを文脈を保持しながら分割できること |
| **優先度** | 必須 (P0) |
| **入力** | テキストファイル、PDF、Markdown、HTML |
| **出力** | チャンクリスト (id, text, metadata, embedding) |
| **分割戦略** | セマンティックチャンキング（下記参照） |
| **受入基準** | 同一トピックの文が同じチャンクに80%以上収まること |

##### チャンク分割戦略

| 戦略 | 説明 | 用途 |
|------|------|------|
| `fixed` | 固定トークン数で分割 | 高速処理、単純なテキスト |
| `sentence` | 文単位で分割（句読点区切り） | 日本語基本対応 |
| `paragraph` | 段落・見出し単位で分割 | 構造化文書 |
| `semantic` | 埋め込み類似度で境界検出 | **推奨**: 高精度検索 |
| `recursive` | 階層的に分割（大→小） | 長文書 |

##### 日本語セマンティックチャンキング (REQ-IDX-001-JA)

| 項目 | 内容 |
|------|------|
| **要件** | 日本語テキストを文脈を保持しながらセマンティックに分割できること |
| **優先度** | 必須 (P0) |
| **処理フロー** | 1. 文分割 → 2. 埋め込み計算 → 3. 境界検出 → 4. チャンク生成 |

**文分割ルール（日本語）**:
- 句点（。）、感嘆符（！）、疑問符（？）で分割
- 改行（\n\n）で段落区切り
- 括弧内は分割しない
- 最小文長: 10文字以上

**境界検出アルゴリズム**:
```
1. 各文の埋め込みベクトルを計算
2. 隣接文間のコサイン類似度を計算
3. 類似度が閾値（デフォルト: 0.5）以下の箇所を境界とする
4. 境界でチャンクを分割（最大トークン数制限あり）
```

**設定パラメータ**:
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `chunk_strategy` | `semantic` | 分割戦略 |
| `max_chunk_tokens` | 300 | チャンク最大トークン数 |
| `min_chunk_tokens` | 50 | チャンク最小トークン数 |
| `overlap_tokens` | 50 | オーバーラップトークン数 |
| `similarity_threshold` | 0.5 | 境界検出閾値 |
| `language` | `auto` | 言語（auto/ja/en） |

#### REQ-IDX-002: ベクトルエンベディング
| 項目 | 内容 |
|------|------|
| **要件** | チャンクのベクトル表現を生成できること |
| **優先度** | 必須 (P0) |
| **入力** | チャンクリスト |
| **出力** | ベクトルインデックス |
| **受入基準** | OpenAI/Azure OpenAI/ローカルモデル対応 |

#### REQ-IDX-003: NLPベース軽量インデックス
| 項目 | 内容 |
|------|------|
| **要件** | LLMを使わずにキーワード/エンティティを抽出できること |
| **優先度** | 高 (P1) |
| **入力** | チャンクリスト |
| **出力** | キーワードインデックス、エンティティリスト |
| **受入基準** | spaCy/NLTK等を使用、LLMコスト0 |

#### REQ-IDX-004: インクリメンタル更新
| 項目 | 内容 |
|------|------|
| **要件** | 新規ドキュメント追加時に全体再構築不要であること |
| **優先度** | 中 (P2) |
| **入力** | 新規ドキュメント |
| **出力** | 更新されたインデックス |
| **受入基準** | 追加分のみ処理、既存インデックス保持 |

### 2.2 クエリ処理機能 (Query) - 既存

#### REQ-QRY-001: LazySearch
| 項目 | 内容 |
|------|------|
| **要件** | 予算制御付きのLazy検索を実行できること |
| **優先度** | 必須 (P0) |
| **ステータス** | ✅ 実装済み |

#### REQ-QRY-002: QueryExpander
| 項目 | 内容 |
|------|------|
| **要件** | クエリを複数のサブクエリに展開できること |
| **優先度** | 必須 (P0) |
| **ステータス** | ✅ 実装済み |

#### REQ-QRY-003: RelevanceTester
| 項目 | 内容 |
|------|------|
| **要件** | チャンクの関連性を0-10でスコアリングできること |
| **優先度** | 必須 (P0) |
| **ステータス** | ✅ 実装済み |

#### REQ-QRY-004: ClaimExtractor
| 項目 | 内容 |
|------|------|
| **要件** | 関連チャンクからクレーム（主張）を抽出できること |
| **優先度** | 必須 (P0) |
| **ステータス** | ✅ 実装済み |

#### REQ-QRY-005: IterativeDeepener
| 項目 | 内容 |
|------|------|
| **要件** | 予算内で反復的に探索を深化できること |
| **優先度** | 必須 (P0) |
| **ステータス** | ✅ 実装済み |

#### REQ-QRY-006: LazyContextBuilder
| 項目 | 内容 |
|------|------|
| **要件** | トークン制限内で最適なコンテキストを構築できること |
| **優先度** | 必須 (P0) |
| **ステータス** | ✅ 実装済み |

#### REQ-QRY-007: グループベースアクセス制御 (Search Scope Restriction)

| 項目 | 内容 |
|------|------|
| **要件** | ユーザーの所属グループに基づいて検索範囲を制限できること |
| **優先度** | 必須 (P0) |
| **ステータス** | 🔲 計画中 |

##### 概要

マルチテナント環境において、ユーザーごとに検索可能なドキュメント範囲を制御する。
グループ（部署、チーム、プロジェクト）単位でのアクセス権限を管理。

##### データモデル

**チャンクメタデータ拡張**:
```python
@dataclass
class ChunkAccessMetadata:
    """チャンクのアクセス制御メタデータ"""
    group_ids: list[str]           # アクセス可能なグループIDリスト
    source: str                    # ドキュメントソース (file path, URL, etc.)
    collection_id: str | None      # コレクション/フォルダID
    created_at: datetime | None    # 作成日時
    owner_id: str | None           # 所有者ID
```

**text_chunks DataFrame 拡張スキーマ**:
| 列名 | 型 | 説明 |
|------|------|------|
| `id` | str | チャンク一意識別子 |
| `text` | str | チャンク本文 |
| `group_ids` | list[str] | アクセス可能グループIDリスト |
| `source` | str | ソースドキュメントパス |
| `collection_id` | str | コレクションID |
| `created_at` | datetime | 作成日時 |

##### フィルタリング方式

**SearchFilter データクラス**:
```python
@dataclass
class SearchFilter:
    """検索フィルタ設定"""
    user_groups: list[str] | None = None    # ユーザーの所属グループ
    source_include: list[str] | None = None # 含めるソースパターン (glob)
    source_exclude: list[str] | None = None # 除外するソースパターン (glob)
    collection_ids: list[str] | None = None # 対象コレクションID
    date_from: datetime | None = None       # 期間開始
    date_to: datetime | None = None         # 期間終了
```

##### 使用例

```python
# 方法1: search() メソッドにフィルタを渡す
filter = SearchFilter(
    user_groups=["engineering", "project-alpha"],
    source_include=["docs/internal/*"],
    source_exclude=["docs/public/*"],
)
result = await search.search(
    query="プロジェクトの進捗は？",
    search_filter=filter,
)

# 方法2: LazySearchConfig でデフォルトフィルタを設定
config = LazySearchConfig(
    default_filter=SearchFilter(user_groups=["engineering"]),
    budget_name="Z500",
)
```

##### フィルタ適用フロー

```
┌─────────────────────────────────────────────────────────────┐
│                    Search Filter Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│  1. フィルタ取得   │ search_filter または default_filter    │
│  2. グループ照合   │ chunk.group_ids ∩ filter.user_groups  │
│  3. ソースパターン │ glob マッチング (include/exclude)      │
│  4. コレクション   │ chunk.collection_id in collection_ids │
│  5. 日付範囲      │ date_from <= created_at <= date_to    │
│  6. 結果          │ フィルタ済み text_chunks DataFrame     │
└─────────────────────────────────────────────────────────────┘
```

##### 受入基準

- [ ] `SearchFilter` でグループIDリストを指定して検索範囲を制限できる
- [ ] グループに属さないチャンクは検索結果に含まれない
- [ ] ソースパターン (glob) で包含/除外指定ができる
- [ ] 日付範囲でフィルタリングできる
- [ ] フィルタなしの場合は全チャンクが対象となる
- [ ] フィルタ適用後もLazySearchの精度が維持される

##### グラフ構造へのフィルタ伝播 (REQ-QRY-007-GRAPH)

LazyGraphRAGの反復深化はコミュニティ階層を探索するため、`text_chunks` のフィルタリングに加えて関連グラフ構造も制限する必要がある。

**フィルタ伝播ルール**:
```
1. text_chunks フィルタリング
   ↓
2. フィルタ済みチャンクに関連する noun_graph_nodes を特定
   ↓
3. 関連ノード間の noun_graph_edges のみを保持
   ↓
4. フィルタ済みグラフでコミュニティ-チャンクマッピングを再計算
```

**実装オプション**:
| 方式 | 説明 | トレードオフ |
|------|------|-------------|
| **事前フィルタ** | グラフ構造を事前にフィルタして渡す | 呼び出し側の責任、柔軟性高 |
| **動的フィルタ** | 検索時に動的にグラフを制限 | 自動化、オーバーヘッドあり |
| **キャッシュ** | グループごとにフィルタ済みグラフをキャッシュ | 高速、メモリ使用増 |

##### 最小データ量保証 (REQ-QRY-007-MIN)

フィルタ適用後のデータ量が不十分な場合、検索品質が著しく低下する可能性がある。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `min_chunks_after_filter` | 10 | フィルタ後の最小チャンク数 |
| `min_chunks_warning` | 50 | 警告を出す閾値 |

**エラー処理**:
```python
class InsufficientDataError(Exception):
    """フィルタ後のデータ量が不十分"""
    pass

# フィルタ後チャンク数 < min_chunks_after_filter の場合
# → InsufficientDataError を発生

# フィルタ後チャンク数 < min_chunks_warning の場合
# → 警告ログを出力、結果に warning フラグを設定
```

##### ベクトル検索との統合 (REQ-QRY-007-VEC)

ベクトル類似度検索を使用する場合、フィルタはベクトルインデックス検索**後**に適用する（post-filtering）か、検索**前**にインデックスをサブセット化する（pre-filtering）かを選択できる。

| 方式 | 説明 | ユースケース |
|------|------|-------------|
| **Post-filtering** | 全インデックス検索後にフィルタ | 小規模フィルタ、簡易実装 |
| **Pre-filtering** | フィルタ済みサブセットで検索 | 大規模フィルタ、厳密な制御 |
| **Hybrid** | メタデータフィルタ付きベクトル検索 | ベクトルDB機能に依存 |

**推奨**: LanceDB/ChromaDB等のメタデータフィルタ機能を活用した **Hybrid** 方式

### 2.3 API/CLI機能

#### REQ-API-001: Python API
| 項目 | 内容 |
|------|------|
| **要件** | Pythonから全機能にアクセスできること |
| **優先度** | 必須 (P0) |
| **受入基準** | `from MONJYU import ...` で利用可能 |

#### REQ-API-002: CLI (Command Line Interface)
| 項目 | 内容 |
|------|------|
| **要件** | コマンドラインから主要機能を実行できること |
| **優先度** | 必須 (P0) |
| **受入基準** | `monjyu index`, `monjyu query` コマンド |

#### REQ-API-003: 非同期サポート
| 項目 | 内容 |
|------|------|
| **要件** | async/awaitによる非同期処理をサポートすること |
| **優先度** | 必須 (P0) |
| **ステータス** | ✅ 実装済み (クエリ部分) |

#### REQ-API-004: ストリーミングレスポンス
| 項目 | 内容 |
|------|------|
| **要件** | LLM応答のストリーミング出力をサポートすること |
| **優先度** | 高 (P1) |
| **ステータス** | 🔲 未実装 |

### 2.4 ストレージ機能

#### REQ-STR-001: ファイルベースストレージ
| 項目 | 内容 |
|------|------|
| **要件** | インデックスをファイルシステムに保存/読込できること |
| **優先度** | 必須 (P0) |
| **フォーマット** | Parquet, JSON |

#### REQ-STR-002: ベクトルストア統合
| 項目 | 内容 |
|------|------|
| **要件** | 外部ベクトルストアと連携できること |
| **優先度** | 高 (P1) |
| **対応** | FAISS, Chroma, LanceDB |

#### REQ-STR-003: キャッシュ
| 項目 | 内容 |
|------|------|
| **要件** | LLM呼び出し結果をキャッシュできること |
| **優先度** | 中 (P2) |
| **目的** | コスト削減、レイテンシ改善 |

### 2.5 LLMプロバイダー

#### REQ-LLM-001: OpenAI対応
| 項目 | 内容 |
|------|------|
| **要件** | OpenAI APIを利用できること |
| **優先度** | 必須 (P0) |
| **モデル** | GPT-4o, GPT-4o-mini, GPT-3.5-turbo |

#### REQ-LLM-002: Azure OpenAI対応
| 項目 | 内容 |
|------|------|
| **要件** | Azure OpenAI Serviceを利用できること |
| **優先度** | 必須 (P0) |

#### REQ-LLM-003: ローカルLLM対応
| 項目 | 内容 |
|------|------|
| **要件** | Ollama等のローカルLLMを利用できること |
| **優先度** | 高 (P1) |
| **モデル** | Llama, Mistral, Gemma等 |

### 2.6 Progressive GraphRAG (段階的GraphRAG)

#### REQ-PRG-001: Progressive GraphRAG アーキテクチャ

| 項目 | 内容 |
|------|------|
| **要件** | 単一インデックスを段階的に深化させ、コスト予算に応じた検索を提供すること |
| **優先度** | 中 (P2) |
| **ステータス** | 🔲 計画中 |
| **参照** | Microsoft Research "GraphRAG Index + LazyGraphRAG Search" を発展 |

##### 概要

**核心思想**: 単一のインデックスを**段階的に深化**させ、クエリの要求レベルに応じて**必要な深さまで動的に処理**する。GraphRAGとLazyGraphRAGを別々に管理するのではなく、1つのインデックスに統合。

##### 設計原則

| 原則 | 説明 |
|------|------|
| **単一インデックス** | GraphRAG/Lazyの二重管理を排除 |
| **段階的深化** | Level 0から必要に応じて深化 |
| **コスト予算制御** | 事前にコストを指定して自動最適化 |
| **適応的構築** | 使用パターンに応じてインデックスを成長 |

##### アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                   Progressive GraphRAG                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Level 0: Raw Text Chunks + Embeddings    [Always available]  │
│       ↓ [On-demand, NLP-based, ~0 cost]                        │
│   Level 1: Noun Graph + Communities        [Lazy Index]        │
│       ↓ [On-demand, LLM-based, low cost]                       │
│   Level 2: Entity Extraction               [Partial GraphRAG]  │
│       ↓ [On-demand, LLM-based, medium cost]                    │
│   Level 3: Community Summaries             [Full GraphRAG]     │
│       ↓ [Background, LLM-based, high cost]                     │
│   Level 4: Pre-extracted Claims            [Enhanced]          │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Query → Cost Budget → Depth Selection → Search          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### REQ-PRG-002: 段階的インデックス (Progressive Index)

| 項目 | 内容 |
|------|------|
| **要件** | 5段階の深度を持つ単一インデックスを管理できること |
| **優先度** | 中 (P2) |
| **ステータス** | 🔲 計画中 |

##### インデックスレベル定義

| Level | 名称 | 内容 | 構築コスト | 検索品質 |
|-------|------|------|-----------|---------|
| **0** | Raw | テキストチャンク + 埋め込み | 低（埋め込みのみ） | 基本 |
| **1** | Lazy | 名詞句グラフ + コミュニティ | 低（NLPのみ） | 良好 |
| **2** | Partial | エンティティ + 関係性 | 中（LLM使用） | 高 |
| **3** | Full | コミュニティサマリー | 高（LLM使用） | 非常に高 |
| **4** | Enhanced | 事前抽出クレーム | 最高（LLM使用） | 最高 |

##### データモデル

```python
@dataclass
class ProgressiveIndex:
    """段階的に深化するインデックス"""
    
    # Level 0: 基盤（必須）
    text_chunks: pd.DataFrame           # id, text, metadata
    embeddings: np.ndarray              # ベクトル埋め込み
    
    # Level 1: 軽量グラフ（NLPベース）
    noun_graph: nx.Graph | None = None
    communities: dict[int, list[str]] | None = None
    
    # Level 2: エンティティ（LLMベース）
    entities: pd.DataFrame | None = None
    relationships: pd.DataFrame | None = None
    
    # Level 3: サマリー（LLMベース）
    community_summaries: dict[int, str] | None = None
    
    # Level 4: 事前抽出（LLMベース）
    pre_extracted_claims: pd.DataFrame | None = None
    
    @property
    def depth(self) -> int:
        """現在のインデックス深度を返す"""
        if self.pre_extracted_claims is not None:
            return 4
        elif self.community_summaries is not None:
            return 3
        elif self.entities is not None:
            return 2
        elif self.noun_graph is not None:
            return 1
        return 0
```

#### REQ-PRG-003: コスト予算ベース検索

| 項目 | 内容 |
|------|------|
| **要件** | コスト予算を指定して検索を実行できること |
| **優先度** | 中 (P2) |
| **ステータス** | 🔲 計画中 |

##### コスト予算レベル

| 予算 | 使用レベル | 推定コスト | ユースケース |
|------|-----------|-----------|-------------|
| `MINIMAL` | Level 0-1 | ~$0 | 探索的、ワンオフ |
| `STANDARD` | Level 0-2 | 低 | 一般的なクエリ |
| `PREMIUM` | Level 0-3 | 中 | 高品質が必要 |
| `UNLIMITED` | Level 0-4 | 高 | 最高品質 |

##### 検索アルゴリズム

```python
class ProgressiveSearch:
    """コスト予算に応じた段階的検索"""
    
    async def search(
        self,
        query: str,
        budget: CostBudget = CostBudget.STANDARD,
        min_quality: float = 0.7,
    ) -> ProgressiveSearchResult:
        """
        コスト予算内で最高品質の検索結果を返す。
        品質が不十分な場合、予算内で段階的に深化。
        """
        
        # 1. 最低レベルから開始
        result = await self._search_at_level(query, level=0)
        
        # 2. 品質が不十分なら段階的に深化
        while result.quality_score < min_quality:
            next_level = result.current_level + 1
            
            # 予算チェック
            if not self._can_use_level(next_level, budget):
                break
            
            # インデックス未構築なら構築
            if self.index.depth < next_level:
                await self._build_level(next_level)
            
            # 深いレベルで再検索
            result = await self._search_at_level(query, level=next_level)
        
        return result
```

#### REQ-PRG-004: 動的インデックス構築

| 項目 | 内容 |
|------|------|
| **要件** | 使用パターンに応じてインデックスを動的に構築できること |
| **優先度** | 低 (P3) |
| **ステータス** | 🔲 計画中 |

##### 構築スコープ

| スコープ | 説明 | コスト | ユースケース |
|---------|------|-------|-------------|
| `FULL` | 全データに対して構築 | 高 | バッチ処理、夜間 |
| `RELEVANT_ONLY` | 関連チャンクのみ | 中 | オンデマンド |
| `HOT_TOPICS` | 頻出トピック周辺 | 低 | 適応的最適化 |

```python
class AdaptiveIndexBuilder:
    """使用パターンに応じたインデックス構築"""
    
    async def build_on_demand(
        self,
        index: ProgressiveIndex,
        target_level: int,
        scope: IndexScope = IndexScope.RELEVANT_ONLY,
    ):
        """必要な部分のみインデックスを構築"""
        
        if scope == IndexScope.FULL:
            await self._build_full(index, target_level)
        
        elif scope == IndexScope.RELEVANT_ONLY:
            relevant_chunks = self._get_recently_accessed_chunks()
            await self._build_partial(index, target_level, relevant_chunks)
        
        elif scope == IndexScope.HOT_TOPICS:
            hot_topics = self._analyze_query_patterns()
            await self._build_around_topics(index, target_level, hot_topics)
```

#### REQ-PRG-005: ストリーミングデータ対応

| 項目 | 内容 |
|------|------|
| **要件** | 新規データを即座にLevel 0として追加し、徐々に深化できること |
| **優先度** | 低 (P3) |
| **ステータス** | 🔲 計画中 |

##### ストリーミング処理フロー

```
新規ドキュメント
    ↓
Level 0 追加（即座）  ← 検索可能になる
    ↓
Level 1 構築（バックグラウンド、NLP）
    ↓
Level 2+ 構築（必要に応じて、LLM）
```

##### 使用例

```python
from MONJYU import ProgressiveGraphRAG, CostBudget

# 初期化（Level 0 のみ必須）
rag = ProgressiveGraphRAG.from_documents(documents)

# コスト重視：最小限のLLM使用
result = await rag.search(
    "プロジェクトの概要は？",
    budget=CostBudget.MINIMAL,
)

# 品質重視：必要に応じて深化
result = await rag.search(
    "詳細な技術要件を教えて",
    budget=CostBudget.PREMIUM,
    min_quality=0.9,
)

# 新規データ追加（即座にLevel 0として検索可能）
await rag.add_documents(new_documents)

# バックグラウンドで事前にインデックス深化
await rag.pre_build(target_level=3)  # 夜間バッチ等
```

##### 実装ロードマップ

| Phase | 内容 | 優先度 | 対応要件 |
|-------|------|--------|---------|
| **Phase 1** | LazyGraphRAG単体 | P0 | ✅ 完了 |
| **Phase 2** | ProgressiveIndex (Level 0-1) | P2 | REQ-PRG-002 |
| **Phase 3** | コスト予算検索 | P2 | REQ-PRG-003 |
| **Phase 4** | 動的インデックス構築 | P3 | REQ-PRG-004 |
| **Phase 5** | Level 2-4 実装 | P3 | REQ-PRG-002 |
| **Phase 6** | ストリーミング対応 | P3 | REQ-PRG-005 |

##### 受入基準

- [ ] `ProgressiveIndex` が5段階の深度を管理できる
- [ ] `CostBudget` 指定で検索コストを事前制御できる
- [ ] 品質が不十分な場合、予算内で自動的に深化する
- [ ] 新規データがLevel 0として即座に検索可能になる
- [ ] バックグラウンドでインデックス深化を実行できる
- [ ] 使用パターンに応じた適応的インデックス構築ができる

##### 旧Unified GraphRAGとの比較

| 項目 | Unified GraphRAG (旧) | Progressive GraphRAG (新) |
|------|----------------------|--------------------------|
| インデックス | 2つ（GraphRAG + Lazy） | 1つ（段階的） |
| モード選択 | ユーザーが明示的に選択 | 自動（予算ベース） |
| コスト制御 | 事後的 | 事前（予算指定） |
| 新データ対応 | 再インデックス | Level 0として即追加 |
| スケーラビリティ | 固定 | 段階的スケール |
| 複雑さ | ルーター + マージロジック | シンプルな深化ロジック |

---

## 3. 非機能要件

### 3.1 パフォーマンス

| ID | 要件 | 目標値 |
|----|------|--------|
| NFR-PERF-001 | クエリレイテンシ | 100万チャンクで3秒以内 |
| NFR-PERF-002 | インデックス作成速度 | 1000チャンク/秒 |
| NFR-PERF-003 | メモリ使用量 | 100万チャンクで4GB以下 |

### 3.2 スケーラビリティ

| ID | 要件 | 目標値 |
|----|------|--------|
| NFR-SCALE-001 | 最大チャンク数 | 1000万チャンク |
| NFR-SCALE-002 | 並列クエリ | 100同時クエリ |
| NFR-SCALE-003 | スケール特性 | 準線形 (O(n log n)) |

### 3.3 信頼性

| ID | 要件 | 目標値 |
|----|------|--------|
| NFR-REL-001 | テストカバレッジ | 80%以上 |
| NFR-REL-002 | CI/CD | GitHub Actions |
| NFR-REL-003 | エラーハンドリング | 全APIで適切な例外処理 |

### 3.4 セキュリティ

| ID | 要件 | 内容 |
|----|------|------|
| NFR-SEC-001 | API キー管理 | 環境変数/設定ファイルで管理 |
| NFR-SEC-002 | データ暗号化 | 保存時暗号化オプション |

### 3.5 互換性

| ID | 要件 | 内容 |
|----|------|------|
| NFR-COMP-001 | Python | 3.10, 3.11, 3.12 |
| NFR-COMP-002 | OS | Linux, macOS, Windows |
| NFR-COMP-003 | GraphRAG | インデックス形式互換 (オプション) |

---

## 4. 依存関係

### 4.1 必須依存

| パッケージ | 用途 | バージョン |
|-----------|------|-----------|
| tiktoken | トークン化 | >=0.5.0 |
| pandas | データ操作 | >=2.0.0 |
| pydantic | 設定バリデーション | >=2.0.0 |
| json-repair | JSON修復 | >=0.25.0 |

### 4.2 オプション依存

| パッケージ | 用途 | バージョン |
|-----------|------|-----------|
| openai | OpenAI API | >=1.0.0 |
| faiss-cpu | ベクトル検索 | >=1.7.0 |
| spacy | NLP処理 | >=3.0.0 |
| chromadb | ベクトルストア | >=0.4.0 |

---

## 5. マイルストーン

### Phase 1: 基盤整備 (v1.1.0) - 1週間
- [ ] インデックス作成基盤
- [ ] ファイルストレージ
- [ ] CLI基本機能

### Phase 2: インデックス機能 (v1.2.0) - 2週間
- [ ] テキストチャンク分割
- [ ] ベクトルエンベディング
- [ ] NLPベースインデックス

### Phase 3: 統合・最適化 (v1.3.0) - 1週間
- [ ] ベクトルストア統合
- [ ] ストリーミングレスポンス
- [ ] パフォーマンス最適化

### Phase 4: エンタープライズ機能 (v2.0.0) - 2週間
- [ ] マルチLLMプロバイダー
- [ ] キャッシュ機能
- [ ] インクリメンタル更新

---

## 6. 用語集

| 用語 | 定義 |
|------|------|
| **チャンク** | ドキュメントを分割した単位テキスト |
| **クレーム** | チャンクから抽出された主張・事実 |
| **予算** | LLM呼び出しの上限数 |
| **Lazy評価** | 必要な時にのみ処理を実行する方式 |
| **プリセット** | 用途別の設定テンプレート (z100, z500, z1500) |

---

## 7. 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 作成者 | - | 2025-12-24 | |
| レビュアー | - | | |
| 承認者 | - | | |

---

**次のステップ**: [02_architecture.md](02_architecture.md) - アーキテクチャ設計書
