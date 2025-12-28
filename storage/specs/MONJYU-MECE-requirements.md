# MONJYU 要件定義書 (MECE版)

**文書番号**: MONJYU-REQ-MECE-001  
**バージョン**: 1.0.0  
**作成日**: 2025-12-25  
**ステータス**: Draft  
**準拠**: MUSUBI SDD Constitutional Article IV (EARS Format)

---

## 概要

本文書は、MONJYU（学術論文向けProgressive GraphRAGシステム）の要件を**MECE原則**（Mutually Exclusive, Collectively Exhaustive）に基づいて再定義したものです。

### MECE構造

```
MONJYU 要件体系 (MECE)
├── 1. 機能要件 (FR: Functional Requirements)
│   ├── 1.1 ドキュメント処理 (Document Processing)
│   │   ├── 入力 (Input)
│   │   ├── 前処理 (Preprocessing)
│   │   └── 出力 (Output)
│   │
│   ├── 1.2 インデックス構築 (Index Building)
│   │   ├── Level 0: Raw (Baseline)
│   │   ├── Level 1: Lazy (NLP-based)
│   │   ├── Level 2: Partial (Entity/Relationship)
│   │   ├── Level 3: Full (Community Reports)
│   │   └── Level 4: Enhanced (Pre-extracted Claims)
│   │
│   ├── 1.3 クエリ処理 (Query Processing)
│   │   ├── Vector Search
│   │   ├── Lazy Search
│   │   ├── Graph Search (Local/Global)
│   │   ├── Hybrid Search
│   │   └── Unified Search (Router)
│   │
│   ├── 1.4 引用ネットワーク (Citation Network)
│   │   ├── 構築 (Building)
│   │   └── 分析 (Analysis)
│   │
│   └── 1.5 外部インターフェース (External Interfaces)
│       ├── Python API
│       ├── CLI
│       ├── MCP Server
│       └── REST API
│
└── 2. 非機能要件 (NFR: Non-Functional Requirements)
    ├── 2.1 パフォーマンス (Performance)
    ├── 2.2 スケーラビリティ (Scalability)
    ├── 2.3 コスト効率 (Cost Efficiency)
    ├── 2.4 品質 (Quality)
    ├── 2.5 セキュリティ (Security)
    ├── 2.6 可用性 (Availability)
    ├── 2.7 運用性 (Operability)
    ├── 2.8 拡張性 (Extensibility)
    ├── 2.9 テスト (Testing)
    └── 2.10 ドキュメント (Documentation)
```

---

## 1. 機能要件 (Functional Requirements)

### 1.1 ドキュメント処理 (Document Processing)

ドキュメント処理は3つの相互排他的なフェーズで構成されます。

#### 1.1.1 入力 (Input)

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-DOC-INP-001 | ファイル形式検出 | Ubiquitous | システムは入力ファイルの形式（拡張子、MIMEタイプ）を検出しなければならない |
| FR-DOC-INP-002 | テキストファイル読込 | Ubiquitous | システムは .txt, .md, .json ファイルを読み込めなければならない |
| FR-DOC-INP-003 | 構造化文書読込 | Ubiquitous | システムは .html, .xml, .csv ファイルを解析・読み込めなければならない |
| FR-DOC-INP-004 | Office文書読込 | Ubiquitous | システムは .docx, .pptx, .xlsx ファイルを解析・読み込めなければならない |
| FR-DOC-INP-005 | PDF読込 | Ubiquitous | システムはPDFファイルを解析・読み込めなければならない |
| FR-DOC-INP-006 | バッチ入力 | Ubiquitous | システムはディレクトリを指定して複数ファイルを一括入力できなければならない |

**対応ファイル形式マトリクス**:

| カテゴリ | 形式 | ライブラリ | 優先度 |
|----------|------|-----------|--------|
| **テキスト** | .txt, .md, .json | unstructured.partition | P0 |
| **構造化** | .html, .xml, .csv | unstructured.partition | P0 |
| **Office** | .docx, .pptx, .xlsx | unstructured.partition | P1 |
| **PDF** | .pdf | Azure Doc Intel / unstructured | P0 |
| **メール** | .eml, .msg | unstructured.partition | P2 |

---

#### 1.1.2 前処理 (Preprocessing)

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-DOC-PRE-001 | 要素分類 | Ubiquitous | システムは文書を構成要素（Title, Section, Paragraph, Table, Figure）に分類しなければならない |
| FR-DOC-PRE-002 | メタデータ抽出 | Ubiquitous | システムはタイトル、著者、日付等のメタデータを抽出しなければならない |
| FR-DOC-PRE-003 | テーブル変換 | Ubiquitous | システムはテーブルをHTML/Markdown形式に変換し構造を保持しなければならない |
| FR-DOC-PRE-004 | 言語検出 | Ubiquitous | システムは文書の言語を自動検出しなければならない |
| FR-DOC-PRE-005 | テキストクリーニング | Ubiquitous | システムはヘッダー/フッター除去、正規化を行わなければならない |

**学術論文専用前処理**:

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-DOC-PRE-101 | IMRaD構造認識 | Conditional | 学術論文の場合、システムはIMRaD構造（Introduction/Methods/Results/Discussion）を認識しなければならない |
| FR-DOC-PRE-102 | 2カラム→1カラム変換 | Conditional | 2カラムレイアウトの場合、システムは1カラムテキストに変換しなければならない |
| FR-DOC-PRE-103 | 数式抽出 | Conditional | 数式が存在する場合、システムはLaTeX形式で抽出しなければならない |
| FR-DOC-PRE-104 | 図表キャプション抽出 | Conditional | 図表が存在する場合、システムはキャプションを抽出しなければならない |
| FR-DOC-PRE-105 | 参考文献抽出 | Conditional | References節が存在する場合、システムは構造化された参考文献リストを抽出しなければならない |
| FR-DOC-PRE-106 | 学術識別子抽出 | Ubiquitous | システムはDOI、arXiv ID、PMIDを抽出しなければならない |
| FR-DOC-PRE-107 | 著者情報抽出 | Ubiquitous | システムは著者名、所属、ORCIDを抽出しなければならない |

---

#### 1.1.3 出力 (Output)

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-DOC-OUT-001 | 標準化文書出力 | Ubiquitous | システムは処理済み文書をAcademicPaperDocument形式で出力しなければならない |
| FR-DOC-OUT-002 | チャンク分割 | Ubiquitous | システムは文書を設定可能なサイズ（デフォルト300トークン）でTextUnitに分割しなければならない |
| FR-DOC-OUT-003 | オーバーラップ設定 | Ubiquitous | システムはチャンク間オーバーラップ（デフォルト100トークン）を設定可能でなければならない |
| FR-DOC-OUT-004 | チャンクメタデータ | Ubiquitous | システムは各TextUnitに元文書ID、位置情報を付与しなければならない |

---

### 1.2 インデックス構築 (Index Building)

インデックスは5つの相互排他的なレベルで構成されます。各レベルは前のレベルに依存します。

```
Level 依存関係:
Level 0 ← Level 1 ← Level 2 ← Level 3 ← Level 4

Level 0: 全アーキテクチャの基盤
Level 1: LazyGraphRAG の基盤
Level 2-4: GraphRAG の段階的拡張
```

#### 1.2.1 Level 0: Raw (Baseline)

**目的**: ベクトル検索の基盤

| ID | 要件名 | EARS Pattern | 説明 | LLMコスト |
|----|--------|--------------|------|----------|
| FR-IDX-L0-001 | TextUnit永続化 | Ubiquitous | システムはTextUnitをParquet形式で永続化しなければならない | $0 |
| FR-IDX-L0-002 | ベクトル埋め込み生成 | Ubiquitous | システムはTextUnitのベクトル埋め込みを生成しなければならない | Embedding |
| FR-IDX-L0-003 | ベクトルインデックス構築 | Ubiquitous | システムはベクトル類似度検索用のインデックスを構築しなければならない | $0 |
| FR-IDX-L0-004 | 文書メタデータ永続化 | Ubiquitous | システムは文書メタデータをParquet形式で永続化しなければならない | $0 |

**Level 0 出力アーティファクト**:

| ファイル | 内容 |
|---------|------|
| `documents.parquet` | 文書メタデータ |
| `text_units.parquet` | チャンクテキスト |
| `embeddings.parquet` | ベクトル埋め込み |
| `vector_index/` | ベクトルDB インデックス |

---

#### 1.2.2 Level 1: Lazy (NLP-based)

**目的**: LLMコストゼロでの軽量グラフ構築（LazyGraphRAG）

| ID | 要件名 | EARS Pattern | 説明 | LLMコスト |
|----|--------|--------------|------|----------|
| FR-IDX-L1-001 | キーワード抽出 | Ubiquitous | システムはNLP技術（TF-IDF/RAKE）でキーワードを抽出しなければならない | $0 |
| FR-IDX-L1-002 | 名詞句抽出 | Ubiquitous | システムはspaCyで名詞句を抽出しなければならない | $0 |
| FR-IDX-L1-003 | 固有表現認識 | Ubiquitous | システムはspaCyでNER（固有表現認識）を実行しなければならない | $0 |
| FR-IDX-L1-004 | 名詞句グラフ構築 | Ubiquitous | システムは名詞句の共起関係からグラフを構築しなければならない | $0 |
| FR-IDX-L1-005 | コミュニティ検出 | Ubiquitous | システムはLeidenアルゴリズムでコミュニティを検出しなければならない | $0 |

**Level 1 出力アーティファクト**:

| ファイル | 内容 |
|---------|------|
| `nlp_features.parquet` | キーワード、名詞句、NER結果 |
| `noun_phrase_graph.parquet` | 名詞句共起グラフ |
| `communities_l1.parquet` | NLPベースコミュニティ |

---

#### 1.2.3 Level 2: Partial (Entity/Relationship)

**目的**: LLMによるエンティティ・関係性抽出（GraphRAG基盤）

| ID | 要件名 | EARS Pattern | 説明 | LLMコスト |
|----|--------|--------------|------|----------|
| FR-IDX-L2-001 | エンティティ抽出 | Conditional | GraphRAGモードの場合、システムはLLMでエンティティを抽出しなければならない | 💰💰💰 |
| FR-IDX-L2-002 | エンティティ分類 | Conditional | システムはエンティティを学術カテゴリ（RESEARCHER, METHOD, MODEL等）に分類しなければならない | 💰 |
| FR-IDX-L2-003 | 関係性抽出 | Conditional | システムはLLMでエンティティ間の関係性を抽出しなければならない | 💰💰💰 |
| FR-IDX-L2-004 | エンティティグラフ構築 | Conditional | システムはエンティティ・関係性からナレッジグラフを構築しなければならない | $0 |
| FR-IDX-L2-005 | エンティティ埋め込み | Conditional | システムはエンティティ説明のベクトル埋め込みを生成しなければならない | Embedding |

**学術論文向けエンティティ分類**:

| カテゴリ | タイプ | 例 |
|---------|--------|-----|
| **人物** | RESEARCHER | Geoffrey Hinton |
| **組織** | ORGANIZATION | Google DeepMind |
| **手法** | METHOD | Attention Mechanism |
| **モデル** | MODEL | GPT-4 |
| **データセット** | DATASET | ImageNet |
| **評価指標** | METRIC | F1-score |
| **タスク** | TASK | Question Answering |
| **概念** | CONCEPT | Self-Attention |
| **ツール** | TOOL | PyTorch |

**Level 2 出力アーティファクト**:

| ファイル | 内容 |
|---------|------|
| `entities.parquet` | エンティティ定義 |
| `relationships.parquet` | 関係性定義 |
| `entity_embeddings.parquet` | エンティティ埋め込み |

---

#### 1.2.4 Level 3: Full (Community Reports)

**目的**: コミュニティ単位のサマリー生成

| ID | 要件名 | EARS Pattern | 説明 | LLMコスト |
|----|--------|--------------|------|----------|
| FR-IDX-L3-001 | 階層的コミュニティ検出 | Conditional | システムはエンティティグラフから階層的コミュニティを検出しなければならない | $0 |
| FR-IDX-L3-002 | コミュニティサマリー生成 | Conditional | システムは各コミュニティのサマリーレポートをLLMで生成しなければならない | 💰💰💰💰 |
| FR-IDX-L3-003 | コミュニティ重要度算出 | Conditional | システムは各コミュニティの重要度スコアを算出しなければならない | $0 |
| FR-IDX-L3-004 | サマリー埋め込み | Conditional | システムはコミュニティサマリーのベクトル埋め込みを生成しなければならない | Embedding |

**Level 3 出力アーティファクト**:

| ファイル | 内容 |
|---------|------|
| `communities_l3.parquet` | 階層的コミュニティ |
| `community_reports.parquet` | コミュニティサマリー |
| `community_embeddings.parquet` | サマリー埋め込み |

---

#### 1.2.5 Level 4: Enhanced (Pre-extracted Claims)

**目的**: 事前クレーム抽出による最高品質

| ID | 要件名 | EARS Pattern | 説明 | LLMコスト |
|----|--------|--------------|------|----------|
| FR-IDX-L4-001 | クレーム事前抽出 | Conditional | システムは全TextUnitからクレーム（主張・事実）をLLMで事前抽出しなければならない | 💰💰💰💰💰 |
| FR-IDX-L4-002 | クレーム分類 | Conditional | システムはクレームをカテゴリ（事実/主張/評価/比較）に分類しなければならない | 💰 |
| FR-IDX-L4-003 | クレーム埋め込み | Conditional | システムはクレームのベクトル埋め込みを生成しなければならない | Embedding |
| FR-IDX-L4-004 | クレームエンティティリンク | Conditional | システムはクレームを関連エンティティにリンクしなければならない | $0 |

**Level 4 出力アーティファクト**:

| ファイル | 内容 |
|---------|------|
| `claims.parquet` | 事前抽出クレーム |
| `claim_embeddings.parquet` | クレーム埋め込み |
| `claim_entity_links.parquet` | クレーム-エンティティ関連 |

---

### 1.3 クエリ処理 (Query Processing)

クエリ処理は5つの相互排他的な検索モードで構成されます。

#### 1.3.1 Vector Search (Baseline RAG)

| ID | 要件名 | EARS Pattern | 説明 | 必要Level |
|----|--------|--------------|------|----------|
| FR-QRY-VEC-001 | クエリ埋め込み | Event-driven | クエリ受信時、システムはクエリのベクトル埋め込みを生成しなければならない | 0 |
| FR-QRY-VEC-002 | 類似度検索 | Event-driven | システムはベクトル類似度でTop-K TextUnitを取得しなければならない | 0 |
| FR-QRY-VEC-003 | コンテキスト構築 | Event-driven | システムは取得したTextUnitからプロンプトコンテキストを構築しなければならない | 0 |
| FR-QRY-VEC-004 | 回答生成 | Event-driven | システムはLLMで回答を生成しなければならない | 0 |

---

#### 1.3.2 Lazy Search (LazyGraphRAG)

| ID | 要件名 | EARS Pattern | 説明 | 必要Level |
|----|--------|--------------|------|----------|
| FR-QRY-LZY-001 | クエリ拡張 | Event-driven | システムはLLMでクエリを複数のサブクエリに拡張しなければならない | 1 |
| FR-QRY-LZY-002 | コミュニティ検索 | Event-driven | システムはNLPコミュニティから関連コミュニティを取得しなければならない | 1 |
| FR-QRY-LZY-003 | 関連性テスト | Event-driven | システムはTextUnitの関連性をLLMでテストしなければならない | 1 |
| FR-QRY-LZY-004 | クレーム抽出 | Event-driven | システムは関連TextUnitからクレームを動的抽出しなければならない | 1 |
| FR-QRY-LZY-005 | 反復深化 | Event-driven | システムは予算内で関連コンテンツを反復的に深掘りしなければならない | 1 |
| FR-QRY-LZY-006 | 予算制御 | Event-driven | システムはコンテキストトークン予算を超えないよう制御しなければならない | 1 |

**Lazy Search パラメータ**:

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `budget` | 10 | 反復深化の最大ラウンド数 |
| `context_tokens` | 8000 | コンテキストの最大トークン数 |
| `relevance_threshold` | 0.7 | 関連性判定の閾値 |

---

#### 1.3.3 Graph Search (GraphRAG)

**Local Search**:

| ID | 要件名 | EARS Pattern | 説明 | 必要Level |
|----|--------|--------------|------|----------|
| FR-QRY-GRP-L01 | エンティティ検索 | Event-driven | システムはクエリに関連するエンティティをベクトル検索で取得しなければならない | 2 |
| FR-QRY-GRP-L02 | 近傍エンティティ取得 | Event-driven | システムは取得エンティティの近傍エンティティをグラフ探索で取得しなければならない | 2 |
| FR-QRY-GRP-L03 | 関係性取得 | Event-driven | システムは関連する関係性を取得しなければならない | 2 |
| FR-QRY-GRP-L04 | ソーステキスト取得 | Event-driven | システムはエンティティに紐づくソースTextUnitを取得しなければならない | 2 |
| FR-QRY-GRP-L05 | ローカル回答生成 | Event-driven | システムは収集コンテキストから回答を生成しなければならない | 2 |

**Global Search**:

| ID | 要件名 | EARS Pattern | 説明 | 必要Level |
|----|--------|--------------|------|----------|
| FR-QRY-GRP-G01 | コミュニティレポート取得 | Event-driven | システムは指定レベルのコミュニティレポートを取得しなければならない | 3 |
| FR-QRY-GRP-G02 | Map処理 | Event-driven | システムは各コミュニティレポートに対してクエリ応答を生成しなければならない | 3 |
| FR-QRY-GRP-G03 | Reduce処理 | Event-driven | システムは個別回答を統合して最終回答を生成しなければならない | 3 |
| FR-QRY-GRP-G04 | 重要度フィルタリング | Event-driven | システムは重要度スコアで使用コミュニティをフィルタリングできなければならない | 3 |

---

#### 1.3.4 Hybrid Search

| ID | 要件名 | EARS Pattern | 説明 | 必要Level |
|----|--------|--------------|------|----------|
| FR-QRY-HYB-001 | 並列実行 | Event-driven | Hybridモード選択時、システムは複数検索エンジンを並列実行しなければならない | 1+ |
| FR-QRY-HYB-002 | RRFマージ | Event-driven | システムはReciprocal Rank Fusionで結果をマージしなければならない | 1+ |
| FR-QRY-HYB-003 | 重複排除 | Event-driven | システムはマージ時に重複コンテンツを排除しなければならない | 1+ |
| FR-QRY-HYB-004 | 統合回答生成 | Event-driven | システムはマージされたコンテキストから統合回答を生成しなければならない | 1+ |

**RRF アルゴリズム**:
```
score(d) = Σ 1/(k + rank_i(d))
k = 60 (デフォルト)
```

---

#### 1.3.5 Unified Search (Query Router)

| ID | 要件名 | EARS Pattern | 説明 | 必要Level |
|----|--------|--------------|------|----------|
| FR-QRY-UNI-001 | クエリ分類 | Event-driven | システムはクエリを分類し最適な検索モードを決定しなければならない | 1+ |
| FR-QRY-UNI-002 | モード自動選択 | Event-driven | mode=AUTO時、システムはQuery Routerの判定に従いモードを選択しなければならない | 1+ |
| FR-QRY-UNI-003 | モード手動指定 | Event-driven | mode指定時、システムは指定されたモードで検索を実行しなければならない | 0+ |
| FR-QRY-UNI-004 | 分類精度保証 | Ubiquitous | Query Routerの分類精度は85%以上でなければならない | - |

**クエリ分類基準（学術論文向け）**:

| クエリパターン | 選択モード | 理由 |
|---------------|-----------|------|
| サーベイ・傾向分析 | GRAPHRAG | 広範な文脈 |
| 手法調査 | LAZY | 探索的 |
| 手法比較 | HYBRID | 網羅性 |
| 具体的事実 | VECTOR | ピンポイント |
| 先行研究調査 | LAZY | 引用ネットワーク |
| ベンチマーク調査 | HYBRID | 表データの網羅 |

---

### 1.4 引用ネットワーク (Citation Network)

引用ネットワークは構築と分析の2つのフェーズで構成されます。

#### 1.4.1 構築 (Building)

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-CIT-BLD-001 | インライン引用検出 | Ubiquitous | システムは本文中の引用（[1], (Smith et al., 2020)等）を検出しなければならない |
| FR-CIT-BLD-002 | 参考文献パース | Ubiquitous | システムはReferences節から構造化参考文献を抽出しなければならない |
| FR-CIT-BLD-003 | DOIマッチング | Ubiquitous | システムは抽出した参考文献をDOIで論文にマッチングしなければならない |
| FR-CIT-BLD-004 | 引用グラフ構築 | Ubiquitous | システムは論文間の引用関係をグラフとして構築しなければならない |
| FR-CIT-BLD-005 | 外部API連携 | Optional | 外部API有効時、システムはSemantic Scholar等から引用情報を取得できなければならない |

**引用エッジ種別**:

| エッジ種別 | 説明 |
|---------|------|
| `cites` | AがBを引用 |
| `cited_by` | AがBに引用される |
| `co_citation` | AとBが同じ論文に引用される |
| `bibliographic_coupling` | AとBが同じ論文を引用 |

---

#### 1.4.2 分析 (Analysis)

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-CIT-ANL-001 | 被引用数取得 | Event-driven | システムは論文の被引用数を取得できなければならない |
| FR-CIT-ANL-002 | 引用先取得 | Event-driven | システムは論文が引用している論文リストを取得できなければならない |
| FR-CIT-ANL-003 | 被引用論文取得 | Event-driven | システムは論文を引用している論文リストを取得できなければならない |
| FR-CIT-ANL-004 | 共引用分析 | Event-driven | システムはCo-citation関係にある論文ペアを取得できなければならない |
| FR-CIT-ANL-005 | 書誌結合分析 | Event-driven | システムはBibliographic coupling関係にある論文ペアを取得できなければならない |
| FR-CIT-ANL-006 | 引用ネットワーク可視化 | Optional | システムは引用ネットワークを可視化できなければならない |

---

### 1.5 外部インターフェース (External Interfaces)

外部インターフェースは4つの相互排他的な種類で構成されます。

#### 1.5.1 Python API

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-API-PY-001 | ファサードAPI | Ubiquitous | システムはMONJYUファサードクラスで統一APIを提供しなければならない |
| FR-API-PY-002 | 非同期対応 | Ubiquitous | 全APIメソッドはasync/awaitに対応しなければならない |
| FR-API-PY-003 | 型ヒント | Ubiquitous | 全APIメソッドは型ヒントを持たなければならない |
| FR-API-PY-004 | インデックス操作 | Ubiquitous | システムはインデックス作成・更新・削除APIを提供しなければならない |
| FR-API-PY-005 | 検索操作 | Ubiquitous | システムは検索API（search, search_papers, get_paper）を提供しなければならない |
| FR-API-PY-006 | 引用ネットワーク操作 | Ubiquitous | システムは引用ネットワークAPI（get_citations, get_cited_by）を提供しなければならない |
| FR-API-PY-007 | ストリーミング | Optional | ストリーミング有効時、システムは応答をAsyncGeneratorで返さなければならない |

---

#### 1.5.2 CLI (Command Line Interface)

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-API-CLI-001 | indexコマンド | Ubiquitous | システムは`monjyu index`コマンドでインデックス構築を実行できなければならない |
| FR-API-CLI-002 | searchコマンド | Ubiquitous | システムは`monjyu search`コマンドで検索を実行できなければならない |
| FR-API-CLI-003 | serveコマンド | Ubiquitous | システムは`monjyu serve`コマンドでMCPサーバーを起動できなければならない |
| FR-API-CLI-004 | upgradeコマンド | Ubiquitous | システムは`monjyu upgrade`コマンドでインデックスレベルをアップグレードできなければならない |
| FR-API-CLI-005 | configコマンド | Ubiquitous | システムは`monjyu config`コマンドで設定を表示・変更できなければならない |
| FR-API-CLI-006 | 進捗表示 | Ubiquitous | 長時間処理時、システムは進捗バーを表示しなければならない |

**CLI コマンド一覧**:

| コマンド | 説明 | 例 |
|---------|------|-----|
| `monjyu index` | インデックス構築 | `monjyu index --input ./papers --level 1` |
| `monjyu search` | 検索実行 | `monjyu search "Transformerの動向"` |
| `monjyu serve` | MCPサーバー起動 | `monjyu serve --port 8000` |
| `monjyu upgrade` | レベルアップグレード | `monjyu upgrade --to-level 2` |
| `monjyu config` | 設定管理 | `monjyu config show` |

---

#### 1.5.3 MCP Server

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-API-MCP-001 | MCP準拠 | Ubiquitous | システムはModel Context Protocol仕様に準拠しなければならない |
| FR-API-MCP-002 | stdio転送 | Ubiquitous | システムはstdio転送をサポートしなければならない |
| FR-API-MCP-003 | SSE転送 | Optional | システムはSSE（Server-Sent Events）転送をサポートできなければならない |
| FR-API-MCP-004 | monjyu_searchツール | Ubiquitous | システムはmonjyu_search MCPツールを提供しなければならない |
| FR-API-MCP-005 | monjyu_indexツール | Ubiquitous | システムはmonjyu_index MCPツールを提供しなければならない |
| FR-API-MCP-006 | monjyu_get_paperツール | Ubiquitous | システムはmonjyu_get_paper MCPツールを提供しなければならない |
| FR-API-MCP-007 | monjyu_citationsツール | Ubiquitous | システムはmonjyu_citations MCPツールを提供しなければならない |
| FR-API-MCP-008 | monjyu_summarizeツール | Ubiquitous | システムはmonjyu_summarize MCPツールを提供しなければならない |
| FR-API-MCP-009 | monjyu_compareツール | Ubiquitous | システムはmonjyu_compare MCPツールを提供しなければならない |
| FR-API-MCP-010 | monjyu_surveyツール | Ubiquitous | システムはmonjyu_survey MCPツールを提供しなければならない |

**MCP ツール一覧**:

| ツール名 | 説明 | 主要パラメータ |
|---------|------|---------------|
| `monjyu_search` | 論文検索 | query, mode, budget, top_k |
| `monjyu_index` | インデックス構築 | input_path, level |
| `monjyu_get_paper` | 論文詳細取得 | doi, arxiv_id |
| `monjyu_citations` | 引用ネットワーク | paper_id, depth |
| `monjyu_summarize` | サマリー生成 | query, scope |
| `monjyu_compare` | 手法比較 | methods[], criteria[] |
| `monjyu_survey` | 文献サーベイ | topic, year_range |

---

#### 1.5.4 REST API

| ID | 要件名 | EARS Pattern | 説明 |
|----|--------|--------------|------|
| FR-API-RST-001 | OpenAPI準拠 | Ubiquitous | システムはOpenAPI 3.0仕様に準拠したREST APIを提供しなければならない |
| FR-API-RST-002 | 検索エンドポイント | Ubiquitous | システムは`POST /search`エンドポイントを提供しなければならない |
| FR-API-RST-003 | インデックスエンドポイント | Ubiquitous | システムは`POST /index`エンドポイントを提供しなければならない |
| FR-API-RST-004 | 論文エンドポイント | Ubiquitous | システムは`GET /papers/{id}`エンドポイントを提供しなければならない |
| FR-API-RST-005 | 引用エンドポイント | Ubiquitous | システムは`GET /citations/{paper_id}`エンドポイントを提供しなければならない |
| FR-API-RST-006 | ヘルスチェック | Ubiquitous | システムは`GET /health`エンドポイントを提供しなければならない |

---

## 2. 非機能要件 (Non-Functional Requirements)

### 2.1 パフォーマンス (Performance)

| ID | 要件名 | 基準値 | 測定方法 |
|----|--------|--------|---------|
| NFR-PERF-001 | Vector Search レイテンシ | < 500ms (p95) | 負荷テスト |
| NFR-PERF-002 | Lazy Search レイテンシ | < 5s (p95) | 負荷テスト |
| NFR-PERF-003 | Graph Search レイテンシ | < 10s (p95) | 負荷テスト |
| NFR-PERF-004 | Hybrid Search レイテンシ | < 10s (p95) | 負荷テスト |
| NFR-PERF-005 | インデックス構築スループット | > 100 docs/min | ベンチマーク |
| NFR-PERF-006 | 同時クエリ処理数 | > 100 concurrent | 負荷テスト |
| NFR-PERF-007 | メモリ使用量 | < 8GB (ローカル) | プロファイリング |

---

### 2.2 スケーラビリティ (Scalability)

| ID | 要件名 | 基準値 | 環境 |
|----|--------|--------|------|
| NFR-SCAL-001 | 最大インスタンス数 | 1-20 replicas | 本番 |
| NFR-SCAL-002 | 同時接続ユーザー数 | > 100 users | 本番 |
| NFR-SCAL-003 | スケールアウト時間 | < 60s | 本番 |
| NFR-SCAL-004 | 最大ドキュメント数 | > 100,000 docs | 本番 |
| NFR-SCAL-005 | 最大インデックスサイズ | > 100GB | 本番 |

---

### 2.3 コスト効率 (Cost Efficiency)

| ID | 要件名 | 基準値 | 測定方法 |
|----|--------|--------|---------|
| NFR-COST-001 | Level 0-1 インデックスコスト | LLMコスト $0 | API使用量 |
| NFR-COST-002 | Lazy vs GraphRAG検索コスト | 1/100 | LLMトークン比 |
| NFR-COST-003 | Unified vs Hybrid検索コスト | 30%削減 | LLMトークン比 |
| NFR-COST-004 | 月間運用コスト（ローカル） | < $10 | 電気代 |
| NFR-COST-005 | 月間運用コスト（本番） | < $500 (Small) | Azure料金 |

---

### 2.4 品質 (Quality)

| ID | 要件名 | 基準値 | 測定方法 |
|----|--------|--------|---------|
| NFR-QUAL-001 | 回答正確性 (RAGAS Faithfulness) | > 0.8 | 評価データセット |
| NFR-QUAL-002 | 回答関連性 (RAGAS Relevancy) | > 0.8 | 評価データセット |
| NFR-QUAL-003 | Query Router分類精度 | > 85% | テストクエリセット |
| NFR-QUAL-004 | PDFテキスト抽出精度 | > 95% | サンプル論文 |
| NFR-QUAL-005 | 参考文献抽出精度 | > 85% | サンプル論文 |
| NFR-QUAL-006 | DOIマッチング率 | > 80% | サンプル論文 |
| NFR-QUAL-007 | IMRaD構造認識精度 | > 80% | サンプル論文 |

---

### 2.5 セキュリティ (Security)

| ID | 要件名 | 説明 | 環境 |
|----|--------|------|------|
| NFR-SEC-001 | 認証 | Azure Entra ID / API Key認証 | 本番 |
| NFR-SEC-002 | 認可 | RBAC（admin/researcher/viewer） | 本番 |
| NFR-SEC-003 | 通信暗号化 | TLS 1.3必須 | 本番 |
| NFR-SEC-004 | 保存データ暗号化 | AES-256 | 本番 |
| NFR-SEC-005 | シークレット管理 | Azure Key Vault使用 | 本番 |
| NFR-SEC-006 | 監査ログ | 全API呼び出し記録 | 本番 |
| NFR-SEC-007 | レート制限 | 100 req/min/user | 本番 |

---

### 2.6 可用性 (Availability)

| ID | 要件名 | 基準値 | 環境 |
|----|--------|--------|------|
| NFR-AVL-001 | SLA | 99.9% | 本番 |
| NFR-AVL-002 | 計画メンテナンス | 月1回、深夜帯 | 本番 |
| NFR-AVL-003 | RTO（目標復旧時間） | < 1時間 | 本番 |
| NFR-AVL-004 | RPO（目標復旧時点） | < 24時間 | 本番 |
| NFR-AVL-005 | リージョン冗長 | Japan East + West | 本番 |
| NFR-AVL-006 | ヘルスチェック間隔 | 30秒 | 本番 |

---

### 2.7 運用性 (Operability)

| ID | 要件名 | 説明 |
|----|--------|------|
| NFR-OPS-001 | バックアップ | 日次自動バックアップ（30日保持） |
| NFR-OPS-002 | リストア | バックアップからの復旧手順整備 |
| NFR-OPS-003 | 監視アラート | CPU/メモリ/エラー率の閾値アラート |
| NFR-OPS-004 | ログ集約 | Azure Monitor / Application Insights |
| NFR-OPS-005 | メトリクス | レイテンシ・スループット・エラー率 |
| NFR-OPS-006 | 分散トレーシング | リクエスト追跡 |
| NFR-OPS-007 | ダッシュボード | 運用監視ダッシュボード |

**アラート閾値**:

| メトリクス | Warning | Critical |
|-----------|---------|----------|
| CPU使用率 | > 70% | > 90% |
| メモリ使用率 | > 70% | > 90% |
| エラー率 | > 1% | > 5% |
| レイテンシ (p95) | > 5s | > 10s |

---

### 2.8 拡張性 (Extensibility)

| ID | 要件名 | 対応オプション |
|----|--------|---------------|
| NFR-EXT-001 | LLMプロバイダー | OpenAI, Azure OpenAI, Ollama |
| NFR-EXT-002 | Embeddingプロバイダー | OpenAI, Azure OpenAI, Ollama |
| NFR-EXT-003 | ストレージバックエンド | File, Azure Blob, S3 |
| NFR-EXT-004 | ベクトルDB | LanceDB, FAISS, Azure AI Search |
| NFR-EXT-005 | PDF処理 | Azure Doc Intel, unstructured |
| NFR-EXT-006 | 外部学術API | Semantic Scholar, CrossRef, OpenAlex, CORE, Unpaywall |
| NFR-EXT-007 | キャッシュ | Local, Redis |

**環境別構成**:

| コンポーネント | ローカル | 本番 |
|----------------|---------|------|
| LLM | Ollama (llama3.2) | Azure OpenAI (GPT-4o) |
| Embedding | Ollama (nomic-embed-text) | Azure OpenAI (text-embedding-3-large) |
| PDF処理 | unstructured | Azure Document Intelligence |
| ベクトルDB | LanceDB | Azure AI Search |
| ストレージ | Local File | Azure Blob Storage |
| キャッシュ | Local Memory | Azure Cache for Redis |

---

### 2.9 テスト (Testing)

| ID | 要件名 | 基準値 |
|----|--------|--------|
| NFR-TST-001 | 単体テストカバレッジ | > 80% |
| NFR-TST-002 | 統合テスト | 主要フロー網羅 |
| NFR-TST-003 | E2Eテスト | ユーザーシナリオベース |
| NFR-TST-004 | パフォーマンステスト | 負荷・ストレステスト |
| NFR-TST-005 | 回帰テスト | CI/CD自動実行 |
| NFR-TST-006 | セキュリティテスト | 脆弱性スキャン |

**テストピラミッド**:

| 層 | 割合 | 内容 |
|-----|------|------|
| Unit | 70% | ロジック検証 |
| Integration | 20% | API・DB連携 |
| E2E | 10% | 主要シナリオ |

---

### 2.10 ドキュメント (Documentation)

| ID | 要件名 | 形式 |
|----|--------|------|
| NFR-DOC-001 | API仕様書 | OpenAPI 3.0 |
| NFR-DOC-002 | MCPツール仕様 | JSON Schema |
| NFR-DOC-003 | ユーザーガイド | Markdown |
| NFR-DOC-004 | 運用マニュアル | Markdown |
| NFR-DOC-005 | 開発者ガイド | Markdown |
| NFR-DOC-006 | CHANGELOG | Keep a Changelog形式 |

---

## 3. 要件トレーサビリティマトリクス

### 3.1 機能要件 → 実装優先度

| 要件カテゴリ | 要件数 | Phase 1 (MVP) | Phase 2 | Phase 3 | Phase 4 |
|-------------|--------|---------------|---------|---------|---------|
| **1.1 ドキュメント処理** | 18 | ✅ 18 | - | - | - |
| **1.2.1 Level 0 (Raw)** | 4 | ✅ 4 | - | - | - |
| **1.2.2 Level 1 (Lazy)** | 5 | ✅ 5 | - | - | - |
| **1.2.3 Level 2 (Partial)** | 5 | - | ✅ 5 | - | - |
| **1.2.4 Level 3 (Full)** | 4 | - | ✅ 4 | - | - |
| **1.2.5 Level 4 (Enhanced)** | 4 | - | - | - | ✅ 4 |
| **1.3.1 Vector Search** | 4 | ✅ 4 | - | - | - |
| **1.3.2 Lazy Search** | 6 | ✅ 6 | - | - | - |
| **1.3.3 Graph Search** | 9 | - | ✅ 9 | - | - |
| **1.3.4 Hybrid Search** | 4 | - | - | - | ✅ 4 |
| **1.3.5 Unified Search** | 4 | - | ✅ 4 | - | - |
| **1.4 引用ネットワーク** | 11 | ✅ 6 | ✅ 5 | - | - |
| **1.5 外部インターフェース** | 25 | ✅ 15 | ✅ 5 | ✅ 5 | - |
| **合計 (FR)** | **103** | **58** | **32** | **5** | **8** |

### 3.2 非機能要件 → 実装優先度

| 要件カテゴリ | 要件数 | Phase 1 (MVP) | Phase 2 | Phase 3 | Phase 4 |
|-------------|--------|---------------|---------|---------|---------|
| **2.1 パフォーマンス** | 7 | ✅ 3 | ✅ 2 | ✅ 2 | - |
| **2.2 スケーラビリティ** | 5 | - | - | ✅ 5 | - |
| **2.3 コスト効率** | 5 | ✅ 3 | ✅ 2 | - | - |
| **2.4 品質** | 7 | ✅ 4 | ✅ 3 | - | - |
| **2.5 セキュリティ** | 7 | - | - | ✅ 7 | - |
| **2.6 可用性** | 6 | - | - | ✅ 6 | - |
| **2.7 運用性** | 7 | - | - | ✅ 7 | - |
| **2.8 拡張性** | 7 | ✅ 4 | ✅ 3 | - | - |
| **2.9 テスト** | 6 | ✅ 3 | ✅ 2 | ✅ 1 | - |
| **2.10 ドキュメント** | 6 | ✅ 3 | ✅ 2 | ✅ 1 | - |
| **合計 (NFR)** | **63** | **20** | **14** | **29** | **0** |

### 3.3 総要件数サマリー

| カテゴリ | 要件数 |
|---------|--------|
| **機能要件 (FR)** | 103 |
| **非機能要件 (NFR)** | 63 |
| **総計** | **166** |

---

## 4. MECE検証

### 4.1 相互排他性 (Mutually Exclusive) 検証

| レイヤー | 検証結果 | 説明 |
|---------|---------|------|
| **ドキュメント処理フェーズ** | ✅ | 入力・前処理・出力は完全に分離 |
| **インデックスレベル** | ✅ | Level 0-4は相互排他的な構築ステップ |
| **検索モード** | ✅ | 5つのモードは相互排他的（同時に1つのみ選択） |
| **引用ネットワーク** | ✅ | 構築と分析は独立したフェーズ |
| **外部インターフェース** | ✅ | Python/CLI/MCP/RESTは独立したアクセス手段 |
| **非機能要件カテゴリ** | ✅ | 10カテゴリは重複なし |

### 4.2 網羅性 (Collectively Exhaustive) 検証

| レイヤー | 検証結果 | 説明 |
|---------|---------|------|
| **ドキュメント処理** | ✅ | 全対応形式・全前処理ステップを網羅 |
| **インデックス構築** | ✅ | Baseline→LazyGraphRAG→GraphRAGの全段階を網羅 |
| **クエリ処理** | ✅ | 全アーキテクチャパターン（Baseline/Lazy/Graph/Hybrid/Unified）を網羅 |
| **引用ネットワーク** | ✅ | 構築・分析の全機能を網羅 |
| **外部インターフェース** | ✅ | 全アクセス手段（プログラム/CLI/AI/REST）を網羅 |
| **非機能要件** | ✅ | ISO 25010品質属性を網羅（一部サブセット） |

---

## 5. 用語集

| 用語 | 定義 |
|------|------|
| **MECE** | Mutually Exclusive, Collectively Exhaustive（相互排他的かつ網羅的） |
| **EARS** | Easy Approach to Requirements Syntax（要件記述パターン） |
| **Progressive GraphRAG** | 段階的インデックス構築＋予算制御を行うアーキテクチャ |
| **Unified GraphRAG** | Query Routerによる動的モード選択を行うアーキテクチャ |
| **LazyGraphRAG** | クエリ時に動的にグラフ情報を抽出する遅延型RAG |
| **MCP** | Model Context Protocol（AIアシスタント統合プロトコル） |
| **RRF** | Reciprocal Rank Fusion（複数検索結果のマージ手法） |
| **IMRaD** | Introduction, Methods, Results, and Discussion（学術論文構造） |

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-12-25 | 初版（MECE原則による再定義） |
