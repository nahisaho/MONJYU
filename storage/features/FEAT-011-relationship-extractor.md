# FEAT-011: RelationshipExtractor - LLM関係抽出

**フェーズ**: Phase 2  
**要件ID**: REQ-IDX-006  
**優先度**: P1  
**見積もり**: 2日  
**依存**: FEAT-010 (EntityExtractor)

---

## 1. 概要

### 1.1 目的

学術論文テキストからLLMを使用してエンティティ間の関係を抽出する。
GraphRAG Level 2の基盤コンポーネント。

### 1.2 スコープ

| 含む | 含まない |
|------|----------|
| LLMベース関係抽出 | エンティティ抽出（FEAT-010） |
| 学術関係タイプ定義 | グラフ構築（FEAT-012） |
| バッチ抽出処理 | コミュニティ検出 |
| 関係のマージ | |

---

## 2. 要件

### 2.1 機能要件

#### FR-011-001: 学術関係タイプ

```python
class AcademicRelationType(Enum):
    """学術関係タイプ"""
    USES = "uses"                  # AがBを使用
    EXTENDS = "extends"            # AがBを拡張
    IMPROVES = "improves"          # AがBを改善
    COMPARES = "compares"          # AとBを比較
    EVALUATES_ON = "evaluates_on"  # AをBで評価
    TRAINED_ON = "trained_on"      # AをBで訓練
    PROPOSED_BY = "proposed_by"    # AをBが提案
    AFFILIATED_WITH = "affiliated_with"  # AがBに所属
    CITES = "cites"                # AがBを引用
    RELATED_TO = "related_to"      # 一般的な関連
```

#### FR-011-002: 関係抽出

- エンティティリストとテキストから関係を抽出
- 関係の方向性を保持（source → target）
- 関係の強度・重要度を付与

#### FR-011-003: バッチ抽出

- チャンク単位の並列処理
- 重複関係のマージ

---

## 3. 設計

### 3.1 データモデル

```python
@dataclass
class Relationship:
    """関係"""
    id: str
    source_entity_id: str
    target_entity_id: str
    type: AcademicRelationType
    description: str
    weight: float = 1.0
    source_chunk_ids: List[str] = field(default_factory=list)
    
    # 追加情報
    evidence: str = ""  # 関係の根拠となるテキスト
    confidence: float = 1.0
```

### 3.2 プロンプトテンプレート

エンティティリストとテキストを入力とし、関係をJSON形式で出力。

---

## 4. ディレクトリ構造

```
monjyu/
├── index/
│   └── relationship_extractor/
│       ├── __init__.py
│       ├── types.py
│       ├── protocol.py
│       ├── llm_extractor.py
│       └── prompts.py
```

---

## 5. 実装タスク

| # | タスク | 見積もり |
|---|--------|----------|
| 1 | types.py - データモデル | 1h |
| 2 | prompts.py - プロンプト | 1h |
| 3 | llm_extractor.py - 実装 | 3h |
| 4 | 単体テスト | 2h |
| 5 | テスト実行・検証 | 1h |

---

## 6. 完了条件

- [ ] 全データモデル実装
- [ ] LLMRelationshipExtractor実装
- [ ] 単体テスト90%以上
- [ ] テスト全件パス
