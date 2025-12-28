# FEAT-014: QueryRouter

## 概要
クエリを分析し、最適な検索モードにルーティングするコンポーネント。

## 対応要件
- REQ-QRY-006: Query Router (Unified)

## 機能
1. **クエリ分類**: クエリタイプを判定（Survey/Exploration/Comparison/Factoid）
2. **モード決定**: 最適な検索モードを選択（VECTOR/LAZY/GRAPHRAG/HYBRID）
3. **ルールベース分類**: パターンマッチングによる高速分類
4. **LLM分類（オプション）**: 低確信度時のLLM支援分類

## データモデル

### SearchMode
```python
class SearchMode(Enum):
    AUTO = "auto"
    VECTOR = "vector"
    LAZY = "lazy"
    GRAPHRAG = "graphrag"
    HYBRID = "hybrid"
```

### QueryType
```python
class QueryType(Enum):
    SURVEY = "survey"           # サーベイ・傾向分析
    EXPLORATION = "exploration" # 手法調査・探索
    COMPARISON = "comparison"   # 手法比較
    FACTOID = "factoid"        # 具体的事実
    UNKNOWN = "unknown"
```

### RoutingDecision
```python
@dataclass
class RoutingDecision:
    mode: SearchMode
    query_type: QueryType
    confidence: float
    reasoning: str
    params: Dict[str, Any]
```

## 分類基準

| クエリタイプ | パターン例 | 選択モード |
|-------------|-----------|-----------|
| SURVEY | 「研究動向」「トレンド」「overview」 | GRAPHRAG/LAZY |
| EXPLORATION | 「〜を使った手法」「実装方法」 | LAZY |
| COMPARISON | 「比較」「違い」「vs」 | HYBRID |
| FACTOID | 「いくつ」「数値」「どこに」 | VECTOR |

## ディレクトリ構造
```
monjyu/query/router/
├── __init__.py
├── types.py      # SearchMode, QueryType, RoutingDecision
└── router.py     # QueryRouter 実装
```

## 依存関係
- monjyu/core/chat_model.py (ChatModelProtocol) - オプションLLM分類用

## テスト計画
1. SearchMode/QueryType 型テスト
2. ルールベース分類テスト
3. モード決定ロジックテスト
4. 日本語/英語クエリテスト
5. LLM分類テスト（モック）
6. 統合テスト
