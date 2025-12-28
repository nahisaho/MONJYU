# FEAT-012: CommunityDetector - コミュニティ検出

**フェーズ**: Phase 2  
**要件ID**: REQ-IDX-007  
**優先度**: P1  
**見積もり**: 2日  
**依存**: FEAT-010, FEAT-011

---

## 1. 概要

### 1.1 目的

エンティティグラフや名詞句グラフからコミュニティ（クラスター）を検出する。
GraphRAG Level 2-3の基盤コンポーネント。

### 1.2 スコープ

| 含む | 含まない |
|------|----------|
| Leiden アルゴリズム | コミュニティレポート生成（FEAT-013） |
| Louvain フォールバック | グラフ構築 |
| 階層的コミュニティ検出 | |
| NetworkX/iGraph対応 | |

---

## 2. 設計

### 2.1 データモデル

```python
@dataclass
class Community:
    """コミュニティ"""
    id: str
    level: int                    # 階層レベル (0=最下位)
    member_ids: List[str]         # メンバーID
    parent_id: Optional[str]      # 親コミュニティID
    child_ids: List[str]          # 子コミュニティID
    
    # メタデータ
    title: Optional[str] = None
    summary: Optional[str] = None
    
    # 統計
    size: int = 0
    density: float = 0.0
```

### 2.2 検出アルゴリズム

1. **Leiden Algorithm** (推奨)
   - leidenalg ライブラリ使用
   - 高品質なコミュニティ検出
   - 階層的検出対応

2. **Louvain Algorithm** (フォールバック)
   - NetworkX 標準
   - leidenalg 未インストール時

---

## 3. ディレクトリ構造

```
monjyu/
├── index/
│   └── community_detector/
│       ├── __init__.py
│       ├── types.py
│       └── detector.py
```

---

## 4. 実装タスク

| # | タスク | 見積もり |
|---|--------|----------|
| 1 | types.py - データモデル | 1h |
| 2 | detector.py - 検出実装 | 3h |
| 3 | 単体テスト | 2h |
| 4 | テスト実行・検証 | 1h |

---

## 5. 完了条件

- [ ] Community データモデル実装
- [ ] CommunityDetector 実装
- [ ] Leiden/Louvain 両対応
- [ ] 単体テスト90%以上
