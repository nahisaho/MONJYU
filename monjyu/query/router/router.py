# Query Router Implementation
"""
FEAT-014: QueryRouter 実装

クエリを分類し最適な検索モードにルーティング
"""

import re
import json
from typing import Protocol, List, Dict, Any, Optional, Tuple, runtime_checkable
from dataclasses import dataclass

from .types import (
    SearchMode,
    QueryType,
    RoutingDecision,
    RoutingContext,
    DEFAULT_MODE_MAPPING,
    LEVEL_FALLBACK_MAPPING,
)


@runtime_checkable
class ChatModelProtocol(Protocol):
    """チャットモデルプロトコル"""
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """メッセージからテキストを生成"""
        ...


class QueryRouterProtocol(Protocol):
    """クエリルータープロトコル"""
    
    async def route(
        self,
        query: str,
        context: Optional[RoutingContext] = None,
    ) -> RoutingDecision:
        """クエリをルーティング"""
        ...


@dataclass
class QueryRouterConfig:
    """クエリルーター設定"""
    use_llm_classification: bool = False
    llm_confidence_threshold: float = 0.7
    default_mode: SearchMode = SearchMode.LAZY
    enable_fallback: bool = True


class QueryRouter:
    """クエリルーター
    
    クエリを分析し、最適な検索モードにルーティング。
    
    Features:
    - ルールベース分類（高速）
    - LLM分類（オプション、高精度）
    - インデックスレベルによるフォールバック
    - 日本語/英語対応
    
    Examples:
        >>> router = QueryRouter()
        >>> decision = await router.route("この分野の研究動向は？")
        >>> decision.mode
        SearchMode.GRAPHRAG
    """
    
    # ==== 分類パターン（日本語・英語対応） ====
    
    SURVEY_PATTERNS = [
        # 日本語
        r'研究動向',
        r'トレンド',
        r'概要',
        r'全体[像的]',
        r'主要な',
        r'どのような.*分野',
        r'現状',
        # 英語
        r'(?i)survey',
        r'(?i)overview',
        r'(?i)trend',
        r'(?i)state[\s-]of[\s-]the[\s-]art',
        r'(?i)main\s+themes?',
        r'(?i)landscape',
        r'(?i)evolution\s+of',
    ]
    
    EXPLORATION_PATTERNS = [
        # 日本語
        r'(?:を|で)使[っう]た',
        r'実装方法',
        r'どう.*実現',
        r'手法',
        r'アプローチ',
        r'(?:の|について)説明',
        # 英語
        r'(?i)how\s+to',
        r'(?i)implementation',
        r'(?i)approach(?:es)?',
        r'(?i)method(?:s|ology)?',
        r'(?i)technique',
        # "what is X?" - but not "what is the difference/accuracy"
        r'(?i)what\s+is\s+(?!the\s+(?:difference|accuracy))',
    ]
    
    COMPARISON_PATTERNS = [
        # 日本語
        r'比較',
        r'違い',
        r'差[異分]',
        r'優れ',
        r'劣',
        r'(?:より|と).*(?:良|悪|高|低)',
        # 英語
        r'(?i)compar(?:e|ison)',
        r'(?i)vs\.?',
        r'(?i)versus',
        r'(?i)differ(?:ence|ent)',
        r'(?i)better|worse',
        r'(?i)advantage|disadvantage',
        r'(?i)difference\s+between',
    ]
    
    FACTOID_PATTERNS = [
        # 日本語
        r'(?:は|の)いくつ',
        r'何[個本件%]',
        r'数値',
        r'精度[はが]',
        r'(?:どこ|何ページ)に',
        r'サイズ[はが]',
        # 英語
        r'(?i)how\s+many',
        r'(?i)how\s+much',
        r'(?i)what\s+(?:is|are)\s+the\s+(?:number|value|size)',
        r'(?i)where\s+(?:is|can)',
        r'(?i)\d+.*(?:accuracy|precision|recall|f1)',
        r'(?i)(?:the\s+)?accuracy\s+(?:on|of|is)',
    ]
    
    CITATION_PATTERNS = [
        # 日本語
        r'最初に.*提案',
        r'元論文',
        r'引用',
        r'先行研究',
        r'誰が.*提案',
        # 英語
        r'(?i)first\s+propos',
        r'(?i)original\s+paper',
        r'(?i)cit(?:e|ation)',
        r'(?i)prior\s+work',
        r'(?i)who\s+(?:first|originally)',
    ]
    
    BENCHMARK_PATTERNS = [
        # 日本語
        r'ベンチマーク',
        r'性能一覧',
        r'評価指標',
        r'ランキング',
        r'リーダーボード',
        # 英語
        r'(?i)benchmark',
        r'(?i)leaderboard',
        r'(?i)ranking',
        r'(?i)evaluation\s+metric',
        r'(?i)performance\s+(?:comparison|table)',
    ]
    
    # LLM分類プロンプト
    LLM_CLASSIFICATION_PROMPT = """Classify the following query into one of these categories:
- SURVEY: Questions about research trends, overviews, state-of-the-art
- EXPLORATION: Questions about specific methods, implementations, approaches
- COMPARISON: Questions comparing methods, models, or approaches
- FACTOID: Questions seeking specific facts, numbers, or values
- CITATION: Questions about original papers, authors, prior work
- BENCHMARK: Questions about benchmarks, rankings, evaluation metrics
- UNKNOWN: Cannot be classified

Query: {query}

Respond with JSON: {{"type": "<TYPE>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""
    
    def __init__(
        self,
        llm_client: Optional[ChatModelProtocol] = None,
        config: Optional[QueryRouterConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or QueryRouterConfig()
        
        # パターンをコンパイル
        self._compiled_patterns = {
            QueryType.SURVEY: [re.compile(p) for p in self.SURVEY_PATTERNS],
            QueryType.EXPLORATION: [re.compile(p) for p in self.EXPLORATION_PATTERNS],
            QueryType.COMPARISON: [re.compile(p) for p in self.COMPARISON_PATTERNS],
            QueryType.FACTOID: [re.compile(p) for p in self.FACTOID_PATTERNS],
            QueryType.CITATION: [re.compile(p) for p in self.CITATION_PATTERNS],
            QueryType.BENCHMARK: [re.compile(p) for p in self.BENCHMARK_PATTERNS],
        }
    
    async def route(
        self,
        query: str,
        context: Optional[RoutingContext] = None,
    ) -> RoutingDecision:
        """クエリをルーティング
        
        Args:
            query: ユーザークエリ
            context: ルーティングコンテキスト
            
        Returns:
            ルーティング決定
        """
        context = context or RoutingContext()
        
        # ユーザー指定モードがあれば優先
        if context.user_preference and context.user_preference != SearchMode.AUTO:
            return RoutingDecision(
                mode=context.user_preference,
                query_type=QueryType.UNKNOWN,
                confidence=1.0,
                reasoning="User specified mode",
            )
        
        # ルールベース分類
        query_type, confidence = self._rule_based_classify(query)
        
        # 確信度が低い場合、LLM分類を試行
        if (
            self.config.use_llm_classification
            and self.llm_client
            and confidence < self.config.llm_confidence_threshold
        ):
            llm_type, llm_confidence = await self._llm_classify(query)
            if llm_confidence > confidence:
                query_type = llm_type
                confidence = llm_confidence
        
        # モード決定
        mode = self._determine_mode(query_type, context)
        
        # フォールバック処理
        fallback_mode = None
        if self.config.enable_fallback:
            mode, fallback_mode = self._apply_fallback(mode, context)
        
        # パラメータ決定
        params = self._determine_params(mode, query_type)
        
        return RoutingDecision(
            mode=mode,
            query_type=query_type,
            confidence=confidence,
            reasoning=self._build_reasoning(query_type, mode, fallback_mode),
            params=params,
            fallback_mode=fallback_mode,
        )
    
    def _rule_based_classify(self, query: str) -> Tuple[QueryType, float]:
        """ルールベース分類
        
        Args:
            query: クエリ文字列
            
        Returns:
            (クエリタイプ, 確信度) のタプル
        """
        scores: Dict[QueryType, int] = {qt: 0 for qt in QueryType}
        
        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[query_type] += 1
        
        # 最高スコアのタイプを選択
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        if max_score == 0:
            return QueryType.UNKNOWN, 0.5
        
        # 確信度を計算（マッチ数に基づく）
        confidence = min(0.5 + (max_score * 0.15), 0.95)
        
        return max_type, confidence
    
    async def _llm_classify(self, query: str) -> Tuple[QueryType, float]:
        """LLM分類
        
        Args:
            query: クエリ文字列
            
        Returns:
            (クエリタイプ, 確信度) のタプル
        """
        if not self.llm_client:
            return QueryType.UNKNOWN, 0.0
        
        try:
            prompt = self.LLM_CLASSIFICATION_PROMPT.format(query=query)
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.llm_client.generate(messages)
            
            # JSONをパース
            data = self._extract_json(response)
            
            type_str = data.get("type", "UNKNOWN").upper()
            confidence = float(data.get("confidence", 0.5))
            
            # 文字列をQueryTypeに変換
            try:
                query_type = QueryType(type_str.lower())
            except ValueError:
                query_type = QueryType.UNKNOWN
            
            return query_type, confidence
            
        except Exception:
            return QueryType.UNKNOWN, 0.0
    
    def _determine_mode(
        self,
        query_type: QueryType,
        context: RoutingContext,
    ) -> SearchMode:
        """検索モードを決定
        
        Args:
            query_type: クエリタイプ
            context: ルーティングコンテキスト
            
        Returns:
            検索モード
        """
        # デフォルトマッピングからモードを取得
        mode = DEFAULT_MODE_MAPPING.get(query_type, self.config.default_mode)
        
        # コンテキストによる調整
        if context.budget == "minimal":
            # 最小予算ではVECTORかLAZYのみ
            if mode in (SearchMode.GRAPHRAG, SearchMode.HYBRID):
                mode = SearchMode.LAZY
        
        return mode
    
    def _apply_fallback(
        self,
        mode: SearchMode,
        context: RoutingContext,
    ) -> Tuple[SearchMode, Optional[SearchMode]]:
        """インデックスレベルに基づくフォールバック
        
        Args:
            mode: 希望モード
            context: ルーティングコンテキスト
            
        Returns:
            (実際のモード, フォールバック元のモード) のタプル
        """
        level = context.index_level
        
        # レベルに応じたフォールバックマッピング
        fallback_map = LEVEL_FALLBACK_MAPPING.get(level, {})
        
        if mode in fallback_map:
            fallback_mode = fallback_map[mode]
            if context.is_mode_available(fallback_mode):
                return fallback_mode, mode
        
        # モードが利用可能か確認
        if not context.is_mode_available(mode):
            # 利用可能なモードにフォールバック
            if SearchMode.LAZY in context.available_modes:
                return SearchMode.LAZY, mode
            if SearchMode.VECTOR in context.available_modes:
                return SearchMode.VECTOR, mode
        
        return mode, None
    
    def _determine_params(
        self,
        mode: SearchMode,
        query_type: QueryType,
    ) -> Dict[str, Any]:
        """モード固有パラメータを決定
        
        Args:
            mode: 検索モード
            query_type: クエリタイプ
            
        Returns:
            パラメータ辞書
        """
        params: Dict[str, Any] = {}
        
        if mode == SearchMode.VECTOR:
            params["top_k"] = 10
            if query_type == QueryType.FACTOID:
                params["top_k"] = 5  # ピンポイント検索は少なめ
        
        elif mode == SearchMode.LAZY:
            params["max_iterations"] = 3
            if query_type == QueryType.EXPLORATION:
                params["max_iterations"] = 5  # 探索は深め
        
        elif mode == SearchMode.GRAPHRAG:
            params["community_level"] = 1
            if query_type == QueryType.SURVEY:
                params["community_level"] = 2  # サーベイは広め
        
        elif mode == SearchMode.HYBRID:
            params["fusion_method"] = "rrf"
            params["engines"] = ["vector", "lazy"]
            if query_type == QueryType.BENCHMARK:
                params["engines"] = ["vector", "lazy", "graphrag"]
        
        return params
    
    def _build_reasoning(
        self,
        query_type: QueryType,
        mode: SearchMode,
        fallback_mode: Optional[SearchMode],
    ) -> str:
        """決定理由を生成"""
        reasoning = f"Query classified as {query_type.value} -> {mode.value} search"
        
        if fallback_mode:
            reasoning += f" (fallback from {fallback_mode.value})"
        
        return reasoning
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """テキストからJSONを抽出"""
        # コードブロック内のJSONを探す
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # { ... } パターン
        brace_match = re.search(r'\{[\s\S]*?\}', text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def classify_sync(self, query: str) -> Tuple[QueryType, float]:
        """同期的にクエリを分類（ルールベースのみ）
        
        Args:
            query: クエリ文字列
            
        Returns:
            (クエリタイプ, 確信度) のタプル
        """
        return self._rule_based_classify(query)


def create_router(
    llm_client: Optional[ChatModelProtocol] = None,
    use_llm: bool = False,
    default_mode: SearchMode = SearchMode.LAZY,
) -> QueryRouter:
    """QueryRouterを作成するファクトリ関数"""
    config = QueryRouterConfig(
        use_llm_classification=use_llm,
        default_mode=default_mode,
    )
    return QueryRouter(llm_client, config)
