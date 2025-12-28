# Community Report Generator Implementation
"""
FEAT-013: CommunityReportGenerator 実装

LLMを使用してコミュニティレポートを生成
"""

import json
import time
import asyncio
import logging
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from dataclasses import dataclass

from .types import CommunityReport, Finding, ReportGenerationResult
from .prompts import build_report_prompt

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatModelProtocol(Protocol):
    """チャットモデルプロトコル（非同期）"""
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """メッセージからテキストを生成"""
        ...


class SyncLLMAdapter:
    """同期LLMクライアントを非同期ChatModelProtocolに適合させるアダプター
    
    MONJYUのLLMClientProtocol（同期）をCommunityReportGeneratorが
    期待するChatModelProtocol（非同期）に変換。
    """
    
    def __init__(self, sync_client: Any):
        """
        Args:
            sync_client: 同期LLMクライアント（generate(prompt) -> str）
        """
        self._sync_client = sync_client
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """非同期でメッセージを処理
        
        Args:
            messages: チャットメッセージリスト
            **kwargs: 追加パラメータ
            
        Returns:
            生成されたテキスト
        """
        # メッセージを単一のプロンプトに結合
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(content)
        
        prompt = "\n\n".join(prompt_parts)
        
        # 同期呼び出しをスレッドプールで実行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._sync_client.generate(prompt, **kwargs)
        )
        return result


class CommunityReportGeneratorProtocol(Protocol):
    """コミュニティレポート生成プロトコル"""
    
    async def generate(
        self,
        community_id: str,
        level: int,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> CommunityReport:
        """単一コミュニティのレポートを生成"""
        ...
    
    async def generate_batch(
        self,
        communities: List[Dict[str, Any]],
    ) -> ReportGenerationResult:
        """複数コミュニティのレポートを一括生成"""
        ...


@dataclass
class CommunityReportGeneratorConfig:
    """コミュニティレポート生成器設定"""
    language: str = "en"
    max_retries: int = 3
    retry_delay: float = 1.0
    max_concurrent: int = 5
    timeout: float = 60.0
    # LLM設定（llm_clientが未指定の場合に使用）
    llm_model: str = "llama3.2"
    ollama_base_url: str = "http://192.168.224.1:11434"


class CommunityReportGenerator:
    """コミュニティレポート生成器
    
    LLMを使用してコミュニティのエグゼクティブサマリーを生成。
    
    Examples:
        >>> # 既存のLLMクライアントを使用
        >>> generator = CommunityReportGenerator(llm_client)
        >>> report = await generator.generate(...)
        
        >>> # デフォルトLLMを自動作成
        >>> generator = CommunityReportGenerator()
        >>> reports = await generator.generate_from_entities(entities, relationships)
    """
    
    def __init__(
        self,
        llm_client: Optional[ChatModelProtocol] = None,
        config: Optional[CommunityReportGeneratorConfig] = None,
    ):
        """初期化
        
        Args:
            llm_client: LLMクライアント（省略時は自動作成）
            config: 設定（省略時はデフォルト）
        """
        self.config = config or CommunityReportGeneratorConfig()
        
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # デフォルトのLLMクライアントを作成
            self.llm_client = self._create_default_llm_client()
    
    def _create_default_llm_client(self) -> ChatModelProtocol:
        """デフォルトのLLMクライアントを作成"""
        from monjyu.api.factory import MockLLMClient
        
        sync_client = MockLLMClient(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
        )
        return SyncLLMAdapter(sync_client)
    
    async def generate(
        self,
        community_id: str,
        level: int,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> CommunityReport:
        """単一コミュニティのレポートを生成
        
        Args:
            community_id: コミュニティID
            level: コミュニティレベル
            entities: エンティティリスト
            relationships: 関係性リスト
            
        Returns:
            生成されたレポート
        """
        prompts = build_report_prompt(
            community_id=community_id,
            level=level,
            entities=entities,
            relationships=relationships,
            language=self.config.language,
        )
        
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]
        
        # LLM呼び出し（リトライ付き）
        response_text = await self._call_llm_with_retry(messages)
        
        # レスポンスをパース
        report = self._parse_response(
            response_text,
            community_id=community_id,
            level=level,
            entity_count=len(entities),
            relationship_count=len(relationships),
        )
        
        return report
    
    async def generate_batch(
        self,
        communities: List[Dict[str, Any]],
    ) -> ReportGenerationResult:
        """複数コミュニティのレポートを一括生成
        
        Args:
            communities: コミュニティ情報のリスト
                各要素: {
                    "community_id": str,
                    "level": int,
                    "entities": List[Dict],
                    "relationships": List[Dict]
                }
                
        Returns:
            生成結果
        """
        start_time = time.time()
        result = ReportGenerationResult(total_communities=len(communities))
        
        # セマフォで並行数を制限
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_community(comm: Dict[str, Any]) -> Optional[CommunityReport]:
            async with semaphore:
                try:
                    return await self.generate(
                        community_id=comm["community_id"],
                        level=comm.get("level", 0),
                        entities=comm.get("entities", []),
                        relationships=comm.get("relationships", []),
                    )
                except Exception as e:
                    result.add_error(
                        f"Failed to generate report for {comm['community_id']}: {str(e)}"
                    )
                    return None
        
        # 並行処理
        tasks = [process_community(c) for c in communities]
        reports = await asyncio.gather(*tasks)
        
        # 結果を収集
        for report in reports:
            if report is not None:
                result.add_report(report)
        
        result.generation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    async def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """リトライ付きLLM呼び出し"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.llm_client.generate(messages)
                return response
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay * (attempt + 1)
                    )
        
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts: {last_error}"
        )
    
    def _parse_response(
        self,
        response_text: str,
        community_id: str,
        level: int,
        entity_count: int,
        relationship_count: int,
    ) -> CommunityReport:
        """LLMレスポンスをパース
        
        Args:
            response_text: LLMからの生テキスト
            community_id: コミュニティID
            level: コミュニティレベル
            entity_count: エンティティ数
            relationship_count: 関係数
            
        Returns:
            パースされたレポート
        """
        # JSONを抽出
        json_data = self._extract_json(response_text)
        
        # Findingsをパース
        findings = []
        for i, f in enumerate(json_data.get("findings", [])):
            if isinstance(f, dict):
                findings.append(Finding(
                    id=f.get("id", f"finding-{i+1}"),
                    summary=f.get("summary", ""),
                    explanation=f.get("explanation", ""),
                    evidence=f.get("evidence", []),
                ))
            elif isinstance(f, str):
                findings.append(Finding(
                    id=f"finding-{i+1}",
                    summary=f,
                ))
        
        return CommunityReport(
            community_id=community_id,
            title=json_data.get("title", "Untitled Community"),
            summary=json_data.get("summary", ""),
            full_content=json_data.get("full_content", ""),
            findings=findings,
            rating=float(json_data.get("rating", 0.0)),
            rating_explanation=json_data.get("rating_explanation", ""),
            entity_count=entity_count,
            relationship_count=relationship_count,
            level=level,
        )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """テキストからJSONを抽出
        
        Args:
            text: JSONを含むテキスト
            
        Returns:
            パースされたJSON辞書
        """
        # コードブロック内のJSONを探す
        import re
        
        # ```json ... ``` パターン
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # { ... } パターン（最も外側の括弧）
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # パースできない場合は空の辞書を返す
        return {}
    
    async def generate_from_entities(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        resolution: float = 1.0,
    ) -> List[CommunityReport]:
        """エンティティと関係性からコミュニティを検出してレポートを生成
        
        Level 2のデータ（entities, relationships）からコミュニティを検出し、
        各コミュニティのレポートを生成する統合メソッド。
        
        Args:
            entities: エンティティリスト
            relationships: 関係性リスト
            resolution: Leidenアルゴリズムの解像度パラメータ
            
        Returns:
            生成されたコミュニティレポートのリスト
        """
        logger.info(f"Generating community reports from {len(entities)} entities and {len(relationships)} relationships")
        
        # エンティティIDからエンティティへのマッピングを作成
        entity_map: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            entity_id = entity.get("id") or entity.get("name")
            if entity_id:
                entity_map[entity_id] = entity
        
        # コミュニティ検出
        communities = self._detect_communities(entities, relationships, resolution)
        logger.info(f"Detected {len(communities)} communities")
        
        if not communities:
            logger.warning("No communities detected")
            return []
        
        # 各コミュニティのレポートを生成
        reports = []
        for community in communities:
            # コミュニティに属するエンティティを取得
            community_entities = []
            for member_id in community.get("member_ids", []):
                if member_id in entity_map:
                    community_entities.append(entity_map[member_id])
            
            # コミュニティに関連する関係性をフィルタリング
            member_ids_set = set(community.get("member_ids", []))
            community_relationships = [
                rel for rel in relationships
                if rel.get("source") in member_ids_set or rel.get("target") in member_ids_set
            ]
            
            try:
                report = await self.generate(
                    community_id=community.get("id", f"comm-{len(reports)+1}"),
                    level=community.get("level", 0),
                    entities=community_entities,
                    relationships=community_relationships,
                )
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to generate report for community {community.get('id')}: {e}")
                continue
        
        logger.info(f"Generated {len(reports)} community reports")
        return reports
    
    def _detect_communities(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        resolution: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """エンティティと関係性からコミュニティを検出
        
        Leidenアルゴリズムを使用してコミュニティを検出。
        
        Args:
            entities: エンティティリスト
            relationships: 関係性リスト
            resolution: 解像度パラメータ
            
        Returns:
            コミュニティのリスト
        """
        try:
            import networkx as nx
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            logger.warning("graspologic not installed, using simple community detection")
            return self._simple_community_detection(entities, relationships)
        
        # グラフを構築
        G = nx.Graph()
        
        # ノードを追加
        for entity in entities:
            node_id = entity.get("id") or entity.get("name")
            if node_id:
                G.add_node(node_id, **entity)
        
        # エッジを追加
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            weight = rel.get("weight", 1.0)
            if source and target and source in G.nodes() and target in G.nodes():
                G.add_edge(source, target, weight=weight)
        
        if len(G.nodes()) == 0:
            return []
        
        # 連結成分が複数ある場合は最大のものを使用
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        if len(G.nodes()) < 2:
            # 単一ノードの場合
            return [{
                "id": "comm-1",
                "level": 0,
                "member_ids": list(G.nodes()),
            }]
        
        # Leiden法でコミュニティ検出
        try:
            community_mapping = hierarchical_leiden(
                G,
                max_cluster_size=len(G.nodes()),
                random_seed=42,
            )
            
            # 結果を整形
            communities: Dict[Any, List[str]] = {}
            for node, comm_id in community_mapping.items():
                comm_key = str(comm_id)
                if comm_key not in communities:
                    communities[comm_key] = []
                communities[comm_key].append(str(node))
            
            return [
                {
                    "id": f"comm-{i+1}",
                    "level": 0,
                    "member_ids": members,
                }
                for i, (_, members) in enumerate(communities.items())
            ]
        except Exception as e:
            logger.warning(f"Leiden failed: {e}, falling back to simple detection")
            return self._simple_community_detection(entities, relationships)
    
    def _simple_community_detection(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """シンプルなコミュニティ検出（フォールバック）
        
        graspologicが利用できない場合の代替手法。
        連結成分をコミュニティとして扱う。
        """
        try:
            import networkx as nx
        except ImportError:
            # NetworkXもない場合は全体を1つのコミュニティとして扱う
            member_ids = [e.get("id") or e.get("name") for e in entities]
            member_ids = [m for m in member_ids if m]
            return [{
                "id": "comm-1",
                "level": 0,
                "member_ids": member_ids,
            }]
        
        # グラフを構築
        G = nx.Graph()
        
        for entity in entities:
            node_id = entity.get("id") or entity.get("name")
            if node_id:
                G.add_node(node_id)
        
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            if source and target and source in G.nodes() and target in G.nodes():
                G.add_edge(source, target)
        
        # 連結成分をコミュニティとして扱う
        communities = []
        for i, component in enumerate(nx.connected_components(G)):
            communities.append({
                "id": f"comm-{i+1}",
                "level": 0,
                "member_ids": list(component),
            })
        
        return communities


def create_report_generator(
    llm_client: Optional[ChatModelProtocol] = None,
    language: str = "en",
    max_concurrent: int = 5,
) -> CommunityReportGenerator:
    """CommunityReportGeneratorを作成するファクトリ関数
    
    Args:
        llm_client: LLMクライアント（省略時は自動作成）
        language: 言語 ("en" or "ja")
        max_concurrent: 最大並行数
        
    Returns:
        設定済みのCommunityReportGenerator
    """
    config = CommunityReportGeneratorConfig(
        language=language,
        max_concurrent=max_concurrent,
    )
    return CommunityReportGenerator(llm_client, config)
