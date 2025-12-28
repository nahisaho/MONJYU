# LLM Relationship Extractor
"""
FEAT-011: LLMベース関係抽出

LLMを使用してエンティティ間の関係を抽出する実装。
"""

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from monjyu.index.relationship_extractor.types import (
    AcademicRelationType,
    Relationship,
    RelationshipExtractionResult,
    BatchRelationshipResult,
)
from monjyu.index.relationship_extractor.protocol import RelationshipExtractorProtocol
from monjyu.index.relationship_extractor.prompts import get_relationship_prompt

if TYPE_CHECKING:
    from monjyu.core.llm import ChatModelProtocol
    from monjyu.index.entity_extractor.types import Entity


class LLMRelationshipExtractor(RelationshipExtractorProtocol):
    """LLMベースの関係抽出
    
    LLMを使用してエンティティ間の関係を抽出する。
    
    Attributes:
        llm_client: LLMクライアント
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）
        language: 抽出言語
    
    Examples:
        >>> from monjyu.core.llm import AzureOpenAIClient
        >>> llm = AzureOpenAIClient(...)
        >>> extractor = LLMRelationshipExtractor(llm)
        >>> result = await extractor.extract(entities, chunk)
    """
    
    def __init__(
        self,
        llm_client: "ChatModelProtocol",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        language: str = "en",
    ):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.language = language
    
    async def extract(
        self,
        entities: List["Entity"],
        chunk: Any  # TextChunk
    ) -> RelationshipExtractionResult:
        """単一チャンクから関係抽出
        
        Args:
            entities: 対象エンティティリスト
            chunk: 抽出対象のテキストチャンク
            
        Returns:
            抽出結果
        """
        start_time = time.time()
        
        # エンティティが2未満なら関係なし
        if len(entities) < 2:
            return RelationshipExtractionResult(
                chunk_id=chunk.id,
                relationships=[],
                extraction_time_ms=(time.time() - start_time) * 1000,
            )
        
        # エンティティ名のセットを作成（検証用）
        entity_names = {e.name.lower() for e in entities}
        entity_map = {e.name.lower(): e for e in entities}
        
        prompt = get_relationship_prompt(entities, chunk.content, self.language)
        
        response = ""
        for attempt in range(self.max_retries):
            try:
                response = await self.llm_client.chat(prompt)
                
                # JSON抽出
                json_match = re.search(r'\{[\s\S]*\}', response)
                if not json_match:
                    raise ValueError("No JSON found in response")
                
                data = json.loads(json_match.group())
                
                relationships = []
                for item in data.get("relationships", []):
                    rel = self._parse_relationship(item, chunk.id, entity_map)
                    if rel:
                        relationships.append(rel)
                
                return RelationshipExtractionResult(
                    chunk_id=chunk.id,
                    relationships=relationships,
                    raw_response=response,
                    extraction_time_ms=(time.time() - start_time) * 1000,
                )
            
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    return RelationshipExtractionResult(
                        chunk_id=chunk.id,
                        relationships=[],
                        raw_response=response,
                        extraction_time_ms=(time.time() - start_time) * 1000,
                        error=str(e),
                    )
                await asyncio.sleep(self.retry_delay)
        
        return RelationshipExtractionResult(
            chunk_id=chunk.id,
            relationships=[],
            raw_response=response,
            extraction_time_ms=(time.time() - start_time) * 1000,
            error="Max retries exceeded",
        )
    
    def _parse_relationship(
        self,
        item: Dict[str, Any],
        chunk_id: str,
        entity_map: Dict[str, "Entity"]
    ) -> Optional[Relationship]:
        """関係を解析
        
        Args:
            item: LLMからの関係データ
            chunk_id: 抽出元チャンクID
            entity_map: エンティティ名→Entity マップ
            
        Returns:
            Relationshipオブジェクト、または解析失敗時はNone
        """
        source_name = item.get("source", "")
        target_name = item.get("target", "")
        
        if not source_name or not target_name:
            return None
        
        # エンティティを検索
        source_entity = entity_map.get(source_name.lower())
        target_entity = entity_map.get(target_name.lower())
        
        # 見つからない場合はスキップ
        if not source_entity or not target_entity:
            return None
        
        # タイプ解析
        type_str = item.get("type", "RELATED_TO").upper()
        type_str = type_str.replace(" ", "_").replace("-", "_")
        
        try:
            rel_type = AcademicRelationType[type_str]
        except KeyError:
            rel_type = AcademicRelationType.RELATED_TO
        
        return Relationship(
            id=str(uuid.uuid4()),
            source_entity_id=source_entity.id,
            target_entity_id=target_entity.id,
            source_entity_name=source_entity.name,
            target_entity_name=target_entity.name,
            type=rel_type,
            description=item.get("description", ""),
            evidence=item.get("evidence", ""),
            source_chunk_ids=[chunk_id],
        )
    
    async def extract_batch(
        self,
        entities: List["Entity"],
        chunks: List[Any],  # List[TextChunk]
        max_concurrent: int = 5
    ) -> List[RelationshipExtractionResult]:
        """バッチ抽出
        
        Args:
            entities: 対象エンティティリスト
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            抽出結果のリスト
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_limit(chunk) -> RelationshipExtractionResult:
            async with semaphore:
                return await self.extract(entities, chunk)
        
        tasks = [extract_with_limit(c) for c in chunks]
        return await asyncio.gather(*tasks)
    
    async def extract_all(
        self,
        entities: List["Entity"],
        chunks: List[Any],
        max_concurrent: int = 5
    ) -> BatchRelationshipResult:
        """全チャンクを抽出しマージ
        
        Args:
            entities: 対象エンティティリスト
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            バッチ抽出結果（マージ済み）
        """
        start_time = time.time()
        
        results = await self.extract_batch(entities, chunks, max_concurrent)
        
        # 全関係収集
        all_relationships = []
        error_count = 0
        for result in results:
            all_relationships.extend(result.relationships)
            if result.error:
                error_count += 1
        
        # マージ
        merged = self.merge_relationships(all_relationships)
        
        return BatchRelationshipResult(
            results=results,
            total_relationships=len(all_relationships),
            merged_relationships=merged,
            total_time_ms=(time.time() - start_time) * 1000,
            error_count=error_count,
        )
    
    def merge_relationships(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """重複関係をマージ
        
        Args:
            relationships: マージ対象の関係リスト
            
        Returns:
            マージ後の関係リスト
        """
        if not relationships:
            return []
        
        # キーでグループ化
        groups: Dict[str, List[Relationship]] = defaultdict(list)
        
        for rel in relationships:
            groups[rel.key].append(rel)
        
        merged: List[Relationship] = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # マージ
                primary = group[0]
                for other in group[1:]:
                    # ソースチャンク統合
                    primary.source_chunk_ids.extend(other.source_chunk_ids)
                    
                    # より長い説明/エビデンスを採用
                    if len(other.description) > len(primary.description):
                        primary.description = other.description
                    if len(other.evidence) > len(primary.evidence):
                        primary.evidence = other.evidence
                
                # 重複除去
                primary.source_chunk_ids = list(set(primary.source_chunk_ids))
                
                # 信頼度更新（出現回数に基づく）
                primary.confidence = min(1.0, len(group) * 0.2 + 0.5)
                primary.weight = min(1.0, len(group) * 0.15 + 0.5)
                
                merged.append(primary)
        
        return merged


class EntityAwareRelationshipExtractor(LLMRelationshipExtractor):
    """エンティティ認識関係抽出
    
    チャンクごとに関連エンティティをフィルタリングして抽出。
    """
    
    async def extract(
        self,
        entities: List["Entity"],
        chunk: Any
    ) -> RelationshipExtractionResult:
        """チャンク関連エンティティのみで抽出"""
        # このチャンクに関連するエンティティをフィルタ
        chunk_entities = [
            e for e in entities
            if chunk.id in getattr(e, 'source_chunk_ids', [])
        ]
        
        # 関連エンティティがない場合は全エンティティを使用
        if len(chunk_entities) < 2:
            chunk_entities = entities
        
        return await super().extract(chunk_entities, chunk)
