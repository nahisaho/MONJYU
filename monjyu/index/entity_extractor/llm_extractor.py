# LLM Entity Extractor
"""
FEAT-010: LLMベースエンティティ抽出

LLMを使用して学術エンティティを抽出する実装。
"""

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional, AsyncIterator, TYPE_CHECKING

from monjyu.index.entity_extractor.types import (
    AcademicEntityType,
    Entity,
    ExtractionResult,
    BatchExtractionResult,
)
from monjyu.index.entity_extractor.protocol import EntityExtractorProtocol
from monjyu.index.entity_extractor.prompts import get_extraction_prompt

if TYPE_CHECKING:
    from monjyu.core.llm import ChatModelProtocol


class LLMEntityExtractor(EntityExtractorProtocol):
    """LLMベースのエンティティ抽出
    
    LLMを使用してテキストチャンクから学術エンティティを抽出する。
    
    Attributes:
        llm_client: LLMクライアント
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）
        language: 抽出言語（"en" or "ja"）
    
    Examples:
        >>> from monjyu.core.llm import AzureOpenAIClient
        >>> llm = AzureOpenAIClient(...)
        >>> extractor = LLMEntityExtractor(llm)
        >>> result = await extractor.extract(chunk)
        >>> print(f"Found {len(result.entities)} entities")
    """
    
    def __init__(
        self,
        llm_client: "ChatModelProtocol",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        language: str = "en",
    ):
        """初期化
        
        Args:
            llm_client: LLMクライアント
            max_retries: 最大リトライ回数
            retry_delay: リトライ間隔（秒）
            language: 抽出言語（"en" or "ja"）
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.language = language
    
    async def extract(
        self,
        chunk: Any  # TextChunk
    ) -> ExtractionResult:
        """単一チャンクからエンティティ抽出
        
        Args:
            chunk: 抽出対象のテキストチャンク
            
        Returns:
            抽出結果
        """
        start_time = time.time()
        prompt = get_extraction_prompt(chunk.content, self.language)
        
        response = ""
        for attempt in range(self.max_retries):
            try:
                response = await self.llm_client.chat(prompt)
                
                # JSON抽出
                json_match = re.search(r'\{[\s\S]*\}', response)
                if not json_match:
                    raise ValueError("No JSON found in response")
                
                data = json.loads(json_match.group())
                
                entities = []
                for item in data.get("entities", []):
                    entity = self._parse_entity(item, chunk.id)
                    if entity:
                        entities.append(entity)
                
                return ExtractionResult(
                    chunk_id=chunk.id,
                    entities=entities,
                    raw_response=response,
                    extraction_time_ms=(time.time() - start_time) * 1000,
                )
            
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    return ExtractionResult(
                        chunk_id=chunk.id,
                        entities=[],
                        raw_response=response,
                        extraction_time_ms=(time.time() - start_time) * 1000,
                        error=str(e),
                    )
                await asyncio.sleep(self.retry_delay)
        
        # Should not reach here, but just in case
        return ExtractionResult(
            chunk_id=chunk.id,
            entities=[],
            raw_response=response,
            extraction_time_ms=(time.time() - start_time) * 1000,
            error="Max retries exceeded",
        )
    
    def _parse_entity(
        self,
        item: Dict[str, Any],
        chunk_id: str
    ) -> Optional[Entity]:
        """エンティティ解析
        
        Args:
            item: LLMからのエンティティデータ
            chunk_id: 抽出元チャンクID
            
        Returns:
            Entityオブジェクト、または解析失敗時はNone
        """
        if not item.get("name"):
            return None
        
        # タイプ解析
        type_str = item.get("type", "CONCEPT").upper()
        try:
            entity_type = AcademicEntityType[type_str]
        except KeyError:
            entity_type = AcademicEntityType.CONCEPT
        
        return Entity(
            id=str(uuid.uuid4()),
            name=item["name"],
            type=entity_type,
            description=item.get("description", ""),
            aliases=item.get("aliases", []),
            source_chunk_ids=[chunk_id],
        )
    
    async def extract_batch(
        self,
        chunks: List[Any],  # List[TextChunk]
        max_concurrent: int = 5
    ) -> List[ExtractionResult]:
        """バッチ抽出
        
        Args:
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            抽出結果のリスト
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_limit(chunk) -> ExtractionResult:
            async with semaphore:
                return await self.extract(chunk)
        
        tasks = [extract_with_limit(c) for c in chunks]
        return await asyncio.gather(*tasks)
    
    async def extract_stream(
        self,
        chunks: List[Any],  # List[TextChunk]
        max_concurrent: int = 5
    ) -> AsyncIterator[ExtractionResult]:
        """ストリーミング抽出
        
        Args:
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Yields:
            各チャンクの抽出結果
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        queue: asyncio.Queue[ExtractionResult] = asyncio.Queue()
        done_event = asyncio.Event()
        
        async def extract_and_queue(chunk):
            async with semaphore:
                result = await self.extract(chunk)
                await queue.put(result)
        
        async def run_all():
            tasks = [asyncio.create_task(extract_and_queue(c)) for c in chunks]
            await asyncio.gather(*tasks)
            done_event.set()
        
        # バックグラウンドでタスク実行
        asyncio.create_task(run_all())
        
        completed = 0
        while completed < len(chunks):
            try:
                result = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield result
                completed += 1
            except asyncio.TimeoutError:
                if done_event.is_set() and queue.empty():
                    break
    
    async def extract_all(
        self,
        chunks: List[Any],  # List[TextChunk]
        max_concurrent: int = 5
    ) -> BatchExtractionResult:
        """全チャンクを抽出しマージ
        
        バッチ抽出後、エンティティをマージして返す。
        
        Args:
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            バッチ抽出結果（マージ済み）
        """
        start_time = time.time()
        
        results = await self.extract_batch(chunks, max_concurrent)
        
        # 全エンティティ収集
        all_entities = []
        error_count = 0
        for result in results:
            all_entities.extend(result.entities)
            if result.error:
                error_count += 1
        
        # マージ
        merged = self.merge_entities(all_entities)
        
        return BatchExtractionResult(
            results=results,
            total_entities=len(all_entities),
            merged_entities=merged,
            total_time_ms=(time.time() - start_time) * 1000,
            error_count=error_count,
        )
    
    def merge_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """重複エンティティをマージ
        
        Args:
            entities: マージ対象のエンティティリスト
            
        Returns:
            マージ後のエンティティリスト
        """
        if not entities:
            return []
        
        # 正規化された名前でグループ化
        groups: Dict[str, List[Entity]] = defaultdict(list)
        
        for entity in entities:
            key = self._normalize_name(entity.name)
            groups[key].append(entity)
        
        merged: List[Entity] = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # マージ
                primary = group[0]
                for other in group[1:]:
                    primary = primary.merge_with(other)
                
                # 信頼度更新（出現回数に基づく）
                primary.confidence = min(1.0, len(group) * 0.2 + 0.5)
                
                merged.append(primary)
        
        return merged
    
    def _normalize_name(self, name: str) -> str:
        """名前の正規化
        
        Args:
            name: 正規化対象の名前
            
        Returns:
            正規化された名前
        """
        # 小文字化
        name = name.lower()
        # 余分な空白除去
        name = re.sub(r'\s+', ' ', name).strip()
        # 特殊文字除去（ハイフンとアンダースコアは保持）
        name = re.sub(r'[^\w\s\-_]', '', name)
        
        return name


class CachedEntityExtractor(LLMEntityExtractor):
    """キャッシュ付きエンティティ抽出
    
    同一チャンクの再抽出を避けるためのキャッシュ機能付き。
    """
    
    def __init__(
        self,
        llm_client: "ChatModelProtocol",
        cache: Optional[Dict[str, ExtractionResult]] = None,
        **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self._cache = cache if cache is not None else {}
    
    async def extract(
        self,
        chunk: Any
    ) -> ExtractionResult:
        """キャッシュ付き抽出"""
        cache_key = f"{chunk.id}:{hash(chunk.content)}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = await super().extract(chunk)
        self._cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """キャッシュサイズ"""
        return len(self._cache)
