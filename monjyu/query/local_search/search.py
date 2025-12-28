"""LocalSearch implementation with entity-centric graph traversal."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set

from .prompts import get_local_search_prompt
from .types import (
    ChunkInfo,
    EntityInfo,
    EntityMatch,
    LocalSearchConfig,
    LocalSearchResult,
    RelationshipInfo,
)


class LLMClientProtocol(Protocol):
    """LLMクライアントのプロトコル"""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """プロンプトから応答を生成"""
        ...

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        ...


class EntityStoreProtocol(Protocol):
    """エンティティストアのプロトコル"""

    def search_entities(
        self, query: str, top_k: int = 10
    ) -> List[EntityInfo]:
        """クエリに関連するエンティティを検索"""
        ...

    def get_entity_by_id(self, entity_id: str) -> Optional[EntityInfo]:
        """IDでエンティティを取得"""
        ...

    def get_entity_by_name(self, name: str) -> Optional[EntityInfo]:
        """名前でエンティティを取得"""
        ...


class RelationshipStoreProtocol(Protocol):
    """リレーションシップストアのプロトコル"""

    def get_relationships_for_entity(
        self, entity_id: str
    ) -> List[RelationshipInfo]:
        """エンティティに関連するリレーションシップを取得"""
        ...

    def get_relationships_between(
        self, entity_ids: List[str]
    ) -> List[RelationshipInfo]:
        """複数エンティティ間のリレーションシップを取得"""
        ...


class ChunkStoreProtocol(Protocol):
    """チャンクストアのプロトコル"""

    def get_chunks_for_entity(
        self, entity_id: str, top_k: int = 10
    ) -> List[ChunkInfo]:
        """エンティティに関連するチャンクを取得"""
        ...

    def search_chunks(
        self, query: str, top_k: int = 20
    ) -> List[ChunkInfo]:
        """クエリに関連するチャンクを検索"""
        ...


@dataclass
class LocalSearch:
    """エンティティ中心のグラフトラバーサルによるローカル検索
    
    特定のエンティティとその関係性に基づいて詳細な質問に回答します。
    
    Attributes:
        llm_client: LLMクライアント
        entity_store: エンティティストア
        relationship_store: リレーションシップストア
        chunk_store: チャンクストア
        config: 検索設定
    """

    llm_client: LLMClientProtocol
    entity_store: EntityStoreProtocol
    relationship_store: RelationshipStoreProtocol
    chunk_store: ChunkStoreProtocol
    config: LocalSearchConfig = field(default_factory=LocalSearchConfig)

    def search(
        self,
        query: str,
        config: Optional[LocalSearchConfig] = None,
    ) -> LocalSearchResult:
        """ローカル検索を実行
        
        Args:
            query: 検索クエリ
            config: 一時的な設定（省略時はインスタンス設定）
            
        Returns:
            LocalSearchResult: 検索結果
        """
        start_time = time.time()
        effective_config = config or self.config

        # 1. クエリから初期エンティティを特定
        initial_entities = self.entity_store.search_entities(
            query=query,
            top_k=effective_config.top_k_entities,
        )

        if not initial_entities:
            return LocalSearchResult(
                query=query,
                answer="No relevant entities found for the query.",
                entities_found=[],
                relationships_used=[],
                chunks_used=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                tokens_used=0,
                hops_traversed=0,
            )

        # 2. グラフトラバーサルで関連エンティティを収集
        entity_matches, relationships, hops_traversed = self._traverse_graph(
            initial_entities=initial_entities,
            max_hops=effective_config.max_hops,
            include_relationships=effective_config.include_relationships,
        )

        # 3. 関連チャンクを収集
        chunks = self._collect_chunks(
            entity_matches=entity_matches,
            query=query,
            top_k=effective_config.top_k_chunks,
        )

        # 4. コンテキストを構築
        entities_context = self._format_entities_context(entity_matches)
        relationships_context = self._format_relationships_context(relationships)
        chunks_context = self._format_chunks_context(chunks)

        # 5. トークン制限チェック
        total_context = entities_context + relationships_context + chunks_context
        context_tokens = self.llm_client.count_tokens(total_context)
        
        if context_tokens > effective_config.max_context_tokens:
            # コンテキストを縮小
            chunks = self._trim_chunks(
                chunks, 
                effective_config.max_context_tokens - self.llm_client.count_tokens(entities_context + relationships_context)
            )
            chunks_context = self._format_chunks_context(chunks)

        # 6. LLMで回答生成
        answer, tokens_used = self._generate_answer(
            query=query,
            entities_context=entities_context,
            relationships_context=relationships_context,
            chunks_context=chunks_context,
            config=effective_config,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return LocalSearchResult(
            query=query,
            answer=answer,
            entities_found=entity_matches,
            relationships_used=relationships,
            chunks_used=chunks,
            processing_time_ms=processing_time_ms,
            tokens_used=tokens_used,
            hops_traversed=hops_traversed,
        )

    def _traverse_graph(
        self,
        initial_entities: List[EntityInfo],
        max_hops: int,
        include_relationships: bool,
    ) -> tuple[List[EntityMatch], List[RelationshipInfo], int]:
        """グラフをトラバースして関連エンティティを収集
        
        Args:
            initial_entities: 初期エンティティリスト
            max_hops: 最大ホップ数
            include_relationships: リレーションシップを含めるか
            
        Returns:
            (エンティティマッチリスト, リレーションシップリスト, トラバースしたホップ数)
        """
        entity_matches: List[EntityMatch] = []
        relationships: List[RelationshipInfo] = []
        visited_ids: Set[str] = set()
        current_entities: List[EntityInfo] = initial_entities
        actual_hops = 0

        # 初期エンティティを追加
        for entity in initial_entities:
            if entity.entity_id not in visited_ids:
                entity_matches.append(
                    EntityMatch(
                        entity=entity,
                        match_score=1.0,  # 直接マッチは最高スコア
                        hop_distance=0,
                        source_query_term="",
                    )
                )
                visited_ids.add(entity.entity_id)

        # ホップごとにトラバース
        for hop in range(1, max_hops + 1):
            next_entities: List[EntityInfo] = []
            
            for entity in current_entities:
                if not include_relationships:
                    continue
                    
                # リレーションシップを取得
                entity_relationships = self.relationship_store.get_relationships_for_entity(
                    entity.entity_id
                )
                
                for rel in entity_relationships:
                    # リレーションシップを追加（重複除外）
                    if rel.relationship_id not in [r.relationship_id for r in relationships]:
                        relationships.append(rel)
                    
                    # 接続先エンティティを取得
                    connected_id = (
                        rel.target_id if rel.source_id == entity.entity_id 
                        else rel.source_id
                    )
                    
                    if connected_id not in visited_ids:
                        connected_entity = self.entity_store.get_entity_by_id(connected_id)
                        if connected_entity:
                            # スコアはホップ距離で減衰
                            score = 1.0 / (hop + 1)
                            entity_matches.append(
                                EntityMatch(
                                    entity=connected_entity,
                                    match_score=score,
                                    hop_distance=hop,
                                    source_query_term="",
                                )
                            )
                            visited_ids.add(connected_id)
                            next_entities.append(connected_entity)

            if next_entities:
                actual_hops = hop
                current_entities = next_entities
            else:
                break

        return entity_matches, relationships, actual_hops

    def _collect_chunks(
        self,
        entity_matches: List[EntityMatch],
        query: str,
        top_k: int,
    ) -> List[ChunkInfo]:
        """エンティティに関連するチャンクを収集
        
        Args:
            entity_matches: エンティティマッチリスト
            query: 元のクエリ
            top_k: 取得するチャンク数
            
        Returns:
            チャンクリスト
        """
        chunks: List[ChunkInfo] = []
        chunk_ids: Set[str] = set()
        
        # 各エンティティからチャンクを収集
        for match in entity_matches:
            entity_chunks = self.chunk_store.get_chunks_for_entity(
                entity_id=match.entity.entity_id,
                top_k=5,  # 各エンティティから最大5チャンク
            )
            
            for chunk in entity_chunks:
                if chunk.chunk_id not in chunk_ids and len(chunks) < top_k:
                    chunks.append(chunk)
                    chunk_ids.add(chunk.chunk_id)

        # 不足分はクエリベースで補完
        if len(chunks) < top_k:
            query_chunks = self.chunk_store.search_chunks(
                query=query,
                top_k=top_k - len(chunks),
            )
            for chunk in query_chunks:
                if chunk.chunk_id not in chunk_ids:
                    chunks.append(chunk)
                    chunk_ids.add(chunk.chunk_id)

        return chunks[:top_k]

    def _format_entities_context(self, entity_matches: List[EntityMatch]) -> str:
        """エンティティをコンテキスト文字列に整形"""
        if not entity_matches:
            return "No entities found."
            
        lines = []
        for match in entity_matches:
            entity = match.entity
            hop_info = f" (hop: {match.hop_distance})" if match.hop_distance > 0 else ""
            lines.append(
                f"- {entity.name} [{entity.entity_type}]{hop_info}: {entity.description}"
            )
        return "\n".join(lines)

    def _format_relationships_context(
        self, relationships: List[RelationshipInfo]
    ) -> str:
        """リレーションシップをコンテキスト文字列に整形"""
        if not relationships:
            return "No relationships found."
            
        lines = []
        for rel in relationships:
            lines.append(
                f"- {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}: {rel.description}"
            )
        return "\n".join(lines)

    def _format_chunks_context(self, chunks: List[ChunkInfo]) -> str:
        """チャンクをコンテキスト文字列に整形"""
        if not chunks:
            return "No supporting text found."
            
        lines = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[{chunk.paper_title}]" if chunk.paper_title else ""
            lines.append(f"[{i}] {source_info}\n{chunk.content}")
        return "\n\n".join(lines)

    def _trim_chunks(
        self, chunks: List[ChunkInfo], max_tokens: int
    ) -> List[ChunkInfo]:
        """トークン制限内にチャンクを縮小"""
        trimmed: List[ChunkInfo] = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self.llm_client.count_tokens(chunk.content)
            if total_tokens + chunk_tokens <= max_tokens:
                trimmed.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
                
        return trimmed

    def _generate_answer(
        self,
        query: str,
        entities_context: str,
        relationships_context: str,
        chunks_context: str,
        config: LocalSearchConfig,
    ) -> tuple[str, int]:
        """LLMで回答を生成
        
        Args:
            query: 検索クエリ
            entities_context: エンティティコンテキスト
            relationships_context: リレーションシップコンテキスト
            chunks_context: チャンクコンテキスト
            config: 検索設定
            
        Returns:
            (回答, 使用トークン数)
        """
        prompt_template = get_local_search_prompt(config.response_language)
        prompt = prompt_template.format(
            entities=entities_context,
            relationships=relationships_context,
            chunks=chunks_context,
            query=query,
        )

        answer = self.llm_client.generate(
            prompt,
            temperature=config.temperature,
        )

        tokens_used = self.llm_client.count_tokens(prompt + answer)

        return answer, tokens_used


# テスト・開発用のインメモリ実装
@dataclass
class InMemoryEntityStore:
    """テスト・開発用のインメモリエンティティストア"""

    entities: Dict[str, EntityInfo] = field(default_factory=dict)

    def add_entity(self, entity: EntityInfo) -> None:
        """エンティティを追加"""
        self.entities[entity.entity_id] = entity

    def search_entities(
        self, query: str, top_k: int = 10
    ) -> List[EntityInfo]:
        """クエリに関連するエンティティを検索（簡易実装）"""
        query_lower = query.lower()
        matches = []
        
        for entity in self.entities.values():
            # 名前または説明にクエリが含まれるか
            if (query_lower in entity.name.lower() or 
                query_lower in entity.description.lower()):
                matches.append(entity)
        
        return matches[:top_k]

    def get_entity_by_id(self, entity_id: str) -> Optional[EntityInfo]:
        """IDでエンティティを取得"""
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[EntityInfo]:
        """名前でエンティティを取得"""
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                return entity
        return None

    def clear(self) -> None:
        """全エンティティを削除"""
        self.entities.clear()


@dataclass
class InMemoryRelationshipStore:
    """テスト・開発用のインメモリリレーションシップストア"""

    relationships: Dict[str, RelationshipInfo] = field(default_factory=dict)
    _entity_index: Dict[str, List[str]] = field(default_factory=dict)

    def add_relationship(self, rel: RelationshipInfo) -> None:
        """リレーションシップを追加"""
        self.relationships[rel.relationship_id] = rel
        
        # エンティティインデックスを更新
        if rel.source_id not in self._entity_index:
            self._entity_index[rel.source_id] = []
        if rel.relationship_id not in self._entity_index[rel.source_id]:
            self._entity_index[rel.source_id].append(rel.relationship_id)
            
        if rel.target_id not in self._entity_index:
            self._entity_index[rel.target_id] = []
        if rel.relationship_id not in self._entity_index[rel.target_id]:
            self._entity_index[rel.target_id].append(rel.relationship_id)

    def get_relationships_for_entity(
        self, entity_id: str
    ) -> List[RelationshipInfo]:
        """エンティティに関連するリレーションシップを取得"""
        rel_ids = self._entity_index.get(entity_id, [])
        return [self.relationships[rid] for rid in rel_ids if rid in self.relationships]

    def get_relationships_between(
        self, entity_ids: List[str]
    ) -> List[RelationshipInfo]:
        """複数エンティティ間のリレーションシップを取得"""
        entity_set = set(entity_ids)
        result = []
        
        for rel in self.relationships.values():
            if rel.source_id in entity_set and rel.target_id in entity_set:
                result.append(rel)
                
        return result

    def clear(self) -> None:
        """全リレーションシップを削除"""
        self.relationships.clear()
        self._entity_index.clear()


@dataclass
class InMemoryChunkStore:
    """テスト・開発用のインメモリチャンクストア"""

    chunks: Dict[str, ChunkInfo] = field(default_factory=dict)
    _entity_index: Dict[str, List[str]] = field(default_factory=dict)

    def add_chunk(self, chunk: ChunkInfo, entity_ids: List[str] = None) -> None:
        """チャンクを追加"""
        self.chunks[chunk.chunk_id] = chunk
        
        # エンティティインデックスを更新
        if entity_ids:
            for entity_id in entity_ids:
                if entity_id not in self._entity_index:
                    self._entity_index[entity_id] = []
                if chunk.chunk_id not in self._entity_index[entity_id]:
                    self._entity_index[entity_id].append(chunk.chunk_id)

    def get_chunks_for_entity(
        self, entity_id: str, top_k: int = 10
    ) -> List[ChunkInfo]:
        """エンティティに関連するチャンクを取得"""
        chunk_ids = self._entity_index.get(entity_id, [])
        chunks = [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]
        return chunks[:top_k]

    def search_chunks(
        self, query: str, top_k: int = 20
    ) -> List[ChunkInfo]:
        """クエリに関連するチャンクを検索（簡易実装）"""
        query_lower = query.lower()
        matches = []
        
        for chunk in self.chunks.values():
            if query_lower in chunk.content.lower():
                matches.append(chunk)
        
        return matches[:top_k]

    def clear(self) -> None:
        """全チャンクを削除"""
        self.chunks.clear()
        self._entity_index.clear()


# テスト用のモックLLMクライアント（GlobalSearchと互換）
@dataclass
class MockLLMClient:
    """テスト用のモックLLMクライアント"""

    responses: Dict[str, str] = field(default_factory=dict)
    default_response: str = "This is a mock response based on the provided entities and relationships."
    tokens_per_char: float = 0.25

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """モック応答を生成"""
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        return self.default_response

    def count_tokens(self, text: str) -> int:
        """簡易トークンカウント"""
        return int(len(text) * self.tokens_per_char)
