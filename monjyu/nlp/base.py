# NLP Base Classes and Protocols
"""
Base classes and protocols for NLP processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@dataclass
class NLPFeatures:
    """NLP抽出特徴量
    
    Attributes:
        text_unit_id: TextUnit ID
        keywords: キーワードリスト
        noun_phrases: 名詞句リスト
        entities: 固有表現リスト (entity, type)
        keyword_scores: キーワードスコア (オプション)
    """
    text_unit_id: str
    keywords: list[str] = field(default_factory=list)
    noun_phrases: list[str] = field(default_factory=list)
    entities: list[tuple[str, str]] = field(default_factory=list)
    keyword_scores: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "text_unit_id": self.text_unit_id,
            "keywords": self.keywords,
            "noun_phrases": self.noun_phrases,
            "entities": self.entities,
            "keyword_scores": self.keyword_scores,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NLPFeatures":
        """辞書から復元"""
        return cls(
            text_unit_id=data["text_unit_id"],
            keywords=data.get("keywords", []),
            noun_phrases=data.get("noun_phrases", []),
            entities=[tuple(e) for e in data.get("entities", [])],
            keyword_scores=data.get("keyword_scores", {}),
        )


@runtime_checkable
class NLPProcessorProtocol(Protocol):
    """NLP処理プロトコル
    
    NLPProcessorが実装すべきインターフェース。
    """
    
    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        """キーワードを抽出
        
        Args:
            text: 入力テキスト
            top_k: 上位k件
            
        Returns:
            キーワードリスト
        """
        ...
    
    def extract_noun_phrases(self, text: str) -> list[str]:
        """名詞句を抽出
        
        Args:
            text: 入力テキスト
            
        Returns:
            名詞句リスト
        """
        ...
    
    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """固有表現を抽出
        
        Args:
            text: 入力テキスト
            
        Returns:
            (entity, type) のリスト
        """
        ...
    
    def process(self, text: str, text_unit_id: str = "") -> NLPFeatures:
        """テキストからNLP特徴量を抽出
        
        Args:
            text: 入力テキスト
            text_unit_id: TextUnit ID
            
        Returns:
            NLP特徴量
        """
        ...


class NLPProcessor(ABC):
    """NLP処理基底クラス
    
    NLPProcessorProtocolを実装する抽象基底クラス。
    """
    
    @abstractmethod
    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        """キーワードを抽出"""
        ...
    
    @abstractmethod
    def extract_noun_phrases(self, text: str) -> list[str]:
        """名詞句を抽出"""
        ...
    
    @abstractmethod
    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """固有表現を抽出"""
        ...
    
    def process(self, text: str, text_unit_id: str = "") -> NLPFeatures:
        """テキストからNLP特徴量を抽出
        
        Args:
            text: 入力テキスト
            text_unit_id: TextUnit ID
            
        Returns:
            NLP特徴量
        """
        return NLPFeatures(
            text_unit_id=text_unit_id,
            keywords=self.extract_keywords(text),
            noun_phrases=self.extract_noun_phrases(text),
            entities=self.extract_entities(text),
        )
    
    def process_batch(
        self,
        texts: list[str],
        text_unit_ids: list[str] | None = None,
    ) -> list[NLPFeatures]:
        """バッチ処理
        
        Args:
            texts: テキストリスト
            text_unit_ids: TextUnit IDリスト
            
        Returns:
            NLP特徴量リスト
        """
        if text_unit_ids is None:
            text_unit_ids = [f"tu_{i}" for i in range(len(texts))]
        
        results = []
        for text, tu_id in zip(texts, text_unit_ids, strict=True):
            results.append(self.process(text, tu_id))
        
        return results
