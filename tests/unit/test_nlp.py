# NLP Unit Tests
"""
Unit tests for NLP components.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from monjyu.nlp.base import NLPFeatures, NLPProcessor
from monjyu.nlp.rake_extractor import RAKEKeywordExtractor


class TestNLPFeatures:
    """NLPFeaturesのテスト"""
    
    def test_create_features(self):
        """特徴量作成"""
        features = NLPFeatures(
            text_unit_id="tu_001",
            keywords=["deep learning", "nlp"],
            noun_phrases=["neural network", "attention mechanism"],
            entities=[("BERT", "MODEL"), ("Google", "ORGANIZATION")],
        )
        
        assert features.text_unit_id == "tu_001"
        assert len(features.keywords) == 2
        assert len(features.noun_phrases) == 2
        assert len(features.entities) == 2
    
    def test_to_dict(self):
        """辞書変換"""
        features = NLPFeatures(
            text_unit_id="tu_001",
            keywords=["deep learning"],
            noun_phrases=["neural network"],
            entities=[("BERT", "MODEL")],
        )
        
        data = features.to_dict()
        
        assert data["text_unit_id"] == "tu_001"
        assert data["keywords"] == ["deep learning"]
        assert data["noun_phrases"] == ["neural network"]
        assert data["entities"] == [("BERT", "MODEL")]
    
    def test_from_dict(self):
        """辞書からの復元"""
        data = {
            "text_unit_id": "tu_001",
            "keywords": ["deep learning"],
            "noun_phrases": ["neural network"],
            "entities": [["BERT", "MODEL"]],
        }
        
        features = NLPFeatures.from_dict(data)
        
        assert features.text_unit_id == "tu_001"
        assert features.keywords == ["deep learning"]
        assert features.entities == [("BERT", "MODEL")]


class TestRAKEKeywordExtractor:
    """RAKEKeywordExtractorのテスト"""
    
    def test_init_defaults(self):
        """デフォルト初期化"""
        extractor = RAKEKeywordExtractor()
        
        assert extractor.min_length == 2
        assert extractor.max_words == 4
        assert "the" in extractor.stopwords
    
    def test_extract_keywords(self):
        """キーワード抽出"""
        extractor = RAKEKeywordExtractor()
        
        text = "Deep learning has revolutionized natural language processing and computer vision."
        keywords = extractor.extract(text, top_k=5)
        
        assert len(keywords) <= 5
        assert isinstance(keywords[0], str)
    
    def test_extract_with_scores(self):
        """スコア付きキーワード抽出"""
        extractor = RAKEKeywordExtractor()
        
        text = "Machine learning algorithms can learn patterns from data."
        scores = extractor.extract_with_scores(text, top_k=3)
        
        assert isinstance(scores, dict)
        assert all(isinstance(v, float) for v in scores.values())
    
    def test_extract_batch(self):
        """バッチキーワード抽出"""
        extractor = RAKEKeywordExtractor()
        
        texts = [
            "Deep learning is a subset of machine learning.",
            "Natural language processing enables computers to understand text.",
        ]
        
        results = extractor.extract_batch(texts, top_k=3)
        
        assert len(results) == 2
        assert all(isinstance(kws, list) for kws in results)
    
    def test_custom_stopwords(self):
        """カスタムストップワード"""
        custom_stopwords = {"the", "a", "an", "is", "are"}
        extractor = RAKEKeywordExtractor(stopwords=custom_stopwords)
        
        assert extractor.stopwords == custom_stopwords
    
    def test_empty_text(self):
        """空テキスト"""
        extractor = RAKEKeywordExtractor()
        
        keywords = extractor.extract("", top_k=5)
        
        assert keywords == []


class TestSpacyNLPProcessor:
    """SpacyNLPProcessorのテスト"""
    
    def test_init_defaults(self):
        """デフォルト初期化"""
        # spaCyモデルがインストールされている場合のみ
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            processor = SpacyNLPProcessor()
            
            assert processor.model_name == "en_core_web_sm"
            assert processor.academic_mode is True
        except RuntimeError:
            pytest.skip("spaCy model not installed")
    
    def test_extract_keywords_fallback(self):
        """フォールバックキーワード抽出"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            processor = SpacyNLPProcessor()
            
            # RAKEがない場合のフォールバック
            keywords = processor._extract_keywords_fallback(
                "Deep learning models achieve state-of-the-art results.",
                top_k=3,
            )
            
            assert isinstance(keywords, list)
        except RuntimeError:
            pytest.skip("spaCy model not installed")
    
    def test_extract_noun_phrases(self):
        """名詞句抽出"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            processor = SpacyNLPProcessor()
            
            text = "The neural network architecture uses attention mechanisms."
            noun_phrases = processor.extract_noun_phrases(text)
            
            assert isinstance(noun_phrases, list)
            # 名詞句は小文字
            assert all(np.islower() for np in noun_phrases)
        except RuntimeError:
            pytest.skip("spaCy model not installed")
    
    def test_extract_entities(self):
        """固有表現抽出"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            processor = SpacyNLPProcessor()
            
            text = "Google released BERT in 2018."
            entities = processor.extract_entities(text)
            
            assert isinstance(entities, list)
            # タプル形式
            assert all(len(e) == 2 for e in entities)
        except RuntimeError:
            pytest.skip("spaCy model not installed")
    
    def test_extract_methods(self):
        """手法名抽出"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            processor = SpacyNLPProcessor()
            
            text = "BERT and GPT-4 are transformer-based models like ResNet."
            methods = processor.extract_methods(text)
            
            assert isinstance(methods, list)
            # CamelCase や ACRONYM を検出
            assert any("BERT" in m or "GPT" in m or "ResNet" in m for m in methods)
        except RuntimeError:
            pytest.skip("spaCy model not installed")
    
    def test_process(self):
        """プロセス"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            processor = SpacyNLPProcessor()
            
            text = "Deep learning has transformed natural language processing."
            features = processor.process(text, "tu_001")
            
            assert features.text_unit_id == "tu_001"
            assert isinstance(features.keywords, list)
            assert isinstance(features.noun_phrases, list)
            assert isinstance(features.entities, list)
        except RuntimeError:
            pytest.skip("spaCy model not installed")
    
    def test_process_batch(self):
        """バッチプロセス"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            processor = SpacyNLPProcessor()
            
            texts = [
                "Deep learning is powerful.",
                "NLP enables text understanding.",
            ]
            tu_ids = ["tu_001", "tu_002"]
            
            results = processor.process_batch(texts, tu_ids)
            
            assert len(results) == 2
            assert results[0].text_unit_id == "tu_001"
            assert results[1].text_unit_id == "tu_002"
        except RuntimeError:
            pytest.skip("spaCy model not installed")
