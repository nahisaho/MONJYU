# spaCy NLP Processor
"""
NLP processor using spaCy for noun phrase extraction and NER.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from monjyu.nlp.base import NLPFeatures, NLPProcessor

if TYPE_CHECKING:
    pass


class SpacyNLPProcessor(NLPProcessor):
    """spaCyベースのNLPプロセッサ
    
    spaCyを使用して名詞句抽出と固有表現認識を行う。
    キーワード抽出はRAKEを併用。
    
    Example:
        >>> processor = SpacyNLPProcessor()
        >>> features = processor.process("Deep learning has revolutionized NLP.")
        >>> print(features.noun_phrases)
        ['deep learning', 'nlp']
    
    Attributes:
        model_name: spaCyモデル名
        is_japanese: 日本語モードかどうか
    """
    
    # 学術論文でよく出る固有表現タイプのマッピング
    ACADEMIC_ENTITY_TYPES = {
        "PERSON": "RESEARCHER",
        "ORG": "ORGANIZATION",
        "PRODUCT": "MODEL",
        "WORK_OF_ART": "PAPER",
        "GPE": "LOCATION",
        "DATE": "DATE",
        "CARDINAL": "NUMBER",
        "PERCENT": "PERCENT",
    }
    
    # 学術用語辞書
    ACADEMIC_TERMS = {
        "transformer", "attention", "bert", "gpt", "lstm", "cnn", "rnn",
        "neural network", "deep learning", "machine learning",
        "natural language processing", "computer vision",
        "reinforcement learning", "supervised learning",
        "unsupervised learning", "transfer learning",
        "embedding", "encoder", "decoder", "tokenizer",
        "fine-tuning", "pre-training", "backpropagation",
        "gradient descent", "optimizer", "loss function",
        "batch normalization", "dropout", "regularization",
        "convolution", "pooling", "softmax", "sigmoid",
        "graph neural network", "knowledge graph",
    }
    
    def __init__(
        self,
        model: str = "en_core_web_sm",
        academic_mode: bool = True,
    ) -> None:
        """初期化
        
        Args:
            model: spaCyモデル名
            academic_mode: 学術論文モードを有効化
        """
        self.model_name = model
        self.academic_mode = academic_mode
        self.is_japanese = "ja" in model
        
        # spaCyモデル読み込み
        self._nlp = None
        
        # RAKE初期化（遅延）
        self._rake = None
    
    @property
    def nlp(self):
        """spaCyモデルを遅延読み込み"""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                # モデルがない場合はダウンロードを案内
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Run: python -m spacy download {self.model_name}"
                )
        return self._nlp
    
    @property
    def rake(self):
        """RAKEを遅延初期化"""
        if self._rake is None:
            try:
                from rake_nltk import Rake
                self._rake = Rake()
            except ImportError:
                # RAKEがない場合は簡易実装を使用
                self._rake = None
        return self._rake
    
    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        """キーワードを抽出（RAKE使用）
        
        Args:
            text: 入力テキスト
            top_k: 上位k件
            
        Returns:
            キーワードリスト
        """
        if self.rake is not None:
            self.rake.extract_keywords_from_text(text)
            keywords = self.rake.get_ranked_phrases()[:top_k]
            return keywords
        else:
            # フォールバック: 名詞ベースのキーワード抽出
            return self._extract_keywords_fallback(text, top_k)
    
    def _extract_keywords_fallback(self, text: str, top_k: int) -> list[str]:
        """RAKEなしのフォールバックキーワード抽出"""
        doc = self.nlp(text)
        
        # 名詞と固有名詞を抽出
        keywords = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                keywords.append(token.lemma_.lower())
        
        # 頻度でソート
        from collections import Counter
        counter = Counter(keywords)
        return [kw for kw, _ in counter.most_common(top_k)]
    
    def extract_noun_phrases(self, text: str) -> list[str]:
        """名詞句を抽出
        
        Args:
            text: 入力テキスト
            
        Returns:
            名詞句リスト（小文字、重複除去）
        """
        doc = self.nlp(text)
        
        noun_phrases = []
        seen = set()
        
        for chunk in doc.noun_chunks:
            # 前後の冠詞や決定詞を除去
            phrase = chunk.text.strip().lower()
            
            # 短すぎるものは除外
            if len(phrase) < 3:
                continue
            
            # ストップワードのみは除外
            if all(token.is_stop for token in chunk):
                continue
            
            # 重複除去
            if phrase not in seen:
                noun_phrases.append(phrase)
                seen.add(phrase)
        
        return noun_phrases
    
    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """固有表現を抽出
        
        Args:
            text: 入力テキスト
            
        Returns:
            (entity, type) のリスト
        """
        doc = self.nlp(text)
        
        entities = []
        seen = set()
        
        # spaCyのNER
        for ent in doc.ents:
            entity_text = ent.text.strip()
            
            # 重複チェック
            if entity_text.lower() in seen:
                continue
            seen.add(entity_text.lower())
            
            # 学術モードの場合はタイプをマッピング
            if self.academic_mode:
                entity_type = self.ACADEMIC_ENTITY_TYPES.get(
                    ent.label_, ent.label_
                )
            else:
                entity_type = ent.label_
            
            entities.append((entity_text, entity_type))
        
        # 学術モードの場合は学術用語も検出
        if self.academic_mode:
            text_lower = text.lower()
            for term in self.ACADEMIC_TERMS:
                if term in text_lower and term not in seen:
                    entities.append((term, "CONCEPT"))
                    seen.add(term)
        
        return entities
    
    def extract_methods(self, text: str) -> list[str]:
        """手法名を抽出（CamelCase, ACRONYMパターン）
        
        Args:
            text: 入力テキスト
            
        Returns:
            手法名リスト
        """
        methods = []
        
        # CamelCase (BERT, ResNet, GPT-4)
        pattern_camel = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'
        methods.extend(re.findall(pattern_camel, text))
        
        # ACRONYM (LSTM, CNN, NLP)
        pattern_acronym = r'\b([A-Z]{2,}(?:-[A-Z0-9]+)*)\b'
        methods.extend(re.findall(pattern_acronym, text))
        
        # 重複除去
        return list(set(methods))
    
    def process(self, text: str, text_unit_id: str = "") -> NLPFeatures:
        """テキストからNLP特徴量を抽出
        
        Args:
            text: 入力テキスト
            text_unit_id: TextUnit ID
            
        Returns:
            NLP特徴量
        """
        # 基本のNLP特徴量を抽出
        features = NLPFeatures(
            text_unit_id=text_unit_id,
            keywords=self.extract_keywords(text),
            noun_phrases=self.extract_noun_phrases(text),
            entities=self.extract_entities(text),
        )
        
        # 学術モードの場合は手法名も追加
        if self.academic_mode:
            methods = self.extract_methods(text)
            for method in methods:
                if (method, "METHOD") not in features.entities:
                    features.entities.append((method, "METHOD"))
        
        return features
    
    def process_batch(
        self,
        texts: list[str],
        text_unit_ids: list[str] | None = None,
        batch_size: int = 50,
    ) -> list[NLPFeatures]:
        """バッチ処理（spaCyパイプライン使用）
        
        Args:
            texts: テキストリスト
            text_unit_ids: TextUnit IDリスト
            batch_size: バッチサイズ
            
        Returns:
            NLP特徴量リスト
        """
        if text_unit_ids is None:
            text_unit_ids = [f"tu_{i}" for i in range(len(texts))]
        
        results = []
        
        # spaCyのパイプライン処理で効率化
        docs = list(self.nlp.pipe(texts, batch_size=batch_size))
        
        for doc, text, tu_id in zip(docs, texts, text_unit_ids, strict=True):
            # docから直接名詞句とエンティティを抽出
            noun_phrases = self._extract_noun_phrases_from_doc(doc)
            entities = self._extract_entities_from_doc(doc, text)
            keywords = self.extract_keywords(text)
            
            features = NLPFeatures(
                text_unit_id=tu_id,
                keywords=keywords,
                noun_phrases=noun_phrases,
                entities=entities,
            )
            
            results.append(features)
        
        return results
    
    def _extract_noun_phrases_from_doc(self, doc) -> list[str]:
        """docから名詞句を抽出"""
        noun_phrases = []
        seen = set()
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            
            if len(phrase) < 3:
                continue
            
            if all(token.is_stop for token in chunk):
                continue
            
            if phrase not in seen:
                noun_phrases.append(phrase)
                seen.add(phrase)
        
        return noun_phrases
    
    def _extract_entities_from_doc(self, doc, text: str) -> list[tuple[str, str]]:
        """docから固有表現を抽出"""
        entities = []
        seen = set()
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            
            if entity_text.lower() in seen:
                continue
            seen.add(entity_text.lower())
            
            if self.academic_mode:
                entity_type = self.ACADEMIC_ENTITY_TYPES.get(
                    ent.label_, ent.label_
                )
            else:
                entity_type = ent.label_
            
            entities.append((entity_text, entity_type))
        
        # 学術用語
        if self.academic_mode:
            text_lower = text.lower()
            for term in self.ACADEMIC_TERMS:
                if term in text_lower and term not in seen:
                    entities.append((term, "CONCEPT"))
                    seen.add(term)
        
        return entities
