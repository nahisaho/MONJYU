# RAKE Keyword Extractor
"""
RAKE (Rapid Automatic Keyword Extraction) implementation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class RAKEKeywordExtractor:
    """RAKEキーワード抽出器
    
    Rapid Automatic Keyword Extraction (RAKE) アルゴリズムを使用して
    テキストからキーフレーズを抽出する。
    
    Example:
        >>> extractor = RAKEKeywordExtractor()
        >>> keywords = extractor.extract("Deep learning has revolutionized NLP.")
        >>> print(keywords)
        ['deep learning', 'nlp']
    
    Attributes:
        stopwords: ストップワードセット
        min_length: 最小キーワード長
        max_words: キーフレーズの最大単語数
    """
    
    # 英語ストップワード
    ENGLISH_STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from",
        "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
        "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
        "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
        "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
        "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll",
        "they're", "they've", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
        "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
        "where's", "which", "while", "who", "who's", "whom", "why", "why's",
        "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves",
        # 追加の一般的なストップワード
        "also", "however", "thus", "therefore", "hence", "moreover", "furthermore",
        "although", "though", "whereas", "nevertheless", "nonetheless",
        "et", "al", "etc", "ie", "eg", "cf", "vs",
    }
    
    def __init__(
        self,
        stopwords: set[str] | None = None,
        min_length: int = 2,
        max_words: int = 4,
        include_numbers: bool = False,
    ) -> None:
        """初期化
        
        Args:
            stopwords: カスタムストップワード（Noneでデフォルト使用）
            min_length: 最小キーワード長
            max_words: キーフレーズの最大単語数
            include_numbers: 数字を含めるか
        """
        self.stopwords = stopwords or self.ENGLISH_STOPWORDS
        self.min_length = min_length
        self.max_words = max_words
        self.include_numbers = include_numbers
    
    def extract(
        self,
        text: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        """キーワードを抽出
        
        Args:
            text: 入力テキスト
            top_k: 上位k件
            return_scores: スコアも返すか
            
        Returns:
            キーワードリスト または (キーワード, スコア) のリスト
        """
        # 1. 候補フレーズを抽出
        phrases = self._extract_candidate_phrases(text)
        
        if not phrases:
            return [] if not return_scores else []
        
        # 2. 単語スコアを計算
        word_scores = self._calculate_word_scores(phrases)
        
        # 3. フレーズスコアを計算
        phrase_scores = self._calculate_phrase_scores(phrases, word_scores)
        
        # 4. スコアでソート
        sorted_phrases = sorted(
            phrase_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        if return_scores:
            return sorted_phrases
        else:
            return [phrase for phrase, _ in sorted_phrases]
    
    def _extract_candidate_phrases(self, text: str) -> list[list[str]]:
        """候補フレーズを抽出
        
        ストップワードで区切られた連続する単語列を抽出。
        """
        # テキストを小文字化してトークン化
        text = text.lower()
        
        # 句読点で分割
        sentences = re.split(r'[.!?,;:\n\t\(\)\[\]\{\}\"\']+', text)
        
        phrases = []
        for sentence in sentences:
            # 単語に分割
            words = re.findall(r'\b[\w\-]+\b', sentence)
            
            current_phrase = []
            for word in words:
                # ストップワードでなければフレーズに追加
                if word not in self.stopwords:
                    # 数字チェック
                    if not self.include_numbers and word.isdigit():
                        if current_phrase:
                            phrases.append(current_phrase)
                            current_phrase = []
                    else:
                        current_phrase.append(word)
                else:
                    # ストップワードでフレーズを区切る
                    if current_phrase:
                        phrases.append(current_phrase)
                        current_phrase = []
            
            # 最後のフレーズ
            if current_phrase:
                phrases.append(current_phrase)
        
        # フィルタリング
        filtered = []
        for phrase in phrases:
            # 最大単語数チェック
            if len(phrase) > self.max_words:
                continue
            
            # 最小長チェック
            phrase_text = " ".join(phrase)
            if len(phrase_text) < self.min_length:
                continue
            
            filtered.append(phrase)
        
        return filtered
    
    def _calculate_word_scores(
        self,
        phrases: list[list[str]],
    ) -> dict[str, float]:
        """単語スコアを計算
        
        RAKE: score(word) = degree(word) / frequency(word)
        """
        word_freq = defaultdict(int)
        word_degree = defaultdict(int)
        
        for phrase in phrases:
            degree = len(phrase) - 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += degree
        
        word_scores = {}
        for word in word_freq:
            # degree / frequency (または degree + frequency)
            word_scores[word] = (
                word_degree[word] + word_freq[word]
            ) / word_freq[word]
        
        return word_scores
    
    def _calculate_phrase_scores(
        self,
        phrases: list[list[str]],
        word_scores: dict[str, float],
    ) -> dict[str, float]:
        """フレーズスコアを計算
        
        RAKE: score(phrase) = sum(score(word) for word in phrase)
        """
        phrase_scores = {}
        
        for phrase in phrases:
            phrase_text = " ".join(phrase)
            score = sum(word_scores.get(word, 0) for word in phrase)
            
            # 既存のフレーズより高いスコアの場合のみ更新
            if phrase_text not in phrase_scores or score > phrase_scores[phrase_text]:
                phrase_scores[phrase_text] = score
        
        return phrase_scores
    
    def extract_with_scores(
        self,
        text: str,
        top_k: int = 10,
    ) -> dict[str, float]:
        """キーワードとスコアを辞書で返す
        
        Args:
            text: 入力テキスト
            top_k: 上位k件
            
        Returns:
            {keyword: score} の辞書
        """
        results = self.extract(text, top_k, return_scores=True)
        return dict(results)
    
    def extract_batch(
        self,
        texts: list[str],
        top_k: int = 10,
    ) -> list[list[str]]:
        """バッチでキーワード抽出
        
        Args:
            texts: テキストリスト
            top_k: 各テキストの上位k件
            
        Returns:
            キーワードリストのリスト
        """
        return [self.extract(text, top_k) for text in texts]


class TFIDFKeywordExtractor:
    """TF-IDFベースのキーワード抽出器
    
    コーパス全体を考慮したTF-IDFスコアでキーワードを抽出。
    """
    
    def __init__(
        self,
        stopwords: set[str] | None = None,
        min_df: int = 1,
        max_df: float = 0.95,
    ) -> None:
        """初期化
        
        Args:
            stopwords: ストップワード
            min_df: 最小文書頻度
            max_df: 最大文書頻度（割合）
        """
        self.stopwords = stopwords or RAKEKeywordExtractor.ENGLISH_STOPWORDS
        self.min_df = min_df
        self.max_df = max_df
        
        self._vectorizer = None
        self._feature_names = None
    
    def fit(self, texts: list[str]) -> "TFIDFKeywordExtractor":
        """コーパスで学習
        
        Args:
            texts: テキストリスト
            
        Returns:
            self
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self._vectorizer = TfidfVectorizer(
            stop_words=list(self.stopwords),
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 3),
        )
        
        self._vectorizer.fit(texts)
        self._feature_names = self._vectorizer.get_feature_names_out()
        
        return self
    
    def extract(
        self,
        text: str,
        top_k: int = 10,
    ) -> list[str]:
        """キーワードを抽出
        
        Args:
            text: 入力テキスト
            top_k: 上位k件
            
        Returns:
            キーワードリスト
        """
        if self._vectorizer is None:
            raise ValueError("fit() must be called before extract()")
        
        tfidf_vector = self._vectorizer.transform([text])
        scores = tfidf_vector.toarray()[0]
        
        # スコアでソート
        indices = scores.argsort()[::-1][:top_k]
        
        return [self._feature_names[i] for i in indices if scores[i] > 0]
    
    def extract_with_scores(
        self,
        text: str,
        top_k: int = 10,
    ) -> dict[str, float]:
        """キーワードとスコアを辞書で返す"""
        if self._vectorizer is None:
            raise ValueError("fit() must be called before extract()")
        
        tfidf_vector = self._vectorizer.transform([text])
        scores = tfidf_vector.toarray()[0]
        
        indices = scores.argsort()[::-1][:top_k]
        
        return {
            self._feature_names[i]: scores[i]
            for i in indices
            if scores[i] > 0
        }
