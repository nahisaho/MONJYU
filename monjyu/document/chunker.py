# Text Chunker
"""
Text chunking for MONJYU.

Splits documents into manageable text units (chunks) with:
- Configurable chunk size (tokens)
- Overlap between chunks
- Section-aware chunking
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import tiktoken

from monjyu.document.models import AcademicPaperDocument, TextUnit


class TextChunkerProtocol(Protocol):
    """テキストチャンカープロトコル"""
    
    def chunk(
        self,
        document: AcademicPaperDocument,
        chunk_size: int = 300,
        overlap: int = 100,
    ) -> list[TextUnit]:
        """ドキュメントをチャンクに分割"""
        ...
    
    def count_tokens(self, text: str) -> int:
        """トークン数をカウント"""
        ...


class TextChunker:
    """テキストチャンカー
    
    ドキュメントを設定可能なサイズのチャンクに分割する。
    セクション境界を考慮した分割が可能。
    
    Example:
        >>> chunker = TextChunker(chunk_size=300, overlap=100)
        >>> text_units = chunker.chunk(document)
    """
    
    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 100,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        """初期化
        
        Args:
            chunk_size: チャンクサイズ（トークン数）
            overlap: オーバーラップ（トークン数）
            tokenizer_name: tiktoken tokenizer名
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    def chunk(
        self,
        document: AcademicPaperDocument,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[TextUnit]:
        """ドキュメントをチャンクに分割
        
        Args:
            document: 対象ドキュメント
            chunk_size: チャンクサイズ（省略時はデフォルト値）
            overlap: オーバーラップ（省略時はデフォルト値）
            
        Returns:
            TextUnitのリスト
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        
        chunks: list[TextUnit] = []
        chunk_index = 0
        
        # ドキュメントID生成
        doc_id = self._generate_document_id(document)
        
        # アブストラクトをチャンク
        if document.abstract:
            abstract_chunks = self._chunk_text(
                text=document.abstract,
                document_id=doc_id,
                chunk_size=chunk_size,
                overlap=overlap,
                section_type="abstract",
                start_index=chunk_index,
                metadata={
                    "title": document.title,
                    "section_heading": "Abstract",
                },
            )
            chunks.extend(abstract_chunks)
            chunk_index += len(abstract_chunks)
        
        # 各セクションをチャンク
        for section in document.sections:
            if not section.content.strip():
                continue
            
            section_chunks = self._chunk_text(
                text=section.content,
                document_id=doc_id,
                chunk_size=chunk_size,
                overlap=overlap,
                section_type=section.section_type,
                start_index=chunk_index,
                page_numbers=section.page_numbers,
                metadata={
                    "title": document.title,
                    "section_heading": section.heading,
                    "section_level": section.level,
                },
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # セクションがない場合はraw_textを使用
        if not chunks and document.raw_text:
            raw_chunks = self._chunk_text(
                text=document.raw_text,
                document_id=doc_id,
                chunk_size=chunk_size,
                overlap=overlap,
                section_type="body",
                start_index=chunk_index,
                metadata={"title": document.title},
            )
            chunks.extend(raw_chunks)
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        chunk_size: int,
        overlap: int,
        section_type: str,
        start_index: int,
        page_numbers: list[int] | None = None,
        metadata: dict | None = None,
    ) -> list[TextUnit]:
        """テキストをチャンクに分割
        
        Args:
            text: 分割対象テキスト
            document_id: ドキュメントID
            chunk_size: チャンクサイズ
            overlap: オーバーラップ
            section_type: セクションタイプ
            start_index: 開始インデックス
            page_numbers: ページ番号リスト
            metadata: メタデータ
            
        Returns:
            TextUnitのリスト
        """
        # テキストを正規化
        text = self._normalize_text(text)
        
        if not text.strip():
            return []
        
        # トークン化
        tokens = self._tokenizer.encode(text)
        
        if len(tokens) <= chunk_size:
            # チャンクサイズ以下なら分割不要
            return [
                TextUnit(
                    id=f"{document_id}_{start_index}",
                    text=text,
                    n_tokens=len(tokens),
                    document_id=document_id,
                    chunk_index=start_index,
                    start_char=0,
                    end_char=len(text),
                    section_type=section_type,
                    page_numbers=page_numbers or [],
                    metadata=metadata or {},
                )
            ]
        
        chunks: list[TextUnit] = []
        token_start = 0
        chunk_idx = start_index
        
        while token_start < len(tokens):
            # チャンク終了位置
            token_end = min(token_start + chunk_size, len(tokens))
            
            # チャンクトークン
            chunk_tokens = tokens[token_start:token_end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            
            # 文字位置を概算（正確な位置は計算コストが高い）
            char_start = len(self._tokenizer.decode(tokens[:token_start]))
            char_end = len(self._tokenizer.decode(tokens[:token_end]))
            
            chunks.append(
                TextUnit(
                    id=f"{document_id}_{chunk_idx}",
                    text=chunk_text,
                    n_tokens=len(chunk_tokens),
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    start_char=char_start,
                    end_char=char_end,
                    section_type=section_type,
                    page_numbers=page_numbers or [],
                    metadata=metadata or {},
                )
            )
            
            chunk_idx += 1
            
            # 次のチャンク開始位置（オーバーラップ考慮）
            if token_end >= len(tokens):
                break
            token_start = token_end - overlap
        
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """テキストを正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化されたテキスト
        """
        # 連続する空白を1つに
        text = re.sub(r"[ \t]+", " ", text)
        # 連続する改行を2つに
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 前後の空白を除去
        text = text.strip()
        return text
    
    def _generate_document_id(self, document: AcademicPaperDocument) -> str:
        """ドキュメントIDを生成
        
        Args:
            document: ドキュメント
            
        Returns:
            ドキュメントID
        """
        # DOIまたはarXiv IDがあれば使用
        if document.doi:
            return f"doi_{document.doi.replace('/', '_')}"
        if document.arxiv_id:
            return f"arxiv_{document.arxiv_id}"
        
        # なければファイル名のハッシュ
        content = f"{document.file_name}_{document.title}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"doc_{hash_value}"
    
    def count_tokens(self, text: str) -> int:
        """トークン数をカウント
        
        Args:
            text: テキスト
            
        Returns:
            トークン数
        """
        return len(self._tokenizer.encode(text))
    
    def estimate_chunks(
        self,
        document: AcademicPaperDocument,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> int:
        """チャンク数を概算
        
        Args:
            document: ドキュメント
            chunk_size: チャンクサイズ
            overlap: オーバーラップ
            
        Returns:
            推定チャンク数
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        
        total_tokens = self.count_tokens(document.full_text)
        
        if total_tokens <= chunk_size:
            return 1
        
        # 概算: (total - overlap) / (chunk_size - overlap)
        effective_chunk_size = chunk_size - overlap
        if effective_chunk_size <= 0:
            return total_tokens // chunk_size
        
        return max(1, (total_tokens - overlap) // effective_chunk_size + 1)
