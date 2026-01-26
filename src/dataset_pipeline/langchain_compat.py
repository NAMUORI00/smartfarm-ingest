"""
LangChain 호환성 패치 모듈

PaddleOCR/paddlex가 구버전 LangChain 모듈을 사용하는 문제를 해결합니다.
이 모듈을 PaddleOCR 임포트 전에 먼저 임포트하면 호환성 문제가 해결됩니다.

필요한 패치:
- langchain.docstore.document.Document
- langchain.text_splitter.RecursiveCharacterTextSplitter

참고: https://github.com/PaddlePaddle/PaddleOCR/issues/17186
"""

import sys
from types import ModuleType


def patch_langchain_modules():
    """
    구버전 langchain 모듈 임포트를 최신 버전으로 패치합니다.
    
    PaddleOCR(paddlex) 내부에서 사용하는 구버전 임포트:
        from langchain.docstore.document import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    최신 LangChain에서는 이 모듈들이 다음으로 이동했습니다:
        langchain_core.documents.Document
        langchain_text_splitters.RecursiveCharacterTextSplitter
    """
    # Document 클래스 가져오기
    Document = None
    try:
        from langchain_core.documents import Document
    except ImportError:
        try:
            from langchain_community.docstore.document import Document
        except ImportError:
            # Document 클래스 직접 정의 (최소 구현)
            class Document:
                """Minimal Document class for compatibility."""
                def __init__(self, page_content: str = "", metadata: dict = None):
                    self.page_content = page_content
                    self.metadata = metadata or {}
    
    # RecursiveCharacterTextSplitter 가져오기
    RecursiveCharacterTextSplitter = None
    CharacterTextSplitter = None
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
        except ImportError:
            # 최소 구현
            class CharacterTextSplitter:
                """Minimal CharacterTextSplitter for compatibility."""
                def __init__(self, separator="\n\n", chunk_size=1000, chunk_overlap=200, **kwargs):
                    self.separator = separator
                    self.chunk_size = chunk_size
                    self.chunk_overlap = chunk_overlap
                
                def split_text(self, text):
                    return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            
            class RecursiveCharacterTextSplitter(CharacterTextSplitter):
                """Minimal RecursiveCharacterTextSplitter for compatibility."""
                def __init__(self, separators=None, **kwargs):
                    super().__init__(**kwargs)
                    self.separators = separators or ["\n\n", "\n", " ", ""]
    
    # langchain.docstore 패치
    _patch_docstore(Document)
    
    # langchain.text_splitter 패치
    _patch_text_splitter(RecursiveCharacterTextSplitter, CharacterTextSplitter)


def _patch_docstore(Document):
    """langchain.docstore 모듈 패치"""
    docstore_module = ModuleType('langchain.docstore')
    docstore_document_module = ModuleType('langchain.docstore.document')
    docstore_document_module.Document = Document
    docstore_module.document = docstore_document_module
    
    # sys.modules에 등록
    sys.modules['langchain.docstore'] = docstore_module
    sys.modules['langchain.docstore.document'] = docstore_document_module
    
    # 기존 langchain 모듈이 있으면 속성 추가
    if 'langchain' in sys.modules and hasattr(sys.modules['langchain'], '__file__'):
        langchain_module = sys.modules['langchain']
        if not hasattr(langchain_module, 'docstore'):
            langchain_module.docstore = docstore_module


def _patch_text_splitter(RecursiveCharacterTextSplitter, CharacterTextSplitter):
    """langchain.text_splitter 모듈 패치"""
    text_splitter_module = ModuleType('langchain.text_splitter')
    text_splitter_module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    text_splitter_module.CharacterTextSplitter = CharacterTextSplitter
    
    # 추가로 필요할 수 있는 클래스들
    text_splitter_module.TextSplitter = CharacterTextSplitter
    
    # sys.modules에 등록
    sys.modules['langchain.text_splitter'] = text_splitter_module
    
    # 기존 langchain 모듈이 있으면 속성 추가
    if 'langchain' in sys.modules and hasattr(sys.modules['langchain'], '__file__'):
        langchain_module = sys.modules['langchain']
        if not hasattr(langchain_module, 'text_splitter'):
            langchain_module.text_splitter = text_splitter_module


# 모듈 임포트 시 자동으로 패치 적용
patch_langchain_modules()
