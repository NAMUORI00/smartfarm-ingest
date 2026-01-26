"""
RAG 커넥터 모듈

RAFT (Zhang et al., 2024) 방법론 기반:
- 문서 청킹 및 임베딩
- 벡터 DB 저장/검색
- 컨텍스트 추출
- PDF/이미지 처리 및 OCR (EasyOCR/PaddleOCR 기반 - 고성능/경량)
"""

import os
import io
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# LangChain 호환성 패치 (PaddleOCR 임포트 전에 먼저 적용)
# PaddleOCR/paddlex가 구버전 langchain.docstore를 사용하는 문제 해결
# ============================================================================
try:
    from . import langchain_compat  # 패치 자동 적용
except ImportError:
    pass  # 독립 실행 시 무시

# ChromaDB 임포트
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# LangChain 임포트
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        DirectoryLoader,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# PDF 처리 라이브러리
try:
    import pypdf
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# 이미지 처리 라이브러리
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# EasyOCR (권장 - 고성능, Python 3.14 호환)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# PaddleOCR (대안 - 최고 성능, Python 3.12 이하 권장)
# 호환성 패치가 먼저 적용되어야 함
# Lazy import로 변경 - numpy 버전 충돌 방지
PADDLEOCR_AVAILABLE = False
PaddleOCR = None

def _check_paddleocr():
    """PaddleOCR 사용 가능 여부를 lazy하게 확인"""
    global PADDLEOCR_AVAILABLE, PaddleOCR
    if PaddleOCR is not None:
        return PADDLEOCR_AVAILABLE
    try:
        from paddleocr import PaddleOCR as _PaddleOCR
        import paddle
        PaddleOCR = _PaddleOCR
        PADDLEOCR_AVAILABLE = True
    except ImportError as e:
        logger.debug(f"PaddleOCR 사용 불가: {e}")
        PADDLEOCR_AVAILABLE = False
    except Exception as e:
        logger.debug(f"PaddleOCR 로드 중 오류: {e}")
        PADDLEOCR_AVAILABLE = False
    return PADDLEOCR_AVAILABLE

# Tesseract OCR (폴백)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# DeepSeek-OCR (고성능 VLM 기반 OCR - GPU 권장)
DEEPSEEK_AVAILABLE = False
_deepseek_model = None
_deepseek_processor = None

def _check_deepseek():
    """DeepSeek-OCR 사용 가능 여부를 lazy하게 확인"""
    global DEEPSEEK_AVAILABLE
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        DEEPSEEK_AVAILABLE = True
    except ImportError as e:
        logger.debug(f"DeepSeek-OCR 사용 불가: {e}")
        DEEPSEEK_AVAILABLE = False
    return DEEPSEEK_AVAILABLE


# ============================================================================
# DeepSeek-OCR 처리 클래스
# ============================================================================

class DeepSeekOCRProcessor:
    """
    DeepSeek-OCR 기반 텍스트 추출 (VLM 기반 고성능 OCR)

    특징:
    - 100+ 언어 지원, 한글 우수
    - 10-20x 토큰 압축으로 효율적
    - 복잡한 레이아웃, 테이블 처리 우수
    - 8GB VRAM (RTX 4060 Ti)에서 tiny/small 모드 구동 가능

    설치: pip install .[deepseek]
    """

    # 모델 크기별 해상도 매핑
    MODEL_SIZES = {
        "tiny": {"resolution": 512, "tokens": 64},
        "small": {"resolution": 640, "tokens": 100},
        "base": {"resolution": 1024, "tokens": 256},
        "large": {"resolution": 1280, "tokens": 400},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: DeepSeek-OCR 설정
                - model_id: HuggingFace 모델 ID (기본: deepseek-ai/DeepSeek-OCR)
                - model_size: tiny, small, base, large
                - quantize: none, 8bit, 4bit
                - device: cuda, cpu
                - use_vllm: vLLM 백엔드 사용 여부
        """
        config = config or {}
        self.model_id = config.get("model_id", "deepseek-ai/DeepSeek-OCR")
        self.model_size = config.get("model_size", "tiny")
        self.quantize = config.get("quantize", "4bit")
        self.device = config.get("device", "cuda")
        self.use_vllm = config.get("use_vllm", True)
        self.max_batch_size = config.get("max_batch_size", 4)

        self._model = None
        self._processor = None
        self._vllm_engine = None

    def _init_model(self):
        """모델 초기화 (Lazy Loading)"""
        if self._model is not None:
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(f"DeepSeek-OCR 모델 로딩 중... (크기: {self.model_size}, 양자화: {self.quantize})")

            # 양자화 설정
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }

            if self.quantize == "4bit":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.quantize == "8bit":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                load_kwargs["torch_dtype"] = torch.float16

            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs
            )

            logger.info(f"DeepSeek-OCR 모델 로딩 완료")
            return True

        except Exception as e:
            logger.error(f"DeepSeek-OCR 모델 로딩 실패: {e}")
            self._model = None
            self._processor = None
            return False

    def _get_resolution(self) -> int:
        """모델 크기에 따른 해상도 반환"""
        return self.MODEL_SIZES.get(self.model_size, self.MODEL_SIZES["tiny"])["resolution"]

    def _preprocess_image(self, image: "Image.Image") -> "Image.Image":
        """이미지 전처리 (해상도 조정)"""
        if not PIL_AVAILABLE:
            return image

        target_res = self._get_resolution()

        # RGB 변환
        if image.mode not in ('RGB',):
            image = image.convert('RGB')

        # 해상도 조정
        width, height = image.size
        max_dim = max(width, height)

        if max_dim > target_res:
            scale = target_res / max_dim
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    def extract_text(self, image_path: str) -> str:
        """이미지에서 텍스트 추출"""
        if not PIL_AVAILABLE:
            logger.error("PIL이 설치되지 않았습니다.")
            return ""

        if not self._init_model():
            logger.error("DeepSeek-OCR 모델 초기화 실패")
            return ""

        try:
            image = Image.open(image_path)
            return self.extract_text_from_pil_image(image)
        except Exception as e:
            logger.error(f"이미지 로드 실패: {image_path}, 오류: {e}")
            return ""

    def extract_text_from_pil_image(self, image: "Image.Image") -> str:
        """PIL Image 객체에서 텍스트 추출"""
        if not self._init_model():
            return ""

        try:
            import torch

            # 이미지 전처리
            image = self._preprocess_image(image)

            # 프롬프트 설정 (마크다운 형식 텍스트 추출)
            prompt = "Extract all text from this image in markdown format. Preserve the layout and structure."

            # 입력 준비
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 추론
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )

            # 디코딩
            generated_text = self._processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            return generated_text.strip()

        except Exception as e:
            logger.error(f"DeepSeek-OCR 추출 실패: {e}")
            return ""

    def get_info(self) -> Dict[str, Any]:
        """현재 설정 정보 반환"""
        return {
            "backend": "deepseek",
            "model_id": self.model_id,
            "model_size": self.model_size,
            "quantize": self.quantize,
            "device": self.device,
            "resolution": self._get_resolution(),
            "available": DEEPSEEK_AVAILABLE,
        }


# ============================================================================
# OCR 및 이미지 처리 클래스
# ============================================================================

class OCRProcessor:
    """
    OCR 처리 클래스 (다중 백엔드 지원)

    이미지에서 텍스트를 추출하며, 엣지 환경을 고려한 경량화 옵션 제공.

    백엔드 우선순위 (auto 모드):
    1. DeepSeek-OCR (GPU) - VLM 기반 고성능, 복잡한 레이아웃/테이블 우수
    2. EasyOCR (권장) - 빠르고 정확, 한글 지원 우수, Python 3.14 호환
    3. PaddleOCR (대안) - 최고 성능, Python 3.12 이하 권장
    4. Tesseract (폴백) - 위 라이브러리 미설치 시
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: OCR 설정 (언어, DPI, 백엔드 등)
        """
        config = config or {}
        self.lang = config.get("lang", "korean")
        self.dpi = config.get("dpi", 200)
        self.preprocessing = config.get("preprocessing", True)
        self.backend = config.get("backend", "auto")  # auto, deepseek, easyocr, paddleocr, tesseract
        self.use_gpu = config.get("use_gpu", False)  # 엣지 환경 기본 CPU

        # DeepSeek-OCR 설정
        self.deepseek_config = config.get("deepseek", {})
        self.deepseek_enabled = self.deepseek_config.get("enabled", False)

        # OCR 엔진 초기화
        self._easyocr_reader = None
        self._paddle_ocr = None
        self._deepseek_ocr = None
        self._init_ocr_engine()

    def _init_ocr_engine(self):
        """OCR 엔진 초기화"""
        if self.backend == "auto":
            # 우선순위: DeepSeek (GPU+enabled) > EasyOCR > PaddleOCR > Tesseract
            if self.deepseek_enabled and self.use_gpu and _check_deepseek():
                self.backend = "deepseek"
            elif EASYOCR_AVAILABLE:
                self.backend = "easyocr"
            elif _check_paddleocr():
                self.backend = "paddleocr"
            elif PYTESSERACT_AVAILABLE:
                self.backend = "tesseract"
            else:
                logger.warning("OCR 엔진이 설치되지 않았습니다. EasyOCR 설치 권장: pip install easyocr")
                self.backend = None

        if self.backend == "deepseek" and _check_deepseek():
            self._init_deepseek()
        elif self.backend == "easyocr" and EASYOCR_AVAILABLE:
            self._init_easyocr()
        elif self.backend == "paddleocr" and _check_paddleocr():
            self._init_paddle_ocr()

    def _init_deepseek(self):
        """DeepSeek-OCR 초기화 (지연 로딩)"""
        if self._deepseek_ocr is None and _check_deepseek():
            try:
                self._deepseek_ocr = DeepSeekOCRProcessor(self.deepseek_config)
                logger.info(f"DeepSeek-OCR 초기화 완료 (모델: {self.deepseek_config.get('model_size', 'tiny')})")
            except Exception as e:
                logger.error(f"DeepSeek-OCR 초기화 실패: {e}")
                self._deepseek_ocr = None
                # 폴백
                if EASYOCR_AVAILABLE:
                    self.backend = "easyocr"
                    self._init_easyocr()
                elif _check_paddleocr():
                    self.backend = "paddleocr"
                    self._init_paddle_ocr()
                elif PYTESSERACT_AVAILABLE:
                    self.backend = "tesseract"
    
    def _init_easyocr(self):
        """EasyOCR 초기화 (지연 로딩)"""
        if self._easyocr_reader is None and EASYOCR_AVAILABLE:
            # 언어 매핑
            lang_map = {
                "korean": ["ko", "en"],
                "kor": ["ko", "en"],
                "kor+eng": ["ko", "en"],
                "eng": ["en"],
                "english": ["en"],
                "chi_sim": ["ch_sim", "en"],
                "chinese": ["ch_sim", "en"],
                "japan": ["ja", "en"],
                "japanese": ["ja", "en"],
            }
            easy_langs = lang_map.get(self.lang.lower(), ["ko", "en"])
            
            try:
                self._easyocr_reader = easyocr.Reader(
                    easy_langs,
                    gpu=self.use_gpu,
                    verbose=False
                )
                logger.info(f"EasyOCR 초기화 완료 (언어: {easy_langs}, GPU: {self.use_gpu})")
            except Exception as e:
                logger.error(f"EasyOCR 초기화 실패: {e}")
                self._easyocr_reader = None
                # 폴백
                if _check_paddleocr():
                    self.backend = "paddleocr"
                    self._init_paddle_ocr()
                elif PYTESSERACT_AVAILABLE:
                    self.backend = "tesseract"
                    logger.info("Tesseract로 폴백")
    
    def _init_paddle_ocr(self):
        """PaddleOCR 초기화 (지연 로딩)"""
        if self._paddle_ocr is None and _check_paddleocr():
            # 한글 지원 언어 매핑
            lang_map = {
                "korean": "korean",
                "kor": "korean",
                "kor+eng": "korean",
                "eng": "en",
                "english": "en",
                "chi_sim": "ch",
                "chinese": "ch",
                "japan": "japan",
                "japanese": "japan",
            }
            paddle_lang = lang_map.get(self.lang.lower(), "korean")
            
            try:
                # PaddleOCR 버전별 호환 초기화
                import inspect
                sig = inspect.signature(PaddleOCR.__init__)
                supported_params = set(sig.parameters.keys())
                
                init_kwargs = {"lang": paddle_lang}
                
                if 'use_angle_cls' in supported_params:
                    init_kwargs['use_angle_cls'] = True
                if 'use_gpu' in supported_params:
                    init_kwargs['use_gpu'] = self.use_gpu
                if 'show_log' in supported_params:
                    init_kwargs['show_log'] = False
                if 'use_textline_orientation' in supported_params:
                    init_kwargs['use_textline_orientation'] = True
                
                self._paddle_ocr = PaddleOCR(**init_kwargs)
                logger.info(f"PaddleOCR 초기화 완료 (언어: {paddle_lang})")
            except Exception as e:
                logger.error(f"PaddleOCR 초기화 실패: {e}")
                self._paddle_ocr = None
                if PYTESSERACT_AVAILABLE:
                    self.backend = "tesseract"
                    logger.info("Tesseract로 폴백")
    
    def _preprocess_image(self, image: "Image.Image") -> "Image.Image":
        """이미지 전처리 (OCR 정확도 향상)"""
        if not PIL_AVAILABLE:
            return image
        
        # RGB 변환 (EasyOCR/PaddleOCR 모두 RGB 선호)
        if image.mode not in ('L', 'RGB'):
            image = image.convert('RGB')
        
        # 이미지 크기 조정 (너무 작으면 확대)
        width, height = image.size
        if width < 800:
            scale = 800 / width
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        이미지 파일에서 텍스트 추출
        
        Args:
            image_path: 이미지 파일 경로
        
        Returns:
            추출된 텍스트
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL이 설치되지 않아 이미지를 처리할 수 없습니다.")
            return ""
        
        try:
            image = Image.open(image_path)
            return self.extract_text_from_pil_image(image)
        except Exception as e:
            logger.error(f"이미지 로드 실패: {image_path}, 오류: {e}")
            return ""
    
    def extract_text_from_pil_image(self, image: "Image.Image") -> str:
        """PIL Image 객체에서 텍스트 추출"""
        if self.preprocessing:
            image = self._preprocess_image(image)

        # DeepSeek-OCR 사용 (VLM 기반 고성능)
        if self.backend == "deepseek" and self._deepseek_ocr is not None:
            return self._extract_with_deepseek(image)

        # EasyOCR 사용 (권장)
        if self.backend == "easyocr" and EASYOCR_AVAILABLE:
            return self._extract_with_easyocr(image)

        # PaddleOCR 사용 (대안)
        if self.backend == "paddleocr" and _check_paddleocr():
            return self._extract_with_paddle(image)

        # Tesseract 폴백
        if self.backend == "tesseract" and PYTESSERACT_AVAILABLE:
            return self._extract_with_tesseract(image)

        logger.warning("사용 가능한 OCR 엔진이 없습니다.")
        return ""

    def _extract_with_deepseek(self, image: "Image.Image") -> str:
        """DeepSeek-OCR로 텍스트 추출"""
        if self._deepseek_ocr is None:
            self._init_deepseek()

        if self._deepseek_ocr is None:
            # 폴백
            if EASYOCR_AVAILABLE:
                return self._extract_with_easyocr(image)
            elif _check_paddleocr():
                return self._extract_with_paddle(image)
            return ""

        try:
            result = self._deepseek_ocr.extract_text_from_pil_image(image)
            return result
        except Exception as e:
            logger.error(f"DeepSeek-OCR 추출 실패: {e}")
            # 폴백 시도
            if EASYOCR_AVAILABLE:
                logger.info("EasyOCR로 폴백 시도")
                return self._extract_with_easyocr(image)
            elif _check_paddleocr():
                logger.info("PaddleOCR로 폴백 시도")
                return self._extract_with_paddle(image)
            return ""
    
    def _extract_with_easyocr(self, image: "Image.Image") -> str:
        """EasyOCR로 텍스트 추출"""
        if self._easyocr_reader is None:
            self._init_easyocr()
        
        if self._easyocr_reader is None:
            return ""
        
        try:
            import numpy as np
            img_array = np.array(image)
            
            # OCR 수행
            result = self._easyocr_reader.readtext(img_array)
            
            # 결과 파싱: [(bbox, text, confidence), ...]
            if not result:
                return ""
            
            lines = [detection[1] for detection in result if detection[1]]
            return "\n".join(lines)
        
        except Exception as e:
            logger.error(f"EasyOCR 추출 실패: {e}")
            # 폴백 시도
            if _check_paddleocr() and self.backend != "paddleocr":
                logger.info("PaddleOCR로 폴백 시도")
                return self._extract_with_paddle(image)
            if PYTESSERACT_AVAILABLE:
                logger.info("Tesseract로 폴백 시도")
                return self._extract_with_tesseract(image)
            return ""
    
    def _extract_with_paddle(self, image: "Image.Image") -> str:
        """PaddleOCR로 텍스트 추출"""
        if self._paddle_ocr is None:
            self._init_paddle_ocr()
        
        if self._paddle_ocr is None:
            return ""
        
        try:
            # PIL Image를 numpy array로 변환
            import numpy as np
            img_array = np.array(image)
            
            # OCR 수행
            result = self._paddle_ocr.ocr(img_array, cls=True)
            
            # 결과 파싱
            if result is None or len(result) == 0:
                return ""
            
            lines = []
            for page in result:
                if page is None:
                    continue
                for line in page:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        lines.append(text)
            
            return "\n".join(lines)
        
        except Exception as e:
            logger.error(f"PaddleOCR 추출 실패: {e}")
            # Tesseract 폴백 시도
            if PYTESSERACT_AVAILABLE:
                logger.info("Tesseract로 폴백 시도")
                return self._extract_with_tesseract(image)
            return ""
    
    def _extract_with_tesseract(self, image: "Image.Image") -> str:
        """Tesseract로 텍스트 추출 (폴백)"""
    def _extract_with_tesseract(self, image: "Image.Image") -> str:
        """Tesseract로 텍스트 추출 (폴백)"""
        if not PYTESSERACT_AVAILABLE:
            return ""
        
        try:
            # Tesseract 언어 코드 매핑
            tesseract_lang_map = {
                "korean": "kor",
                "kor": "kor",
                "kor+eng": "kor+eng",
                "eng": "eng",
                "english": "eng",
                "chi_sim": "chi_sim",
                "chinese": "chi_sim",
                "japan": "jpn",
                "japanese": "jpn",
            }
            tess_lang = tesseract_lang_map.get(self.lang.lower(), "kor+eng")
            
            text = pytesseract.image_to_string(image, lang=tess_lang)
            return text.strip()
        
        except Exception as e:
            logger.error(f"Tesseract OCR 실패: {e}")
            return ""
    
    def get_backend_info(self) -> Dict[str, Any]:
        """현재 OCR 백엔드 정보 반환"""
        info = {
            "backend": self.backend,
            "lang": self.lang,
            "use_gpu": self.use_gpu,
            "deepseek_available": _check_deepseek(),
            "deepseek_enabled": self.deepseek_enabled,
            "easyocr_available": EASYOCR_AVAILABLE,
            "paddleocr_available": _check_paddleocr(),
            "tesseract_available": PYTESSERACT_AVAILABLE,
        }
        # DeepSeek 상세 정보 추가
        if self._deepseek_ocr is not None:
            info["deepseek_info"] = self._deepseek_ocr.get_info()
        return info


class PDFProcessor:
    """
    PDF 처리 클래스 (EasyOCR/PaddleOCR 기반)
    
    텍스트 기반 PDF와 이미지 기반 PDF(스캔 문서) 모두 처리.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: PDF 처리 설정
        """
        config = config or {}
        self.ocr_processor = OCRProcessor(config.get("ocr", {}))
        self.use_ocr_fallback = config.get("use_ocr_fallback", True)
        self.min_text_length = config.get("min_text_length", 50)  # OCR 폴백 임계값
        self.dpi = config.get("dpi", 200)
        self.extract_tables = config.get("extract_tables", True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDF에서 텍스트 추출 (페이지별)
        
        Args:
            pdf_path: PDF 파일 경로
        
        Returns:
            페이지별 텍스트 리스트 [{"page": 1, "text": ..., "method": ...}, ...]
        """
        pages = []
        
        # 1차: pypdf로 텍스트 추출 시도
        if PYPDF_AVAILABLE:
            try:
                pages = self._extract_with_pypdf(pdf_path)
            except Exception as e:
                logger.warning(f"pypdf 추출 실패: {e}")
        
        # 2차: pdfplumber로 테이블 포함 추출 시도
        if not pages and PDFPLUMBER_AVAILABLE:
            try:
                pages = self._extract_with_pdfplumber(pdf_path)
            except Exception as e:
                logger.warning(f"pdfplumber 추출 실패: {e}")
        
        # 3차: OCR 폴백 (텍스트가 부족한 경우)
        if self.use_ocr_fallback:
            pages = self._apply_ocr_fallback(pdf_path, pages)
        
        return pages
    
    def _extract_with_pypdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """pypdf로 텍스트 추출"""
        reader = PdfReader(pdf_path)
        pages = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({
                "page": i + 1,
                "text": text.strip(),
                "method": "pypdf",
                "has_text": len(text.strip()) >= self.min_text_length
            })
        
        return pages
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """pdfplumber로 텍스트 및 테이블 추출"""
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text_parts = []
                
                # 일반 텍스트
                text = page.extract_text() or ""
                text_parts.append(text)
                
                # 테이블 추출
                if self.extract_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = self._table_to_text(table)
                            text_parts.append(f"\n[표]\n{table_text}")
                
                combined_text = "\n".join(text_parts).strip()
                pages.append({
                    "page": i + 1,
                    "text": combined_text,
                    "method": "pdfplumber",
                    "has_text": len(combined_text) >= self.min_text_length
                })
        
        return pages
    
    def _table_to_text(self, table: List[List[Any]]) -> str:
        """테이블을 텍스트로 변환"""
        rows = []
        for row in table:
            cells = [str(cell) if cell else "" for cell in row]
            rows.append(" | ".join(cells))
        return "\n".join(rows)
    
    def _apply_ocr_fallback(self, pdf_path: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """텍스트가 부족한 페이지에 OCR 적용"""
        if not PDF2IMAGE_AVAILABLE:
            return pages
        
        # 텍스트가 부족한 페이지 확인
        pages_needing_ocr = [
            i for i, p in enumerate(pages) 
            if not p.get("has_text", False)
        ]
        
        if not pages_needing_ocr:
            return pages
        
        logger.info(f"OCR 필요 페이지: {len(pages_needing_ocr)}개")
        
        try:
            # PDF를 이미지로 변환
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi,
                first_page=min(pages_needing_ocr) + 1 if pages_needing_ocr else 1,
                last_page=max(pages_needing_ocr) + 1 if pages_needing_ocr else len(pages)
            )
            
            # OCR 수행
            image_idx = 0
            for page_idx in pages_needing_ocr:
                if image_idx < len(images):
                    ocr_text = self.ocr_processor.extract_text_from_pil_image(images[image_idx])
                    if ocr_text:
                        pages[page_idx]["text"] = ocr_text
                        pages[page_idx]["method"] = "ocr"
                        pages[page_idx]["has_text"] = True
                    image_idx += 1
        
        except Exception as e:
            logger.error(f"OCR 폴백 실패: {e}")
        
        return pages


class ImageLoader:
    """
    이미지 문서 로더 (RAGAnything 스타일)
    
    다양한 이미지 형식을 지원하며 OCR로 텍스트 추출.
    """
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp'}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.ocr_processor = OCRProcessor(config.get("ocr", {}))
    
    def load(self, path: str) -> List[Dict[str, Any]]:
        """
        이미지 파일 또는 디렉토리 로드
        
        Args:
            path: 이미지 파일 또는 디렉토리 경로
        
        Returns:
            문서 리스트 [{"content": ..., "metadata": ...}, ...]
        """
        documents = []
        
        if os.path.isdir(path):
            for file_path in Path(path).rglob("*"):
                if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    doc = self._load_single_image(str(file_path))
                    if doc:
                        documents.append(doc)
        else:
            doc = self._load_single_image(path)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _load_single_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """단일 이미지 로드"""
        text = self.ocr_processor.extract_text_from_image(image_path)
        
        if not text:
            logger.warning(f"이미지에서 텍스트를 추출하지 못함: {image_path}")
            return None
        
        return {
            "content": text,
            "metadata": {
                "source": image_path,
                "type": "image",
                "format": Path(image_path).suffix.lower()
            }
        }

# 간단한 인메모리 벡터 DB (chromadb 대안)
class SimpleVectorDB:
    """간단한 인메모리 벡터 DB"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add(self, ids, documents, embeddings=None, metadatas=None):
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.ids.append(ids[i])
            self.metadatas.append(metadatas[i] if metadatas else {})
            self.embeddings.append(embeddings[i] if embeddings else None)
    
    def query(self, query_texts, n_results=5):
        # 간단한 키워드 기반 검색 (실제로는 임베딩 유사도 사용)
        results = []
        query = query_texts[0].lower()
        
        for i, doc in enumerate(self.documents):
            if any(word in doc.lower() for word in query.split()):
                results.append((i, doc))
                if len(results) >= n_results:
                    break
        
        return {
            "documents": [[doc for _, doc in results]],
            "ids": [[self.ids[i] for i, _ in results]],
            "metadatas": [[self.metadatas[i] for i, _ in results]],
        }
    
    def get(self):
        return {
            "documents": self.documents,
            "ids": self.ids,
            "metadatas": self.metadatas,
        }


@dataclass
class RAGConfig:
    """RAG 설정 데이터 클래스"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db: str = "chromadb"
    top_k: int = 5
    # PDF/이미지 처리 설정 (PaddleOCR 기반)
    use_ocr_fallback: bool = True
    ocr_backend: str = "auto"      # auto, paddleocr, tesseract
    ocr_lang: str = "korean"       # PaddleOCR 언어 코드
    ocr_dpi: int = 200
    ocr_use_gpu: bool = False      # 엣지 환경 기본 CPU
    extract_tables: bool = True


class RAGConnector:
    """
    RAG 컨텍스트 관리자 (PaddleOCR 기반 - 최고 성능/경량)
    
    RAFT 방법론 기반으로 문서를 처리하고
    질문에 관련된 컨텍스트를 검색합니다.
    
    지원 형식:
    - 텍스트: .txt, .md
    - PDF: .pdf (텍스트 + PaddleOCR 폴백)
    - 이미지: .png, .jpg, .jpeg, .tiff, .bmp, .gif, .webp (PaddleOCR)
    """
    
    # 지원 파일 형식
    TEXT_FORMATS = {'.txt', '.md'}
    PDF_FORMATS = {'.pdf'}
    IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp'}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: settings.yaml의 rag 섹션
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain이 필요합니다: pip install langchain langchain-community")
        
        self.config = RAGConfig(
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 50),
            embedding_model=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            vector_db=config.get("vector_db", "chromadb"),
            top_k=config.get("top_k", 5),
            use_ocr_fallback=config.get("use_ocr_fallback", True),
            ocr_backend=config.get("ocr_backend", "auto"),
            ocr_lang=config.get("ocr_lang", "korean"),
            ocr_dpi=config.get("ocr_dpi", 200),
            ocr_use_gpu=config.get("ocr_use_gpu", False),
            extract_tables=config.get("extract_tables", True),
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        
        # PDF/이미지 프로세서 초기화 (PaddleOCR 기반)
        ocr_config = {
            "backend": self.config.ocr_backend,
            "lang": self.config.ocr_lang,
            "dpi": self.config.ocr_dpi,
            "use_gpu": self.config.ocr_use_gpu,
        }
        pdf_config = {
            "ocr": ocr_config,
            "use_ocr_fallback": self.config.use_ocr_fallback,
            "dpi": self.config.ocr_dpi,
            "extract_tables": self.config.extract_tables,
        }
        self.pdf_processor = PDFProcessor(pdf_config)
        self.image_loader = ImageLoader({"ocr": ocr_config})
        
        # 벡터 DB 초기화
        if CHROMADB_AVAILABLE and self.config.vector_db == "chromadb":
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )
            self.client = chromadb.Client()
            self.collection = None
            self.use_chromadb = True
        else:
            self.vector_db = SimpleVectorDB()
            self.use_chromadb = False
    
    def load_documents(self, path: str) -> List[Dict[str, Any]]:
        """
        문서 로드 및 청킹 (RAGAnything 스타일)
        
        다양한 형식의 문서를 자동 감지하여 처리합니다:
        - 텍스트 파일: 직접 로드
        - PDF: 텍스트 추출 + OCR 폴백
        - 이미지: OCR로 텍스트 추출
        
        Args:
            path: 문서 파일 또는 디렉토리 경로
        
        Returns:
            청크 리스트 [{"id": ..., "content": ..., "metadata": ...}, ...]
        """
        raw_documents = []
        
        if os.path.isdir(path):
            raw_documents = self._load_directory(path)
        else:
            raw_documents = self._load_single_file(path)
        
        # 청킹
        chunks = []
        for doc_idx, doc in enumerate(raw_documents):
            content = doc.get("content", "")
            if not content or not content.strip():
                continue
            
            splits = self.text_splitter.split_text(content)
            for chunk_idx, chunk_content in enumerate(splits):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                })
                
                chunks.append({
                    "id": f"doc_{doc_idx}_chunk_{chunk_idx}",
                    "content": chunk_content,
                    "metadata": metadata,
                })
        
        logger.info(f"총 {len(raw_documents)}개 문서에서 {len(chunks)}개 청크 생성")
        return chunks
    
    def _load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """디렉토리 내 모든 지원 파일 로드"""
        documents = []
        dir_path = Path(dir_path)
        
        all_formats = self.TEXT_FORMATS | self.PDF_FORMATS | self.IMAGE_FORMATS
        
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in all_formats:
                try:
                    docs = self._load_single_file(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"파일 로드 실패: {file_path}, 오류: {e}")
        
        return documents
    
    def _load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """단일 파일 로드 (형식 자동 감지)"""
        suffix = Path(file_path).suffix.lower()
        
        # 텍스트 파일
        if suffix in self.TEXT_FORMATS:
            return self._load_text_file(file_path)
        
        # PDF 파일
        elif suffix in self.PDF_FORMATS:
            return self._load_pdf_file(file_path)
        
        # 이미지 파일
        elif suffix in self.IMAGE_FORMATS:
            return self._load_image_file(file_path)
        
        else:
            logger.warning(f"지원하지 않는 파일 형식: {file_path}")
            return []
    
    def _load_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """텍스트 파일 로드"""
        try:
            # 여러 인코딩 시도
            for encoding in ['utf-8', 'cp949', 'euc-kr', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"텍스트 파일 인코딩 실패: {file_path}")
                return []
            
            return [{
                "content": content,
                "metadata": {
                    "source": file_path,
                    "type": "text",
                    "format": Path(file_path).suffix.lower()
                }
            }]
        except Exception as e:
            logger.error(f"텍스트 파일 로드 실패: {file_path}, 오류: {e}")
            return []
    
    def _load_pdf_file(self, file_path: str) -> List[Dict[str, Any]]:
        """PDF 파일 로드 (텍스트 + OCR)"""
        documents = []
        
        try:
            pages = self.pdf_processor.extract_text_from_pdf(file_path)
            
            for page_data in pages:
                text = page_data.get("text", "").strip()
                if text:
                    documents.append({
                        "content": text,
                        "metadata": {
                            "source": file_path,
                            "type": "pdf",
                            "page": page_data.get("page", 0),
                            "extraction_method": page_data.get("method", "unknown")
                        }
                    })
            
            logger.info(f"PDF 로드 완료: {file_path} ({len(documents)}페이지)")
        
        except Exception as e:
            logger.error(f"PDF 로드 실패: {file_path}, 오류: {e}")
            
            # LangChain 폴백
            if LANGCHAIN_AVAILABLE:
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        documents.append({
                            "content": doc.page_content,
                            "metadata": {
                                "source": file_path,
                                "type": "pdf",
                                "page": doc.metadata.get("page", 0),
                                "extraction_method": "langchain"
                            }
                        })
                except Exception as e2:
                    logger.error(f"LangChain PDF 폴백 실패: {e2}")
        
        return documents
    
    def _load_image_file(self, file_path: str) -> List[Dict[str, Any]]:
        """이미지 파일 로드 (OCR)"""
        documents = self.image_loader.load(file_path)
        
        if documents:
            logger.info(f"이미지 OCR 완료: {file_path}")
        
        return documents
    
    def index_documents(self, chunks: List[Dict[str, Any]], collection_name: str = "smartfarm"):
        """
        문서 인덱싱 (벡터 DB 저장)
        
        Args:
            chunks: 청크 리스트
            collection_name: 컬렉션 이름
        """
        if self.use_chromadb:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
            )
            
            self.collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["content"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
        else:
            # SimpleVectorDB 사용
            self.vector_db.add(
                ids=[c["id"] for c in chunks],
                documents=[c["content"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        관련 컨텍스트 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
        
        Returns:
            검색 결과 리스트
        """
        k = top_k or self.config.top_k
        
        if self.use_chromadb:
            if self.collection is None:
                raise ValueError("먼저 index_documents()를 호출하세요.")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
            )
            
            contexts = []
            for i, doc in enumerate(results["documents"][0]):
                contexts.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i] if results["ids"] else f"result_{i}",
                })
        else:
            results = self.vector_db.query([query], n_results=k)
            
            contexts = []
            for i, doc in enumerate(results["documents"][0]):
                contexts.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i] if results["ids"] else f"result_{i}",
                })
        
        return contexts
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """모든 청크 반환 (질문 생성용)"""
        if self.use_chromadb:
            if self.collection is None:
                return []
            
            results = self.collection.get()
            chunks = []
            for i, doc in enumerate(results["documents"]):
                chunks.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        else:
            results = self.vector_db.get()
            chunks = []
            for i, doc in enumerate(results["documents"]):
                chunks.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        
        return chunks