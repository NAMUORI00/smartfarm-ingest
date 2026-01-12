"""
재현성 보장을 위한 유틸리티 모듈

학술적 표준을 충족하기 위해 모든 랜덤 생성을 제어합니다.
"""

from __future__ import annotations

import random
import os
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int] = None) -> int:
    """
    모든 랜덤 시드를 고정하여 재현성 보장.
    
    Args:
        seed: 시드 값 (None이면 42 사용)
    
    Returns:
        설정된 시드 값
    
    Example:
        >>> from dataset_pipeline.reproducibility import set_seed
        >>> set_seed(42)
        42
    """
    if seed is None:
        seed = 42
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # 환경 변수로도 설정 (일부 라이브러리용)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed


def get_llm_generation_params(config: dict, seed: Optional[int] = None) -> dict:
    """
    재현 가능한 LLM 생성 파라미터 반환.
    
    OpenAI, Anthropic 등은 seed 파라미터를 지원하여
    deterministic generation이 가능합니다.
    
    Args:
        config: LLM 설정 딕셔너리
        seed: LLM API용 seed (None이면 config에서 가져옴)
    
    Returns:
        seed가 포함된 generation params
    
    Example:
        >>> params = get_llm_generation_params(config, seed=42)
        >>> response = client.chat.completions.create(..., **params)
    """
    params = {
        'temperature': config.get('temperature', 0.7),
        'max_tokens': config.get('max_tokens', 2048),
    }
    
    # Seed 지원 (OpenAI, Anthropic)
    if seed is not None:
        params['seed'] = seed
    elif 'llm_seed' in config:
        params['seed'] = config['llm_seed']
    
    return params


def is_deterministic_mode(config: dict) -> bool:
    """
    결정론적 모드 활성화 여부 확인.
    
    Args:
        config: 전체 설정 딕셔너리
    
    Returns:
        deterministic 모드 여부
    """
    reproducibility = config.get('reproducibility', {})
    return reproducibility.get('deterministic', False)
