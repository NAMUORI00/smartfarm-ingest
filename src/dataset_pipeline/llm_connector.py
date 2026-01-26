"""
LLM 커넥터 모듈

OpenAI-compatible API 규격으로 다양한 프로바이더 지원:
- OpenAI (GPT-4o, GPT-4-turbo 등)
- Feather AI (Qwen, Llama 등 오픈소스 모델)
- 기타 OpenAI-compatible 서버

사용법:
    # 기존 방식 (하위 호환성 유지)
    config = yaml.safe_load(open("config/settings.yaml"))
    connector = LLMConnector(config["llm"])
    
    # 새로운 방식 (ConfigManager 사용)
    from dataset_pipeline.config import ConfigManager
    config = ConfigManager()
    connector = LLMConnector.from_config(config)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from dataclasses import dataclass

from openai import OpenAI

if TYPE_CHECKING:
    from .config import ConfigManager

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None


@dataclass
class LLMConfig:
    """LLM 설정 데이터 클래스"""
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2048


class LLMConnector:
    """
    OpenAI-compatible API 커넥터
    
    generator와 judge 두 역할을 지원하며,
    각각 다른 프로바이더/모델을 설정할 수 있습니다.
    
    사용법:
        # 방법 1: 딕셔너리 설정 (기존 방식)
        connector = LLMConnector({"generator": {...}, "judge": {...}})
        
        # 방법 2: ConfigManager 사용 (권장)
        from dataset_pipeline.config import ConfigManager
        connector = LLMConnector.from_config(ConfigManager())
    """
    
    @classmethod
    def from_config(cls, config_manager: "ConfigManager") -> "LLMConnector":
        """
        ConfigManager에서 LLMConnector 생성 (권장 방식).
        
        Args:
            config_manager: ConfigManager 인스턴스
        
        Returns:
            LLMConnector 인스턴스
        
        Example:
            >>> from dataset_pipeline.config import ConfigManager
            >>> config = ConfigManager()
            >>> connector = LLMConnector.from_config(config)
        """
        generator_cfg = config_manager.get_llm_config("generator")
        judge_cfg = config_manager.get_llm_config("judge")
        
        return cls({
            "generator": generator_cfg,
            "judge": judge_cfg,
        })
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: settings.yaml의 llm 섹션
        """
        self._load_dotenv_files()
        self.generator_config = self._parse_config(config.get("generator", {}))
        self.judge_config = self._parse_config(config.get("judge", {}))
        
        self.generator_client = self._create_client(self.generator_config)
        self.judge_client = self._create_client(self.judge_config)

    def _load_dotenv_files(self) -> None:
        """Load .env from common locations (repo root and dataset/)."""
        if load_dotenv is None:
            return
        try:
            repo_root = Path(__file__).resolve().parents[2]
            load_dotenv(repo_root / ".env", override=False)
            load_dotenv(repo_root / "dataset" / ".env", override=False)
        except Exception:
            # .env loading is best-effort; don't block runtime on env parsing issues.
            return
    
    def _parse_config(self, config: Dict[str, Any]) -> LLMConfig:
        """설정 파싱 및 환경변수 치환"""
        def resolve_env(value: str) -> str:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.environ.get(env_var, "")
            return value

        def resolve_base_url(value: str) -> str:
            # Prefer environment override for sensitive/internal endpoints.
            return (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("API_BASE_URL")
                or resolve_env(value)
            )

        def resolve_api_key(value: str) -> str:
            resolved = resolve_env(value)
            if resolved:
                return resolved
            # Backward/compat: allow common env var names.
            return os.environ.get("API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        
        return LLMConfig(
            base_url=resolve_base_url(config.get("base_url", "https://api.openai.com/v1")),
            model=config.get("model", "gpt-4o"),
            api_key=resolve_api_key(config.get("api_key", "")),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
        )
    
    def _create_client(self, config: LLMConfig) -> OpenAI:
        """OpenAI 클라이언트 생성"""
        return OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
    
    def generate(
        self,
        prompt: str,
        role: str = "generator",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            role: "generator" 또는 "judge"
            system_prompt: 시스템 프롬프트 (선택)
            **kwargs: 추가 파라미터
        
        Returns:
            생성된 텍스트
        """
        if role == "generator":
            client = self.generator_client
            config = self.generator_config
        elif role == "judge":
            client = self.judge_client
            config = self.judge_config
        else:
            raise ValueError(f"Unknown role: {role}. Use 'generator' or 'judge'.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=kwargs.get("temperature", config.temperature),
            max_tokens=kwargs.get("max_tokens", config.max_tokens),
        )
        
        return response.choices[0].message.content
    
    def generate_batch(
        self,
        prompts: List[str],
        role: str = "generator",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """배치 생성 (순차 처리)"""
        return [
            self.generate(prompt, role, system_prompt, **kwargs)
            for prompt in prompts
        ]
