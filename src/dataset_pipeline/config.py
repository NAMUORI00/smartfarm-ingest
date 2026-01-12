"""
Configuration Management for Dataset Pipeline

설정 로딩 우선순위:
1. 환경 변수 (최우선)
2. config/secrets.yaml (민감 정보)
3. config/settings.yaml (기본 설정)

Example:
    >>> from dataset_pipeline.config import ConfigManager
    >>> config = ConfigManager()
    >>> model = config.get('llm.generator.model')
    >>> llm_config = config.get_llm_config('generator')
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigManager:
    """통합 설정 관리자 - secrets 및 환경변수 지원."""

    def __init__(
        self,
        config_path: str | Path = "config/settings.yaml",
        secrets_path: Optional[str | Path] = None,
        auto_load_secrets: bool = True,
    ):
        """
        ConfigManager 초기화.

        Args:
            config_path: settings.yaml 경로 (기본: config/settings.yaml)
            secrets_path: secrets.yaml 경로 (기본: config/secrets.yaml)
            auto_load_secrets: secrets.yaml 자동 로드 여부 (기본: True)
        """
        # 프로젝트 루트 기준 상대 경로 처리
        if not Path(config_path).is_absolute():
            project_root = self._find_project_root()
            self.config_path = project_root / config_path
        else:
            self.config_path = Path(config_path)

        if secrets_path:
            self.secrets_path = Path(secrets_path)
        else:
            self.secrets_path = self.config_path.parent / "secrets.yaml"

        # 기본 설정 로드
        self._base_config = self._load_yaml(self.config_path)

        # secrets.yaml 로드 (존재하고 auto_load 활성화 시)
        self._secrets: Dict[str, Any] = {}
        if auto_load_secrets and self.secrets_path.exists():
            self._secrets = self._load_yaml(self.secrets_path)

        # 설정 병합 (secrets가 base를 오버라이드)
        self._config = self._merge_configs(self._base_config, self._secrets)

        # 환경변수 치환 수행
        self._config = self._substitute_env_vars(self._config)

    def _find_project_root(self) -> Path:
        """setup.py 또는 pyproject.toml을 찾아 프로젝트 루트 탐색."""
        current = Path(__file__).resolve().parent

        # 최대 5단계 상위까지 탐색
        for _ in range(5):
            if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
                return current
            current = current.parent

        # 폴백: 모듈 상위 디렉토리
        return Path(__file__).resolve().parent.parent.parent

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """YAML 파일 로드."""
        if not path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """두 설정 딕셔너리 깊은 병합 (override가 우선)."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        재귀적으로 ${VAR} 및 ${VAR:-default} 치환.

        Examples:
            "${API_KEY}" -> os.environ['API_KEY']
            "${API_KEY:-default-key}" -> os.environ.get('API_KEY', 'default-key')
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._resolve_env_var(config)
        else:
            return config

    def _resolve_env_var(self, value: str) -> str:
        """${VAR} 또는 ${VAR:-default} 문자열 내 치환."""
        # 패턴: ${VAR} 또는 ${VAR:-default}
        pattern = r"\$\{([^}:]+)(?::-(.*?))?\}"

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else None

            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # 기본값 없는 미설정 변수는 빈 문자열 반환 (하위 호환성)
                return ""

        return re.sub(pattern, replacer, value)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        점 표기법으로 설정 값 조회.

        Args:
            key_path: 점으로 구분된 경로 (예: 'llm.generator.model')
            default: 키가 없을 때 반환할 기본값

        Returns:
            설정 값 또는 기본값

        Examples:
            >>> config.get('llm.generator.model')
            'gemini-2.5-flash'
            >>> config.get('llm.generator.temperature', 0.7)
            0.7
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_llm_config(self, role: str = "generator") -> Dict[str, Any]:
        """
        특정 역할에 대한 병합된 LLM 설정 조회.

        Args:
            role: 'generator', 'judge', 또는 커스텀 역할명

        Returns:
            {base_url, model, api_key, temperature, max_tokens, ...} 딕셔너리

        우선순위:
            1. 환경 변수 (API_KEY, OPENAI_BASE_URL)
            2. secrets.yaml (llm.<role>.api_key, llm.<role>.base_url)
            3. settings.yaml (llm.<role>.*)

        Examples:
            >>> config.get_llm_config('generator')
            {'base_url': 'https://...', 'model': 'gemini-2.5-flash', 'api_key': '***', ...}
        """
        # settings에서 기본 설정 조회
        llm_config = self.get(f"llm.{role}", {})

        if not llm_config:
            raise ValueError(f"LLM 설정을 찾을 수 없습니다 (role: {role})")

        # 환경 변수로 오버라이드 (최우선)
        result = llm_config.copy()

        # API Key 우선순위: API_KEY env > OPENAI_API_KEY env > config
        if "API_KEY" in os.environ:
            result["api_key"] = os.environ["API_KEY"]
        elif "OPENAI_API_KEY" in os.environ:
            result["api_key"] = os.environ["OPENAI_API_KEY"]

        # Base URL 우선순위: OPENAI_BASE_URL env > API_BASE_URL env > config
        if "OPENAI_BASE_URL" in os.environ:
            result["base_url"] = os.environ["OPENAI_BASE_URL"]
        elif "API_BASE_URL" in os.environ:
            result["base_url"] = os.environ["API_BASE_URL"]

        return result

    def get_huggingface_token(self) -> Optional[str]:
        """
        CGIAR 데이터셋 접근용 HuggingFace 토큰 조회.

        우선순위:
            1. HF_TOKEN 환경 변수
            2. HUGGINGFACE_HUB_TOKEN 환경 변수
            3. secrets.yaml (huggingface.token)
            4. None

        Returns:
            HuggingFace 토큰 또는 None
        """
        # 환경 변수 확인
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            return token

        # secrets 설정 확인
        return self.get("huggingface.token")

    def get_mqm_judges(self) -> List[Dict[str, Any]]:
        """
        MQM 판정 모델 설정 목록 조회.

        Returns:
            판정 모델 설정 리스트, mqm.judges 미정의 시 [llm.judge]로 폴백
        """
        judges = self.get("mqm.judges")

        if judges:
            return judges

        # 단일 judge로 폴백
        judge_config = self.get_llm_config("judge")
        return [{"name": "judge", **judge_config}]

    def debug_dump(self, mask_secrets: bool = True) -> str:
        """
        디버그용 설정 출력 (선택적 시크릿 마스킹).

        Args:
            mask_secrets: API 키 및 토큰 마스킹 여부 (기본: True)

        Returns:
            YAML 문자열 표현
        """
        config_copy = self._config.copy()

        if mask_secrets:
            config_copy = self._mask_secrets(config_copy)

        return yaml.dump(config_copy, default_flow_style=False, allow_unicode=True)

    def _mask_secrets(self, config: Any) -> Any:
        """재귀적으로 민감 값 마스킹."""
        sensitive_keys = {"api_key", "token", "password", "secret"}

        if isinstance(config, dict):
            return {
                k: "***MASKED***" if k.lower() in sensitive_keys else self._mask_secrets(v)
                for k, v in config.items()
            }
        elif isinstance(config, list):
            return [self._mask_secrets(item) for item in config]
        else:
            return config

    @property
    def raw_config(self) -> Dict[str, Any]:
        """환경변수 치환 후 원본 설정 딕셔너리 반환."""
        return self._config.copy()


# 전역 인스턴스를 위한 편의 함수
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    전역 ConfigManager 인스턴스 조회 (싱글톤 패턴).

    Returns:
        ConfigManager 인스턴스

    Example:
        >>> from dataset_pipeline.config import get_config
        >>> config = get_config()
        >>> model = config.get('llm.generator.model')
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config
