#!/usr/bin/env python3
"""
search_verify 配置模块
从 .env 文件加载 API Key 和运行参数
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """search_verify 配置类"""

    def __init__(self, env_file: Optional[str] = None):
        """
        初始化配置，加载环境变量

        Args:
            env_file: .env 文件路径。若未指定，默认使用本文件所在目录的 .env
        """
        env_path = Path(env_file) if env_file else Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # LLM 配置
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Jina 配置
        self.jina_api_key: str = os.getenv("JINA_API_KEY", "")

        # 运行参数
        self.max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
        self.max_search_queries: int = int(os.getenv("MAX_SEARCH_QUERIES", "3"))
        self.jina_top_k: int = int(os.getenv("JINA_TOP_K", "8"))

    def validate(self) -> bool:
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.jina_api_key:
            missing.append("JINA_API_KEY")
        if missing:
            print(f"[Config] 缺少必要配置: {', '.join(missing)}")
            print("请在 .env 文件中设置这些值。")
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  openai_base_url={self.openai_base_url},\n"
            f"  openai_model={self.openai_model},\n"
            f"  openai_api_key={'*' * 8 if self.openai_api_key else 'NOT SET'},\n"
            f"  jina_api_key={'*' * 8 if self.jina_api_key else 'NOT SET'},\n"
            f"  max_workers={self.max_workers},\n"
            f"  max_search_queries={self.max_search_queries},\n"
            f"  jina_top_k={self.jina_top_k}\n"
            f")"
        )


_config: Optional[Config] = None


def get_config(env_file: Optional[str] = None) -> Config:
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config


def reset_config() -> None:
    global _config
    _config = None
