#!/usr/bin/env python3
"""
LLM 客户端封装
支持 OpenAI-compatible 接口，带重试逻辑
"""
import json
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from config import Config, get_config


class LLMClient:
    """LLM 客户端，带重试和 JSON 解析"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )
        self.model = self.config.openai_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Optional[str]:
        """
        发送对话请求，返回文本回复

        Args:
            messages: 消息列表 [{"role": ..., "content": ...}]
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）

        Returns:
            回复文本，失败返回 None
        """
        last_error: Any = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as exc:
                last_error = exc
                print(f"  [LLM] 第 {attempt + 1}/{max_retries} 次请求失败: {exc}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        print(f"  [LLM] 全部重试失败，最后错误: {last_error}")
        return None

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """发送对话请求并解析 JSON 回复"""
        raw = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if raw is None:
            return None
        try:
            content = raw.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError as exc:
            print(f"  [LLM] JSON 解析失败: {exc}")
            print(f"  [LLM] 原始回复预览: {raw[:300]}")
            return None

    def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Optional[str]:
        """简化接口：单条 user prompt"""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def simple_chat_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """简化接口：单条 user prompt，返回解析后的 JSON"""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat_json(messages, temperature=temperature, max_tokens=max_tokens)
