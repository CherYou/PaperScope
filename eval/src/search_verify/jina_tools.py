#!/usr/bin/env python3
"""
Jina Search / Reader 封装
提供网络搜索和网页内容读取功能
"""
import re
import time
from typing import Any, Dict, List, Optional

import requests

from config import Config, get_config


class SearchResult:
    """单条搜索结果"""

    def __init__(
        self,
        title: str,
        url: str,
        description: str = "",
        date: str = "",
    ):
        self.title = title
        self.url = url
        self.description = description
        self.date = date
        self.arxiv_id: str = self._extract_arxiv_id(url)

    def _extract_arxiv_id(self, url: str) -> str:
        patterns = [
            r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})',
            r'arxiv\.org/(?:abs|pdf)/([a-z\-]+/\d{7})',
        ]
        for pat in patterns:
            m = re.search(pat, url)
            if m:
                return m.group(1)
        return ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "date": self.date,
            "arxiv_id": self.arxiv_id,
        }

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title[:60]!r}, arxiv_id={self.arxiv_id!r})"


class JinaSearcher:
    """Jina Search API 封装"""

    SEARCH_URL = "https://s.jina.ai/"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.api_key = self.config.jina_api_key

    def search(
        self,
        query: str,
        top_k: int = 8,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> List[SearchResult]:
        """
        使用 Jina Search 搜索，返回结构化结果列表

        Args:
            query: 搜索关键词
            top_k: 最多返回的结果数
            max_retries: 重试次数
            retry_delay: 重试间隔（秒）

        Returns:
            SearchResult 列表
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Respond-With": "no-content",
        }
        params = {"q": query}
        last_error: Any = None

        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    self.SEARCH_URL,
                    headers=headers,
                    params=params,
                    timeout=60,
                )
                if resp.status_code == 200:
                    return self._parse(resp.text, top_k)
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"  [Jina] 第 {attempt + 1}/{max_retries} 次搜索失败: {last_error}")
            except Exception as exc:
                last_error = exc
                print(f"  [Jina] 第 {attempt + 1}/{max_retries} 次搜索异常: {exc}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

        print(f"  [Jina] 全部重试失败，最后错误: {last_error}")
        return []

    def _parse(self, text: str, top_k: int) -> List[SearchResult]:
        results: List[SearchResult] = []
        current: Dict[str, str] = {}
        current_idx: Optional[int] = None

        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r'\[(\d+)\]\s*(.*)', line)
            if not m:
                continue
            idx = int(m.group(1))
            content = m.group(2)

            if idx != current_idx and current:
                results.append(self._make(current))
                current = {}
            current_idx = idx

            if content.startswith("Title:"):
                current["title"] = content[6:].strip()
            elif content.startswith("URL Source:"):
                current["url"] = content[11:].strip()
            elif content.startswith("Description:"):
                current["description"] = content[12:].strip()
            elif content.startswith("Date:"):
                current["date"] = content[5:].strip()

        if current:
            results.append(self._make(current))

        return results[:top_k]

    def _make(self, data: Dict[str, str]) -> SearchResult:
        return SearchResult(
            title=data.get("title", ""),
            url=data.get("url", ""),
            description=data.get("description", ""),
            date=data.get("date", ""),
        )


class JinaReader:
    """Jina Reader API 封装，用于读取网页全文"""

    READ_URL = "https://r.jina.ai/"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.api_key = self.config.jina_api_key

    def read(
        self,
        url: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Optional[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        full_url = f"{self.READ_URL}{url}"
        last_error: Any = None

        for attempt in range(max_retries):
            try:
                resp = requests.get(full_url, headers=headers, timeout=120)
                if resp.status_code == 200:
                    return resp.text
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"  [JinaReader] 第 {attempt + 1}/{max_retries} 次读取失败: {last_error}")
            except Exception as exc:
                last_error = exc
                print(f"  [JinaReader] 第 {attempt + 1}/{max_retries} 次读取异常: {exc}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

        print(f"  [JinaReader] 全部重试失败，最后错误: {last_error}")
        return None


class JinaTools:
    """Jina 搜索 + 读取 组合工具"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.searcher = JinaSearcher(self.config)
        self.reader = JinaReader(self.config)

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        k = top_k if top_k is not None else self.config.jina_top_k
        return self.searcher.search(query, top_k=k)

    def read(self, url: str) -> Optional[str]:
        return self.reader.read(url)

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """将搜索结果格式化为 LLM 可读的文本"""
        if not results:
            return "（无搜索结果）"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] 标题: {r.title}")
            lines.append(f"    URL: {r.url}")
            if r.description:
                lines.append(f"    摘要: {r.description[:300]}")
            if r.date:
                lines.append(f"    日期: {r.date}")
            if r.arxiv_id:
                lines.append(f"    ArXiv ID: {r.arxiv_id}")
            lines.append("")
        return "\n".join(lines)
