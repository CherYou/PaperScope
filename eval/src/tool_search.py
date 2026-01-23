import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid

import os


BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")
BOCHA_API_URL = os.getenv("BOCHA_API_URL", "")


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    
    def bocha_web_search(self, query: str, count: int = 10, summary: bool = True):
        """
        使用 bocha.cn API 进行网页搜索
        
        Args:
            query: 搜索查询字符串
            count: 返回结果数量，默认10
            summary: 是否返回摘要，默认True
        
        Returns:
            格式化的搜索结果字符串
        """
        payload = json.dumps({
            "query": query,
            "summary": summary,
            "count": count
        })
        
        headers = {
            'Authorization': f'Bearer {BOCHA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        for i in range(5):
            try:
                response = requests.post(BOCHA_API_URL, headers=headers, data=payload, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                print(f"Search attempt {i+1} failed: {e}")
                if i == 4:
                    return f"Web search timeout, please try again later. Error: {str(e)}"
                continue
        
        try:
            results = response.json()
            
            # 检查 API 返回状态
            if results.get('code') != 200:
                error_msg = results.get('msg', 'Unknown error')
                return f"Search API error: {error_msg}"
            
            # 获取搜索结果
            data = results.get('data', {})
            web_pages = data.get('webPages', {})
            pages = web_pages.get('value', [])
            
            if not pages:
                return f"No results found for '{query}'. Try with a more general query."
            
            web_snippets = []
            for idx, page in enumerate(pages, 1):
                title = page.get('name', 'No title')
                url = page.get('url', '')
                display_url = page.get('displayUrl', url)
                snippet = page.get('snippet', '')
                
                # 获取额外信息（如果有）
                date_published = ""
                if 'datePublished' in page:
                    date_published = f"\nDate published: {page['datePublished']}"
                elif 'dateLastCrawled' in page:
                    date_published = f"\nLast crawled: {page['dateLastCrawled']}"
                
                site_name = ""
                if 'siteName' in page:
                    site_name = f"\nSource: {page['siteName']}"
                
                # 格式化结果
                redacted_version = f"{idx}. [{title}]({url}){date_published}{site_name}\n{snippet}"
                web_snippets.append(redacted_version)
            
            content = f"A web search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
            
        except json.JSONDecodeError:
            return f"Failed to parse search results for '{query}'."
        except Exception as e:
            return f"Error processing search results for '{query}': {str(e)}"

    def search_with_bocha(self, query: str):
        """搜索入口方法"""
        result = self.bocha_web_search(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # 单个查询
            response = self.search_with_bocha(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_bocha(q))
            response = "\n=======\n".join(responses)
            
        return response
