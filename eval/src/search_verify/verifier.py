#!/usr/bin/env python3
"""
核心验证逻辑
对 induction / summary / solution 三类问题，判断网络搜索是否能直接找到答案相关文献。

验证流程：
  1. 根据数据类型和内容，用 LLM 生成 1~N 条搜索 query
  2. 用 Jina Search 执行搜索，收集结果
  3. 再次调用 LLM 判断：搜索结果中是否能直接找到答案所需的关键文献 / 信息
  4. 返回带标记的 VerifyResult
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from config import Config, get_config
from jina_tools import JinaTools, SearchResult
from llm_client import LLMClient


# ─────────────────────────── 数据结构 ───────────────────────────

@dataclass
class VerifyResult:
    """单条数据的验证结果"""
    # 原始数据字段（透传）
    question: str
    answer: Any
    data_type: str                    # "induction" | "summary" | "solution"
    original_index: int               # 在原文件中的行号（0-based）

    # 验证相关字段
    verified: bool = False            # True = 搜索可直接找到答案相关文献
    confidence: str = "unknown"       # "high" | "medium" | "low" | "unknown"
    reason: str = ""                  # LLM 给出的判断理由
    search_queries: List[str] = field(default_factory=list)   # 使用的搜索 query
    search_results: List[Dict] = field(default_factory=list)  # 搜索结果摘要

    # 额外元数据（可选）
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ─────────────────────────── 提示模板 ───────────────────────────

SYSTEM_PROMPT_QUERY_GEN = (
    "你是一名科研信息检索专家。"
    "请根据给定的问题和答案，生成最有效的网络搜索 query，"
    "目标是通过 Google/Bing 等搜索引擎直接找到答案相关的学术论文或文献。"
    "只返回 JSON 数组，格式为 [\"query1\", \"query2\", ...]，不要其他内容。"
)

SYSTEM_PROMPT_JUDGE = (
    "你是一名严格的科研数据质量审核员。"
    "你的任务是：根据搜索结果，判断这些搜索结果是否足以让人通过网络公开资料"
    "直接找到答案所需的关键文献或信息（无需依赖私有数据库）。"
    "请只返回如下 JSON 格式，不要其他内容：\n"
    '{"verified": true/false, "confidence": "high"/"medium"/"low", "reason": "简短理由"}'
)

# ── induction ──
QUERY_GEN_TEMPLATE_INDUCTION = """\
【任务类型】induction（文献归纳检索）
【问题】
{question}

【标准答案（论文标题列表）】
{answer_titles}

请生成 {n} 条搜索 query，优先使用答案中的论文标题直接搜索。
每条 query 应能独立找到对应论文。直接返回 JSON 数组。"""

JUDGE_TEMPLATE_INDUCTION = """\
【任务类型】induction（文献归纳检索）
【问题】
{question}

【标准答案（论文标题列表）】
{answer_titles}

【网络搜索结果】
{search_results_text}

请判断：上述搜索结果中，是否能直接找到标准答案中列出的大部分（≥50%）论文？
如果搜索结果的标题或描述中明确出现了答案论文，即视为"能找到"。
返回 JSON 判断结果。"""

# ── summary ──
QUERY_GEN_TEMPLATE_SUMMARY = """\
【任务类型】summary（论文趋势总结）
【问题】
{question}

【涉及的核心论文（来源标题）】
{source_titles}

请生成 {n} 条搜索 query，目标是通过搜索找到问题涉及的这些论文的公开资料（arxiv/摘要等）。
直接返回 JSON 数组。"""

JUDGE_TEMPLATE_SUMMARY = """\
【任务类型】summary（论文趋势总结）
【问题】
{question}

【涉及的核心论文（来源标题）】
{source_titles}

【网络搜索结果】
{search_results_text}

请判断：上述搜索结果中，是否能找到问题所涉及的大部分（≥50%）论文的公开资料（如 arxiv 页面、官网等）？
返回 JSON 判断结果。"""

# ── solution ──
QUERY_GEN_TEMPLATE_SOLUTION = """\
【任务类型】solution（问题求解）
【问题（前500字）】
{question_snippet}

【答案关键词/方法名（从答案中提取）】
{answer_snippet}

请生成 {n} 条搜索 query，目标是找到答案中涉及的关键论文/技术文档。
直接返回 JSON 数组。"""

JUDGE_TEMPLATE_SOLUTION = """\
【任务类型】solution（问题求解）
【问题（前500字）】
{question_snippet}

【答案关键信息（前500字）】
{answer_snippet}

【网络搜索结果】
{search_results_text}

请判断：上述搜索结果中，是否能找到答案所依据的关键论文或技术文档？
（即：搜索结果中出现了与答案核心方法/论文名称明确匹配的内容）
返回 JSON 判断结果。"""


# ─────────────────────────── 验证器 ───────────────────────────

class QuestionVerifier:
    """
    使用 LLM + Jina Search 对单条数据进行验证
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.jina = JinaTools(self.config)
        self.llm = LLMClient(self.config)
        self.max_queries = self.config.max_search_queries

    # ── 公共入口 ──────────────────────────────────────────────

    def verify(self, item: Dict[str, Any], data_type: str, index: int) -> VerifyResult:
        """
        对单条数据进行验证

        Args:
            item: 原始数据字典（含 question / answer 等字段）
            data_type: "induction" | "summary" | "solution"
            index: 在文件中的行号（0-based）

        Returns:
            VerifyResult
        """
        question = item.get("question", "")
        answer = item.get("answer", "")

        result = VerifyResult(
            question=question,
            answer=answer,
            data_type=data_type,
            original_index=index,
            extra={k: v for k, v in item.items() if k not in ("question", "answer")},
        )

        try:
            if data_type == "induction":
                self._verify_induction(item, result)
            elif data_type == "summary":
                self._verify_summary(item, result)
            elif data_type == "solution":
                self._verify_solution(item, result)
            else:
                result.reason = f"未知数据类型: {data_type}"
        except Exception as exc:
            result.reason = f"验证过程异常: {exc}"
            result.verified = False
            result.confidence = "unknown"

        return result

    # ── induction ─────────────────────────────────────────────

    def _verify_induction(self, item: Dict[str, Any], result: VerifyResult) -> None:
        answer = item.get("answer", [])
        answer_titles = [
            a.get("title", "") for a in answer if isinstance(a, dict) and a.get("title")
        ] if isinstance(answer, list) else []

        if not answer_titles:
            result.reason = "答案中未找到论文标题列表"
            return

        titles_text = "\n".join(f"- {t}" for t in answer_titles)

        # 生成搜索 query
        queries = self._gen_queries(
            QUERY_GEN_TEMPLATE_INDUCTION.format(
                question=item.get("question", "")[:800],
                answer_titles=titles_text,
                n=min(self.max_queries, len(answer_titles)),
            )
        )
        # 若 LLM 生成失败，直接用论文标题搜索
        if not queries:
            queries = [f'arxiv "{t}"' for t in answer_titles[: self.max_queries]]

        result.search_queries = queries
        search_results = self._do_search(queries)
        result.search_results = [r.to_dict() for r in search_results]

        # LLM 判断
        results_text = self.jina.format_results_for_llm(search_results)
        judgement = self._judge(
            JUDGE_TEMPLATE_INDUCTION.format(
                question=item.get("question", "")[:800],
                answer_titles=titles_text,
                search_results_text=results_text,
            )
        )
        self._apply_judgement(result, judgement)

    # ── summary ───────────────────────────────────────────────

    def _verify_summary(self, item: Dict[str, Any], result: VerifyResult) -> None:
        source_titles = item.get("source_titles", [])
        # summary 数据可能把 source_titles 放在顶层或 answer 同组
        if not source_titles and isinstance(item.get("answer"), list):
            source_titles = [
                a.get("title", "") for a in item["answer"]
                if isinstance(a, dict) and a.get("title")
            ]

        if not source_titles:
            # 退化：直接用问题关键词搜索
            source_titles_text = "（无法解析来源论文标题）"
            queries = self._gen_queries(
                QUERY_GEN_TEMPLATE_SOLUTION.format(
                    question_snippet=item.get("question", "")[:500],
                    answer_snippet=str(item.get("answer", ""))[:300],
                    n=self.max_queries,
                )
            )
        else:
            source_titles_text = "\n".join(f"- {t}" for t in source_titles)
            queries = self._gen_queries(
                QUERY_GEN_TEMPLATE_SUMMARY.format(
                    question=item.get("question", "")[:800],
                    source_titles=source_titles_text,
                    n=min(self.max_queries, len(source_titles)),
                )
            )
            if not queries:
                queries = [f'arxiv "{t}"' for t in source_titles[: self.max_queries]]

        result.search_queries = queries
        search_results = self._do_search(queries)
        result.search_results = [r.to_dict() for r in search_results]

        results_text = self.jina.format_results_for_llm(search_results)
        judgement = self._judge(
            JUDGE_TEMPLATE_SUMMARY.format(
                question=item.get("question", "")[:800],
                source_titles=source_titles_text,
                search_results_text=results_text,
            )
        )
        self._apply_judgement(result, judgement)

    # ── solution ──────────────────────────────────────────────

    def _verify_solution(self, item: Dict[str, Any], result: VerifyResult) -> None:
        question_snippet = str(item.get("question", ""))[:500]
        answer_raw = item.get("answer", "")
        # answer 可能是 dict/str
        if isinstance(answer_raw, dict):
            answer_snippet = str(answer_raw.get("answer", answer_raw))[:500]
        else:
            answer_snippet = str(answer_raw)[:500]

        queries = self._gen_queries(
            QUERY_GEN_TEMPLATE_SOLUTION.format(
                question_snippet=question_snippet,
                answer_snippet=answer_snippet,
                n=self.max_queries,
            )
        )
        if not queries:
            queries = [question_snippet[:200]]

        result.search_queries = queries
        search_results = self._do_search(queries)
        result.search_results = [r.to_dict() for r in search_results]

        results_text = self.jina.format_results_for_llm(search_results)
        judgement = self._judge(
            JUDGE_TEMPLATE_SOLUTION.format(
                question_snippet=question_snippet,
                answer_snippet=answer_snippet,
                search_results_text=results_text,
            )
        )
        self._apply_judgement(result, judgement)

    # ── 内部工具方法 ───────────────────────────────────────────

    def _gen_queries(self, prompt: str) -> List[str]:
        """调用 LLM 生成搜索 query 列表"""
        raw = self.llm.simple_chat_json(prompt, system_prompt=SYSTEM_PROMPT_QUERY_GEN)
        if isinstance(raw, list):
            return [str(q) for q in raw if q]
        return []

    def _do_search(self, queries: List[str]) -> List[SearchResult]:
        """执行多条 query 的搜索，去重合并结果"""
        seen_urls: set = set()
        all_results: List[SearchResult] = []
        for q in queries:
            print(f"    [Search] {q!r}")
            results = self.jina.search(q)
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)
        return all_results

    def _judge(self, prompt: str) -> Optional[Dict[str, Any]]:
        """调用 LLM 进行最终判断，返回解析后的 JSON"""
        raw = self.llm.simple_chat_json(prompt, system_prompt=SYSTEM_PROMPT_JUDGE)
        if isinstance(raw, dict):
            return raw
        return None

    def _apply_judgement(
        self, result: VerifyResult, judgement: Optional[Dict[str, Any]]
    ) -> None:
        if judgement is None:
            result.verified = False
            result.confidence = "unknown"
            result.reason = "LLM 判断失败或返回格式错误"
            return
        result.verified = bool(judgement.get("verified", False))
        result.confidence = str(judgement.get("confidence", "unknown"))
        result.reason = str(judgement.get("reason", ""))
