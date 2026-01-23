#!/usr/bin/env python3
"""
根据论文标题查找论文的 metadata 与 pdf_link。

支持两种数据源：
- 使用已生成的 JSONL 映射文件（推荐）
- 直接递归扫描指定目录下的 metadata.json（兜底）

命令行示例：
python code/paper_process/paper_lookup.py \
  --title "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization" \
  --jsonl ./paper_process/paper_map/paper_map.jsonl
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List


def _normalize_title(title: str) -> str:
    return (title or "").strip().lower()


class PaperLookup:
    """论文标题查找类"""

    def __init__(self, jsonl_path: Optional[str] = None, input_dir: Optional[str] = None):
        if not jsonl_path and not input_dir:
            raise ValueError("必须提供 jsonl_path 或 input_dir 之一")

        self.index: Dict[str, Dict[str, Any]] = {}
        if jsonl_path:
            self._load_from_jsonl(jsonl_path)
        else:
            self._load_from_dir(input_dir)  # type: ignore[arg-type]

    def _load_from_jsonl(self, jsonl_path: str) -> None:
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                title = obj.get("title")
                if not title:
                    continue
                key = _normalize_title(title)
                metadata = obj.get("metadata", {})
                item = {
                    "title": title,
                    "paper_path": obj.get("paper_path"),
                    "metadata_path": obj.get("metadata_path"),
                    "pdf_path": obj.get("pdf_path"),
                    "metadata": metadata,
                    "pdf_link": metadata.get("pdf_link"),
                }
                self.index[key] = item

    def _load_from_dir(self, input_dir: str) -> None:
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"输入目录不存在或不是目录: {input_dir}")
        for root, _, files in os.walk(input_dir):
            if "metadata.json" in files:
                meta_path = os.path.join(root, "metadata.json")
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    title = metadata.get("title")
                    if not title:
                        continue
                    key = _normalize_title(title)
                    pdf_link = metadata.get("pdf_link")
                    # 目录下尝试找 pdf
                    pdf_path = None
                    paper_id = metadata.get("id")
                    if paper_id:
                        candidate = os.path.join(root, f"{paper_id}.pdf")
                        if os.path.exists(candidate):
                            pdf_path = candidate
                    if not pdf_path:
                        for fname in os.listdir(root):
                            if fname.lower().endswith(".pdf"):
                                pdf_path = os.path.join(root, fname)
                                break
                    self.index[key] = {
                        "title": title,
                        "paper_path": root,
                        "metadata_path": meta_path,
                        "pdf_path": pdf_path,
                        "metadata": metadata,
                        "pdf_link": pdf_link,
                    }
                except Exception:
                    continue

    def find_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """精确查找（大小写与前后空格不敏感）"""
        key = _normalize_title(title)
        if key in self.index:
            return self.index[key]
        # 简单兜底：尝试包含匹配
        for k, v in self.index.items():
            if key in k or k in key:
                return v
        return None

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """模糊搜索：返回最多 limit 条包含匹配的候选"""
        q = _normalize_title(query)
        out: List[Dict[str, Any]] = []
        for k, v in self.index.items():
            if q in k:
                out.append(v)
                if len(out) >= limit:
                    break
        return out


def main():
    parser = argparse.ArgumentParser(description="根据论文标题查找 metadata 与 pdf_link")
    parser.add_argument("--title", required=True, help="论文标题")
    parser.add_argument("--jsonl", help="JSONL 映射文件路径")
    parser.add_argument("--input-dir", help="递归扫描的论文根目录（备选）")
    args = parser.parse_args()

    lookup = PaperLookup(jsonl_path=args.jsonl, input_dir=args.input_dir)
    result = lookup.find_by_title(args.title)
    if not result:
        candidates = lookup.search(args.title)
        print(json.dumps({
            "found": False,
            "message": "未找到精确匹配，返回部分候选",
            "candidates": candidates,
        }, ensure_ascii=False))
        return

    print(json.dumps({
        "found": True,
        "title": result.get("title"),
        "metadata_path": result.get("metadata_path"),
        "paper_path": result.get("paper_path"),
        "pdf_path": result.get("pdf_path"),
        "pdf_link": result.get("pdf_link"),
        "metadata": result.get("metadata"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()