#!/usr/bin/env python3
"""
生成 paper 标题到 paper 路径和 metadata 信息的映射，并写入 JSONL 文件。

使用方式示例：
python paper_process/paper_map_generator.py \
  -i ./data/organized_papers/papers \
  --output-file ./paper_process/paper_map/paper_map.jsonl
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional


def find_metadata_files(input_dir: str) -> List[str]:
    """
    递归查找指定目录下的所有 metadata.json 文件
    """
    metadata_files: List[str] = []
    for root, dirs, files in os.walk(input_dir):
        # 跳过聚合文件夹（如果有）
        if os.path.basename(root) in {"paper_map", "results"}:
            continue
        if "metadata.json" in files:
            metadata_files.append(os.path.join(root, "metadata.json"))
    metadata_files.sort()
    return metadata_files


def find_pdf_file(paper_dir: str, metadata: Dict[str, Any]) -> Optional[str]:
    """
    在论文目录中查找 PDF 文件：优先使用 id 命名的文件，其次第一个 .pdf 文件
    """
    try:
        paper_id = metadata.get("id")
        if paper_id:
            candidate = os.path.join(paper_dir, f"{paper_id}.pdf")
            if os.path.exists(candidate):
                return candidate
        # 退化为目录下第一个 .pdf 文件
        for fname in os.listdir(paper_dir):
            if fname.lower().endswith(".pdf"):
                return os.path.join(paper_dir, fname)
    except Exception:
        pass
    return None


def build_mapping_item(metadata_path: str) -> Optional[Dict[str, Any]]:
    """
    构建单条映射记录：{title, paper_path, metadata_path, pdf_path, metadata}
    """
    try:
        paper_dir = os.path.dirname(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        title = metadata.get("title")
        if not title:
            # 无标题则跳过
            return None

        pdf_path = find_pdf_file(paper_dir, metadata)

        item = {
            "title": title,
            "paper_path": paper_dir,
            "metadata_path": metadata_path,
            "pdf_path": pdf_path,
            "metadata": metadata,
        }
        return item
    except Exception as e:
        print(f"读取 metadata 失败: {metadata_path} -> {e}")
        return None


def derive_output_path(base_output: str, input_dir: str) -> str:
    """
    与现有脚本保持一致的命名：<conference>_<basename(input_dir)>.jsonl
    conference 从 input_dir 的上一级目录名中提取并按 '_' 分割取首段
    """
    try:
        # 例如：.../NeurIPS2024_papers/natural_language_processing -> conference_dir = NeurIPS2024_papers
        conference_dir = input_dir.rstrip("/").split("/")[-2]
        confer_name = conference_dir.split("_")[0]
    except Exception:
        confer_name = "papers"

    suffix = f"{confer_name}_{os.path.basename(input_dir.rstrip('/'))}.jsonl"
    return base_output.replace(".jsonl", suffix)


def main():
    parser = argparse.ArgumentParser(description="生成论文标题到路径和元数据的映射(JSONL)")
    parser.add_argument(
        "--input-dir",
        "-i",
        default="./data/organized_papers",
        help="指定扫描的论文根目录(递归查找 metadata.json)",
    )
    parser.add_argument(
        "--output-file",
        default="./paper_process/paper_map/paper_map.jsonl",
        help="输出 JSONL 文件路径(文件名会自动补全)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"输入目录不存在或不是目录: {args.input_dir}")
        return

    # 组装输出路径并确保目录存在
    output_file = derive_output_path(args.output_file, args.input_dir)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 收集 metadata.json
    metadata_files = find_metadata_files(args.input_dir)
    if not metadata_files:
        print(f"未在目录中找到 metadata.json: {args.input_dir}")
        return

    total = 0
    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for meta in metadata_files:
            total += 1
            item = build_mapping_item(meta)
            if item is None:
                continue
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"扫描完成: {args.input_dir}")
    print(f"发现 metadata.json: {total}，成功写入: {written}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()