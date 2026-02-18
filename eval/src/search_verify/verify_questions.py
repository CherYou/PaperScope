#!/usr/bin/env python3
"""
主运行脚本：对 induction / summary / solution 三类问题进行网络搜索验证

用法示例：
  # 验证 induction 数据，使用默认输入路径
  python verify_questions.py --type induction

  # 指定输入文件和输出目录
  python verify_questions.py --type solution \
      --input /path/to/solution_all_sampled_500.jsonl \
      --output-dir /path/to/output

  # 并发数 & 只验证前 N 条（用于测试）
  python verify_questions.py --type summary --workers 8 --limit 20

输出文件（写入 --output-dir，默认 ./verified_results/）：
  {type}_verified.jsonl          所有数据的验证结果（含标记）
  {type}_web_findable.jsonl      仅保留 verified=True 的条目（可通过网搜找到）
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 将本目录加入 sys.path，保证能 import 同目录模块 ──
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from config import Config, get_config, reset_config
from verifier import QuestionVerifier, VerifyResult

# ─────────────────────────── 默认路径 ───────────────────────────

_REPO_ROOT = _HERE.parents[4]          # /share/project/xionglei
_QA_DIR = _REPO_ROOT / "code" / "qa_constructor"

DEFAULT_INPUTS: Dict[str, str] = {
    "induction": str(
        _QA_DIR / "induction_data_constructor" / "results"
        / "induction_all_sampled_500.jsonl"
    ),
    "summary": str(
        _QA_DIR / "summary_data_constructor" / "results"
        / "summary_all_sampled_500.jsonl"
    ),
    "solution": str(
        _QA_DIR / "solution_data_constructor" / "results"
        / "solution_all_sampled_500.jsonl"
    ),
}

DEFAULT_OUTPUT_DIR = str(_HERE / "verified_results")

# ─────────────────────────── I/O 工具 ───────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [警告] {path}:{lineno} JSON 解析失败，已跳过: {exc}")
    return items


def save_jsonl(results: List[VerifyResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"  [保存] {path}  ({len(results)} 条)")


# ─────────────────────────── 核心逻辑 ───────────────────────────

def verify_dataset(
    data_type: str,
    input_path: str,
    output_dir: str,
    config: Config,
    workers: int = 4,
    limit: Optional[int] = None,
    resume: bool = True,
) -> None:
    """
    对整个数据集进行验证并写出结果

    Args:
        data_type: "induction" | "summary" | "solution"
        input_path: 输入 .jsonl 文件路径
        output_dir: 输出目录
        config: 配置对象
        workers: 并发线程数
        limit: 若设置，只处理前 N 条（用于测试）
        resume: 若为 True，跳过已写出结果中已处理的条目（断点续跑）
    """
    print(f"\n{'='*60}")
    print(f"  数据类型: {data_type}")
    print(f"  输入文件: {input_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  并发线程: {workers}")
    if limit:
        print(f"  限制条数: {limit}")
    print(f"{'='*60}\n")

    if not Path(input_path).exists():
        print(f"[错误] 输入文件不存在: {input_path}")
        return

    items = load_jsonl(input_path)
    if limit:
        items = items[:limit]
    print(f"[加载] 共 {len(items)} 条数据")

    # 断点续跑：读取已完成结果
    all_output_path = str(Path(output_dir) / f"{data_type}_verified.jsonl")
    done_indices: set = set()
    existing_results: List[VerifyResult] = []
    if resume and Path(all_output_path).exists():
        raw_done = load_jsonl(all_output_path)
        for rd in raw_done:
            vr = VerifyResult(
                question=rd.get("question", ""),
                answer=rd.get("answer", ""),
                data_type=rd.get("data_type", data_type),
                original_index=rd.get("original_index", -1),
                verified=rd.get("verified", False),
                confidence=rd.get("confidence", "unknown"),
                reason=rd.get("reason", ""),
                search_queries=rd.get("search_queries", []),
                search_results=rd.get("search_results", []),
                extra=rd.get("extra", {}),
            )
            existing_results.append(vr)
            done_indices.add(vr.original_index)
        print(f"[断点续跑] 已完成 {len(done_indices)} 条，跳过")

    # 过滤待处理条目
    pending = [
        (idx, item) for idx, item in enumerate(items)
        if idx not in done_indices
    ]
    print(f"[待处理] {len(pending)} 条\n")

    if not pending:
        print("[完成] 全部条目已处理。")
        _write_outputs(existing_results, data_type, output_dir)
        return

    verifier = QuestionVerifier(config)
    all_results: List[VerifyResult] = list(existing_results)
    completed = 0
    start_time = time.time()

    def _task(idx_item):
        idx, item = idx_item
        print(f"  >> [{data_type}] 处理第 {idx} 条...")
        return verifier.verify(item, data_type, idx)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_task, p): p[0] for p in pending}
        for future in as_completed(futures):
            try:
                vr = future.result()
                all_results.append(vr)
                completed += 1
                elapsed = time.time() - start_time
                avg = elapsed / completed
                remaining = avg * (len(pending) - completed)
                tag = "✓ 可找到" if vr.verified else "✗ 无法找到"
                print(
                    f"  [{completed}/{len(pending)}] idx={vr.original_index} "
                    f"{tag} (置信: {vr.confidence}) "
                    f"| 剩余约 {remaining:.0f}s"
                )
            except Exception as exc:
                print(f"  [错误] future 异常: {exc}")

    # 按原始索引排序后写出
    all_results.sort(key=lambda r: r.original_index)
    _write_outputs(all_results, data_type, output_dir)


def _write_outputs(
    all_results: List[VerifyResult], data_type: str, output_dir: str
) -> None:
    """将全量结果和仅标记为 verified 的子集分别写出"""
    verified_only = [r for r in all_results if r.verified]
    unverified = [r for r in all_results if not r.verified]

    all_path = str(Path(output_dir) / f"{data_type}_verified.jsonl")
    findable_path = str(Path(output_dir) / f"{data_type}_web_findable.jsonl")

    save_jsonl(all_results, all_path)
    save_jsonl(verified_only, findable_path)

    print(f"\n[统计] {data_type}:")
    print(f"  总条数: {len(all_results)}")
    print(f"  可通过网搜找到 (verified=True):  {len(verified_only)} 条 "
          f"({len(verified_only)/max(len(all_results),1)*100:.1f}%)")
    print(f"  无法通过网搜找到 (verified=False): {len(unverified)} 条 "
          f"({len(unverified)/max(len(all_results),1)*100:.1f}%)")

    # 置信度分布
    conf_dist: Dict[str, int] = {}
    for r in all_results:
        conf_dist[r.confidence] = conf_dist.get(r.confidence, 0) + 1
    print(f"  置信度分布: {conf_dist}")


# ─────────────────────────── CLI ───────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 LLM + Jina Search 验证问题数据集是否可通过网络搜索找到答案"
    )
    parser.add_argument(
        "--type",
        choices=["induction", "summary", "solution", "all"],
        default="all",
        help="数据类型（默认: all，即全部三类）",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="输入 .jsonl 文件路径（--type=all 时忽略，使用默认路径）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并发线程数（默认读取 .env MAX_WORKERS）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个数据集只处理前 N 条（用于测试）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续跑，从头重新处理",
    )
    parser.add_argument(
        "--env",
        default=None,
        help=".env 文件路径（默认使用 search_verify/.env）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    reset_config()
    env_path = args.env or str(_HERE / ".env")
    config = get_config(env_file=env_path)
    print(config)

    if not config.validate():
        sys.exit(1)

    workers = args.workers or config.max_workers

    # 确定待处理的数据类型列表
    if args.type == "all":
        types_to_run = list(DEFAULT_INPUTS.keys())
    else:
        types_to_run = [args.type]

    for dt in types_to_run:
        # 若 --type 非 all 且用户指定了 --input，使用用户路径
        if args.type != "all" and args.input:
            input_path = args.input
        else:
            input_path = DEFAULT_INPUTS[dt]

        verify_dataset(
            data_type=dt,
            input_path=input_path,
            output_dir=args.output_dir,
            config=config,
            workers=workers,
            limit=args.limit,
            resume=not args.no_resume,
        )

    print("\n[全部完成]")


if __name__ == "__main__":
    main()
