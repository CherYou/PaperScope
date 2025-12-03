#!/usr/bin/env python3
"""
论文数据统计脚本
用于统计/share/project/xionglei/data/NeuraIPS2023_papers目录下的论文数据
并更新all_papers.json, index.csv, scraping_progress.json文件
"""

import os
import json
import csv
import glob
from datetime import datetime
from pathlib import Path
import argparse


class PaperStatistics:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.all_papers_file = self.data_dir / "all_papers.json"
        self.index_file = self.data_dir / "index.csv"
        self.progress_file = self.data_dir / "scraping_progress.json"
        
    def scan_paper_directories(self):
        """扫描所有论文目录，收集论文信息"""
        papers = []
        index_data = []
        scraped_papers = set()
        
        # 获取所有论文目录（排除文件）
        paper_dirs = [d for d in self.data_dir.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        print(f"发现 {len(paper_dirs)} 个论文目录")
        
        for paper_dir in paper_dirs:
            try:
                # 检查是否有metadata.json文件
                metadata_file = paper_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 检查是否有PDF文件
                    pdf_files = list(paper_dir.glob("*.pdf"))
                    has_pdf = len(pdf_files) > 0
                    
                    # 构建论文信息
                    paper_info = {
                        "title": metadata.get("title", paper_dir.name),
                        "authors": metadata.get("authors", []),
                        "abstract": metadata.get("abstract", ""),
                        "pdf_link": metadata.get("pdf_link", ""),
                        "folder_path": str(paper_dir.relative_to(self.data_dir)),
                        "has_pdf": has_pdf,
                        "pdf_files": [f.name for f in pdf_files],
                        "scraped_at": metadata.get("scraped_at", "")
                    }
                    
                    papers.append(paper_info)
                    
                    # 添加到索引数据
                    index_data.append({
                        "paper_number": len(index_data) + 1,
                        "title": paper_info["title"],
                        "folder_path": paper_info["folder_path"]
                    })
                    
                    # 如果有PDF文件，认为已完成抓取
                    if has_pdf:
                        scraped_papers.add(paper_dir.name)
                        
                else:
                    print(f"警告: {paper_dir.name} 目录缺少metadata.json文件")
                    
            except Exception as e:
                print(f"处理目录 {paper_dir.name} 时出错: {e}")
                continue
        
        return papers, index_data, scraped_papers
    
    def update_all_papers_json(self, papers):
        """更新all_papers.json文件"""
        print(f"更新 {self.all_papers_file}")
        with open(self.all_papers_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"已更新all_papers.json，包含 {len(papers)} 篇论文")
    
    def update_index_csv(self, index_data):
        """更新index.csv文件"""
        print(f"更新 {self.index_file}")
        with open(self.index_file, 'w', newline='', encoding='utf-8') as f:
            if index_data:
                writer = csv.DictWriter(f, fieldnames=["paper_number", "title", "folder_path"])
                writer.writeheader()
                writer.writerows(index_data)
        print(f"已更新index.csv，包含 {len(index_data)} 条记录")
    
    def update_scraping_progress(self, scraped_papers):
        """更新scraping_progress.json文件"""
        print(f"更新 {self.progress_file}")
        
        # 读取现有进度（如果存在）
        existing_progress = {}
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    existing_progress = json.load(f)
            except:
                existing_progress = {}
        
        # 更新进度信息
        progress_data = {
            "scraped_papers": list(scraped_papers),
            "total_scraped": len(scraped_papers),
            "last_updated": datetime.now().isoformat()
        }
        
        # 保留现有的其他信息
        if "start_time" in existing_progress:
            progress_data["start_time"] = existing_progress["start_time"]
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        print(f"已更新scraping_progress.json，记录 {len(scraped_papers)} 篇已抓取论文")
    
    def generate_statistics_report(self, papers):
        """生成统计报告"""
        total_papers = len(papers)
        papers_with_pdf = sum(1 for p in papers if p["has_pdf"])
        papers_without_pdf = total_papers - papers_with_pdf
        
        # 统计作者信息
        all_authors = []
        for paper in papers:
            all_authors.extend(paper.get("authors", []))
        unique_authors = len(set(all_authors))
        
        # 统计摘要长度
        abstract_lengths = [len(p.get("abstract", "")) for p in papers if p.get("abstract")]
        avg_abstract_length = sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0
        
        report = f"""
=== NeuraIPS2023 论文数据统计报告 ===
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

论文总数: {total_papers}
已下载PDF: {papers_with_pdf}
未下载PDF: {papers_without_pdf}
下载完成率: {papers_with_pdf/total_papers*100:.1f}%

作者统计:
- 总作者数（去重）: {unique_authors}
- 平均每篇论文作者数: {len(all_authors)/total_papers:.1f}

摘要统计:
- 有摘要的论文数: {len(abstract_lengths)}
- 平均摘要长度: {avg_abstract_length:.0f} 字符

文件状态:
- all_papers.json: 已更新
- index.csv: 已更新  
- scraping_progress.json: 已更新
"""
        
        print(report)
        
        # 保存报告到文件
        report_file = self.data_dir / "statistics_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"统计报告已保存到: {report_file}")
    
    def run(self):
        """运行统计分析"""
        print(f"开始分析论文数据: {self.data_dir}")
        
        # 扫描论文目录
        papers, index_data, scraped_papers = self.scan_paper_directories()
        
        # 更新文件
        self.update_all_papers_json(papers)
        self.update_index_csv(index_data)
        self.update_scraping_progress(scraped_papers)
        
        # 生成统计报告
        self.generate_statistics_report(papers)
        
        print("论文数据统计完成！")


def main():
    parser = argparse.ArgumentParser(description="论文数据统计脚本")
    parser.add_argument(
        "--data-dir", 
        default="/share/project/xionglei/data/NeuraIPS2023_papers",
        help="论文数据目录路径"
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return
    
    # 运行统计
    stats = PaperStatistics(args.data_dir)
    stats.run()


if __name__ == "__main__":
    main()