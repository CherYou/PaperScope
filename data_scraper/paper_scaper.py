#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Scraper - 优化版本
支持断点续传、进度条显示、重试机制等功能
"""

import requests
import os
import json
import csv
import time
import re
import argparse
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperScraper:
    def __init__(self, args):
        self.json_url = args.json_url
        self.output_dir = args.output_dir
        self.max_retries = args.max_retries
        self.delay = args.delay
        self.resume = args.resume
        
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 断点续传相关文件
        self.progress_file = os.path.join(self.output_dir, "scraping_progress.json")
        self.scraped_papers = self.load_progress() if self.resume else set()
        
        self.all_papers = []
        self.csv_rows = []
        # 根据已爬取的论文数量设置起始索引
        self.index = len(self.scraped_papers) + 1 if self.resume else 1

    def load_progress(self):
        """加载已爬取的论文记录"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    return set(progress_data.get('scraped_papers', []))
            except Exception as e:
                logger.warning(f"无法加载进度文件: {e}")
        return set()

    def save_progress(self, paper_id):
        """保存爬取进度"""
        self.scraped_papers.add(paper_id)
        progress_data = {
            'scraped_papers': list(self.scraped_papers),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存进度失败: {e}")

    def sanitize_filename(self, name):
        """移除文件名中的非法字符"""
        return re.sub(r'[\\/:"*?<>|]+', "_", name)

    def download_with_retry(self, url, max_retries=None, **kwargs):
        """带重试机制的下载函数"""
        if max_retries is None:
            max_retries = self.max_retries
            
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=self.headers, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"下载失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}, {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"下载最终失败: {e}")
                    raise

    def fetch_papers_data(self):
        """获取论文数据"""
        logger.info(f"正在获取论文数据: {self.json_url}")
        try:
            resp = self.download_with_retry(self.json_url, timeout=30)
            papers = resp.json()
            logger.info(f"成功获取 {len(papers)} 篇论文数据")
            return papers
        except Exception as e:
            logger.error(f"无法获取 JSON 数据: {e}")
            return []

    def process_paper(self, paper, idx):
        """处理单篇论文"""
        # 提取论文字段
        title = paper.get("title", "").strip()
        session = paper.get("primary_area", "").strip()
        authors = paper.get("author", [])
        abstract = paper.get("abstract", [])
        status = paper.get("status", "").strip()
        pdf_link = paper.get("pdf", "").strip() if "pdf" in paper else None
        paper_id = paper.get("id", "").strip()
        track = paper.get("track", "").strip() if "track" in paper else None

        # 创建以标题命名的文件夹路径
        folder_name = self.sanitize_filename(title)
        folder_path = os.path.join(self.output_dir, folder_name)
        
        # 增强的跳过逻辑：检查是否已经下载
        if self.resume:
            # 检查进度记录
            if paper_id in self.scraped_papers:
                logger.info(f"跳过已记录的论文: {title}")
                return False
            
            # 检查文件夹和PDF是否存在
            if os.path.exists(folder_path):
                metadata_path = os.path.join(folder_path, "metadata.json")
                pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                
                # 如果有metadata.json和PDF文件，认为已完成下载
                if os.path.exists(metadata_path) and pdf_files:
                    logger.info(f"跳过已下载的论文: {title} (发现PDF文件: {pdf_files[0]})")
                    # 将其添加到已爬取列表中，避免重复检查
                    self.scraped_papers.add(paper_id)
                    self.save_progress(paper_id)
                    return False
                elif os.path.exists(metadata_path):
                    logger.info(f"发现部分下载的论文: {title} (仅有metadata，继续下载PDF)")

        # 跳过撤回或拒绝的论文
        if status.lower() in ["withdraw", "reject"]:
            logger.info(f"跳过已撤回或拒绝的论文: {title} ({status})")
            return False

        # 构造元数据字典
        metadata = {
            "title": title,
            "session": session,
            "authors": authors,
            "abstract": abstract,
            "status": status,
            "pdf_link": pdf_link or "",
            "id": paper_id,
            "track": track or "",
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 创建文件夹（如果不存在）
        os.makedirs(folder_path, exist_ok=True)

        # 写入 metadata.json
        metadata_path = os.path.join(folder_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 尝试下载 PDF
        pdf_downloaded = False
        if pdf_link:
            try:
                pdf_resp = self.download_with_retry(
                    pdf_link, 
                    stream=True, 
                    timeout=30
                )
                pdf_filename = os.path.basename(paper_id) + ".pdf"
                pdf_path = os.path.join(folder_path, pdf_filename)
                
                with open(pdf_path, "wb") as pf:
                    for chunk in pdf_resp.iter_content(chunk_size=8192):
                        pf.write(chunk)
                
                pdf_downloaded = True
                logger.info(f"已下载 PDF: {self.index}, {title}")
                
            except Exception as e:
                logger.error(f"下载 PDF 失败 ({idx}, {title}): {e}")

        # 汇总信息
        self.all_papers.append(metadata)
        self.csv_rows.append([idx + 1, title, folder_path])  # 使用传入的索引

        # 保存进度
        self.save_progress(paper_id)
        
        return True

    def save_results(self):
        """保存最终结果"""
        # 写入 all_papers.json
        all_papers_path = os.path.join(self.output_dir, "all_papers.json")
        with open(all_papers_path, "w", encoding="utf-8") as af:
            json.dump(self.all_papers, af, ensure_ascii=False, indent=2)

        # 写入 index.csv
        index_csv_path = os.path.join(self.output_dir, "index.csv")
        with open(index_csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["编号", "标题", "文件夹路径"])
            writer.writerows(self.csv_rows)

        logger.info(f"结果已保存到: {self.output_dir}")

    def run(self):
        """运行爬虫"""
        logger.info("开始爬取论文...")
        
        # 获取论文数据
        papers = self.fetch_papers_data()
        if not papers:
            logger.error("没有获取到论文数据，退出程序")
            return

        # 过滤需要处理的论文（如果启用断点续传）
        if self.resume:
            papers_to_process = [p for p in papers if p.get("id", "") not in self.scraped_papers]
            logger.info(f"断点续传模式：需要处理 {len(papers_to_process)} 篇论文（总共 {len(papers)} 篇）")
            logger.info(f"从索引 {self.index} 开始继续爬取")
        else:
            papers_to_process = papers
            logger.info(f"全新爬取模式：需要处理 {len(papers_to_process)} 篇论文")

        # 使用进度条处理论文
        processed_count = 0
        with tqdm(total=len(papers_to_process), desc="爬取进度", unit="篇") as pbar:
            for idx, paper in enumerate(papers_to_process):
                try:
                    # 传递正确的全局索引而不是局部索引
                    current_index = self.index + idx
                    if self.process_paper(paper, current_index):
                        processed_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        '已处理': processed_count,
                        '索引': current_index,
                        '当前': paper.get('title', '')[:30] + '...' if len(paper.get('title', '')) > 30 else paper.get('title', '')
                    })
                    
                    # 延迟请求，避免过快
                    time.sleep(self.delay)
                    
                except KeyboardInterrupt:
                    logger.info("用户中断，保存当前进度...")
                    break
                except Exception as e:
                    logger.error(f"处理论文时出错: {e}")
                    continue

        # 保存最终结果
        self.save_results()
        
        logger.info(f"爬取完成！共处理论文: {processed_count}")
        if self.resume:
            logger.info(f"总计已爬取论文: {len(self.scraped_papers)}")


def main():
    parser = argparse.ArgumentParser(description='论文爬虫 - 支持断点续传和进度显示')
    
    parser.add_argument(
        '--json_url', 
        type=str, 
        default="https://raw.githubusercontent.com/Papercopilot/paperlists/main/icml/icml2025.json",
        help='论文数据的JSON URL'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="/share/project/xionglei/data/ICML2025_papers",
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--max_retries', 
        type=int, 
        default=3,
        help='最大重试次数'
    )
    
    parser.add_argument(
        '--delay', 
        type=float, 
        default=2.0,
        help='请求间隔时间（秒）'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        default=False,
        help='启用断点续传模式'
    )
    
    args = parser.parse_args()
    
    # 创建并运行爬虫
    scraper = PaperScraper(args)
    scraper.run()


if __name__ == "__main__":
    main()