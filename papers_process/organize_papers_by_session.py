#!/usr/bin/env python3
"""
论文按session分类整理脚本
处理ICLR2025_papers, NeurIPS2024_papers, ICML2025_papers中的论文
将每个会议的论文按照session分类，分类后每个session一个文件夹
该文件夹下每个论文一个文件夹：包含pdf文件和metadata.json文件，文件名为论文的id
"""

import os
import json
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperOrganizer:
    def __init__(self, data_root: str, output_root: str):
        """
        初始化论文整理器
        
        Args:
            data_root: 原始数据根目录
            output_root: 输出根目录
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.conferences = ['NeurIPS2023_papers']
        
    def clean_session_name(self, session: str) -> str:
        """
        清理session名称，使其适合作为文件夹名称
        
        Args:
            session: 原始session名称
            
        Returns:
            清理后的session名称
        """
        # 移除特殊字符，替换为下划线
        cleaned = re.sub(r'[^\w\s-]', '_', session)
        # 替换空格和多个下划线为单个下划线
        cleaned = re.sub(r'[\s_]+', '_', cleaned)
        # 移除开头和结尾的下划线
        cleaned = cleaned.strip('_')
        # 限制长度
        if len(cleaned) > 100:
            cleaned = cleaned[:100]
        return cleaned
    
    def extract_session_info(self, metadata: Dict) -> str:
        """
        从metadata中提取session信息
        
        Args:
            metadata: 论文元数据
            
        Returns:
            session名称
        """
        session = metadata.get('session', 'unknown')
        if not session or session.strip() == '':
            return 'unknown'
        return self.clean_session_name(session)
    
    def find_pdf_file(self, paper_dir: Path) -> Path:
        """
        在论文目录中查找PDF文件
        
        Args:
            paper_dir: 论文目录路径
            
        Returns:
            PDF文件路径，如果未找到返回None
        """
        pdf_files = list(paper_dir.glob('*.pdf'))
        if pdf_files:
            return pdf_files[0]  # 返回第一个找到的PDF文件
        return None
    
    def process_conference(self, conference: str) -> Dict[str, int]:
        """
        处理单个会议的论文
        
        Args:
            conference: 会议名称
            
        Returns:
            统计信息字典
        """
        logger.info(f"开始处理会议: {conference}")
        
        conference_dir = self.data_root / conference
        output_conference_dir = self.output_root / conference
        
        if not conference_dir.exists():
            logger.warning(f"会议目录不存在: {conference_dir}")
            return {'processed': 0, 'errors': 0, 'sessions': 0}
        
        # 创建输出目录
        output_conference_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'processed': 0, 'errors': 0, 'sessions': set()}
        
        # 遍历所有论文文件夹
        for paper_dir in conference_dir.iterdir():
            if not paper_dir.is_dir():
                continue
                
            # 跳过特殊文件夹
            if paper_dir.name in ['all_papers.json', 'index.csv']:
                continue
                
            try:
                # 读取metadata.json
                metadata_file = paper_dir / 'metadata.json'
                if not metadata_file.exists():
                    logger.warning(f"未找到metadata.json: {paper_dir}")
                    stats['errors'] += 1
                    continue
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 提取session信息
                session = self.extract_session_info(metadata)
                stats['sessions'].add(session)
                
                # 创建session目录
                session_dir = output_conference_dir / session
                session_dir.mkdir(parents=True, exist_ok=True)
                
                # 获取论文ID
                paper_id = metadata.get('id', paper_dir.name)
                
                # 创建论文目录
                paper_output_dir = session_dir / paper_id
                paper_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 复制metadata.json
                shutil.copy2(metadata_file, paper_output_dir / 'metadata.json')
                
                # 查找并复制PDF文件
                pdf_file = self.find_pdf_file(paper_dir)
                if pdf_file:
                    shutil.copy2(pdf_file, paper_output_dir / f"{paper_id}.pdf")
                    logger.debug(f"复制PDF: {pdf_file} -> {paper_output_dir / f'{paper_id}.pdf'}")
                else:
                    logger.warning(f"未找到PDF文件: {paper_dir}")
                
                stats['processed'] += 1
                logger.debug(f"处理完成: {paper_dir.name} -> {session}/{paper_id}")
                
            except Exception as e:
                logger.error(f"处理论文时出错 {paper_dir}: {e}")
                stats['errors'] += 1
        
        stats['sessions'] = len(stats['sessions'])
        logger.info(f"会议 {conference} 处理完成: {stats['processed']} 篇论文, {stats['sessions']} 个session, {stats['errors']} 个错误")
        return stats
    
    def generate_summary_report(self, all_stats: Dict[str, Dict]) -> None:
        """
        生成汇总报告
        
        Args:
            all_stats: 所有会议的统计信息
        """
        report_file = self.output_root / 'organization_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("论文按Session分类整理报告\n")
            f.write("=" * 50 + "\n\n")
            
            total_processed = 0
            total_errors = 0
            total_sessions = 0
            
            for conference, stats in all_stats.items():
                f.write(f"会议: {conference}\n")
                f.write(f"  处理论文数: {stats['processed']}\n")
                f.write(f"  Session数: {stats['sessions']}\n")
                f.write(f"  错误数: {stats['errors']}\n\n")
                
                total_processed += stats['processed']
                total_errors += stats['errors']
                total_sessions += stats['sessions']
            
            f.write("总计:\n")
            f.write(f"  总处理论文数: {total_processed}\n")
            f.write(f"  总Session数: {total_sessions}\n")
            f.write(f"  总错误数: {total_errors}\n")
        
        logger.info(f"汇总报告已生成: {report_file}")
    
    def organize_all_papers(self) -> None:
        """
        整理所有会议的论文
        """
        logger.info("开始整理所有会议的论文...")
        
        all_stats = {}
        
        for conference in self.conferences:
            stats = self.process_conference(conference)
            all_stats[conference] = stats
        
        # 生成汇总报告
        self.generate_summary_report(all_stats)
        
        logger.info("所有论文整理完成!")

def main():
    """主函数"""
    # 配置路径
    data_root = os.getenv("DATA_ROOT", "./data")
    output_root = os.getenv("OUTPUT_ROOT", "./data/organized_papers")
    
    # 创建整理器并执行
    organizer = PaperOrganizer(data_root, output_root)
    organizer.organize_all_papers()

if __name__ == "__main__":
    main()