#!/usr/bin/env python3
"""
优化的随机游走文章选择器
针对30k节点2M边的大规模论文实体图进行优化
主要优化策略：
1. 预处理索引和缓存
2. 分层采样策略
3. 基于图结构的智能游走
4. 内存优化和并行处理
5. 集成性能监控功能
"""

import networkx as nx
import numpy as np
import json
import argparse
import time
import psutil
import os
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import combinations
import random
import heapq
from dataclasses import dataclass
from contextlib import nullcontext
from performance_monitor import PerformanceMonitor, TimedContext, benchmark_function


@dataclass
class GraphStats:
    """图统计信息"""
    total_nodes: int
    total_edges: int
    title_nodes: int
    entity_nodes: int
    avg_title_degree: float
    avg_entity_degree: float
    max_degree: int
    min_degree: int


class OptimizedRandomWalkSelector:
    """优化的随机游走文章选择器"""
    
    def __init__(self, graph_path: str, use_cache: bool = True, enable_monitoring: bool = True):
        """
        初始化优化的随机游走选择器
        
        Args:
            graph_path: GraphML文件路径
            use_cache: 是否使用缓存优化
            enable_monitoring: 是否启用性能监控
        """
        self.graph_path = graph_path
        self.use_cache = use_cache
        self.graph = None
        self.stats = None
        
        # 节点分类
        self.title_nodes = set()
        self.entity_nodes = set()
        
        # 预处理索引
        self.title_to_entities = {}  # 标题 -> 连接的实体集合
        self.entity_to_titles = defaultdict(set)  # 实体 -> 连接的标题集合
        self.entity_frequency = {}  # 实体频率
        self.title_entity_count = {}  # 标题的实体数量
        
        # 邻居缓存（用于加速随机游走）
        self.neighbor_cache = {}
        
        # 高频实体索引（用于智能采样）
        self.high_freq_entities = []
        self.entity_importance_scores = {}
        
        # 性能监控
        self.monitor = PerformanceMonitor(monitor_interval=0.5) if enable_monitoring else None
        
        # 内存优化设置
        self.max_walk_cache_size = 10000  # 最大缓存的游走路径数
        self.batch_size = 1000            # 批处理大小
        
        self.load_and_preprocess()
    
    def load_and_preprocess(self):
        """加载图并进行预处理"""
        if self.monitor:
            self.monitor.start_monitoring()
        
        with TimedContext(self.monitor, "graph_loading") if self.monitor else nullcontext():
            print(f"加载图文件: {self.graph_path}")
            start_time = time.time()
            
            self.graph = nx.read_graphml(self.graph_path)
            
            print(f"图加载完成，用时: {time.time() - start_time:.2f}秒")
            print("开始预处理...")
        
        with TimedContext(self.monitor, "node_classification") if self.monitor else nullcontext():
            self._classify_nodes()
        
        with TimedContext(self.monitor, "index_building") if self.monitor else nullcontext():
            self._build_indices()
        
        with TimedContext(self.monitor, "statistics_calculation") if self.monitor else nullcontext():
            self._calculate_stats()
        
        with TimedContext(self.monitor, "importance_scoring") if self.monitor else nullcontext():
            self._build_importance_scores()
        
        print(f"预处理完成，总用时: {time.time() - start_time:.2f}秒")
        self._print_stats()
        if self.monitor:
            self.monitor.add_custom_metric("preprocessing_memory", self.monitor.get_current_memory())
    
    def _classify_nodes(self):
        """分类节点"""
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type == 'title':
                self.title_nodes.add(node)
            else:
                self.entity_nodes.add(node)
    
    def _build_indices(self):
        """构建预处理索引"""
        print("构建索引...")
        
        # 构建标题-实体映射
        for title in self.title_nodes:
            connected_entities = set()
            for neighbor in self.graph.neighbors(title):
                if neighbor in self.entity_nodes:
                    connected_entities.add(neighbor)
                    self.entity_to_titles[neighbor].add(title)
            
            self.title_to_entities[title] = connected_entities
            self.title_entity_count[title] = len(connected_entities)
        
        # 计算实体频率
        for entity, titles in self.entity_to_titles.items():
            self.entity_frequency[entity] = len(titles)
        
        # 构建邻居缓存
        if self.use_cache:
            for node in self.graph.nodes():
                self.neighbor_cache[node] = list(self.graph.neighbors(node))
    
    def _calculate_stats(self):
        """计算图统计信息"""
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        title_degrees = [self.graph.degree(node) for node in self.title_nodes]
        entity_degrees = [self.graph.degree(node) for node in self.entity_nodes]
        
        self.stats = GraphStats(
            total_nodes=self.graph.number_of_nodes(),
            total_edges=self.graph.number_of_edges(),
            title_nodes=len(self.title_nodes),
            entity_nodes=len(self.entity_nodes),
            avg_title_degree=np.mean(title_degrees) if title_degrees else 0,
            avg_entity_degree=np.mean(entity_degrees) if entity_degrees else 0,
            max_degree=max(degrees) if degrees else 0,
            min_degree=min(degrees) if degrees else 0
        )
    
    def _build_importance_scores(self):
        """构建实体重要性分数"""
        print("计算实体重要性分数...")
        
        # 基于频率和度中心性计算重要性
        for entity in self.entity_nodes:
            frequency = self.entity_frequency.get(entity, 1)
            degree = self.graph.degree(entity)
            
            # 重要性 = 频率权重 * 度中心性权重
            importance = frequency * 0.7 + degree * 0.3
            self.entity_importance_scores[entity] = importance
        
        # 获取高频实体（前20%）
        sorted_entities = sorted(
            self.entity_importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_20_percent = max(1, len(sorted_entities) // 5)
        self.high_freq_entities = [entity for entity, _ in sorted_entities[:top_20_percent]]
        
        print(f"识别出 {len(self.high_freq_entities)} 个高重要性实体")
    
    def _print_stats(self):
        """打印统计信息"""
        print(f"\n=== 图统计信息 ===")
        print(f"总节点数: {self.stats.total_nodes:,}")
        print(f"总边数: {self.stats.total_edges:,}")
        print(f"标题节点数: {self.stats.title_nodes:,}")
        print(f"实体节点数: {self.stats.entity_nodes:,}")
        print(f"平均标题度: {self.stats.avg_title_degree:.2f}")
        print(f"平均实体度: {self.stats.avg_entity_degree:.2f}")
        print(f"最大度: {self.stats.max_degree}")
        print(f"最小度: {self.stats.min_degree}")
        
        # 内存使用情况
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"当前内存使用: {memory_mb:.2f} MB")
    
    def smart_random_walk(self, start_node: str, walk_length: int = 10, 
                         bias_towards_high_freq: float = 0.3) -> List[str]:
        """
        智能随机游走，偏向高频实体
        
        Args:
            start_node: 起始节点
            walk_length: 游走长度
            bias_towards_high_freq: 偏向高频实体的概率
            
        Returns:
            游走路径
        """
        if start_node not in self.graph:
            return []
        
        path = [start_node]
        current_node = start_node
        
        for _ in range(walk_length - 1):
            # 使用缓存的邻居
            neighbors = self.neighbor_cache.get(current_node, [])
            if not neighbors:
                break
            
            # 智能选择下一个节点
            if random.random() < bias_towards_high_freq and current_node in self.title_nodes:
                # 如果当前是标题节点，偏向选择高频实体
                high_freq_neighbors = [n for n in neighbors if n in self.high_freq_entities]
                if high_freq_neighbors:
                    next_node = random.choice(high_freq_neighbors)
                else:
                    next_node = random.choice(neighbors)
            else:
                # 普通随机选择
                next_node = random.choice(neighbors)
            
            path.append(next_node)
            current_node = next_node
        
        return path
    
    def stratified_sampling_walks(self, num_walks: int = 1000, walk_length: int = 10,
                                num_threads: int = 4) -> Dict[str, Set[str]]:
        """
        分层采样随机游走
        
        Args:
            num_walks: 总游走次数
            walk_length: 游走长度
            num_threads: 线程数
            
        Returns:
            文章到实体的映射
        """
        print(f"开始分层采样随机游走: {num_walks} 次游走，{walk_length} 步长")
        start_time = time.time()
        
        # 分层采样起始节点
        # 70% 从标题节点开始，30% 从高频实体开始
        title_starts = int(num_walks * 0.7)
        entity_starts = num_walks - title_starts
        
        start_nodes = []
        start_nodes.extend(random.choices(list(self.title_nodes), k=title_starts))
        start_nodes.extend(random.choices(self.high_freq_entities, k=entity_starts))
        
        # 并行执行游走
        article_entities = defaultdict(set)
        
        def process_walks(start_nodes_batch):
            local_article_entities = defaultdict(set)
            for start_node in start_nodes_batch:
                path = self.smart_random_walk(start_node, walk_length)
                
                # 提取路径中的文章和实体
                articles_in_path = [node for node in path if node in self.title_nodes]
                entities_in_path = [node for node in path if node in self.entity_nodes]
                
                # 建立文章-实体关联
                for article in articles_in_path:
                    for entity in entities_in_path:
                        local_article_entities[article].add(entity)
            
            return local_article_entities
        
        # 分批处理
        batch_size = max(1, num_walks // num_threads)
        batches = [start_nodes[i:i + batch_size] for i in range(0, len(start_nodes), batch_size)]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_walks, batches))
        
        # 合并结果
        for result in results:
            for article, entities in result.items():
                article_entities[article].update(entities)
        
        print(f"随机游走完成，用时: {time.time() - start_time:.2f}秒")
        print(f"收集到 {len(article_entities)} 篇文章的实体信息")
        
        return dict(article_entities)
    
    def find_articles_with_shared_entities_optimized(self, 
                                                   article_entities: Dict[str, Set[str]],
                                                   min_shared_entities: int = 3,
                                                   target_articles: int = 3,
                                                   max_combinations: int = 1000) -> List[Dict]:
        """
        优化的共享实体文章查找
        
        Args:
            article_entities: 文章到实体的映射
            min_shared_entities: 最少共享实体数
            target_articles: 目标文章数
            max_combinations: 最大组合数
            
        Returns:
            文章组合列表
        """
        print(f"寻找有 {min_shared_entities}+ 个共享实体的 {target_articles} 篇文章组合")
        start_time = time.time()
        
        # 预过滤：只考虑有足够实体的文章
        filtered_articles = {
            article: entities for article, entities in article_entities.items()
            if len(entities) >= min_shared_entities
        }
        
        print(f"预过滤后剩余 {len(filtered_articles)} 篇文章")
        
        if len(filtered_articles) < target_articles:
            print("文章数量不足，无法形成组合")
            return []
        
        # 使用堆来维护最佳组合
        best_combinations = []
        
        # 限制组合数量以避免内存爆炸
        articles = list(filtered_articles.keys())
        max_articles_to_check = min(len(articles), 500)  # 限制文章数量
        articles = articles[:max_articles_to_check]
        
        combinations_checked = 0
        
        for article_combo in combinations(articles, target_articles):
            if combinations_checked >= max_combinations:
                break
            
            # 计算共同实体
            common_entities = set(filtered_articles[article_combo[0]])
            for article in article_combo[1:]:
                common_entities &= filtered_articles[article]
            
            if len(common_entities) >= min_shared_entities:
                # 计算组合质量分数
                quality_score = self._calculate_combination_quality(
                    article_combo, common_entities, filtered_articles
                )
                
                combination_info = {
                    'articles': [
                        {
                            'title': article,
                            'total_entities': len(filtered_articles[article]),
                            'article_count': self.graph.nodes[article].get('article_count', 1),
                            'frequency': self.graph.nodes[article].get('frequency', 1)
                        }
                        for article in article_combo
                    ],
                    'common_entities': list(common_entities),
                    'common_entity_count': len(common_entities),
                    'quality_score': quality_score,
                    'total_articles': len(article_combo)
                }
                
                # 使用堆维护最佳结果
                if len(best_combinations) < max_combinations:
                    heapq.heappush(best_combinations, (quality_score, combination_info))
                elif quality_score > best_combinations[0][0]:
                    heapq.heapreplace(best_combinations, (quality_score, combination_info))
            
            combinations_checked += 1
            
            if combinations_checked % 10000 == 0:
                print(f"已检查 {combinations_checked} 个组合...")
        
        # 提取结果并按质量排序
        results = [combo for _, combo in sorted(best_combinations, reverse=True)]
        
        print(f"找到 {len(results)} 个符合条件的组合，用时: {time.time() - start_time:.2f}秒")
        return results
    
    def _calculate_combination_quality(self, articles: Tuple[str], 
                                     common_entities: Set[str],
                                     article_entities: Dict[str, Set[str]]) -> float:
        """
        计算文章组合的质量分数
        
        Args:
            articles: 文章元组
            common_entities: 共同实体
            article_entities: 文章实体映射
            
        Returns:
            质量分数
        """
        # 基于多个因素计算质量分数
        
        # 1. 共同实体数量权重
        common_count_score = len(common_entities) * 2
        
        # 2. 共同实体重要性权重
        importance_score = sum(
            self.entity_importance_scores.get(entity, 1) 
            for entity in common_entities
        ) / len(common_entities) if common_entities else 0
        
        # 3. 文章实体丰富度权重
        richness_score = sum(
            len(article_entities[article]) for article in articles
        ) / len(articles)
        
        # 4. 实体覆盖率权重
        total_unique_entities = set()
        for article in articles:
            total_unique_entities.update(article_entities[article])
        
        coverage_score = len(common_entities) / len(total_unique_entities) if total_unique_entities else 0
        
        # 综合分数
        quality_score = (
            common_count_score * 0.4 +
            importance_score * 0.3 +
            richness_score * 0.2 +
            coverage_score * 0.1
        )
        
        return quality_score
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def run_optimized_selection(self, 
                              num_walks: int = 2000,
                              walk_length: int = 12,
                              min_shared_entities: int = 5,
                              target_articles: int = 3,
                              max_results: int = 100,
                              num_threads: int = 4) -> List[Dict]:
        """
        运行优化的文章选择流程
        
        Args:
            num_walks: 随机游走次数
            walk_length: 游走长度
            min_shared_entities: 最少共享实体数
            target_articles: 目标文章数
            max_results: 最大结果数
            num_threads: 线程数
            
        Returns:
            优化的文章组合列表
        """
        print(f"\n=== 开始优化的文章选择流程 ===")
        print(f"参数: walks={num_walks}, length={walk_length}, min_entities={min_shared_entities}")
        
        start_memory = self.get_memory_usage()
        total_start_time = time.time()
        
        if self.monitor:
            self.monitor.set_phase("random_walks")
        
        # 1. 执行分层采样随机游走
        article_entities, walk_time = benchmark_function(
            self.stratified_sampling_walks,
            num_walks=num_walks,
            walk_length=walk_length,
            num_threads=num_threads,
            monitor=self.monitor,
            phase_name="stratified_walks"
        ) if self.monitor else (self.stratified_sampling_walks(
            num_walks=num_walks,
            walk_length=walk_length,
            num_threads=num_threads
        ), 0)
        
        if self.monitor:
            self.monitor.set_phase("article_combination_search")
            self.monitor.add_custom_metric("walk_results_count", len(article_entities))
        
        # 2. 寻找共享实体的文章组合
        combinations, search_time = benchmark_function(
            self.find_articles_with_shared_entities_optimized,
            article_entities=article_entities,
            min_shared_entities=min_shared_entities,
            target_articles=target_articles,
            max_combinations=max_results * 10,  # 搜索更多组合以获得更好结果
            monitor=self.monitor,
            phase_name="combination_search"
        ) if self.monitor else (self.find_articles_with_shared_entities_optimized(
            article_entities=article_entities,
            min_shared_entities=min_shared_entities,
            target_articles=target_articles,
            max_combinations=max_results * 10  # 搜索更多组合以获得更好结果
        ), 0)
        
        # 3. 限制结果数量
        if len(combinations) > max_results:
            combinations = combinations[:max_results]
        
        if self.monitor:
            self.monitor.set_phase("result_enrichment")
            self.monitor.add_custom_metric("combinations_found", len(combinations))
        
        # 4. 添加实体详细信息
        enriched_combinations, enrich_time = benchmark_function(
            self._enrich_with_entity_details,
            combinations,
            monitor=self.monitor,
            phase_name="result_enrichment"
        ) if self.monitor else (self._enrich_with_entity_details(combinations), 0)
        
        end_memory = self.get_memory_usage()
        total_time = time.time() - total_start_time
        
        if self.monitor:
            self.monitor.set_phase("completed")
            self.monitor.add_custom_metric("final_results_count", len(enriched_combinations))
        
        print(f"\n=== 优化选择完成 ===")
        print(f"总用时: {total_time:.2f}秒")
        print(f"内存增长: {end_memory - start_memory:.2f} MB")
        print(f"最终结果: {len(enriched_combinations)} 个文章组合")
        
        return enriched_combinations
    
    def _enrich_with_entity_details(self, combinations: List[Dict]) -> List[Dict]:
        """为组合添加实体详细信息"""
        print("添加实体详细信息...")
        
        for combo in combinations:
            entity_details = []
            for entity_name in combo['common_entities']:
                entity_info = {
                    'name': entity_name,
                    'node_type': self.graph.nodes[entity_name].get('node_type', 'unknown'),
                    'frequency': self.entity_frequency.get(entity_name, 1),
                    'article_count': len(self.entity_to_titles.get(entity_name, set())),
                    'importance_score': self.entity_importance_scores.get(entity_name, 0)
                }
                entity_details.append(entity_info)
            
            # 按重要性分数排序
            entity_details.sort(key=lambda x: x['importance_score'], reverse=True)
            combo['entity_details'] = entity_details
        
        return combinations
    
    def save_results(self, combinations: List[Dict], output_path: str):
        """保存结果到文件"""
        print(f"保存结果到: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for combo in combinations:
                json.dump(combo, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"结果已保存，共 {len(combinations)} 个组合")
    
    def print_summary(self, combinations: List[Dict], top_n: int = 5):
        """打印结果摘要"""
        print(f"\n=== 优化结果摘要 (前 {top_n} 个) ===")
        
        for i, combo in enumerate(combinations[:top_n]):
            print(f"\n组合 {i+1} (质量分数: {combo.get('quality_score', 0):.2f}):")
            print(f"  共同实体数: {combo['common_entity_count']}")
            print(f"  文章数: {combo['total_articles']}")
            
            print("  文章列表:")
            for article in combo['articles']:
                print(f"    - {article['title']} (实体数: {article['total_entities']})")
            
            print("  重要共同实体 (前5个):")
            for entity in combo['entity_details'][:5]:
                print(f"    - {entity['name']} (重要性: {entity['importance_score']:.2f}, "
                      f"频率: {entity['frequency']})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化的随机游走文章选择器')
    parser.add_argument('--graph_path','-g', type=str, required=True, help='GraphML文件路径')
    parser.add_argument('--output_path', type=str, default='', help='输出文件路径')
    parser.add_argument('--num_walks', type=int, default=10000, help='随机游走次数')
    parser.add_argument('--walk_length', type=int, default=100, help='随机游走长度')
    parser.add_argument('--min_common_entities', type=int, default=3, help='最小共同实体数')
    parser.add_argument('--min_articles', type=int, default=5, help='最小文章数')
    parser.add_argument('--max_results', type=int, default=100, help='最大结果数')
    parser.add_argument('--use_parallel','-up', action='store_true', help='使用并行处理')
    parser.add_argument('--random_seed', type=int, default=16, help='随机种子')
    parser.add_argument('--disable_monitoring','-dm', action='store_true', help='禁用性能监控')
    parser.add_argument('--performance_output', type=str, default='output/performance_2', help='性能监控输出目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if not args.disable_monitoring:
        os.makedirs(args.performance_output, exist_ok=True)
    
    try:
        # 初始化选择器
        selector = OptimizedRandomWalkSelector(
            args.graph_path, 
            enable_monitoring=not args.disable_monitoring
        )
        
        # 运行优化选择
        results = selector.run_optimized_selection(
            num_walks=args.num_walks,
            walk_length=args.walk_length,
            min_shared_entities=args.min_common_entities,
            target_articles=args.min_articles,
            max_results=args.max_results,
            num_threads=args.use_parallel if isinstance(args.use_parallel, int) else 4
        )
        
        # 保存结果
        selector.save_results(results, args.output_path)
        
        # 打印摘要
        selector.print_summary(results)
        
        # 保存性能监控结果
        if selector.monitor:
            selector.monitor.stop_monitoring()
            selector.monitor.print_summary()
            
            # 保存性能指标
            performance_json = os.path.join(args.performance_output, 'performance_metrics.json')
            selector.monitor.save_metrics(performance_json)
            
            # 生成性能图表
            selector.monitor.plot_metrics(args.performance_output)
            
            print(f"\n性能监控结果已保存到: {args.performance_output}")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        if 'selector' in locals() and selector.monitor:
            selector.monitor.stop_monitoring()
        raise


if __name__ == "__main__":
    main()