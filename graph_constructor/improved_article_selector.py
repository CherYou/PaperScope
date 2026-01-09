#!/usr/bin/env python3
"""
改进的文章选择器
基于实体频率和文章对共同实体来寻找相关文章组合
"""

import networkx as nx
import json
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import os
from itertools import combinations

class ImprovedArticleSelector:
    def __init__(self, graph_path: str):
        """
        初始化改进的文章选择器
        
        Args:
            graph_path: GraphML文件路径
        """
        self.graph_path = graph_path
        self.graph = None
        self.title_nodes = set()
        self.entity_nodes = set()
        self.article_entities = {}
        self.entity_articles = defaultdict(set)
        self.load_graph()
    
    def load_graph(self):
        """加载GraphML图并建立索引"""
        print(f"加载图文件: {self.graph_path}")
        self.graph = nx.read_graphml(self.graph_path)
        
        # 分类节点
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type == 'title':
                self.title_nodes.add(node)
            else:
                self.entity_nodes.add(node)
        
        # 建立文章-实体映射
        for title in self.title_nodes:
            connected_entities = set()
            for neighbor in self.graph.neighbors(title):
                if neighbor in self.entity_nodes:
                    connected_entities.add(neighbor)
                    self.entity_articles[neighbor].add(title)
            self.article_entities[title] = connected_entities
        
        print(f"图加载完成:")
        print(f"  总节点数: {self.graph.number_of_nodes()}")
        print(f"  总边数: {self.graph.number_of_edges()}")
        print(f"  标题节点数: {len(self.title_nodes)}")
        print(f"  实体节点数: {len(self.entity_nodes)}")
    
    def find_high_frequency_entities(self, min_articles: int = 2) -> List[Tuple[str, int]]:
        """
        找到出现在多篇文章中的高频实体
        
        Args:
            min_articles: 最少出现的文章数
            
        Returns:
            (实体名, 文章数) 的列表，按文章数降序排列
        """
        high_freq_entities = []
        
        for entity, articles in self.entity_articles.items():
            if len(articles) >= min_articles:
                high_freq_entities.append((entity, len(articles)))
        
        high_freq_entities.sort(key=lambda x: x[1], reverse=True)
        return high_freq_entities
    
    def find_articles_by_entity_overlap(self, 
                                      min_shared_entities: int = 2,
                                      target_articles: int = 3) -> List[Dict]:
        """
        基于实体重叠寻找文章组合
        
        Args:
            min_shared_entities: 最少共享实体数
            target_articles: 目标文章数量
            
        Returns:
            文章组合列表
        """
        print(f"寻找基于实体重叠的 {target_articles} 篇文章组合 (最少 {min_shared_entities} 个共享实体)")
        
        # 首先找到高频实体
        high_freq_entities = self.find_high_frequency_entities(min_articles=target_articles)
        
        if not high_freq_entities:
            print("没有找到出现在多篇文章中的实体")
            return []
        
        print(f"找到 {len(high_freq_entities)} 个高频实体")
        
        valid_combinations = []
        
        # 基于高频实体构建文章组合
        for entity, article_count in high_freq_entities:
            if article_count >= target_articles:
                articles_with_entity = list(self.entity_articles[entity])
                
                # 从包含该实体的文章中选择组合
                for article_combo in combinations(articles_with_entity, target_articles):
                    # 计算这些文章的共同实体
                    common_entities = set(self.article_entities[article_combo[0]])
                    for article in article_combo[1:]:
                        common_entities &= self.article_entities[article]
                    
                    if len(common_entities) >= min_shared_entities:
                        # 计算每篇文章的详细信息
                        article_info = []
                        for article in article_combo:
                            article_data = self.graph.nodes[article]
                            article_info.append({
                                'title': article,
                                'total_entities': len(self.article_entities[article]),
                                'article_count': article_data.get('article_count', 1),
                                'frequency': article_data.get('frequency', 1)
                            })
                        
                        combination = {
                            'articles': article_info,
                            'common_entities': list(common_entities),
                            'common_entity_count': len(common_entities),
                            'total_articles': len(article_combo),
                            'seed_entity': entity,
                            'seed_entity_frequency': article_count
                        }
                        
                        # 检查是否已存在相同的组合
                        combo_key = tuple(sorted([a['title'] for a in article_info]))
                        if not any(tuple(sorted([a['title'] for a in existing['articles']])) == combo_key 
                                 for existing in valid_combinations):
                            valid_combinations.append(combination)
        
        # 按共同实体数量排序
        valid_combinations.sort(key=lambda x: x['common_entity_count'], reverse=True)
        
        print(f"找到 {len(valid_combinations)} 个符合条件的文章组合")
        return valid_combinations
    
    def find_articles_by_pairwise_expansion(self, 
                                          min_shared_entities: int = 2,
                                          target_articles: int = 3) -> List[Dict]:
        """
        基于文章对扩展寻找文章组合
        
        Args:
            min_shared_entities: 最少共享实体数
            target_articles: 目标文章数量
            
        Returns:
            文章组合列表
        """
        print(f"基于文章对扩展寻找 {target_articles} 篇文章组合")
        
        # 首先找到所有有共同实体的文章对
        article_pairs = []
        articles = list(self.article_entities.keys())
        
        for i, article1 in enumerate(articles):
            for j, article2 in enumerate(articles[i+1:], i+1):
                common_entities = self.article_entities[article1] & self.article_entities[article2]
                if len(common_entities) >= min_shared_entities:
                    article_pairs.append({
                        'articles': [article1, article2],
                        'common_entities': common_entities,
                        'common_count': len(common_entities)
                    })
        
        # 按共同实体数量排序
        article_pairs.sort(key=lambda x: x['common_count'], reverse=True)
        print(f"找到 {len(article_pairs)} 个有共同实体的文章对")
        
        valid_combinations = []
        
        # 尝试扩展每个文章对
        for pair in article_pairs[:50]:  # 只处理前50个最好的文章对
            base_articles = set(pair['articles'])
            base_common = pair['common_entities']
            
            # 寻找与这两篇文章都有共同实体的第三篇文章
            for candidate_article in articles:
                if candidate_article not in base_articles:
                    candidate_entities = self.article_entities[candidate_article]
                    
                    # 计算三篇文章的共同实体
                    three_way_common = base_common & candidate_entities
                    
                    if len(three_way_common) >= min_shared_entities:
                        final_articles = list(base_articles) + [candidate_article]
                        
                        # 构建文章信息
                        article_info = []
                        for article in final_articles:
                            article_data = self.graph.nodes[article]
                            article_info.append({
                                'title': article,
                                'total_entities': len(self.article_entities[article]),
                                'article_count': article_data.get('article_count', 1),
                                'frequency': article_data.get('frequency', 1)
                            })
                        
                        combination = {
                            'articles': article_info,
                            'common_entities': list(three_way_common),
                            'common_entity_count': len(three_way_common),
                            'total_articles': len(final_articles),
                            'expansion_method': 'pairwise'
                        }
                        
                        # 检查是否已存在相同的组合
                        combo_key = tuple(sorted([a['title'] for a in article_info]))
                        if not any(tuple(sorted([a['title'] for a in existing['articles']])) == combo_key 
                                 for existing in valid_combinations):
                            valid_combinations.append(combination)
        
        # 按共同实体数量排序
        valid_combinations.sort(key=lambda x: x['common_entity_count'], reverse=True)
        
        print(f"通过文章对扩展找到 {len(valid_combinations)} 个文章组合")
        return valid_combinations
    
    def get_entity_details(self, entity_name: str) -> Dict:
        """获取实体的详细信息"""
        if entity_name not in self.graph:
            return {}
        
        entity_data = self.graph.nodes[entity_name]
        return {
            'name': entity_name,
            'node_type': entity_data.get('node_type', 'unknown'),
            'frequency': entity_data.get('frequency', 1),
            'article_count': entity_data.get('article_count', 1),
            'articles': list(self.entity_articles[entity_name])
        }
    
    def enrich_combinations_with_entity_details(self, combinations: List[Dict]) -> List[Dict]:
        """为文章组合添加实体详细信息"""
        print("为文章组合添加实体详细信息...")
        
        enriched_combinations = []
        
        for combo in combinations:
            enriched_combo = combo.copy()
            
            # 添加实体详细信息
            entity_details = []
            for entity_name in combo['common_entities']:
                entity_info = self.get_entity_details(entity_name)
                entity_details.append(entity_info)
            
            # 按实体在文章中的出现次数排序
            entity_details.sort(key=lambda x: len(x.get('articles', [])), reverse=True)
            
            enriched_combo['entity_details'] = entity_details
            enriched_combinations.append(enriched_combo)
        
        return enriched_combinations
    
    def save_results_to_jsonl(self, combinations: List[Dict], output_path: str):
        """将结果保存为JSONL格式"""
        print(f"保存结果到: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for combo in combinations:
                json.dump(combo, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"结果已保存，共 {len(combinations)} 个文章组合")
    
    def print_summary(self, combinations: List[Dict], top_n: int = 5):
        """打印结果摘要"""
        print(f"\n=== 结果摘要 (前 {top_n} 个) ===")
        
        for i, combo in enumerate(combinations[:top_n]):
            print(f"\n组合 {i+1}:")
            print(f"  共同实体数: {combo['common_entity_count']}")
            print(f"  文章数: {combo['total_articles']}")
            if 'seed_entity' in combo:
                print(f"  种子实体: {combo['seed_entity']} (出现在 {combo['seed_entity_frequency']} 篇文章中)")
            if 'expansion_method' in combo:
                print(f"  发现方法: {combo['expansion_method']}")
            
            print("  文章列表:")
            for article in combo['articles']:
                print(f"    - {article['title'][:60]}... (实体数: {article['total_entities']})")
            
            print("  共同实体:")
            for entity in combo['entity_details'][:8]:
                articles_count = len(entity.get('articles', []))
                print(f"    - {entity['name']} (出现在 {articles_count} 篇文章中)")


def main():
    parser = argparse.ArgumentParser(description='改进的文章选择器')
    parser.add_argument('--graph', '-g', default='output/graphs/merged_global_graph.graphml', 
                       help='GraphML图文件路径')
    parser.add_argument('--output', '-o', default='output/selected_papers/improved_selected_articles.jsonl', 
                       help='输出JSONL文件路径')
    parser.add_argument('--min_shared_entities', '-e', type=int, default=2, 
                       help='最少共享实体数量')
    parser.add_argument('--target_articles', '-a', type=int, default=10, 
                       help='目标文章数量')
    parser.add_argument('--max_results', '-m', type=int, default=20, 
                       help='最大结果数量')
    parser.add_argument('--method', choices=['entity_overlap', 'pairwise_expansion', 'both'], 
                       default='both', help='选择方法')
    
    args = parser.parse_args()
    
    # 检查图文件是否存在
    if not os.path.exists(args.graph):
        print(f"错误: 图文件不存在: {args.graph}")
        return
    
    # 创建选择器
    selector = ImprovedArticleSelector(args.graph)
    
    all_combinations = []
    
    # 基于实体重叠的方法
    if args.method in ['entity_overlap', 'both']:
        overlap_combinations = selector.find_articles_by_entity_overlap(
            args.min_shared_entities, 
            args.target_articles
        )
        all_combinations.extend(overlap_combinations)
    
    # 基于文章对扩展的方法
    if args.method in ['pairwise_expansion', 'both']:
        pairwise_combinations = selector.find_articles_by_pairwise_expansion(
            args.min_shared_entities, 
            args.target_articles
        )
        all_combinations.extend(pairwise_combinations)
    
    # 去重并排序
    unique_combinations = []
    seen_combos = set()
    
    for combo in all_combinations:
        combo_key = tuple(sorted([a['title'] for a in combo['articles']]))
        if combo_key not in seen_combos:
            seen_combos.add(combo_key)
            unique_combinations.append(combo)
    
    # 按共同实体数量排序
    unique_combinations.sort(key=lambda x: x['common_entity_count'], reverse=True)
    
    # 限制结果数量
    if len(unique_combinations) > args.max_results:
        unique_combinations = unique_combinations[:args.max_results]
        print(f"限制结果数量为前 {args.max_results} 个")
    
    # 添加实体详细信息
    enriched_combinations = selector.enrich_combinations_with_entity_details(unique_combinations)
    
    args.output = args.output.replace('.jsonl', f'_{args.method}_{args.min_shared_entities}_{args.target_articles}.jsonl')
    # 保存结果
    selector.save_results_to_jsonl(enriched_combinations, args.output)
    
    # 打印摘要
    selector.print_summary(enriched_combinations)
    
    print(f"\n处理完成！结果已保存到: {args.output}")


if __name__ == "__main__":
    main()