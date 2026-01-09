#!/usr/bin/env python3
"""
图构建脚本
"""
import json
import networkx as nx
from collections import defaultdict, Counter
import argparse
import os
import re
import torch
from vllm import LLM
from tqdm import tqdm
import faiss
import numpy as np
import multiprocessing as mp
from typing import Any, Dict, List, Set, Tuple
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class UnionFind:
    def __init__(self, nodes):
        # 初始化，每个节点的父节点都是它自己
        self.parent = {node: node for node in nodes}

    def find(self, i):
        # 查找根节点，并进行路径压缩
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # 合并两个节点所在的集合
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i # 将 j 的根指向 i 的根

class GraphBuilder:
    def __init__(self, stopwords_file: str = None, merge_similar: bool = False, similarity_threshold: float = 0.8):
        self.entity_types = [
            'research_background', 'classification_tags', 'key_contributions',
            'methodology', 'datasets', 'results', 'metrics', 'figure','table','algorithm',
            'formulas','limitations'
        ]
        self.stopwords = self.load_stopwords(stopwords_file) if stopwords_file else set()
        self.merge_similar = merge_similar
        self.similarity_threshold = similarity_threshold
        # 只有在需要合并相似实体时才初始化LLM模型
        self.model = None
        if self.merge_similar:
            try:
                self.model = LLM(model="models/Qwen/Qwen3-Embedding-8B", task="embed")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM model: {e}")
                print("Disabling merge_similar functionality")
                self.merge_similar = False
        
    def load_stopwords(self, stopwords_file: str) -> Set[str]:
        """加载停用词表"""
        stopwords = set()
        if os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释行
                    if line and not line.startswith('#'):
                        stopwords.add(line.lower())  # 转换为小写进行匹配
        return stopwords
    
    def is_stopword(self, entity: str) -> bool:
        """检查实体是否为停用词"""
        return entity.lower().strip() in self.stopwords
    
    def normalize_text(self, text: str) -> str:
        """标准化文本，用于相似度比较"""
        if type(text) != str:
            return ''
        # 转换为小写
        text = text.lower()
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        # 移除标点符号（保留字母、数字、空格）
        text = re.sub(r'[^\w\s]', '', text)
        return text
    def get_detailed_instruct(self,task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nEntity:{query}'

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        # 如果没有LLM模型，使用简单的字符串相似度
        if self.model is None:
            norm1 = self.normalize_text(text1)
            norm2 = self.normalize_text(text2)
            
            # 使用Jaccard相似度作为简单的相似度度量
            set1 = set(norm1.lower().split())
            set2 = set(norm2.lower().split())
            
            if len(set1) == 0 and len(set2) == 0:
                return 1.0
            if len(set1) == 0 or len(set2) == 0:
                return 0.0
                
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0
        
        # 使用LLM模型计算相似度
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        task = 'Given a scientific paper entity, retrieve the semantically similar entity' 
        queries = [
            self.get_detailed_instruct(task, norm1),
        ]
        # No need to add instruction for retrieval documents
        documents = [
            norm2
        ]
        input_texts = queries + documents
        
        outputs = self.model.embed(input_texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        scores = (embeddings[:1] @ embeddings[1:].T)
        #print(scores.tolist())
        final_sim = scores.tolist()[0][0]
        return final_sim

    @staticmethod
    def remap_edges_chunk(chunk, node_to_rep):
        edge_dict = defaultdict(float)
        for u, v, data in chunk:
            ru, rv = node_to_rep[u], node_to_rep[v]
            if ru != rv:
                edge_dict[(ru, rv)] += data.get('weight', 1.0)
        return edge_dict

    def merge_nodes_with_unionfind_fast(self,graph, uf, num_workers=None):
        print("\n步骤 6: 根据并查集结果创建合并分组...")

        # Step 1. 构建节点代表映射
        print("构建节点代表映射...")
        node_to_rep = {node: uf.find(node) for node in tqdm(graph.nodes())}

        # Step 2. 拆分边列表以并行映射
        edges = list(graph.edges(data=True))
        num_workers = num_workers or max(1, mp.cpu_count() - 1)
        chunks = np.array_split(edges, num_workers)

        print(f"并行重映射边集合... (workers={num_workers})")
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(GraphBuilder.remap_edges_chunk, [(chunk, node_to_rep) for chunk in chunks])

        # Step 3. 合并局部结果
        print("合并边集合结果...")
        new_edges = defaultdict(float)
        for r in results:
            for k, v in r.items():
                new_edges[k] += v

        # Step 4. 创建新的合并后图
        print("构建新图...")
        merged_graph = nx.Graph()
        
        # 收集每个代表节点的属性信息
        print("合并节点属性...")
        rep_node_attrs = {}
        for node, rep in tqdm(node_to_rep.items(), desc="处理节点属性"):
            if rep not in rep_node_attrs:
                # 初始化代表节点的属性
                rep_node_attrs[rep] = {
                    'node_type': graph.nodes[node].get('node_type', ''),
                    'frequency': 0,
                    'articles': set(),
                    'article_count': 0
                }
            
            # 累加频次
            rep_node_attrs[rep]['frequency'] += graph.nodes[node].get('frequency', 0)
            
            # 合并文章集合
            articles_str = graph.nodes[node].get('articles', '')
            if articles_str:
                articles_set = set(articles_str.split('|'))
                rep_node_attrs[rep]['articles'].update(articles_set)
        
        # 添加节点及其属性到新图
        for rep_node, attrs in rep_node_attrs.items():
            # 转换文章集合为字符串
            articles_str = '|'.join(sorted(attrs['articles'])) if attrs['articles'] else ''
            merged_graph.add_node(
                rep_node,
                node_type=attrs['node_type'],
                frequency=attrs['frequency'],
                articles=articles_str,
                article_count=len(attrs['articles'])
            )

        # 可以分块添加边，防止内存瞬间暴涨
        items = list(new_edges.items())
        for i in tqdm(range(0, len(items), 500000), desc="添加边到新图"):
            batch = items[i:i+500000]
            merged_graph.add_weighted_edges_from((u, v, w) for (u, v), w in batch)

        print(f"✅ 合并完成: 原节点数={len(graph):,}, 新节点数={len(merged_graph):,}, 边数={len(merged_graph.edges()):,}")
        return merged_graph

    def merge_similar_entities(self, graph: nx.Graph,ann_method:str) -> Dict[str, List[str]]:
        """找到图中相似的实体节点，返回合并映射"""
        

        #使用qwen-embedding-8b模型计算节点嵌入/考虑使用小模型，embedding维度为256更小，矩阵计算更高效
        nodes_list = list(graph.nodes())
        node_embeddings = self.model.embed(nodes_list)
        node_embeddings = torch.tensor([o.outputs.embedding for o in node_embeddings])
        node_embeddings = np.ascontiguousarray(node_embeddings, dtype=np.float32)
        # 保存节点列表以便后续映射索引到实际节点
        #node_list = list(graph.nodes())
        print(f"节点嵌入生成完毕，向量维度为: {node_embeddings.shape}")
        print("\n步骤 3: 构建 Faiss 索引...")
        embedding_dimension = node_embeddings.shape[1]
        if ann_method == 'hnsw':
            M = 32  
            # 初始化hnsw索引
            index = faiss.IndexHNSWFlat(embedding_dimension, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 30
            # 训练索引
            #index.train(node_embeddings)
            # 添加向量到索引
            #print(f"正在向索引中添加 {num_database_vectors} 个向量...")
            start_time = time.time()
            index.add(node_embeddings)
            end_time = time.time()
            print(f"添加向量完成，耗时: {end_time - start_time:.4f} 秒")
            print(f"索引中的向量总数: {index.ntotal}")

        else:
            index = faiss.IndexFlatL2(embedding_dimension)
            # 将我们的向量数据添加到索引中
            index.add(node_embeddings)


        print(f"Faiss 索引构建完成，共索引了 {index.ntotal} 个向量。")
        print("\n步骤 4: 按照节点类型执行近似最近邻搜索...")
        # 假设我们想为每个节点找到 5 个最相似的邻居
        k = 20
        
        # 先根据节点类型分组记录索引
        type_list_nodes = {}
        for node_id in nodes_list:
            node_type = graph.nodes[node_id]['node_type']
            if node_type not in type_list_nodes:
                type_list_nodes[node_type] = []
            type_list_nodes[node_type].append(node_id)
        
        
        type_search_results = {}
        # 对每个节点类型分别执行搜索
        for node_type, node_ids in type_list_nodes.items():
            print(f"\n节点类型: {node_type}")
            print(f"节点数量: {len(node_ids)}")
            
            # 提取当前类型的节点嵌入
            type_embedding = node_embeddings[[nodes_list.index(nid) for nid in node_ids]]
            
            # 执行最近邻搜索
            distances, indices = index.search(type_embedding, k)
            
            # 保存搜索结果
            type_search_results[node_type] = {
                'distances': distances,
                'indices': indices
            }
            

        print(f"为每个节点找到了 {k} 个最近邻居。")
        # --- 5. 生成并展示候选对 ---
        print("\n步骤 5: 生成并展示候选对...")
        candidate_pairs = set()

        # indices 的每一行对应一个节点，行中的元素是该节点的最近邻居ID
        for node_type, results in type_search_results.items():
            distances = results['distances']
            indices = results['indices']
            for node_idx in range(len(indices)):
                # indices[i, 0] 总是节点自身，所以我们从 1 开始
                for i in range(1, k): 
                    neighbor_idx = indices[node_idx][i]
                    
                    # 将索引转换回实际的节点ID
                    node_id = nodes_list[node_idx]
                    neighbor_id = nodes_list[neighbor_idx]
                
                    # 避免添加重复的对，如 (1, 5) 和 (5, 1)
                    # 通过排序来规范化
                    pair = tuple(sorted((node_id, neighbor_id)))
                    candidate_pairs.add(pair)

        print(f"共生成了 {len(candidate_pairs)} 个候选相似对。")

        #遍历候选对，计算相似度
        # 遍历候选对，合并相似实体
        # 设置一个语义相似度阈值
        THRESHOLD = 0.55


        # 初始化并查集
        uf = UnionFind(graph.nodes())

        # 遍历所有由 ANN 生成的候选对，根据语义相似度合并相似实体
        count = 0
        for u, v in tqdm(candidate_pairs):
            # 从原始图中获取节点名称，计算语义相似度
            node1, node2 = u, v  # u和v已经是实际的节点名称
            if count % 100 == 0:
                print(f"当前处理到第 {count} 个候选对")
            count += 1
            if graph.nodes[node1]['node_type'] != graph.nodes[node2]['node_type']:
                #print(f"节点类型不同，无法合并: {graph.nodes[node1]['node_type']} vs {graph.nodes[node2]['node_type']}")
                continue
            similarity = self.calculate_similarity(node1, node2)
                
            # 如果相似度超过阈值，则将它们标记为待合并（在并查集中）
            if similarity >= THRESHOLD:
                uf.union(u, v)
                # print(f"标记合并: ({u}, {v}), Jaccard 相似度: {similarity:.2f}")


        print("\n步骤 6: 根据并查集结果创建合并分组...")


        return self.merge_nodes_with_unionfind_fast(graph, uf, num_workers=None)
        

        
    def load_entities(self, jsonl_file: str) -> List[Dict]:
        """加载实体数据"""
        entities = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                entities.append(json.loads(line.strip()))
        return entities
    
    def build_article_graph(self, article_data: Dict) -> nx.Graph:
        """为单篇文章构建实体图"""
        G = nx.Graph()
        
        # 添加文章标题作为中心节点
        title = article_data['title']
        G.add_node(title, node_type='title', article=title)
        
        # 收集所有实体
        all_entities = []
        for entity_type in self.entity_types:
            if entity_type in article_data and article_data[entity_type]:
                entities = article_data[entity_type]
                if isinstance(entities, list):
                    for entity in entities:
                        if entity and entity.strip():  # 过滤空实体
                            entity_clean = entity.strip()
                            # 检查是否为停用词，如果是则跳过
                            if self.is_stopword(entity_clean):
                                continue
                            G.add_node(entity_clean, node_type=entity_type, article=title)
                            # 连接到文章标题
                            G.add_edge(title, entity_clean, edge_type='contains')
                            all_entities.append((entity_clean, entity_type))
        
        # 在同一类型的实体之间添加连接（表示它们属于同一篇文章）
        for entity_type in self.entity_types:
            if entity_type in article_data and article_data[entity_type]:
                # 过滤停用词和空实体
                entities = [e.strip() for e in article_data[entity_type] 
                           if e and e.strip() and not self.is_stopword(e.strip())]
                # 同类型实体之间建立连接
                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        G.add_edge(entities[i], entities[j], edge_type='same_type')
        
        # 不同类型实体之间建立连接（表示它们来自同一篇文章）
        for i in range(len(all_entities)):
            for j in range(i+1, len(all_entities)):
                entity1, type1 = all_entities[i]
                entity2, type2 = all_entities[j]
                if type1 != type2:  # 不同类型的实体
                    G.add_edge(entity1, entity2, edge_type='cross_type')
        
        return G
    
    def merge_graphs(self, graphs: List[nx.Graph]) -> nx.Graph:
        """合并多个图为一个全局图"""
        global_graph = nx.Graph()
        
        # 统计实体出现频次（包括标题节点）
        entity_counts = Counter()
        entity_types = {}
        entity_articles = defaultdict(set)
        
        for graph in graphs:
            for node, data in graph.nodes(data=True):
                # 保留所有节点，包括标题节点
                entity_counts[node] += 1
                entity_types[node] = data.get('node_type', 'unknown')
                if data.get('node_type') == 'title':
                    # 对于标题节点，它本身就是文章
                    entity_articles[node].add(node)
                else:
                    # 对于实体节点，记录它出现在哪些文章中
                    entity_articles[node].add(data.get('article', 'unknown'))
        
        # 添加节点到全局图
        for entity, count in entity_counts.items():
            global_graph.add_node(
                entity,
                node_type=entity_types[entity],
                frequency=count,
                articles="|".join(list(entity_articles[entity])),  # 转换为字符串
                article_count=len(entity_articles[entity])
            )
        
        # 添加边：首先保留原始图中的边关系
        for graph in graphs:
            for edge in graph.edges(data=True):
                node1, node2, edge_data = edge
                # 只有当两个节点都在全局图中时才添加边
                if global_graph.has_node(node1) and global_graph.has_node(node2):
                    if not global_graph.has_edge(node1, node2):
                        global_graph.add_edge(node1, node2, **edge_data)
        
        # 添加额外的边：基于共现关系（仅对非标题节点）
        entities = [e for e in entity_counts.keys() if entity_types[e] != 'title']
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                entity1, entity2 = entities[i], entities[j]
                
                # 计算共现文章数
                common_articles = entity_articles[entity1] & entity_articles[entity2]
                if len(common_articles) > 0:
                    # 如果边不存在，添加共现边
                    if not global_graph.has_edge(entity1, entity2):
                        # 计算共现强度
                        cooccurrence_strength = len(common_articles)
                        jaccard_similarity = len(common_articles) / len(entity_articles[entity1] | entity_articles[entity2])
                        
                        global_graph.add_edge(
                            entity1, entity2,
                            edge_type='cooccurrence',
                            weight=cooccurrence_strength,
                            jaccard_similarity=jaccard_similarity,
                            common_articles="|".join(list(common_articles))  # 转换为字符串
                        )
        
        return global_graph
    
    def save_graph(self, graph: nx.Graph, output_path: str, format_type: str = 'graphml'):
        """保存图到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建图的副本并清理不兼容的数据类型
        clean_graph = graph.copy()
        
        # 清理节点属性中的字典类型数据
        for node, data in clean_graph.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, dict):
                    # 将字典转换为字符串
                    data[key] = str(value)
                elif isinstance(value, list):
                    # 将列表转换为字符串
                    data[key] = "|".join(map(str, value))
        
        # 清理边属性中的字典类型数据
        for u, v, data in clean_graph.edges(data=True):
            for key, value in list(data.items()):
                if isinstance(value, dict):
                    # 将字典转换为字符串
                    data[key] = str(value)
                elif isinstance(value, list):
                    # 将列表转换为字符串
                    data[key] = "|".join(map(str, value))
        
        if format_type == 'graphml':
            nx.write_graphml(clean_graph, output_path)
        elif format_type == 'gexf':
            nx.write_gexf(clean_graph, output_path)
        elif format_type == 'json':
            # 转换为JSON格式
            data = nx.node_link_data(clean_graph)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"图已保存到: {output_path}")
        print(f"节点数: {clean_graph.number_of_nodes()}")
        print(f"边数: {clean_graph.number_of_edges()}")
    
    def print_graph_stats(self, graph: nx.Graph, name: str = "图"):
        """打印图的统计信息"""
        print(f"\n{name}统计信息:")
        print(f"  节点数: {graph.number_of_nodes()}")
        print(f"  边数: {graph.number_of_edges()}")
        
        if graph.number_of_nodes() > 0:
            # 节点类型统计
            node_types = defaultdict(int)
            for _, data in graph.nodes(data=True):
                node_types[data.get('node_type', 'unknown')] += 1
            
            print("  节点类型分布:")
            for node_type, count in sorted(node_types.items()):
                print(f"    {node_type}: {count}")
            
            # 连通性
            if nx.is_connected(graph):
                print("  图是连通的")
                print(f"  平均度: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")
            else:
                components = list(nx.connected_components(graph))
                print(f"  图有 {len(components)} 个连通分量")
                print(f"  最大连通分量大小: {len(max(components, key=len))}")


def main():
    parser = argparse.ArgumentParser(description='简化的实体图构建器')
    parser.add_argument('--input', '-i', required=True, help='输入的实体JSONL文件路径')
    parser.add_argument('--output_dir', '-o', default='output/graphs', help='输出目录')
    parser.add_argument('--format', '-f', choices=['graphml', 'gexf', 'json'], default='graphml', help='输出格式')
    parser.add_argument('--save_individual', action='store_true', help='是否保存单独的文章图')
    parser.add_argument('--stopwords', '-s', default='stopwords.txt', help='停用词文件路径')
    parser.add_argument('--merge_similar', action='store_true', help='是否合并相似的实体节点')
    parser.add_argument('--similarity_threshold', type=float, default=0.70, help='相似度阈值 (0.0-1.0)')
    
    args = parser.parse_args()
    
    # 创建图构建器，传入停用词文件和实体合并参数
    builder = GraphBuilder(
        stopwords_file=args.stopwords,
        merge_similar=args.merge_similar,
        similarity_threshold=args.similarity_threshold
    )
    
    # 如果停用词文件存在，打印加载信息
    if os.path.exists(args.stopwords):
        print(f"已加载停用词文件: {args.stopwords} (共 {len(builder.stopwords)} 个停用词)")
    else:
        print(f"停用词文件不存在: {args.stopwords}，将不进行停用词过滤")
    
    # 打印实体合并设置
    if args.merge_similar:
        print(f"启用实体合并功能，相似度阈值: {args.similarity_threshold}")
    else:
        print("未启用实体合并功能")
    
    # 加载数据
    print(f"加载实体数据: {args.input}")
    articles_data = builder.load_entities(args.input)
    print(f"加载了 {len(articles_data)} 篇文章的数据")
    
    # 为每篇文章构建图
    print("\n构建各文章的实体图...")
    article_graphs = []
    
    for i, article_data in enumerate(articles_data):
        article_graph = builder.build_article_graph(article_data)
        article_graphs.append(article_graph)
        
        if args.save_individual:
            # 保存单独的文章图
            safe_title = "".join(c for c in article_data['title'][:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            individual_path = os.path.join(args.output_dir, 'individual', f'article_{i:03d}_{safe_title}.{args.format}')
            builder.save_graph(article_graph, individual_path, args.format)
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(articles_data)} 篇文章")
    
    # 合并所有图
    print("\n合并所有文章图为全局图...")
    global_graph = builder.merge_graphs(article_graphs)
    
    # 如果启用了实体合并，进行相似实体合并
    output_name = args.input.split('/')[-1].split('.')[0].split('extracted_entities_')[-1]
    if args.merge_similar:
        print("\n执行相似实体合并...")
        global_graph = builder.merge_similar_entities(global_graph,'hnsw')
        output_name += '_merged'
    
    # 保存全局图
    global_path = os.path.join(args.output_dir, f'{output_name}_global_graph.{args.format}')
    builder.save_graph(global_graph, global_path, args.format)
    
    # 打印统计信息
    builder.print_graph_stats(global_graph, "全局图")
    
    # 保存一些高频实体信息
    print("\n高频实体 (出现在3篇以上文章中):")
    high_freq_entities = []
    for node, data in global_graph.nodes(data=True):
        if data.get('article_count', 0) >= 3:
            high_freq_entities.append((node, data.get('article_count', 0), data.get('node_type', 'unknown')))
    
    high_freq_entities.sort(key=lambda x: x[1], reverse=True)
    for entity, count, entity_type in high_freq_entities[:20]:
        print(f"  {entity} ({entity_type}): {count} 篇文章")


if __name__ == "__main__":
    main()