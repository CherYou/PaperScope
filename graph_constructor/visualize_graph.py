#!/usr/bin/env python3
"""
Graph Visualization Script
For visualizing constructed entity graphs
Optimized version: supports more entity types, improved color schemes and layouts
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import argparse
import os
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Set font for Chinese characters support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GraphVisualizer:
    def __init__(self):
        # Optimized entity type color mapping - using more intuitive and distinguishable colors
        self.entity_colors = {
            # Core content types - warm colors
            'title': '#2E54A1',                    # Bright red - title (most important)
            'key_contributions': '#1E386B',        # Orange - key contributions
            'methodology': '#0986CA',              # Blue - methodology
            'algorithm': '#7CCEF6',                # Purple - algorithm
            
            # Background and classification - neutral colors
            'research_background': '#00327E',      # Green - research background
            'classification_tags': '#005CA9',      # Teal - classification tags
            'limitations': '#419DDC',              # Gray - limitations
            
            # Data and results - cool colors
            'datasets': '#C7EFF7',                 # Dark blue-gray - datasets
            'results': '#9A98FD',                  # Dark teal - results
            'metrics': '#C0DEFA',                  # Dark green - metrics
            
            # Technical details - special colors
            'formulas': '#B1C6FE',                 # Dark purple - formulas
            'figure': '#4599FB',                   # Dark orange - figures
            'table': '#48F6F8',                    # Dark red - tables
            
            # Default and unknown
            'unknown': '#BDC3C7'                   # Light gray - unknown
        }
        
        # Entity type English name mapping
        self.entity_names_en = {
            'title': 'Title',
            'research_background': 'Research Background',
            'classification_tags': 'Classification Tags',
            'key_contributions': 'Key Contributions',
            'methodology': 'Methodology',
            'algorithm': 'Algorithm',
            'datasets': 'Datasets',
            'results': 'Results',
            'metrics': 'Metrics',
            'formulas': 'Formulas',
            'figure': 'Figure',
            'table': 'Table',
            'limitations': 'Limitations',
            'unknown': 'Unknown'
        }
        
        # Entity type importance weights (for node size and layout)
        self.entity_importance = {
            'title': 10,
            'key_contributions': 9,
            'methodology': 8,
            'algorithm': 8,
            'research_background': 7,
            'results': 7,
            'datasets': 6,
            'metrics': 6,
            'formulas': 5,
            'classification_tags': 5,
            'figure': 4,
            'table': 4,
            'limitations': 3,
            'unknown': 1
        }
        
    def load_graph(self, graph_path: str) -> nx.Graph:
        """Load graph file"""
        if graph_path.endswith('.graphml'):
            return nx.read_graphml(graph_path)
        elif graph_path.endswith('.gexf'):
            return nx.read_gexf(graph_path)
        else:
            raise ValueError(f"Unsupported graph file format: {graph_path}")
    
    def filter_high_degree_nodes(self, graph: nx.Graph, min_degree: int = 5) -> nx.Graph:
        """Filter subgraph with high degree nodes"""
        high_degree_nodes = [node for node, degree in graph.degree() if degree >= min_degree]
        return graph.subgraph(high_degree_nodes).copy()
    
    def filter_high_frequency_nodes(self, graph: nx.Graph, min_frequency: int = 3) -> nx.Graph:
        """Filter subgraph with high frequency nodes"""
        high_freq_nodes = []
        for node, data in graph.nodes(data=True):
            frequency = data.get('frequency', 1)
            article_count = data.get('article_count', 1)
            if frequency >= min_frequency or article_count >= min_frequency:
                high_freq_nodes.append(node)
        return graph.subgraph(high_freq_nodes).copy()
    
    def filter_by_entity_types(self, graph: nx.Graph, entity_types: List[str]) -> nx.Graph:
        """Filter graph by Entity Type"""
        filtered_nodes = []
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type in entity_types:
                filtered_nodes.append(node)
        return graph.subgraph(filtered_nodes).copy()
    
    def get_node_colors(self, graph: nx.Graph) -> list:
        """Get node color list"""
        colors = []
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            colors.append(self.entity_colors.get(node_type, self.entity_colors['unknown']))
        return colors
    
    def get_node_sizes(self, graph: nx.Graph, base_size: int = 100) -> list:
        """Calculate node size based on frequency and importance"""
        sizes = []
        for node, data in graph.nodes(data=True):
            frequency = data.get('frequency', 1)
            article_count = data.get('article_count', 1)
            node_type = data.get('node_type', 'unknown')
            
            # Combine frequency, article count and entity importance
            freq_factor = max(frequency, article_count)
            importance_factor = self.entity_importance.get(node_type, 1)
            
            size = base_size + freq_factor * 15 + importance_factor * 10
            sizes.append(min(size, 800))  # Limit maximum size
        return sizes
    
    def create_hierarchical_layout(self, graph: nx.Graph) -> Dict:
        """Create hierarchical layout based on Entity Type"""
        pos = {}
        
        # Group by Entity Type
        type_groups = defaultdict(list)
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            type_groups[node_type].append(node)
        
        # Define hierarchical structure
        layers = {
            0: ['title'],  # Top level: titles
            1: ['key_contributions', 'methodology'],  # Second level: core content
            2: ['research_background', 'algorithm'],  # Third level: background and algorithms
            3: ['datasets', 'results', 'metrics'],  # Fourth level: data and results
            4: ['formulas', 'figure', 'table'],  # Fifth level: technical details
            5: ['classification_tags', 'limitations', 'unknown']  # Bottom level: others
        }
        
        y_positions = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
        
        for layer, entity_types in layers.items():
            y = y_positions[layer]
            nodes_in_layer = []
            for entity_type in entity_types:
                nodes_in_layer.extend(type_groups.get(entity_type, []))
            
            if nodes_in_layer:
                # Distribute nodes evenly within this layer
                x_positions = np.linspace(-len(nodes_in_layer)/2, len(nodes_in_layer)/2, len(nodes_in_layer))
                for i, node in enumerate(nodes_in_layer):
                    pos[node] = (x_positions[i], y + np.random.normal(0, 0.1))  # Add small random offset
        
        return pos
    
    def create_clustered_layout(self, graph: nx.Graph) -> Dict:
        """Create clustered layout based on Entity Type"""
        pos = {}
        
        # Group by Entity Type
        type_groups = defaultdict(list)
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            type_groups[node_type].append(node)
        
        # foreachEntity Typedefineclusterclasscenter
        cluster_centers = {
            'title': (0, 0),
            'key_contributions': (3, 2),
            'methodology': (-3, 2),
            'algorithm': (0, 3),
            'research_background': (-2, -1),
            'classification_tags': (2, -1),
            'datasets': (-4, 0),
            'results': (4, 0),
            'metrics': (1, -3),
            'formulas': (-1, -3),
            'figure': (3, -2),
            'table': (-3, -2),
            'limitations': (0, -4),
            'unknown': (0, 4)
        }
        
        # ateachclusterclasscenteraroundrangedistributionnode
        for entity_type, nodes in type_groups.items():
            center_x, center_y = cluster_centers.get(entity_type, (0, 0))
            
            if len(nodes) == 1:
                pos[nodes[0]] = (center_x, center_y)
            else:
                # useusecircleshapedistribution
                angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
                radius = 0.5 + len(nodes) * 0.1
                
                for i, node in enumerate(nodes):
                    x = center_x + radius * np.cos(angles[i])
                    y = center_y + radius * np.sin(angles[i])
                    pos[node] = (x, y)
        
        return pos
    
    def create_interactive_visualization(self, graph: nx.Graph, output_path: str, 
                                      filter_type: str = 'frequency', min_threshold: int = 3):
        """createInteractive visualization"""
        # filtergraph
        if filter_type == 'frequency':
            filtered_graph = self.filter_high_frequency_nodes(graph, min_threshold)
        elif filter_type == 'degree':
            filtered_graph = self.filter_high_degree_nodes(graph, min_threshold)
        else:
            filtered_graph = graph
        
        # getlayout
        pos = nx.spring_layout(filtered_graph, k=3, iterations=50)
        
        # prepareNodesdata
        node_trace = []
        node_info = []
        
        for node, data in filtered_graph.nodes(data=True):
            x, y = pos[node]
            node_type = data.get('node_type', 'unknown')
            frequency = data.get('frequency', 1)
            article_count = data.get('article_count', 1)
            
            node_trace.append(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=max(10, frequency * 5 + article_count * 3),
                    color=self.entity_colors.get(node_type, self.entity_colors['unknown']),
                    line=dict(width=2, color='white')
                ),
                text=node[:20] + '...' if len(node) > 20 else node,
                textposition="middle center",
                textfont=dict(size=8),
                hovertemplate=f"<b>{node}</b><br>" +
                             f"classtype: {self.entity_names_en.get(node_type, node_type)}<br>" +
                             f"frequency: {frequency}<br>" +
                             f"articlenumber: {article_count}<extra></extra>",
                name=self.entity_names_en.get(node_type, node_type),
                showlegend=True
            ))
        
        # prepareEdgesdata
        edge_x = []
        edge_y = []
        
        for edge in filtered_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # creategraphshape
        fig = go.Figure(data=[edge_trace] + node_trace,
                       layout=go.Layout(
                           title=dict(
                               text='realsolidoffsystemgraph - Interactive visualization',
                               x=0.5,
                               font=dict(size=20)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="dragdragnodeenterrowinteractive，hoverstopviewdetailed information",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        # saveinteractivestylegraphshape
        fig.write_html(output_path.replace('.png', '_interactive.html'))
        print(f"Interactive visualizationalreadysaveto: {output_path.replace('.png', '_interactive.html')}")
    
    def visualize_global_graph(self, graph: nx.Graph, output_path: str, 
                              filter_type: str = 'frequency', min_threshold: int = 3,
                              layout: str = 'spring', figsize: tuple = (20, 16),
                              enable_interactive: bool = True) -> None:
        """
        visualizationentirelocalgraph
        
        Args:
            graph: NetworkXgraphobject
            output_path: outputpath
            filter_type: filterclasstype ('frequency', 'degree', 'none')
            min_threshold: minimumthresholdvalue
            layout: layoutclasstype ('spring', 'circular', 'kamada_kawai', 'spectral', 'hierarchical', 'clustered')
            figsize: graphshapesize
            enable_interactive: yesnoGenerateInteractive visualization
        """
        
        # filtergraph
        if filter_type == 'frequency':
            filtered_graph = self.filter_high_frequency_nodes(graph, min_threshold)
        elif filter_type == 'degree':
            filtered_graph = self.filter_high_degree_nodes(graph, min_threshold)
        else:
            filtered_graph = graph
        
        print(f"originalgraph: {graph.number_of_nodes()}  nodes, {graph.number_of_edges()}  edges")
        print(f"Filtered graph: {filtered_graph.number_of_nodes()}  nodes, {filtered_graph.number_of_edges()}  edges")
        
        if filtered_graph.number_of_nodes() == 0:
            print("warn: filterback graphforempty，pleasereducelowthresholdvalue")
            return
        
        # selectlayout
        if layout == 'spring':
            pos = nx.spring_layout(filtered_graph, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(filtered_graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(filtered_graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(filtered_graph)
        elif layout == 'hierarchical':
            pos = self.create_hierarchical_layout(filtered_graph)
        elif layout == 'clustered':
            pos = self.create_clustered_layout(filtered_graph)
        else:
            pos = nx.spring_layout(filtered_graph, k=3, iterations=50)
        
        # getnodecolorandsize
        node_colors = self.get_node_colors(filtered_graph)
        node_sizes = self.get_node_sizes(filtered_graph, base_size=200)
        
        # creategraphshape
        plt.figure(figsize=figsize)
        
        # drawcreateedge
        nx.draw_networkx_edges(filtered_graph, pos, alpha=0.3, width=0.5, edge_color='gray')
        
        # drawcreatenode
        nx.draw_networkx_nodes(filtered_graph, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.8,
                              linewidths=1,
                              edgecolors='white')
        
        # addlabel（onlyforhighfrequencynode）
        high_freq_nodes = {}
        for node, data in filtered_graph.nodes(data=True):
            frequency = data.get('frequency', 1)
            article_count = data.get('article_count', 1)
            if frequency >= min_threshold * 2 or article_count >= min_threshold * 2:
                # cutbreaklonglabel
                label = node[:15] + '...' if len(node) > 15 else node
                high_freq_nodes[node] = label
        
        nx.draw_networkx_labels(filtered_graph, pos, high_freq_nodes, 
                               font_size=8, font_weight='bold')
        
        # creategraphexample
        legend_elements = []
        for entity_type, color in self.entity_colors.items():
            # checkthisclasstypeyesnoatwhenfrontgraphcentersaveat
            type_exists = any(data.get('node_type') == entity_type 
                            for _, data in filtered_graph.nodes(data=True))
            if type_exists:
                legend_elements.append(
                    mpatches.Patch(color=color, 
                                 label=f"{self.entity_names_en.get(entity_type, entity_type)}")
                )
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title(f'realsolidoffsystemgraph - {layout.upper()}layout\n'
                 f'filtercondition: {filter_type} >= {min_threshold}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # savegraphshape
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"graphshapealreadysaveto: {output_path}")
        
        # generateInteractive visualization
        if enable_interactive:
            self.create_interactive_visualization(filtered_graph, output_path, filter_type, min_threshold)

    def create_entity_type_analysis(self, graph: nx.Graph, output_dir: str):
        """CreateEntity TypeAnalyzefigure"""
        # statisticseachEntity Type Node Countandfrequency
        type_stats = defaultdict(lambda: {'count': 0, 'total_frequency': 0, 'total_articles': 0})
        
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            frequency = data.get('frequency', 1)
            article_count = data.get('article_count', 1)
            
            type_stats[node_type]['count'] += 1
            type_stats[node_type]['total_frequency'] += frequency
            type_stats[node_type]['total_articles'] += article_count
        
        # createDataFrame
        df_data = []
        for entity_type, stats in type_stats.items():
            df_data.append({
                'entity type': self.entity_names_en.get(entity_type, entity_type),
                'node count': stats['count'],
                'total frequency': stats['total_frequency'],
                'total articles': stats['total_articles'],
                'average frequency': stats['total_frequency'] / stats['count'] if stats['count'] > 0 else 0
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('node count', ascending=False)
        
        # createsubgraph
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Entity type distributionpiegraph
        colors = [self.entity_colors.get(list(type_stats.keys())[i], '#CCCCCC') 
                 for i in range(len(df))]
        ax1.pie(df['node count'], labels=df['entity type'], autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax1.set_title('entity type distribution', fontsize=14, fontweight='bold')
        
        # 2. Node Countbarstategraph
        bars = ax2.bar(range(len(df)), df['node count'], color=colors)
        ax2.set_xlabel('entity type')
        ax2.set_ylabel('node count')
        ax2.set_title('node count per entity type', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['entity type'], rotation=45, ha='right')
        
        # addnumbervaluelabel
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. totalfrequencycontrast
        bars = ax3.bar(range(len(df)), df['total frequency'], color=colors)
        ax3.set_xlabel('entity type')
        ax3.set_ylabel('total frequency')
        ax3.set_title('total frequency per entity type', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df['entity type'], rotation=45, ha='right')
        
        # 4. averagefrequencycontrast
        bars = ax4.bar(range(len(df)), df['average frequency'], color=colors)
        ax4.set_xlabel('entity type')
        ax4.set_ylabel('average frequency')
        ax4.set_title('average frequency per entity type', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels(df['entity type'], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # savefigure
        analysis_path = os.path.join(output_dir, 'entity_type_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # savestatisticsdata
        csv_path = os.path.join(output_dir, 'entity_type_statistics.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"Entity TypeanalyzeChart saved to: {analysis_path}")
        print(f"Entity Typestatisticsdataalreadysaveto: {csv_path}")
        
        return df

    def analyze_graph_statistics(self, graph: nx.Graph, output_dir: str = None) -> dict:
        """Analyzegraph statisticsinformation"""
        stats = {}
        
        # basicstatistics
        stats['nodes'] = graph.number_of_nodes()
        stats['edges'] = graph.number_of_edges()
        stats['density'] = nx.density(graph)
        stats['is_connected'] = nx.is_connected(graph)
        
        # Degree Distribution
        degrees = [d for n, d in graph.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees) if degrees else 0
        stats['min_degree'] = min(degrees) if degrees else 0
        
        # connectthroughproperty
        if nx.is_connected(graph):
            stats['diameter'] = nx.diameter(graph)
            stats['avg_path_length'] = nx.average_shortest_path_length(graph)
        else:
            # pairinnonconnectthroughgraph，calculatemaximumConnected Components statistics
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_subgraph = graph.subgraph(largest_cc)
            stats['diameter'] = nx.diameter(largest_subgraph)
            stats['avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
            stats['connected_components'] = nx.number_connected_components(graph)
        
        # nodeclasstypedistribution
        node_types = defaultdict(int)
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] += 1
        stats['node_types'] = dict(node_types)
        
        # frequencystatistics
        frequencies = []
        article_counts = []
        for node, data in graph.nodes(data=True):
            frequencies.append(data.get('frequency', 1))
            article_counts.append(data.get('article_count', 1))
        
        stats['avg_frequency'] = np.mean(frequencies)
        stats['max_frequency'] = max(frequencies) if frequencies else 0
        stats['avg_article_count'] = np.mean(article_counts)
        stats['max_article_count'] = max(article_counts) if article_counts else 0
        
        # savestatisticsinformationtofile
        stats_file = os.path.join(output_dir, 'graph_statistics.json')
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return stats

    def add_network_analysis_metrics(self, graph: nx.Graph) -> dict:
        """addNetwork analysis metrics"""
        metrics = {}
        
        # centralitymetrics
        print("calculatecentralitymetrics...")
        metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
        metrics['closeness_centrality'] = nx.closeness_centrality(graph)
        metrics['degree_centrality'] = nx.degree_centrality(graph)
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(graph, max_iter=1000)
        
        # PageRank
        metrics['pagerank'] = nx.pagerank(graph)
        
        # Clustering Coefficient
        metrics['clustering'] = nx.clustering(graph)
        
        return metrics
    
    def create_network_analysis_visualization(self, graph: nx.Graph, metrics: dict, output_dir: str):
        """Create network analysis visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. DegreeCentrality Distribution
        degree_values = list(metrics['degree_centrality'].values())
        axes[0, 0].hist(degree_values, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Degree Centrality Distribution')
        axes[0, 0].set_xlabel('Degree Centrality')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. betweennumberCentrality Distribution
        betweenness_values = list(metrics['betweenness_centrality'].values())
        axes[0, 1].hist(betweenness_values, bins=30, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Betweenness Centrality Distribution')
        axes[0, 1].set_xlabel('Betweenness Centrality')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. PageRankdistribution
        pagerank_values = list(metrics['pagerank'].values())
        axes[0, 2].hist(pagerank_values, bins=30, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('PageRank Distribution')
        axes[0, 2].set_xlabel('PageRank')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Topnode - Degreecentrality
        top_degree = sorted(metrics['degree_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]
        nodes, values = zip(*top_degree)
        short_nodes = [n[:15] + '...' if len(n) > 15 else n for n in nodes]
        axes[1, 0].barh(range(len(short_nodes)), values, color='skyblue')
        axes[1, 0].set_yticks(range(len(short_nodes)))
        axes[1, 0].set_yticklabels(short_nodes)
        axes[1, 0].set_title('Top 10 Nodes by Degree Centrality')
        axes[1, 0].set_xlabel('Degree Centrality')
        
        # 5. Topnode - Betweenness Centrality
        top_betweenness = sorted(metrics['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]
        nodes, values = zip(*top_betweenness)
        short_nodes = [n[:15] + '...' if len(n) > 15 else n for n in nodes]
        axes[1, 1].barh(range(len(short_nodes)), values, color='lightcoral')
        axes[1, 1].set_yticks(range(len(short_nodes)))
        axes[1, 1].set_yticklabels(short_nodes)
        axes[1, 1].set_title('Top 10 Nodes by Betweenness Centrality')
        axes[1, 1].set_xlabel('Betweenness Centrality')
        
        # 6. Topnode - PageRank
        top_pagerank = sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:10]
        nodes, values = zip(*top_pagerank)
        short_nodes = [n[:15] + '...' if len(n) > 15 else n for n in nodes]
        axes[1, 2].barh(range(len(short_nodes)), values, color='lightgreen')
        axes[1, 2].set_yticks(range(len(short_nodes)))
        axes[1, 2].set_yticklabels(short_nodes)
        axes[1, 2].set_title('Top 10 Nodes by PageRank')
        axes[1, 2].set_xlabel('PageRank')
        
        plt.tight_layout()
        
        # savefigure
        metrics_path = os.path.join(output_dir, 'network_analysis_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # saveTopNodesdata
        top_nodes_data = {
            'top_degree_centrality': top_degree,
            'top_betweenness_centrality': top_betweenness,
            'top_pagerank': top_pagerank
        }
        
        import pandas as pd
        df_list = []
        for metric_name, node_list in top_nodes_data.items():
            for node, value in node_list:
                df_list.append({
                    'metric': metric_name,
                    'node': node,
                    'value': value
                })
        
        df = pd.DataFrame(df_list)
        csv_path = os.path.join(output_dir, 'top_nodes_analysis.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"Network analysis metricsChart saved to: {metrics_path}")
        print(f"Top nodes analysisdataalreadysaveto: {csv_path}")
    
    def create_temporal_analysis(self, graph_files: list, output_dir: str, time_labels: list = None):
        """Create temporal analysis functionality"""
        if not graph_files:
            print("Warning: No graph files provided for temporal analysis")
            return
        
        print(f"Starting temporal analysis with{len(graph_files)} time points")
        
        # If no time labels provided, use file names
        if time_labels is None:
            time_labels = [f"T{i+1}" for i in range(len(graph_files))]
        
        # Load all graphs and calculate metrics
        temporal_data = []
        all_graphs = []
        
        for i, graph_file in enumerate(graph_files):
            print(f"Processing time point {time_labels[i]}: {graph_file}")
            try:
                graph = self.load_graph(graph_file)
                all_graphs.append(graph)
                
                # Calculate basic statistical metrics
                stats = {
                    'time_label': time_labels[i],
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'density': nx.density(graph),
                    'avg_clustering': nx.average_clustering(graph),
                    'connected_components': nx.number_connected_components(graph)
                }
                
                # Calculate average centrality metrics
                if graph.number_of_nodes() > 0:
                    degree_centrality = nx.degree_centrality(graph)
                    stats['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
                    
                    if graph.number_of_nodes() <= 1000:  # Limit computation for large graphs
                        betweenness_centrality = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes())))
                        stats['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
                    else:
                        stats['avg_betweenness_centrality'] = 0
                
                # Entity type distribution
                entity_types = {}
                for node in graph.nodes():
                    node_type = graph.nodes[node].get('node_type', 'unknown')
                    entity_types[node_type] = entity_types.get(node_type, 0) + 1
                
                stats['entity_types'] = entity_types
                stats['num_entity_types'] = len(entity_types)
                
                temporal_data.append(stats)
                
            except Exception as e:
                print(f"Warning: Unable to load graph file {graph_file}: {e}")
                continue
        
        if not temporal_data:
            print("Error: Failed to load any graph files")
            return
        
        # Create temporal visualization
        self._create_temporal_plots(temporal_data, output_dir)
        
        # Create entity type evolution analysis
        self._create_entity_evolution_analysis(temporal_data, output_dir)
        
        # saveTimeordercolumndata
        df = pd.DataFrame(temporal_data)
        csv_path = os.path.join(output_dir, 'temporal_analysis.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Temporal analysis data saved to: {csv_path}")
        
        return temporal_data
    
    def _create_temporal_plots(self, temporal_data: list, output_dir: str):
        """Create temporal plots"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Graph Network Temporal Analysis', fontsize=16, fontweight='bold')
        
        time_labels = [data['time_label'] for data in temporal_data]
        
        # 1. Node Count Changes
        nodes = [data['nodes'] for data in temporal_data]
        axes[0, 0].plot(time_labels, nodes, marker='o', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_title('Node Count Changes')
        axes[0, 0].set_ylabel('Node Count')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Edge Count Changes
        edges = [data['edges'] for data in temporal_data]
        axes[0, 1].plot(time_labels, edges, marker='s', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_title('Edge Count Changes')
        axes[0, 1].set_ylabel('Edge Count')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Graph Density Changes
        densities = [data['density'] for data in temporal_data]
        axes[0, 2].plot(time_labels, densities, marker='^', linewidth=2, markersize=8, color='green')
        axes[0, 2].set_title('Graph Density Changes')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Average Clustering Coefficient Changes
        clustering = [data['avg_clustering'] for data in temporal_data]
        axes[1, 0].plot(time_labels, clustering, marker='d', linewidth=2, markersize=8, color='purple')
        axes[1, 0].set_title('Average Clustering Coefficient Changes')
        axes[1, 0].set_ylabel('Clustering Coefficient')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Connected Components Changes
        components = [data['connected_components'] for data in temporal_data]
        axes[1, 1].plot(time_labels, components, marker='v', linewidth=2, markersize=8, color='orange')
        axes[1, 1].set_title('Connected Components Changes')
        axes[1, 1].set_ylabel('Connected Components')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Entity Type Count Changes
        entity_type_counts = [data['num_entity_types'] for data in temporal_data]
        axes[1, 2].plot(time_labels, entity_type_counts, marker='*', linewidth=2, markersize=10, color='brown')
        axes[1, 2].set_title('Entity Type Count Changes')
        axes[1, 2].set_ylabel('Entity Types')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # savefigure
        temporal_path = os.path.join(output_dir, 'temporal_analysis.png')
        plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal analysis chart saved to: {temporal_path}")
    
    def _create_entity_evolution_analysis(self, temporal_data: list, output_dir: str):
        """Create entity type evolution analysis"""
        # Collect all entity types
        all_entity_types = set()
        for data in temporal_data:
            all_entity_types.update(data['entity_types'].keys())
        
        all_entity_types = sorted(list(all_entity_types))
        time_labels = [data['time_label'] for data in temporal_data]
        
        # Create entity type evolution matrix
        evolution_matrix = []
        for entity_type in all_entity_types:
            counts = []
            for data in temporal_data:
                counts.append(data['entity_types'].get(entity_type, 0))
            evolution_matrix.append(counts)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. stackstackareagraph
        ax1.stackplot(time_labels, *evolution_matrix, labels=all_entity_types, alpha=0.8)
        ax1.set_title('Entity Type Evolution - Stacked Area Chart', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Entity Count')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. hotforcegraph
        evolution_array = np.array(evolution_matrix)
        im = ax2.imshow(evolution_array, cmap='YlOrRd', aspect='auto')
        ax2.set_title('Entity Type Evolution - Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Entity Type')
        ax2.set_xticks(range(len(time_labels)))
        ax2.set_xticklabels(time_labels, rotation=45)
        ax2.set_yticks(range(len(all_entity_types)))
        ax2.set_yticklabels(all_entity_types)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Entity Count')
        
        # Add values on heatmap
        for i in range(len(all_entity_types)):
            for j in range(len(time_labels)):
                text = ax2.text(j, i, evolution_array[i, j], 
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        # savefigure
        evolution_path = os.path.join(output_dir, 'entity_evolution_analysis.png')
        plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Entity evolution analysis chart saved to: {evolution_path}")
    
    def create_comprehensive_dashboard(self, graph: nx.Graph, output_dir: str, 
                                    temporal_data: list = None, metrics: dict = None):
        """Create comprehensive dashboard"""
        print("positiveatCreate comprehensive dashboard...")
        
        # Create a large comprehensive chart
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Graph Basic Information (leftupangle)
        ax1 = fig.add_subplot(gs[0, 0])
        basic_info = [
            f"Nodes: {graph.number_of_nodes()}",
            f"Edges: {graph.number_of_edges()}",
            f"Density: {nx.density(graph):.4f}",
            f"Connected Components: {nx.number_connected_components(graph)}",
            f"averageClustering Coefficient: {nx.average_clustering(graph):.4f}"
        ]
        ax1.text(0.1, 0.9, "Graph Basic Information", fontsize=14, fontweight='bold', transform=ax1.transAxes)
        for i, info in enumerate(basic_info):
            ax1.text(0.1, 0.7 - i*0.12, info, fontsize=12, transform=ax1.transAxes)
        ax1.axis('off')
        
        # 2. Degree Distribution (rightupangle)
        ax2 = fig.add_subplot(gs[0, 1:3])
        degrees = [d for n, d in graph.degree()]
        ax2.hist(degrees, bins=min(30, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Degree Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Entity type distribution (rightupanglepiegraph)
        ax3 = fig.add_subplot(gs[0, 3])
        entity_types = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get('node_type', 'unknown')
            entity_types[node_type] = entity_types.get(node_type, 0) + 1
        
        if entity_types:
            colors = [self.entity_colors.get(et, '#cccccc') for et in entity_types.keys()]
            wedges, texts, autotexts = ax3.pie(entity_types.values(), labels=entity_types.keys(), 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax3.set_title('Entity type distribution', fontsize=14, fontweight='bold')
        
        # 4. Network analysis metrics (ifprovidemetrics)
        if metrics:
            # Centrality Distribution
            ax4 = fig.add_subplot(gs[1, 0])
            degree_centrality = list(metrics['degree_centrality'].values())
            ax4.hist(degree_centrality, bins=20, alpha=0.7, color='lightcoral')
            ax4.set_title('DegreeCentrality Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Degreecentrality')
            ax4.set_ylabel('Frequency')
            
            ax5 = fig.add_subplot(gs[1, 1])
            betweenness_centrality = list(metrics['betweenness_centrality'].values())
            ax5.hist(betweenness_centrality, bins=20, alpha=0.7, color='lightgreen')
            ax5.set_title('betweennumberCentrality Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Betweenness Centrality')
            ax5.set_ylabel('Frequency')
            
            # Topnode
            ax6 = fig.add_subplot(gs[1, 2:])
            top_nodes = sorted(metrics['degree_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]
            nodes, values = zip(*top_nodes)
            short_nodes = [n[:20] + '...' if len(n) > 20 else n for n in nodes]
            ax6.barh(range(len(short_nodes)), values, color='gold')
            ax6.set_yticks(range(len(short_nodes)))
            ax6.set_yticklabels(short_nodes)
            ax6.set_title('Top 10 node (Degreecentrality)', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Degreecentrality')
        
        # 5. Timeordercolumnanalyze (ifprovidetemporal_data)
        if temporal_data and len(temporal_data) > 1:
            ax7 = fig.add_subplot(gs[2, :2])
            time_labels = [data['time_label'] for data in temporal_data]
            nodes_count = [data['nodes'] for data in temporal_data]
            edges_count = [data['edges'] for data in temporal_data]
            
            ax7_twin = ax7.twinx()
            line1 = ax7.plot(time_labels, nodes_count, 'b-o', label='Nodes', linewidth=2)
            line2 = ax7_twin.plot(time_labels, edges_count, 'r-s', label='Edges', linewidth=2)
            
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Nodes', color='b')
            ax7_twin.set_ylabel('Edges', color='r')
            ax7.set_title('networkscaleTimedemoize', fontsize=12, fontweight='bold')
            ax7.tick_params(axis='x', rotation=45)
            ax7.grid(True, alpha=0.3)
            
            # mergegraphexample
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax7.legend(lines, labels, loc='upper left')
            
            ax8 = fig.add_subplot(gs[2, 2:])
            densities = [data['density'] for data in temporal_data]
            clustering = [data['avg_clustering'] for data in temporal_data]
            
            ax8_twin = ax8.twinx()
            line3 = ax8.plot(time_labels, densities, 'g-^', label='Density', linewidth=2)
            line4 = ax8_twin.plot(time_labels, clustering, 'm-d', label='Clustering Coefficient', linewidth=2)
            
            ax8.set_xlabel('Time')
            ax8.set_ylabel('Density', color='g')
            ax8_twin.set_ylabel('Clustering Coefficient', color='m')
            ax8.set_title('Network Structure Metrics Evolution', fontsize=12, fontweight='bold')
            ax8.tick_params(axis='x', rotation=45)
            ax8.grid(True, alpha=0.3)
            
            lines = line3 + line4
            labels = [l.get_label() for l in lines]
            ax8.legend(lines, labels, loc='upper left')
        
        # 6. networkvisualizationpreview (bottom)
        ax9 = fig.add_subplot(gs[3, :])
        
        # createoneitemsimplify networklayoutuseinpreview
        if graph.number_of_nodes() <= 100:
            # smallgraphshowallnode
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
            preview_graph = graph
        else:
            # biggraphonlyshowhighDegreenumbernode
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:50]
            preview_nodes = [node for node, _ in top_nodes]
            preview_graph = graph.subgraph(preview_nodes)
            pos = nx.spring_layout(preview_graph, k=2, iterations=50, seed=42)
        
        # drawcreatenetwork
        node_colors = [self.entity_colors.get(preview_graph.nodes[node].get('node_type', ''), '#cccccc') 
                      for node in preview_graph.nodes()]
        node_sizes = [max(50, min(500, preview_graph.degree(node) * 20)) for node in preview_graph.nodes()]
        
        nx.draw_networkx_nodes(preview_graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8, ax=ax9)
        nx.draw_networkx_edges(preview_graph, pos, alpha=0.3, width=0.5, ax=ax9)
        
        ax9.set_title('Network Structure Preview', fontsize=14, fontweight='bold')
        ax9.axis('off')
        
        # addtotaltitle
        fig.suptitle('Graph Network Comprehensive Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # savedashboard
        dashboard_path = os.path.join(output_dir, 'comprehensive_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive dashboard saved to: {dashboard_path}")
        
        return dashboard_path
    
    def create_statistics_plot(self, graph: nx.Graph, output_dir: str):
        """Createstatisticsinformationfigure"""
        stats = self.analyze_graph_statistics(graph)
        
        # createsubgraph
        fig = plt.figure(figsize=(20, 12))
        
        # 1. nodeclasstypedistributionpiegraph
        ax1 = plt.subplot(2, 3, 1)
        node_types = stats['node_types']
        labels = [self.entity_names_en.get(t, t) for t in node_types.keys()]
        colors = [self.entity_colors.get(t, self.entity_colors['unknown']) for t in node_types.keys()]
        sizes = list(node_types.values())
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('node type distribution', fontsize=14, fontweight='bold')
        
        # 2. Degree Distributiondirectmethodgraph
        ax2 = plt.subplot(2, 3, 2)
        degrees = [d for n, d in graph.degree()]
        ax2.hist(degrees, bins=min(30, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('node degree')
        ax2.set_ylabel('node count')
        ax2.set_title('degree distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. highfrequencyrealsolidTop10
        ax3 = plt.subplot(2, 3, 3)
        node_freq = [(node, data.get('frequency', 1)) for node, data in graph.nodes(data=True)]
        node_freq.sort(key=lambda x: x[1], reverse=True)
        top_nodes = node_freq[:10]
        
        if top_nodes:
            nodes, freqs = zip(*top_nodes)
            # cutbreaklongnodename
            short_nodes = [n[:15] + '...' if len(n) > 15 else n for n in nodes]
            bars = ax3.barh(range(len(short_nodes)), freqs, color='lightcoral')
            ax3.set_yticks(range(len(short_nodes)))
            ax3.set_yticklabels(short_nodes)
            ax3.set_xlabel('frequency')
            ax3.set_title('top 10 frequent entities', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # addnumbervaluelabel
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}', ha='left', va='center')
        
        # 4. basicstatisticsinformationtext
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        stats_text = f"""
        graphstatisticsinformation
        ═══════════════════
        Node Count: {stats['nodes']:,}
        Edge Count: {stats['edges']:,}
        graphDensity: {stats['density']:.4f}
        
        Degreestatistics
        ───────────────────
        averageDegreenumber: {stats['avg_degree']:.2f}
        maximumDegreenumber: {stats['max_degree']}
        minimumDegreenumber: {stats['min_degree']}
        
        connectthroughproperty
        ───────────────────
        yesnoconnectthrough: {'yes' if stats['is_connected'] else 'no'}
        """
        
        if 'diameter' in stats:
            stats_text += f"graph diameter: {stats['diameter']}\n"
            stats_text += f"average path length: {stats['avg_path_length']:.2f}\n"
        
        if 'connected_components' in stats:
            stats_text += f"Connected Components: {stats['connected_components']}\n"
        
        stats_text += f"""
        frequencystatistics
        ───────────────────
        averagefrequency: {stats['avg_frequency']:.2f}
        maximumfrequency: {stats['max_frequency']}
        averagearticlenumber: {stats['avg_article_count']:.2f}
        maximumarticlenumber: {stats['max_article_count']}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 5. Connected Componentsanalyze
        ax5 = plt.subplot(2, 3, 5)
        if not stats['is_connected']:
            components = list(nx.connected_components(graph))
            component_sizes = [len(comp) for comp in components]
            component_sizes.sort(reverse=True)
            
            # onlyshowfront20itemConnected Components
            top_components = component_sizes[:20]
            ax5.bar(range(len(top_components)), top_components, color='lightgreen')
            ax5.set_xlabel('connected component rank')
            ax5.set_ylabel('component size')
            ax5.set_title('connected component size distribution', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'graphyesconnectthrough \nnoneneedanalyzeConnected Components', 
                    ha='center', va='center', transform=ax5.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax5.set_title('connected component analysis', fontsize=14, fontweight='bold')
        
        # 6. Entity Typeimportanceanalyze
        ax6 = plt.subplot(2, 3, 6)
        type_importance = []
        for entity_type, count in stats['node_types'].items():
            importance = self.entity_importance.get(entity_type, 1)
            weighted_importance = count * importance
            type_importance.append((entity_type, weighted_importance))
        
        type_importance.sort(key=lambda x: x[1], reverse=True)
        
        if type_importance:
            types, importances = zip(*type_importance)
            type_labels = [self.entity_names_en.get(t, t) for t in types]
            colors = [self.entity_colors.get(t, self.entity_colors['unknown']) for t in types]
            
            bars = ax6.bar(range(len(type_labels)), importances, color=colors)
            ax6.set_xlabel('entity type')
            ax6.set_ylabel('weighted importance')
            ax6.set_title('entity type importance analysis', fontsize=14, fontweight='bold')
            ax6.set_xticks(range(len(type_labels)))
            ax6.set_xticklabels(type_labels, rotation=45, ha='right')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # savestatisticsfigure
        stats_path = os.path.join(output_dir, 'graph_statistics.png')
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"statisticsChart saved to: {stats_path}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='visualizationrealsolidoffsystemgraph')
    parser.add_argument('--input', '-i', required=True, help='inputgraphfilepath (.graphml or .gexf)')
    parser.add_argument('--output_dir', '-o', required=True, help='outputdirectory')
    parser.add_argument('--filter_type', '-f', choices=['frequency', 'degree', 'none'], 
                       default='frequency', help='filterclasstype')
    parser.add_argument('--min_threshold', '-t', type=int, default=3, help='minimumthresholdvalue')
    parser.add_argument('--layout', '-l', 
                       choices=['spring', 'circular', 'kamada_kawai', 'spectral', 'hierarchical', 'clustered'],
                       default='spring', help='layoutalgorithm')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 16], help='graphshapesize (wide high)')
    parser.add_argument('--entity_types', nargs='*', help='specifyneedshow Entity Type')
    parser.add_argument('--disable_interactive', action='store_true', help='disableInteractive visualization')
    parser.add_argument('--analysis_only', action='store_true', help='Only perform statistical analysis, do not generate graphs')
    parser.add_argument('--temporal_files', nargs='*', help='Timeordercolumnanalyze graphfilelist')
    parser.add_argument('--time_labels', nargs='*', help='Timeordercolumnanalyze Timelabel')
    parser.add_argument('--create_dashboard', action='store_true', help='Create comprehensive dashboard')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # initializevisualizationtool
    visualizer = GraphVisualizer()
    
    # Timeordercolumnanalyzemode
    if args.temporal_files:
        print("enterTimeordercolumnanalyzemode...")
        temporal_data = visualizer.create_temporal_analysis(
            args.temporal_files, 
            args.output_dir, 
            args.time_labels
        )
        
        if temporal_data and args.create_dashboard:
            # useusefirstitemgraphfilecreatedashboard
            main_graph = visualizer.load_graph(args.temporal_files[0])
            metrics = visualizer.add_network_analysis_metrics(main_graph)
            visualizer.create_comprehensive_dashboard(main_graph, args.output_dir, temporal_data, metrics)
        
        print(f"\nTimeordercolumnanalyzeaccomplish！allFile saved to: {args.output_dir}")
        return
    
    # loadgraph
    print(f"Loading graph file: {args.input}")
    graph = visualizer.load_graph(args.input)
    print(f"Successfully loaded graph: {graph.number_of_nodes()}  nodes, {graph.number_of_edges()}  edges")
    
    # ifspecifyEntity Type，firstenterrowfilter
    if args.entity_types:
        print(f"byEntity Typefilter: {args.entity_types}")
        graph = visualizer.filter_by_entity_types(graph, args.entity_types)
        print(f"Filtered graph: {graph.number_of_nodes()}  nodes, {graph.number_of_edges()}  edges")
    
    # generatestatisticsanalyze
    print("Performing graph statistical analysis...")
    stats = visualizer.analyze_graph_statistics(graph, args.output_dir)
    
    # generateEntity Typeanalyze
    print("positiveatenterrowEntity Typeanalyze...")
    visualizer.create_entity_type_analysis(graph, args.output_dir)
    
    # addNetwork analysis metrics
    print("Performing network analysis...")
    metrics = visualizer.add_network_analysis_metrics(graph)
    visualizer.create_network_analysis_visualization(graph, metrics, args.output_dir)
    
    # Create comprehensive dashboard
    if args.create_dashboard:
        visualizer.create_comprehensive_dashboard(graph, args.output_dir, None, metrics)
    
    # ifonlyenterrowanalyze，skipgraphshapegenerate
    if args.analysis_only:
        print(f"\nAnalysis completed! All analysis results saved to: {args.output_dir}")
        return
    
    # generateMain visualization graph
    output_path = os.path.join(args.output_dir, f'graph_visualization_{args.layout}.png')
    print(f"Generating visualization graph...")
    
    visualizer.visualize_global_graph(
        graph=graph,
        output_path=output_path,
        filter_type=args.filter_type,
        min_threshold=args.min_threshold,
        layout=args.layout,
        figsize=tuple(args.figsize),
        enable_interactive=not args.disable_interactive
    )
    
    # ifyespartlayerorclustered layout，amountoutsidegeneratespringlayoutmakeforcontrast
    if args.layout in ['hierarchical', 'clustered']:
        spring_path = os.path.join(args.output_dir, 'graph_visualization_spring_comparison.png')
        print("Generating Spring layout comparison...")
        visualizer.visualize_global_graph(
            graph=graph,
            output_path=spring_path,
            filter_type=args.filter_type,
            min_threshold=args.min_threshold,
            layout='spring',
            figsize=tuple(args.figsize),
            enable_interactive=False
        )

    print(f"\nVisualization completed! All files saved to: {args.output_dir}")
    print(f"Generated files include:")
    print(f"  - Statistical analysis chart: graph_statistics.png")
    print(f"  - Entity Type analyze: entity_type_analysis.png")
    print(f"  - Entity Type statistics: entity_type_statistics.csv")
    print(f"  - Network analysis metrics: network_analysis_metrics.png")
    print(f"  - Top nodes analysis: top_nodes_analysis.csv")

    if args.create_dashboard:
        print(f"  - Comprehensive dashboard: comprehensive_dashboard.png")

    if not args.analysis_only:
        print(f"  - Main visualization graph: graph_visualization_{args.layout}.png")
        if not args.disable_interactive:
            print(f"  - Interactive visualization: graph_visualization_{args.layout}_interactive.html")

if __name__ == "__main__":
    main()