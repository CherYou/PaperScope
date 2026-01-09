#!/usr/bin/env python3
"""
运行评分系统的主脚本
"""

from eval_score import EvalScorer
import os
import sys

def main():
    """主函数"""
    print("开始运行评分系统...")
    
    # 初始化评分器（不提供API key用于演示）
    scorer = EvalScorer()
    
    # 获取所有结果目录
    results_dir = "eval/src/results/"
    
    if not os.path.exists(results_dir):
        print(f"错误：结果目录不存在 {results_dir}")
        return
    
    # 遍历所有模型目录
    for model_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        print(f"\n处理模型: {model_name}")
        
        # 遍历所有问题类型目录
        for question_dir in os.listdir(model_path):
            question_path = os.path.join(model_path, question_dir)
            if not os.path.isdir(question_path):
                continue
                
            # 提取问题类型
            question_type = question_dir.split('_')[0]
            print(f"  问题类型: {question_type} ({question_dir})")
            
            # 查找结果文件
            result_files = []
            for file_name in os.listdir(question_path):
                if file_name.endswith('.jsonl'):
                    result_files.append(os.path.join(question_path, file_name))
            
            if not result_files:
                print(f"    警告：未找到结果文件")
                continue
            
            # 处理每个结果文件
            for result_file in result_files:
                print(f"    处理文件: {os.path.basename(result_file)}")
                
                try:
                    # 评估文件
                    results = scorer.evaluate_file(result_file, question_type)
                    
                    # 显示结果摘要
                    if 'summary' in results:
                        summary = results['summary']
                        print(f"      总样本数: {summary.get('total_samples', 0)}")
                        print(f"      平均分数: {summary.get('average_score', 0):.3f}")
                        
                        if question_type == 'reasoning':
                            print(f"      准确率: {summary.get('accuracy', 0):.3f}")
                        elif question_type == 'induction':
                            print(f"      Recall@3: {summary.get('recall_at_3', 0):.3f}")
                            print(f"      NDCG@3: {summary.get('ndcg_at_3', 0):.3f}")
                        elif question_type in ['summary', 'solution']:
                            print(f"      GPT评分: {summary.get('average_score', 0):.3f}")
                    
                except Exception as e:
                    print(f"      错误：处理文件失败 - {e}")

if __name__ == "__main__":
    main()