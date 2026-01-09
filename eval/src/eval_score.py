#!/usr/bin/env python3
"""
Agent结果评分系统
支持不同类型问题的评分：reasoning（exact match）、induction（recall@k/ndcg@k）、
summary（GPT-score）、solution（GPT-5判断）
"""

import json5
import os
import re
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from openai import OpenAI
from eval_prompt import *
import math
import glob


class EvalScorer:
    def __init__(self):
        """
        初始化评分器
        
        Args:
            results_dir: 结果文件目录
            knowledge_file: solution类问题的知识文件路径
            openai_api_key: OpenAI API密钥，如果不提供则从环境变量获取
        """
        self.results_dir = ""
        self.knowledge_file = ""
        self.knowledge_data = self._load_knowledge_data()
        
        self.client = OpenAI(
            api_key="", # 
            base_url="",
        )

    def _load_knowledge_data(self) -> Dict[str, Any]:
        """加载solution类问题的知识数据"""
        knowledge_data = {}
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json5.loads(line.strip())
                    # 使用question作为key来匹配
                    #print(len(data['extracted_solutions']))
                    #print(data['extracted_solutions'][0]['analytical knowledge'])
                    knowledge_data[data['question']] = {
                        'analytical_knowledge': [item.get('analytical_knowledge', []) for item in data['extracted_solutions']],
                        'technical_knowledge': [item.get('technical_knowledge', []) for item in data['extracted_solutions']],
                        'explanation': [item.get('explanation', []) for item in data['extracted_solutions']],
                        'solution': data['answer']
                    }
        return knowledge_data
    
    def exact_match_score(self, prediction: str, answer: str) -> float:
        """
        Reasoning类问题的exact match评分
        
        Args:
            prediction: 模型预测
            answer: 标准答案
            
        Returns:
           float: 1.0 if exact match, 0.0 otherwise
        """
        prompt = REASONING_PROMPT.format(reference=answer, model_output=prediction)
        
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        
        content = response.choices[0].message.content
        #print(content)
        #提取json
        try:
            json_content = json5.loads(content)
            result = json_content.get("score", 0)
            return float(result)
        except Exception as e:
            print(e)
            return 0
        
    
    def recall_at_k(self, prediction_list: List[str], answer_list: List[str], k: int = 5) -> float:
        """
        计算recall@k
        
        Args:
            prediction_list: 预测结果列表
            answer_list: 标准答案列表
            k: top-k
            
        Returns:
            float: recall@k score
        """
        metric = f"recall@{k}"
        prompt = INDUCTION_PROMPT.format(metric=metric, answer_list=answer_list, prediction_list=prediction_list)
        
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        
        content = response.choices[0].message.content
        #print(content)
        #提取json
        try:
            json_content = json5.loads(content)
            result = json_content.get(metric, 0)
            return float(result)
        except Exception as e:
            print(e)
            return 0

    
    def ndcg_at_k(self, prediction_list: List[str], answer_list: List[str], k: int = 5) -> float:
        """
        计算NDCG@k
        
        Args:
            prediction_scores: 预测分数列表
            answer_relevance: 标准答案相关性列表
            k: top-k
            
        Returns:
            float: NDCG@k score
        """
        metric = f"ndcg@{k}"
        prompt = INDUCTION_PROMPT.format(metric=metric, answer_list=answer_list, prediction_list=prediction_list)
        
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],

        )
        content = response.choices[0].message.content
        #print(content)
        #提取json
        try:
            json_content = json5.loads(content)
            result = json_content.get(metric, 0)
            return float(result)
        except Exception as e:
            print(e)
            return 0
    
    def gpt_score_evaluation(self, prediction: str, reference: str) -> Dict[str, Any]:
        """
        使用GPT-5进行summary类问题的评分
        
        Args:
            prediction: 模型预测
            reference: 参考答案
            
        Returns:
            Dict: 包含各维度分数的字典
        """
        example_output_json = {
            "fluency": 0,
            "relevance": 0,
            "accuracy": 0,
            "creativity": 0,
            "overall_quality": 0,
            "average_score": 0.0,
            "comments": ""
            }
        prompt = GPT_SCORE_PROMPT.format(
            reference=reference,
            model_output=prediction,
            example_output_json=example_output_json
        )
        #print(prompt)

        response = self.client.chat.completions.create(
            model="gpt-5",  # 使用gpt-5作为评估模型
            messages=[
                {"role": "system", "content": "You are an expert evaluator for text quality assessment."},
                {"role": "user", "content": prompt}
            ],

        )
        #print(response.choices[0].message.content)
        # 解析JSON响应
        result_text = response.choices[0].message.content
        if result_text.startswith('```json'):
            result_text = response_text[7:]
        if result_text.endswith('```'):
            result_text = response_text[:-3]
        result_text = result_text.strip()
        #print(result_text)  

        try:
            json_content = json5.loads(result_text)
            return json_content
        except :
            print(f"无法解析JSON: {result_text}")
            # 如果无法解析JSON，返回默认分数
            return {result_text}
                
    
    def solution_evaluation(self, question: str, prediction: str, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用GPT-5进行solution类问题评分
        
        Args:
            question: 问题
            prediction: 模型预测结果
            knowledge_data: 知识数据
            
        Returns:
            包含分析分数和技术分数的字典
        """


        # 从knowledge_data中提取所需信息
        #print(knowledge_data['solution'])

        
        # 使用第一个solution作为参考（可以根据需要调整）
        solution_data = knowledge_data
        
        prompt = SOLUTION_PROMPT.format(
            task=question,
            solution=prediction,
            analysis_knowledge=solution_data.get('analytical knowledge', ''),
            technology_knowledge=solution_data.get('technical knowledge', ''),
            golden_explanation=solution_data.get('explanation', ''),
            golden_solution=solution_data.get('solution', '')
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert solution evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            #print(content)
            # 解析评分结果

            try:
                scores = json5.loads(content)
                analysis_score = scores.get("Analysis Score", 0)
                technology_score = scores.get("Technology Score", 0)
                average_score = (analysis_score + technology_score) / 2
                
                return {
                    "analysis_score": analysis_score,
                    "technology_score": technology_score,
                    "average_score": average_score
                }
            except Exception as e:
                print(e)
                pass
            

        except Exception as e:
            print(f"Solution评分出错: {e}")
            return {"analysis_score": 50, "technology_score": 50, "average_score": 50.0}
    
    def evaluate_file(self, file_path: str, question_type: str) -> Dict[str, Any]:
        """
        评估单个结果文件
        
        Args:
            file_path: 结果文件路径
            question_type: 问题类型 (reasoning/induction/summary/solution)
            
        Returns:
            Dict: 评估结果
        """
        results = []
        print("begin evaluate file")
        #print(self.knowledge_data)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json5.loads(line)
                    answer = data.get('answer', '')
                    #print(answer)
                    prediction = data.get('prediction', '')
                    #print(prediction)
                    question = data.get('question', '')
                    #print(question)
                    
                    if question_type == 'reasoning':
                        score = self.exact_match_score(prediction, answer)
                        #print(score)
                        results.append({
                            'line': line_num,
                            'score': score,
                            'method': 'exact_match'
                        })
                    
                    elif question_type == 'induction':
                        # 假设answer和prediction都是列表格式的字符串
                        try:
                            #print(type(answer))
                            #answer_json = json5.loads(answer)
                            answer_list = answer.get('titles', [])
                            #print(answer_list)
                            prediction_list = [prediction]
                            if not isinstance(answer_list, list):
                                answer_list = [answer_list]
                            if not isinstance(prediction_list, list):
                                prediction_list = [prediction_list]
                            
                            recall_5 = self.recall_at_k(prediction_list, answer_list, k=5)
                            #recall_10 = self.recall_at_k(prediction_list, answer_list, k=10)
                            
                            # 对于NDCG，需要相关性分数，这里简化处理

                            ndcg_5 = self.ndcg_at_k(prediction_list, answer_list, k=5)
                            #ndcg_10 = self.ndcg_at_k(prediction_list, answer_list, k=10)
                            
                            results.append({
                                'line': line_num,
                                'recall@5': recall_5 * 100,
                                #'recall@10': int(recall_10 * 100),
                                'ndcg@5': ndcg_5 * 100,
                                #'ndcg@10': int(ndcg_10 * 100),
                                'method': 'recall_ndcg'
                            })
                        except Exception as e:
                            print(f"处理induction数据出错 (line {line_num}): {e}")
                            results.append({
                                'line': line_num,
                                'error': str(e),
                                'method': 'recall_ndcg'
                            })
                    
                    elif question_type == 'summary':
                        gpt_scores = self.gpt_score_evaluation(prediction, answer)
                        results.append({
                            'line': line_num,
                            'gpt_scores': gpt_scores,
                            'method': 'gpt_score'
                        })
                        
                    elif question_type == 'solution':
                        # 获取对应的知识数据
                        knowledge_data = self.knowledge_data.get(question, {})
                        
                        if not knowledge_data:
                            # 如果找不到精确匹配，尝试模糊匹配
                            for key in self.knowledge_data.keys():
                                if question in key or key in question:
                                    knowledge_data = self.knowledge_data[key]
                                    break
                        
                        if knowledge_data:
                            solution_scores = self.solution_evaluation(question, prediction, knowledge_data)
                        else:
                            print(f"未找到问题的知识数据: {question[:100]}...")
                            solution_scores = {"analysis_score": 0, "technology_score": 0, "average_score": 0.0}
                        
                        results.append({
                            'line': line_num,
                            'solution_scores': solution_scores,
                            'method': 'solution_evaluation'
                        })
                    
                except Exception as e:
                    print(f"处理第{line_num}行数据出错: {e}")
                    results.append({
                        'line': line_num,
                        'error': str(e)
                    })
        
        return {
            'file_path': file_path,
            'question_type': question_type,
            'results': results,
            'summary': self._calculate_summary(results, question_type)
        }
    
    def _calculate_summary(self, results: List[Dict], question_type: str) -> Dict[str, Any]:
        """计算评估结果的汇总统计"""
        if not results:
            return {}
        
        valid_results = [r for r in results if 'error' not in r]
        
        if question_type == 'reasoning':
            scores = [r['score'] for r in valid_results if 'score' in r]
            return {
                'total_samples': len(results),
                'valid_samples': len(valid_results),
                'accuracy': sum(scores) / len(scores) if scores else 0.0,
                'correct_count': sum(scores) if scores else 0,
                'average_score': sum(scores) / len(scores) if scores else 0.0
            }
        
        elif question_type == 'induction':
            recall_5_scores = [r['recall@5'] for r in valid_results if 'recall@5' in r]
            recall_10_scores = [r['recall@10'] for r in valid_results if 'recall@10' in r]
            ndcg_5_scores = [r['ndcg@5'] for r in valid_results if 'ndcg@5' in r]
            ndcg_10_scores = [r['ndcg@10'] for r in valid_results if 'ndcg@10' in r]
            
            return {
                'total_samples': len(results),
                'valid_samples': len(valid_results),
                'recall_at_5': sum(recall_5_scores) / len(recall_5_scores) if recall_5_scores else 0.0,
                'ndcg_at_5': sum(ndcg_5_scores) / len(ndcg_5_scores) if ndcg_5_scores else 0.0,
                'average_score': (sum(recall_5_scores) / len(recall_5_scores) + sum(ndcg_5_scores) / len(ndcg_5_scores)) / 2 if recall_5_scores and ndcg_5_scores else 0.0
            }
        
        elif question_type == 'summary':
            gpt_scores = [r['gpt_scores'] for r in valid_results if 'gpt_scores' in r and 'average_score' in r['gpt_scores']]
            avg_scores = [s['average_score'] for s in gpt_scores]
            
            return {
                'total_samples': len(results),
                'valid_samples': len(valid_results),
                'average_score': sum(avg_scores) / len(avg_scores) if avg_scores else 0.0,
                'avg_fluency': sum([s['fluency'] for s in gpt_scores]) / len(gpt_scores) if gpt_scores else 0.0,
                'avg_relevance': sum([s['relevance'] for s in gpt_scores]) / len(gpt_scores) if gpt_scores else 0.0,
                'avg_accuracy': sum([s['accuracy'] for s in gpt_scores]) / len(gpt_scores) if gpt_scores else 0.0,
                'avg_creativity': sum([s['creativity'] for s in gpt_scores]) / len(gpt_scores) if gpt_scores else 0.0,
                'avg_overall_quality': sum([s['overall_quality'] for s in gpt_scores]) / len(gpt_scores) if gpt_scores else 0.0
            }
        
        elif question_type == 'solution':
            solution_scores = [r['solution_scores'] for r in valid_results if 'solution_scores' in r]
            analysis_scores = [s.get('analysis_score', 0) for s in solution_scores]
            technology_scores = [s.get('technology_score', 0) for s in solution_scores]
            
            return {
                'total_samples': len(results),
                'valid_samples': len(valid_results),
                'avg_analysis_score': sum(analysis_scores) / len(analysis_scores) if analysis_scores else 0.0,
                'avg_technology_score': sum(technology_scores) / len(technology_scores) if technology_scores else 0.0,
                'average_score': sum([a + t for a, t in zip(analysis_scores, technology_scores)]) / (2 * len(analysis_scores)) if analysis_scores and technology_scores else 0.0
            }
        
        return {}
    
    def evaluate_all_results(self) -> Dict[str, Any]:
        """评估所有结果文件"""
        all_results = {}
        
        # 遍历所有模型目录
        for model_dir in os.listdir(self.results_dir):
            model_path = os.path.join(self.results_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            
            all_results[model_dir] = {}
            
            # 遍历每个模型的问题类型目录
            for question_dir in os.listdir(model_path):
                question_path = os.path.join(model_path, question_dir)
                if not os.path.isdir(question_path):
                    continue
                
                # 从目录名提取问题类型
                question_type = question_dir.split('_')[0]
                
                # 查找结果文件
                result_files = glob.glob(os.path.join(question_path, "*.jsonl"))
                
                if result_files:
                    # 评估第一个找到的结果文件
                    result_file = result_files[0]
                    print(f"评估: {model_dir}/{question_dir}")
                    
                    evaluation_result = self.evaluate_file(result_file, question_type)
                    all_results[model_dir][question_dir] = evaluation_result
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存评估结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json5.dump(results, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存到: {output_file}")


def main():
    """主函数"""
    # 初始化评分器
    scorer = EvalScorer()
    
    # 评估所有结果
    print("开始评估所有结果...")
    all_results = scorer.evaluate_all_results()
    
    # 保存结果
    output_file = "/share/project/xionglei/code/eval/src/evaluation_results.json"
    scorer.save_results(all_results, output_file)
    
    # 打印汇总信息
    print("\n=== 评估汇总 ===")
    for model_name, model_results in all_results.items():
        print(f"\n模型: {model_name}")
        for question_type, result in model_results.items():
            summary = result.get('summary', {})
            print(f"  {question_type}: {summary}")


if __name__ == "__main__":
    main()