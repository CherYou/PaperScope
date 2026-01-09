#!/usr/bin/env python3
"""
Summary QA Generator
基于induction_xxx_xxx.jsonl文件生成summary问题的脚本
"""
import json
import argparse
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI
import sys
sys.path.append("/share/project/xionglei/code/paper_process")
from paper_lookup import PaperLookup


class SummaryQAGenerator:
    """Summary问题生成器"""
    
    def __init__(self, api_key: str, base_url: str , 
                 model: str, lookup_jsonl: str = None):
        """
        初始化生成器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型
            lookup_jsonl: 用于查找PDF链接的JSONL文件路径
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.paper_lookup = PaperLookup(lookup_jsonl) if lookup_jsonl else print("没有提供lookup_jsonl文件")
        
        # 问题后缀列表
        self.question_suffix_list = [
            "Please summarize the development trend of their methods.",
            "Please analyze the development defects of these methods.", 
            "Please compare and analyze the experimental results of these methods."
        ]
    
    def read_induction_file(self, file_path: str) -> List[Dict[str, Any]]:
        """读取induction JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"解析JSON失败: {e}")
                        continue
        return data
    
    def get_pdf_links_for_titles(self, titles: List[str]) -> List[str]:
        """根据标题获取PDF链接"""
        if not self.paper_lookup:
            # 如果没有lookup数据，返回占位符链接
            print("没有提供lookup_jsonl文件，返回占位符链接")
            return [f"placeholder_link_for_{title.replace(' ', '_')}.pdf" for title in titles]
        
        pdf_links = []
        for title in titles:
            #print(title)
            if self.paper_lookup.find_by_title(title):
                #print(self.paper_lookup.find_by_title(title).get("pdf_link", []))
                pdf_links.append(self.paper_lookup.find_by_title(title).get("pdf_link", []))
            else :
                pdf_links.append(f"placeholder_link_for_{title.replace(' ', '_')}.pdf")
        return pdf_links
    
    def create_summary_prompt(self, query: str, num_papers: int, common_entities: List[str], 
                             prompt_type: str = "trend") -> str:
        """
        创建summary prompt
        
        Args:
            query: 原始查询
            num_papers: 论文数量
            common_entities: 共同实体
            pdf_links: PDF链接列表
            prompt_type: prompt类型 (trend, gap, results_comparison)
        """
        example_json = {"answer": "xxxx"}
        
        
        if prompt_type == "trend":
            prompt = f"""
I have {num_papers} scientific literature papers that share common paper entities. Now, I have summarized a specific query from these {num_papers} papers. Below is the full text of these {num_papers} papers, the common entities, and the query. 
Please integrate this heterogeneous knowledge based on this information: Please summarize the development trend of their methods.
Requirements: 
1. The development trend you summarize can only come from the content of the scientific literature itself and the related information I have given you. You cannot fabricate or arbitrarily add content; 
2. The development trend should reflect certain characteristics, for example: the scalability of the method; the changing pattern of the method over time; the change in the field of the method, etc. (If there is a time characteristic, it must be reflected)
Common entities: {common_entities}
Query: {query}
Please output your answer in json format:
{example_json}
"""
        elif prompt_type == "gap":
            prompt = f"""
I have {num_papers} scientific literature papers that share common paper entities. Now, a specific query has been summarized from these {num_papers} papers. Below is the full text of these {num_papers} papers, the common entities, and the query. 
Please integrate this heterogeneous knowledge based on this information: Please summarize the development defects of their methods.
Requirements: 
1. The development defects you summarize can only come from the content of the scientific literature itself and the related information I have given you. You cannot fabricate or arbitrarily add content; 
2. The defects should be complete and detailed, and each method needs to have its defects clearly identified.
Common entities: {common_entities}
Query: {query}
Please output your answer in json format:
{example_json}
"""
        else:  # results_comparison
            prompt = f"""
I have {num_papers} scientific literature papers that share common paper entities. Now, a specific query has been summarized from these {num_papers} papers. Below is the full text of these {num_papers} papers, the interpretation of their figures and tables, the common entities, and the topic. 
Please integrate this heterogeneous knowledge based on this information: Please conduct a detailed comparison and summary of the experimental results of their methods.
Requirements: 
1. The comparison of experimental results you summarize must only come from the content of the scientific literature itself and the related information I have provided. You cannot fabricate or arbitrarily add content; 
2. The experimental results must be derived from the information in the figures and tables and must be compared; 
3. Your summary should include the following: the performance of all methods on any identical datasets, for example, which method performs better and which performs worse; the performance of all methods on similar types of datasets; and the performance of all methods on specific datasets. (If there are identical datasets, you must point this out.)
Common entities: {common_entities}
Query: {query}
Please output your answer in json format:
{example_json}
"""
        
        return prompt
    
    def generate_summary_answer(self, pdf_links: List[str], prompt: str) -> str:
        """使用OpenAI API生成summary答案"""
        try:
            file_links = [{"type": "input_file", "file_url": file} for file in pdf_links]
            input_data = [
                {
                    "role": "user", 
                    "content": [
                        { 
                            "type": "input_text",
                            "text": prompt
                        }
                    ] + file_links
                }
            ]
            
            response = self.client.responses.create(
            model="gpt-5",
            input=input_data,
            temperature=1.0,
        )
            print(response)
            # 尝试解析JSON格式的回答
            try:
                json_response = json.loads(response.output_text)
                return json_response.get("answer", response)
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接返回文本
                return response.output_text
        except Exception as e:
            print(f"生成答案时出错: {e}")
            return f"Error generating answer: {str(e)}"
    
    def read_processed_entries(self, output_file: str) -> set:
        """
        读取已处理的条目，用于断点续传
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            已处理条目的标识集合（使用original_question、titles和prompt_type作为标识）
        """
        processed = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                # 使用original_question、titles和prompt_type组合作为唯一标识
                                # 因为一个条目会生成3种不同类型的QA对
                                identifier = (
                                    item.get("original_question", ""),
                                    tuple(sorted(item.get("titles", []))),
                                    item.get("prompt_type", "")
                                )
                                processed.add(identifier)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"读取已处理条目时出错: {e}")
        return processed
    
    def write_qa_pair_incremental(self, output_file: str, qa_pair: Dict[str, Any]):
        """
        增量写入QA对到文件
        
        Args:
            output_file: 输出文件路径
            qa_pair: 要写入的QA对
        """
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
    
    def process_induction_file(self, input_file: str, output_file: str, max_entries: Optional[int] = None):
        """
        处理induction文件并生成summary QA对（支持增量输出和断点续传）
        
        Args:
            input_file: 输入的induction JSONL文件
            output_file: 输出的QA对JSONL文件
            max_entries: 最大处理条目数（用于测试）
        """
        print(f"读取induction文件: {input_file}")
        induction_data = self.read_induction_file(input_file)
        
        if max_entries:
            induction_data = induction_data[:max_entries]
            print(f"限制处理条目数为: {max_entries}")
        
        print(f"共找到 {len(induction_data)} 个条目")
        
        # 读取已处理的条目（断点续传）
        print(f"检查输出文件: {output_file}")
        processed_entries = self.read_processed_entries(output_file)
        print(f"已处理 {len(processed_entries)} 个QA对，将跳过这些QA对")
        
        success_count = 0
        skip_count = 0
        
        for i, item in enumerate(induction_data):
            print(f"\n处理第 {i+1}/{len(induction_data)} 个条目...")
            
            question = item.get("question", "")
            answer_data = item.get("answer", {})
            
            titles = item.get("source_titles", [])
            common_entities = item.get("common_entities", [])

            num_papers = len(titles)
            
            # 获取PDF链接
            pdf_links = self.get_pdf_links_for_titles(titles)
            print(f"  找到 {len(pdf_links)} 个PDF链接")
            print(f"  PDF链接: {pdf_links}")
            
            # 为每个prompt类型生成答案
            prompt_types = ["trend", "gap", "results_comparison"]
            
            for j, prompt_type in enumerate(prompt_types):
                # 检查是否已处理（断点续传）
                identifier = (question, tuple(sorted(titles)), prompt_type)
                if identifier in processed_entries:
                    print(f"  ⏭️  {prompt_type} 类型的QA对已处理，跳过")
                    skip_count += 1
                    continue
                
                print(f"  生成 {prompt_type} 类型的答案...")
                
                # 创建prompt
                prompt = self.create_summary_prompt(
                    query=question,
                    num_papers=num_papers,
                    common_entities=common_entities,
                    prompt_type=prompt_type
                )
                
                # 生成答案
                try:
                    summary_answer = self.generate_summary_answer(pdf_links, prompt)
                except Exception as e:
                    print(f"  ❌ 生成 {prompt_type} 类型答案时出错: {e}")
                    continue
                
                # 创建问题（原问题 + 后缀）
                summary_question = question + self.question_suffix_list[j]
                
                # 创建QA对
                qa_pair = {
                    "question": summary_question,
                    "answer": summary_answer,
                    "original_question": question,
                    "prompt_type": prompt_type,
                    "titles": titles,
                    "session": item.get("type", "unknown"),
                    "num_papers": num_papers,
                    "common_entities": common_entities,
                    "pdf_links": pdf_links
                }
                
                # 增量写入文件
                try:
                    self.write_qa_pair_incremental(output_file, qa_pair)
                    success_count += 1
                    print(f"  ✅ 已写入第 {success_count} 个QA对 ({prompt_type})")
                except Exception as e:
                    print(f"  ❌ 写入QA对时出错: {e}")
                    continue
        
        print(f"\n" + "="*60)
        print(f"处理完成！")
        print(f"  总条目数: {len(induction_data)}")
        print(f"  预期生成QA对数: {len(induction_data) * 3}")
        print(f"  跳过QA对: {skip_count}")
        print(f"  新生成QA对: {success_count}")
        print(f"  累计QA对: {len(processed_entries) + success_count}")
        print(f"  输出文件: {output_file}")
        print(f"="*60)


def main():
    parser = argparse.ArgumentParser(description="生成基于induction文件的summary QA对")
    parser.add_argument("--input", "-i", required=True, help="输入的induction JSONL文件路径")
    parser.add_argument("--output", "-o", default='./results/summary.jsonl', help="输出的QA对JSONL文件路径")
    parser.add_argument("--api_key", default='sk-Gvc3mDz4v6wJ9bevnPZgQpuTgsPard4OJBYDc3VMnxnl24HH', help="OpenAI API密钥")
    parser.add_argument("--base_url", default="https://api.key77qiqi.com/v1", help="OpenAI API基础URL")
    parser.add_argument("--model", default="gpt-5", help="使用的模型名称")
    parser.add_argument("--lookup_jsonl","-l", default="/share/project/xionglei/code/paper_process/paper_map/sampled_papers_500data_sampled_papers_500.jsonl", help="用于查找PDF链接的JSONL文件路径")
    parser.add_argument("--max_entries", type=int, help="最大处理条目数（用于测试）")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 创建生成器
    generator = SummaryQAGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        lookup_jsonl=args.lookup_jsonl
    )
    output_name = args.input.split('/')[-1].split('.')[0].split('induction_')[-1]
    # 处理文件
    generator.process_induction_file(args.input, args.output.replace('.jsonl', f'_{output_name}.jsonl'), args.max_entries)


if __name__ == "__main__":
    main()