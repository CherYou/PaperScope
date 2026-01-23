#!/usr/bin/env python3
"""
Solution QA Generator
基于induction_xxx_xxx.jsonl文件生成solution问题的脚本
"""
import json
import argparse
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "paper_process"))
from paper_lookup import PaperLookup


class SolutionQAGenerator:
    """Solution问题生成器"""
    
    def __init__(self, api_key: str, base_url: str, 
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
        
        # 定义extract_prompt
        self.extract_prompt = """
The above is a solution for a complex task.
You need to extract the following content from the input document, with the following requirements:
1. Extraction must strictly follow the template.
2. Extraction must strictly be based on the document content; no additional information may be added.
3. The extracted content must be detailed, including specific values and professional terms.

{
"title": "", # The title of the document
"requirement": "Under these conditions of..., complete the task of...", # A detailed and comprehensive description of the task in the document, covering all the conditions mentioned in the above analysis
"solution": "This paper proposes the solution of..., specifically, first, considering the conditions of..., and the challenges faced, using... technologies, through... achieved...; secondly,...", # Based on the explanation, a detailed and comprehensive introduction to the solution in the document
"analytical knowledge": [ # Extract the analysis of the task from the document, with specific values and professional terms
    { "idx": "analysis0",
    "condition": "Condition is...", # The description of task conditions in the document
    "challenge": "Under this condition, conducting... will face challenges such as..., which may lead to... consequences" # Based on the document's analysis, the challenges and potential negative consequences encountered in the task due to this condition
    },
    ...
],
"technical knowledge": [ # Extract the solutions for the task from the document, with specific values and professional terms
    { "idx": "technology0", "name": "...technology", # A key technology in the solution in the document
    "detail": "This technology applies to..., and can achieve..." # The scope and effects of the technology mentioned in the document, with specific values and professional terms
    },
    ...
],
"explanation": [ # Find the reasoning from the task to the solution in the document, corresponding to the analysis. For each analysis, an explanation must be provided; one analysis may correspond to one or more technologies
    { "idx": "explanation0",
    "content": "In response to analysis0, considering the conditions of... and the challenges that may arise, the use of technology0... achieves...", # Find the explanation in the document that describes how the mentioned technologies solve the challenges from the analysis
    },
    ...
],
}
"""
    
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
            if self.paper_lookup.find_by_title(title):
                pdf_links.append(self.paper_lookup.find_by_title(title).get("pdf_link", []))
            else:
                pdf_links.append(f"placeholder_link_for_{title.replace(' ', '_')}.pdf")
        return pdf_links
    
    def extract_solution_content(self, pdf_links: List[str]) -> List[Dict[str, Any]]:
        """使用extract_prompt从PDF中提取解决方案内容"""
        extracted_solutions = []
        print(pdf_links)
        for i, pdf_link in enumerate(pdf_links):
            try:
                file_links = [{"type": "input_file", "file_url": pdf_link}]
                input_data = [
                    {
                        "role": "user", 
                        "content": [
                            { 
                                "type": "input_text",
                                "text": self.extract_prompt
                            }
                        ] + file_links
                    }
                ]
                
                response = self.client.responses.create(
                    model=self.model,
                    input=input_data,
                    temperature=0.7,
                )
                
                # 尝试解析JSON格式的回答
                try:
                    json_response = json.loads(response.output_text)
                    extracted_solutions.append(json_response)
                    print(f"成功提取第 {i+1} 个PDF的解决方案内容")
                except json.JSONDecodeError:
                    print(f"第 {i+1} 个PDF提取结果不是有效JSON格式，跳过")
                    continue
                    
            except Exception as e:
                print(f"提取第 {i+1} 个PDF内容时出错: {e}")
                continue
        
        return extracted_solutions
    
    def create_qa_prompt(self, extracted_solutions: List[Dict[str, Any]]) -> str:
        """创建QA生成prompt"""
        example_json = {
            "query": "The generated query",
            "answer": "The comprehensive solution"
        }
        
        # 将提取的解决方案转换为JSONL格式字符串
        jsonl_files = "\n".join([json.dumps(solution, ensure_ascii=False) for solution in extracted_solutions])
        
        qa_prompt = f"""
Please refer to the analysis JSONL file containing the extracted solutions from two scientific papers. Select three different conditions and combine them to generate a specific scientific problem as the query. Then, based on the solutions corresponding to each condition, summarize a comprehensive solution as the answer.
Requirements:
The query should be simple, clear, and easy to understand, but must explicitly indicate the problem and its associated conditions.
The problem must be one that can only be solved using the proposed solution.
{jsonl_files}
Your answer must be a JSON object with the following format:
{example_json}
"""
        return qa_prompt
    
    def generate_solution_qa(self, qa_prompt: str) -> Dict[str, str]:
        """使用OpenAI API生成solution QA对"""
        try:
            input_data = [
                {
                    "role": "user", 
                    "content": [
                        { 
                            "type": "input_text",
                            "text": qa_prompt
                        }
                    ]
                }
            ]
            
            response = self.client.responses.create(
                model=self.model,
                input=input_data,
                temperature=1.0,
            )
            
            # 尝试解析JSON格式的回答
            try:
                json_response = json.loads(response.output_text)
                return json_response
            except json.JSONDecodeError:
                # 如果不是JSON格式，返回默认格式
                return {
                    "query": "Generated query (parsing error)",
                    "answer": response.output_text
                }
        except Exception as e:
            print(f"生成QA对时出错: {e}")
            return {
                "query": f"Error generating query: {str(e)}",
                "answer": f"Error generating answer: {str(e)}"
            }
    
    def read_processed_entries(self, output_file: str) -> set:
        """
        读取已处理的条目，用于断点续传
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            已处理条目的标识集合（使用original_question作为标识）
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
                                # 使用original_question和titles组合作为唯一标识
                                identifier = (
                                    item.get("original_question", ""),
                                    tuple(sorted(item.get("titles", [])))
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
        处理induction文件并生成solution QA对（支持增量输出和断点续传）
        
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
        print(f"已处理 {len(processed_entries)} 个条目，将跳过这些条目")
        
        success_count = 0
        skip_count = 0
        
        for i, item in enumerate(induction_data):
            print(f"\n处理第 {i+1}/{len(induction_data)} 个条目...")
            
            question = item.get("question", "")
            answer_data = item.get("answer", [])
            titles = []
            for answer in answer_data:
                title = answer.get("title", "")
                titles.append(title)
            
            # 检查是否已处理（断点续传）
            identifier = (question, tuple(sorted(titles)))
            if identifier in processed_entries:
                print(f"  ⏭️  条目已处理，跳过")
                skip_count += 1
                continue
            
            # 获取PDF链接
            pdf_links = self.get_pdf_links_for_titles(titles)
            print(f"  找到 {len(pdf_links)} 个PDF链接")
            
            # 从PDF中提取解决方案内容
            print("  提取PDF解决方案内容...")
            extracted_solutions = self.extract_solution_content(pdf_links)
            
            if not extracted_solutions:
                print("  ⚠️  没有成功提取到解决方案内容，跳过此条目")
                continue
            
            print(f"  成功提取 {len(extracted_solutions)} 个解决方案")
            
            # 创建QA生成prompt
            qa_prompt = self.create_qa_prompt(extracted_solutions)
            
            # 生成solution QA对
            print("  生成solution QA对...")
            solution_qa = self.generate_solution_qa(qa_prompt)
            
            # 创建完整的QA对记录
            qa_pair = {
                "question": solution_qa.get("query", "Generated query"),
                "answer": solution_qa.get("answer", "Generated answer"),
                "original_question": question,
                "titles": titles,
                "session": item.get("type", "unknown"),
                "num_papers":  len(titles),
                "common_entities": item.get("common_entities", []),
                "pdf_links": pdf_links,
                "extracted_solutions": extracted_solutions
            }
            
            # 增量写入文件
            try:
                self.write_qa_pair_incremental(output_file, qa_pair)
                success_count += 1
                print(f"  ✅ 已写入第 {success_count} 个QA对")
            except Exception as e:
                print(f"  ❌ 写入QA对时出错: {e}")
                continue
        
        print(f"\n" + "="*60)
        print(f"处理完成！")
        print(f"  总条目数: {len(induction_data)}")
        print(f"  跳过条目: {skip_count}")
        print(f"  新生成QA对: {success_count}")
        print(f"  累计QA对: {len(processed_entries) + success_count}")
        print(f"  输出文件: {output_file}")
        print(f"="*60)


def main():
    parser = argparse.ArgumentParser(description="生成基于induction文件的solution QA对")
    parser.add_argument("--input", "-i", required=True, help="输入的induction JSONL文件路径")
    parser.add_argument("--output", "-o", default='./results/solution.jsonl', help="输出的QA对JSONL文件路径")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""), help="OpenAI API密钥")
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", ""), help="OpenAI API基础URL")
    parser.add_argument("--model", default="gpt-4", help="使用的模型名称")
    parser.add_argument("--lookup_jsonl", "-l", default="./paper_process/paper_map/paper_map.jsonl", help="用于查找PDF链接的JSONL文件路径")
    parser.add_argument("--max_entries", type=int, help="最大处理条目数（用于测试）")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 创建生成器
    generator = SolutionQAGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        lookup_jsonl=args.lookup_jsonl
    )
    
    # 从输入文件名生成输出文件名
    output_name = args.input.split('/')[-1].split('.')[0].split('induction_')[-1]
    output_file = args.output.replace('.jsonl', f'_{output_name}.jsonl')
    
    # 处理文件
    generator.process_induction_file(args.input, output_file, args.max_entries)


if __name__ == "__main__":
    main()