#!/usr/bin/env python3
"""
Induction QA Generator
基于induction data和PDF链接生成显式和隐式QA对的脚本
"""
import json
import argparse
import os
import sys
from typing import Dict, List, Any, Optional
from openai import OpenAI

# 添加paper_lookup路径
sys.path.append("paper_process")
try:
    from paper_lookup import PaperLookup
except ImportError:
    # 简单的PaperLookup实现作为后备
    print("Warning: Could not import PaperLookup from ./paper_process. Using fallback implementation.")
    class PaperLookup:
        def __init__(self, jsonl_path):
            self.index = {}
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    title = item.get('title', '').strip().lower()
                    if title:
                        self.index[title] = item
        
        def find_by_title(self, title):
            return self.index.get(title.strip().lower())

class InductionQAGenerator:
    """Induction QA生成器"""
    
    def __init__(self, api_key: str, base_url: str, 
                 model: str, lookup_jsonl: str):
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
        self.lookup_jsonl = lookup_jsonl
        print(f"正在加载PaperLookup索引: {lookup_jsonl}")
        self.paper_lookup = PaperLookup(lookup_jsonl)
        
        # 定义generation_prompt
        self.generation_prompt = """
You are an expert researcher and data generator. Based on the provided scientific papers (PDFs), their titles, and common entities, your task is to generate two types of queries (Explicit and Implicit) that can be answered by these papers collectively.

Input Information:
- Paper Titles: {titles}
- Common Entities: {common_entities}

Task:
Generate two distinct queries where the answer is the provided list of papers.

1. Explicit Theme Query:
   - Must contain core information from the 'Common Entities'.
   - Extract the shared theme from the papers' content.
   - The question must be explicit and clear (e.g., "Help find papers about xxx").
   - Generalize specific technical terms (fuzzy matching) to prevent direct keyword search hits. Focus on the concept rather than exact strings.

2. Implicit Theme Query:
   - Embed the specific problem within a practical, real-world scenario (e.g., "Find works that can help me process long video").
   - Integrate the core information/theme into the scenario.
   - Generalize specific technical terms (fuzzy matching) to prevent direct keyword search hits.

Output Requirement:
You must output a single valid JSON object containing exactly two fields: "explicit_query" and "implicit_query". Do not include any markdown formatting or explanation outside the JSON.

Example Output Format:
{{
  "explicit_query": "Help me find papers discussing optimization techniques for large-scale neural network training...",
  "implicit_query": "I am working on a project that requires efficient model updates with limited memory, are there any works..."
}}
"""
    
    def read_input_file(self, file_path: str) -> List[Dict[str, Any]]:
        """读取输入 JSONL文件"""
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
        pdf_links = []
        for title in titles:
            result = self.paper_lookup.find_by_title(title)
            if result and result.get("pdf_link"):
                pdf_links.append(result.get("pdf_link"))
            elif result and result.get("metadata") and result["metadata"].get("pdf_link"):
                pdf_links.append(result["metadata"]["pdf_link"])
            else:
                # 如果找不到链接，不添加或者添加None，这里选择跳过以便只发送有效的链接
                print(f"Warning: PDF link not found for title: {title}")
                pass
        return pdf_links
    
    def generate_queries(self, titles: List[str], common_entities: List[str], pdf_links: List[str]) -> Dict[str, str]:
        """调用模型生成查询"""
        
        # 构造prompt
        prompt = self.generation_prompt.format(
            titles=json.dumps(titles, ensure_ascii=False),
            common_entities=json.dumps(common_entities, ensure_ascii=False)
        )
        
        file_inputs = []
        for link in pdf_links:
            # 确保链接有效且是PDF (简单检查)
            if link and link.startswith("http"):
                file_inputs.append({"type": "input_file", "file_url": link})
        
        # 限制文件数量，防止超出模型限制（假设只用前5个）
        if len(file_inputs) > 5:
            file_inputs = file_inputs[:5]
            
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ] + file_inputs
                }
            ]
            
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature=0.7,
            )
            
            output_text = response.output_text
            # 清理markdown代码块标记
            if output_text.startswith("```json"):
                output_text = output_text[7:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]
            output_text = output_text.strip()
            
            return json.loads(output_text)
            
        except Exception as e:
            print(f"Error generating queries: {e}")
            # 返回空结果或错误信息
            return {
                "explicit_query": f"Error generating explicit query: {str(e)}",
                "implicit_query": f"Error generating implicit query: {str(e)}"
            }

    def process_file(self, input_file: str, output_file: str, max_entries: Optional[int] = None):
        """主处理流程"""
        print(f"读取输入文件: {input_file}")
        input_data = self.read_input_file(input_file)
        
        if max_entries:
            input_data = input_data[:max_entries]
            print(f"限制处理条目数为: {max_entries}")
        
        print(f"共找到 {len(input_data)} 个条目")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 打开输出文件准备增量写入
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, item in enumerate(input_data):
                print(f"\n处理第 {i+1}/{len(input_data)} 个条目...")
                
                articles = item.get("articles", [])
                common_entities = item.get("common_entities", [])
                
                titles = [article.get("title") for article in articles if article.get("title")]
                
                if not titles:
                    print("  没有找到文章标题，跳过")
                    continue
                
                # 获取PDF链接
                pdf_links = self.get_pdf_links_for_titles(titles)
                print(f"  找到 {len(pdf_links)} 个PDF链接")
                
                if not pdf_links:
                    print("  没有有效的PDF链接，无法根据内容生成，跳过")
                    continue
                
                # 生成查询
                queries = self.generate_queries(titles, common_entities, pdf_links)
                
                if "explicit_query" in queries and "implicit_query" in queries:
                    # 构造两个QA对
                    qa_explicit = {
                        "question": queries["explicit_query"],
                        "answer": articles,
                        "type": "explicit",
                        "source_titles": titles,
                        "common_entities": common_entities
                    }
                    
                    qa_implicit = {
                        "question": queries["implicit_query"],
                        "answer": articles,
                        "type": "implicit",
                        "source_titles": titles,
                        "common_entities": common_entities
                    }
                    
                    # 写入文件
                    f_out.write(json.dumps(qa_explicit, ensure_ascii=False) + '\n')
                    f_out.write(json.dumps(qa_implicit, ensure_ascii=False) + '\n')
                    f_out.flush() # 确保写入
                    print("  ✅ 成功生成并写入 2 个QA对")
                else:
                    print("  ❌ 生成查询失败或格式错误")

def main():
    parser = argparse.ArgumentParser(description="生成Induction QA数据")
    parser.add_argument("--input", "-i", default="xxxx.jsonl", help="输入文件路径")
    parser.add_argument("--output", "-o", default="./results/induction_xxxx.jsonl", help="输出文件路径")
    parser.add_argument("--api_key", default='', help="OpenAI API密钥")
    parser.add_argument("--base_url", default="", help="OpenAI API基础URL")
    parser.add_argument("--model", default="gpt-5", help="使用的模型名称")
    parser.add_argument("--lookup_jsonl", "-l", default="paper_process/paper_map/xxxx.jsonl", help="用于查找PDF链接的JSONL文件路径")
    parser.add_argument("--max_entries", type=int, help="最大处理条目数（用于测试）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
        
    generator = InductionQAGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        lookup_jsonl=args.lookup_jsonl
    )
    
    generator.process_file(args.input, args.output, args.max_entries)

if __name__ == "__main__":
    main()

