"""No Retrieval Tool - 根据原始问题直接从预定义数据集中查找相关文档并返回

input:
- question: 用户原始问题字符串

output:
- 返回与问题相关的文档内容，直接从数据集中查找answer_docs并返回对应的MD文档
"""

import logging
import os
import json
from typing import List, Dict, Any
from qwen_agent.tools.base import BaseTool, register_tool


class NoRetrievalEngine:
    def __init__(self, qa_file_path: str, docs_base_path: str):
        """
        初始化 NoRetrievalEngine
        
        Args:
            qa_file_path: QA数据集的jsonl文件路径
            docs_base_path: MD文档的基础目录路径（可以是单个目录或父目录）
        """
        self.qa_file_path = qa_file_path
        self.docs_base_path = docs_base_path
        self.qa_data = {}  # question -> answer_docs 映射
        
        # 获取所有可能的文档目录
        self.doc_dirs = self._get_doc_directories()
        
        self._load_qa_data()
    
    def _get_doc_directories(self) -> List[str]:
        """获取所有可能的文档目录"""
        doc_dirs = []
        
        # 如果指定的路径存在，直接使用
        if os.path.exists(self.docs_base_path):
            doc_dirs.append(self.docs_base_path)
        
        # 检查父目录下的所有子目录
        parent_dir = os.path.dirname(self.docs_base_path)
        if os.path.exists(parent_dir):
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path) and (item.startswith('ICLR') or item.startswith('NeurIPS')):
                    doc_dirs.append(item_path)
        
        logging.info(f"Found {len(doc_dirs)} document directories: {doc_dirs}")
        return doc_dirs
    
    def _load_qa_data(self):
        """加载QA数据集，建立问题到answer_docs的映射"""
        if not os.path.exists(self.qa_file_path):
            logging.warning(f"QA file not found: {self.qa_file_path}")
            return
        
        try:
            with open(self.qa_file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            question = data.get('question', '')
                            answer_docs = data.get('answer_docs', [])
                            if question and answer_docs:
                                self.qa_data[question] = answer_docs
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skipping line {line_number} due to JSON error: {e}")
                            continue
            
            logging.info(f"Loaded {len(self.qa_data)} QA pairs from {self.qa_file_path}")
        except Exception as e:
            logging.error(f"Error loading QA data: {e}")
    
    def _load_document(self, doc_id: str) -> str:
        """
        根据文档ID加载对应的MD文档，在所有可能的目录中搜索
        
        Args:
            doc_id: 文档ID（论文ID）
            
        Returns:
            文档内容字符串，如果文件不存在则返回错误信息
        """
        # 在所有文档目录中搜索
        for base_dir in self.doc_dirs:
            # MD文档路径: {base_dir}/{doc_id}/vlm/{doc_id}.md
            doc_path = os.path.join(base_dir, doc_id, 'vlm', f'{doc_id}.md')
            
            if os.path.exists(doc_path):
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logging.info(f"Found document {doc_id} at {doc_path}")
                    return content
                except Exception as e:
                    logging.error(f"Error reading document {doc_id}: {e}")
                    return f"[Error reading document {doc_id}: {str(e)}]"
        
        # 如果在所有目录中都找不到
        logging.warning(f"Document {doc_id} not found in any directory")
        return f"[Document {doc_id} not found in any directory]"
    
    def get_documents(self, question: str) -> str:
        """
        根据问题获取相关文档
        
        Args:
            question: 用户问题
            
        Returns:
            所有相关文档的内容，拼接成一个字符串
        """
        # 查找问题对应的answer_docs
        answer_docs = self.qa_data.get(question, [])
        
        if not answer_docs:
            return f"未找到与问题相关的文档。问题：{question}"
        
        # 加载所有answer_docs对应的MD文档
        all_content = []
        for doc_id in answer_docs:
            content = self._load_document(doc_id)
            all_content.append(f"\n{'='*80}\n文档ID: {doc_id}\n{'='*80}\n{content}")
        
        result = "\n".join(all_content)
        logging.info(f"Found {len(answer_docs)} documents for question")
        
        return result


@register_tool('NoRetrievalTool')
class NoRetrievalTool(BaseTool):
    name = "NoRetrievalTool"
    description = 'Directly retrieve pre-defined documents based on the original question without any retrieval process. Returns the parsed markdown documents corresponding to the question.'
    parameters = [
        {
            'name': 'question',
            'type': 'string',
            'description': 'The original question provided to the model',
            'required': True
        }
    ]
    
    def __init__(self):
        super().__init__()
        # QA数据集路径
        qa_file_path = os.getenv("QA_FILE_PATH", "./qa_constructor/reasoning_data_constructor/results/reasoning_questions.jsonl")
        # MD文档基础路径
        docs_base_path = os.getenv("DOCS_BASE_PATH", "./doc_parse/output")
        
        self.engine = NoRetrievalEngine(qa_file_path, docs_base_path)
    
    def call(self, params: str, **kwargs) -> str:
        """
        根据问题获取相关文档
        
        Args:
            params: JSON string containing 'question'
            
        Returns:
            相关文档内容的字符串
        """
        try:
            # 解析参数
            if isinstance(params, str):
                params_dict = json.loads(params)
            else:
                params_dict = params
            
            question = params_dict.get('question', '')
            
            if not question:
                return json.dumps({
                    "error": "Question parameter is required"
                }, ensure_ascii=False)
            
            # 获取文档
            result = self.engine.get_documents(question)
            
            return result
            
        except Exception as e:
            return f"Error: {str(e)}"


# Async wrapper for compatibility
async def no_retrieval_search(question: str) -> str:
    """Async wrapper for no retrieval functionality"""
    tool = NoRetrievalTool()
    params = {"question": question}
    return tool.call(params)


# Test the tool
if __name__ == "__main__":
    tool = NoRetrievalTool()
    
    # Test with a sample question from the dataset
    test_question = """Across the collection, focusing on Walker2d in each study's setting, how much does the proposed method beat its primary baseline on the reported metric, and which method achieves the largest gain?"""
    
    result = tool.call(json.dumps({"question": test_question}))
    print("="*80)
    print("Test Question:")
    print(test_question)
    print("\n" + "="*80)
    print("Result (first 2000 chars):")
    print(result[:2000] + "..." if len(result) > 2000 else result)

