"""
论文实体抽取模块
从JSONL文件中提取论文的各种实体信息，包括title、research_background、classification_tags等12个实体类型
支持使用远程API或本地vLLM进行实体提取
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
from openai import OpenAI
from transformers import AutoTokenizer

# vLLM相关导入（仅在使用本地模型时需要）
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("vLLM未安装，无法使用本地模型加速。请安装vLLM: pip install vllm")

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0,1")
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperEntityExtractor:
    """论文实体抽取器"""
    
    def __init__(self, jsonl_dir: str, pdf_dir: str, use_local_model: bool = False):
        """
        初始化实体抽取器
        
        Args:
            jsonl_dir: JSONL文件目录路径
            pdf_dir: PDF文件目录路径
            use_local_model: 是否使用本地vLLM模型，默认为False使用远程API
        """
        self.jsonl_dir = Path(jsonl_dir)
        self.pdf_dir = Path(pdf_dir)
        self.use_local_model = use_local_model
        
        # 检查vLLM可用性
        if use_local_model and not VLLM_AVAILABLE:
            logger.error("vLLM未安装，无法使用本地模型。将回退到远程API模式。")
            self.use_local_model = False
        
        # 根据选择初始化不同的模型
        if not self.use_local_model:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""), 
                base_url=os.getenv("OPENAI_BASE_URL", ""),
            )
            self.model_name = os.getenv("MODEL_NAME", "")
        else:
            # 本地模型将在首次调用时初始化
            self.vllm_model = None
        # 定义需要抽取的实体类型
        self.entity_types = [
            'title', 'research_background', 'classification_tags', 'key_contributions',
            'methodology', 'datasets', 'results', 'metrics',
            'formulas', 'algorithm', 'figure', 'table', 'limitations'
        ]
    def call_llm_api(self, entity_type: str, text: str) -> str:
        """
        调用LLM API获取实体，支持重试机制
        
        Args:
            entity_type: 实体类型
            text: 输入文本
            
        Returns:
            模型生成的实体字符串
        """
        import time
        
        max_retries = 3
        retry_delay = 2  # 等待2秒
        print(text)
        example_json = {
            "entities":['xxx','xxx','xxx']
        }
        prompt = f"""
        You are a professional paper entity extractor. Your answer should be in JSON format.
        I will give you a paper paragraph, and you need to extract the {entity_type} entities from it.
        example json format:{example_json}
        """
        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {
                            "role": "assistant", 
                            "content": prompt
                        },
                        {
                            "role": "user", 
                            "content": text
                        }
                    ]
                )
                print(response)
                
                # 解析响应
                try:
                    response_text = response.output_text.strip() 
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    # 解析 JSON
                    summary_json = json.loads(response_text) if response_text else {"entities": []}
                    return summary_json
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误 (尝试 {attempt + 1}/{max_retries}): {response_text}")
                    if attempt == max_retries - 1:
                        return {"entities": []}
                    
            except Exception as e:
                logger.error(f"API调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"entities": []}
            
            # 等待后重试
            if attempt < max_retries - 1:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
        
        return {"entities": []}
    def call_llm_local(self, entity_type: str, text: str) -> Dict[str, Any]:
        """
        调用本地vLLM获取实体，支持离线加速
        
        Args:
            entity_type: 实体类型
            text: 输入文本
            
        Returns:
            模型生成的实体字典
        """
        import json
        
        # 检查vLLM是否可用
        if not VLLM_AVAILABLE:
            logger.error("vLLM不可用，回退到远程API")
            return self.call_llm_api(entity_type, text)
        
        try:
            # 初始化vLLM模型（如果还未初始化）
            model_name = os.getenv("VLLM_MODEL_PATH", "Qwen/Qwen3-32B")
            if not hasattr(self, 'vllm_model') or self.vllm_model is None:

                
                logger.info(f"初始化vLLM模型: {model_name}")
                self.vllm_model = LLM(
                    model=model_name,
                    tensor_parallel_size=2,  # 使用2个GPU进行并行推理
                    gpu_memory_utilization=0.9,
                    max_model_len=32768,
                    trust_remote_code=True
                )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # 准备提示词
            example_json = {
                "entities": ['xxx', 'xxx', 'xxx']
            }
            prompt = f"""You are a professional paper entity extractor. Your answer should be in JSON format.
            I will give you a paper paragraph, and you need to extract the {entity_type} entities from it.
            example json format: {example_json}
            paper paragraph: {text}
            """
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=2048,
                stop=["</s>", "<|endoftext|>"]
            )
            
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
            tokenize=False, 
            enable_thinking=False,
            add_generation_prompt=True
            )

            # 生成响应
            outputs = self.vllm_model.generate([text], sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            print("vLLM raw response:", response_text)
            # 解析JSON响应
            try:
                start_index = response_text.find("```json")
                end_index = response_text.rfind("```")
                response_text = response_text[start_index + 7:end_index]
                response_text = response_text.strip()
                print("vLLM response:", response_text)

                # 解析JSON
                result = json.loads(response_text) if response_text else {"entities": []}
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"vLLM JSON解析错误: {response_text}")
                return {"entities": []}
                
        except Exception as e:
            logger.error(f"vLLM调用错误: {e}")
            # 如果vLLM失败，回退到远程API
            logger.info("回退到远程API模式")
            #return self.call_llm_api(entity_type, text)

    def extract_entities_from_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从单篇论文数据中提取实体
        
        Args:
            paper_data: 论文数据字典
            
        Returns:
            提取的实体字典
        """
        entities = {}
        
        # 验证输入数据
        if not isinstance(paper_data, dict):
            logger.warning("论文数据不是字典格式，跳过处理")
            return {}
        
        try:
            # 1. 提取标题
            text = paper_data.get('title', '')
            entities['title'] = text
            
            # 2. 提取研究背景
            text = paper_data.get('research_background', '')
            if self.use_local_model:
                summary_json = self.call_llm_local('research_background', text)
            else:
                summary_json = self.call_llm_api('research_background', text)
            entities['research_background'] = summary_json.get('entities', [])
            
            # 3. 提取分类标签
            text = paper_data.get('classification_tags', '')
            if self.use_local_model:
                summary_json = self.call_llm_local('classification_tags', text)
            else:
                summary_json = self.call_llm_api('classification_tags', text)
            entities['classification_tags'] = summary_json.get('entities', [])
            
            # 5. 提取关键贡献
            text = paper_data.get('key_contributions', '')
            if self.use_local_model:
                summary_json = self.call_llm_local('key_contributions', text)
            else:
                summary_json = self.call_llm_api('key_contributions', text)
            entities['key_contributions'] = summary_json.get('entities', [])
            
            # 7. 提取方法论
            text = paper_data.get('mmethodology', '')
            if self.use_local_model:
                summary_json = self.call_llm_local('methodology', text)
            else:
                summary_json = self.call_llm_api('methodology', text)
            entities['methodology'] = summary_json.get('entities', [])
            
            # 6. 提取数据集
            _text = paper_data.get('experiments_and_results', '')
            text = _text.get('datasets', '')
            if self.use_local_model:
                summary_json = self.call_llm_local('datasets', text)
            else:
                summary_json = self.call_llm_api('datasets', text)
            entities['datasets'] = summary_json.get('entities', [])
            
            # 7. 提取结果
            _text = paper_data.get('experiments_and_results', '')
            text = _text.get('results', '')
            if self.use_local_model:
                summary_json = self.call_llm_local('results', text)
            else:
                summary_json = self.call_llm_api('results', text)
            entities['results'] = summary_json.get('entities', [])
            
            # 8. 提取评估指标
            _text = paper_data.get('experiments_and_results', '')
            text = _text.get('metrics', '')
            if self.use_local_model:
                summary_json = self.call_llm_local('metrics', text)
            else:
                summary_json = self.call_llm_api('metrics', text)
            entities['metrics'] = summary_json.get('entities', [])
            
            
            # 9. 提取算法
            _text = paper_data.get('formulas_or_pseudocode', [])
            text = [t.get('pseudocode', '') for t in _text]
            text = [t for t in text if t != ''] 
            if self.use_local_model:
                summary_json = self.call_llm_local('algorithm', text)
            else:
                summary_json = self.call_llm_api('algorithm', text)
            entities['algorithm'] = summary_json.get('entities', [])
            
            # 10. 提取图片
            _text = paper_data.get('figures_or_tables_interpretation', [])
            text = [t.get('interpretation', '') if t.get('type') == 'figure' else '' for t in _text]
            #删除空字符串
            text = [t for t in text if t != '']
            if self.use_local_model:
                summary_json = self.call_llm_local('figure', text)
            else:
                summary_json = self.call_llm_api('figure', text)
            entities['figure'] = summary_json.get('entities', [])
            
            # 11. 提取表格
            _text = paper_data.get('figures_or_tables_interpretation', [])
            text = [t.get('interpretation', '') if t.get('type') == 'table' else '' for t in _text]
            text = [t for t in text if t != '']
            if self.use_local_model:
                summary_json = self.call_llm_local('table', text)
            else:
                summary_json = self.call_llm_api('table', text)
            entities['table'] = summary_json.get('entities', [])
            
            
            # 12. 提取公式
            _text = paper_data.get('formulas_or_pseudocode', [])
            text = [t.get('formula', '') for t in _text]
            text = [t for t in text if t != '']
            if self.use_local_model:
                summary_json = self.call_llm_local('formulas', text)
            else:
                summary_json = self.call_llm_api('formulas', text)
            entities['formulas'] = summary_json.get('entities', [])
            
        
            
            # 13. 提取局限性
            _text = paper_data.get('"conclusion_limitations_future', '')
            text = _text['limitations']
            if self.use_local_model:
                summary_json = self.call_llm_local('limitations', text)
            else:
                summary_json = self.call_llm_api('limitations', text)
            entities['limitations'] = summary_json.get('entities', [])
            

            
        except Exception as e:
            logger.error(f"提取实体时发生错误: {e}")
            # 返回基本的实体结构，避免完全失败
        
        return entities
    
    def extract_from_jsonl_file(self, jsonl_file_path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件中提取所有论文的实体
        
        Args:
            jsonl_file_path: JSONL文件路径
            
        Returns:
            提取的实体列表
        """
        extracted_entities = []
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        paper_data = json.loads(line.strip())
                        entities = self.extract_entities_from_paper(paper_data)
                        extracted_entities.append(entities)
                        
                        if line_num % 10 == 0:
                            logger.info(f"已处理 {line_num} 篇论文")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"第 {line_num} 行JSON解析错误: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"第 {line_num} 行处理错误: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"文件未找到: {jsonl_file_path}")
        except Exception as e:
            logger.error(f"文件处理错误: {e}")
        
        logger.info(f"总共提取了 {len(extracted_entities)} 篇论文的实体")
        return extracted_entities
    
    def extract_from_directory(self) -> List[Dict[str, Any]]:
        """
        从目录中的所有JSONL文件提取实体
        
        Returns:
            所有提取的实体列表
        """
        all_entities = []
        
        # 查找所有JSONL文件
        jsonl_files = list(self.jsonl_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.warning(f"在目录 {self.jsonl_dir} 中未找到JSONL文件")
            return all_entities
        
        logger.info(f"找到 {len(jsonl_files)} 个JSONL文件")
        
        for jsonl_file in jsonl_files:
            logger.info(f"处理文件: {jsonl_file}")
            entities = self.extract_from_jsonl_file(str(jsonl_file))
            all_entities.extend(entities)
        
        logger.info(f"总共提取了 {len(all_entities)} 篇论文的实体")
        return all_entities


def main():
    """主函数，用于测试实体提取功能"""
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='论文实体提取工具')
    parser.add_argument('--use-local', action='store_true', 
                       help='使用本地vLLM模型进行实体提取（需要安装vLLM）')
    parser.add_argument('--jsonl-dir', type=str, 
                       default="./paper_process/results/all_sampled_papers_500",
                       help='JSONL文件目录路径')
    parser.add_argument('--pdf-dir', type=str,
                       default="./data/papers", 
                       help='PDF文件目录路径')
    parser.add_argument('--output', type=str,
                       default="./output/entities/extracted_entities.jsonl",
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 配置路径
    jsonl_dir = args.jsonl_dir
    pdf_dir = args.pdf_dir
    use_local_model = args.use_local
    
    logger.info(f"使用模式: {'本地vLLM模型' if use_local_model else '远程API'}")
    logger.info(f"JSONL目录: {jsonl_dir}")
    logger.info(f"PDF目录: {pdf_dir}")
    
    # 创建实体提取器
    extractor = PaperEntityExtractor(jsonl_dir, pdf_dir, use_local_model=use_local_model)
    
    # 提取实体
    entities = extractor.extract_from_directory()
    
    # 保存结果为JSONL格式
    output_file = args.output.replace(".jsonl", f"_{os.path.basename(jsonl_dir)}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entity in entities:
            json.dump(entity, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"实体提取完成，结果保存到: {output_file}")
    logger.info(f"总共处理了 {len(entities)} 篇论文")

if __name__ == "__main__":
    main()