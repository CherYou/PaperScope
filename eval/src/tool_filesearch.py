"""File Search Tool - 从本地file_corpus目录中查找与用户query最相关的内容

input:
- query: 用户查询字符串

output:
- 返回最相关的文档列表，包含文件名、相关性分数和内容摘要
"""

import logging
import os
import re
import base64
import json
import glob
from tkinter import NO
import numpy as np
import faiss
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
from qwen_agent.tools.base import BaseTool, register_tool
import pytesseract
from ops_mm_embedding_v1 import OpsMMEmbeddingV1, fetch_image
from modelscope import AutoModel, AutoTokenizer

class FileSearchEngine:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.documents = {}  # filename -> content
        self.file_paths = {}  # filename -> full path
        self.imgs = {} # filename -> img
        self.texts = []  # List of document texts for FAISS
        self.filenames = []  # Corresponding filenames
        self.total_docs = 0
        
        # Initialize DeepSeek-OCR
        ocr_model_name = './models/deepseek-ai/DeepSeek-OCR'
        self.orc_model = AutoModel.from_pretrained(ocr_model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
        self.ocr_tokenizer = AutoTokenizer.from_pretrained(ocr_model_name, trust_remote_code=True)
        self.ocr_model = self.orc_model.eval().cuda().to(torch.bfloat16)

        # Initialize Ops-MM-embedding model for multimodal embeddings
        model_path = "./models/opensearch-ai/Ops-MM-embedding-v1-2B"
        logging.info(f"Initializing Ops-MM-embedding model from {model_path}")
        self.embedding_model = OpsMMEmbeddingV1(
            model_name=model_path,
            device="cuda:3" if torch.cuda.is_available() else "cpu",
            attn_implementation="flash_attention_2"
        )
        
        # FAISS index
        self.index = None
        
        self._load_documents()
        self._build_index()
    
    def ocr_image(self,image):
        prompt = "<image>\nCaption this image. "
        #print(image)
        output_path = "eval/src/ocroutput"
        res = self.ocr_model.infer(self.ocr_tokenizer, 
        prompt=prompt, 
        image_file=image, 
        output_path = output_path, 
        base_size = 1024, 
        image_size = 640, 
        crop_mode=True, 
        save_results = False, 
        test_compress = True,
        eval_mode=True
        )
        return res

    def extract_text_from_md(self, file_path: str) -> str:
        """Extract text from markdown files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from images using DeepSeek-OCR"""
        try:
            return fetch_image(file_path)
        except Exception as e:
            print(f"Error extracting text from image {file_path}: {e}")
            return f"[IMAGE_FILE]{file_path}"
    
    def _load_documents(self):
        """Load all files from the corpus directory, focusing on paper md documents and images"""
        if not os.path.exists(self.corpus_path):
            print(f"Warning: Corpus path {self.corpus_path} does not exist")
            return
        
        # Get all markdown and image files
        file_patterns = [
            os.path.join(self.corpus_path, '**', '*.md'),
            os.path.join(self.corpus_path, '**', '*.png'),
            os.path.join(self.corpus_path, '**', '*.jpg'),
            os.path.join(self.corpus_path, '**', '*.jpeg'),
            os.path.join(self.corpus_path, '**', '*.gif'),
            os.path.join(self.corpus_path, '**', '*.bmp'),
            os.path.join(self.corpus_path, '**', '*.tiff')
        ]
        
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(pattern, recursive=True))
        
        for file_path in all_files:
            try:
                relative_path = os.path.relpath(file_path, self.corpus_path)
                filename = os.path.basename(file_path)
                
                # Extract text based on file type
                if file_path.lower().endswith('.md'):
                    text = self.extract_text_from_md(file_path)
                    img = None
                elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                    text = None 
                    img = self.extract_text_from_image(file_path)
                else:
                    continue
                
                # Store document information
                path_parts = relative_path.split(os.sep)
                if len(path_parts) >= 3:
                    conference = path_parts[0]
                    paper_id = path_parts[1]
                    paper_key = f"{conference}/{paper_id}"
                    
                    if file_path.lower().endswith('.md'):
                        doc_key = paper_key
                    else:
                        doc_key = f"{paper_key}/images/{filename}"
                    
                    self.imgs[doc_key] = img
                    self.documents[doc_key] = text
                    self.file_paths[doc_key] = file_path
                    self.texts.append(text)
                    self.filenames.append(doc_key)
                    self.total_docs += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    def _build_index(self):
        """Build FAISS index for all documents using Ops-MM-embedding multimodal embeddings"""
        if not self.texts:
            print("No documents to index")
            return
        
        print(f"Building embeddings for {len(self.texts)} documents...")
        
        # Separate text and image documents for different processing
        text_docs = []
        image_docs = []
        text_indices = []
        image_indices = []
        
        for i, filename in enumerate(self.filenames):
            logging.info(f"Processing {i} file: {filename}")
            if '/images/' in filename:
                # This is an image document, load the actual image
                image_path = self.file_paths[filename]
                try:
                    image = self.imgs[filename]
                    image_docs.append(image)
                    image_indices.append(i)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    # Fallback to text processing

            else:
                if self.documents[filename]:
                    text = self.documents[filename]
                    text_docs.append(text)
                    text_indices.append(i)
        
        # Generate embeddings
        all_embeddings = []
        # Process text documents
        if text_docs:
            print(f"Processing {len(text_docs)} text documents...")
            text_embeddings = self.embedding_model.get_text_embeddings(
                texts=text_docs,
                batch_size=4,

            )
            # Convert to float32 for numpy compatibility
            text_embeddings = text_embeddings.float().cpu().numpy()
            all_embeddings.extend([(text_embeddings[i], text_indices[i]) 
                                 for i in range(len(text_docs))])
        
        # Process image documents
        if image_docs:
            print(f"Processing {len(image_docs)} image documents...")
            image_embeddings = self.embedding_model.get_image_embeddings(
                images=image_docs,
            )
            # Convert to float32 for numpy compatibility
            image_embeddings = image_embeddings.float().cpu().numpy()
            all_embeddings.extend([(image_embeddings[i], image_indices[i]) 
                                 for i in range(len(image_docs))])
        
        # Sort embeddings by original document order
        all_embeddings.sort(key=lambda x: x[1])
        embeddings = np.array([emb[0] for emb in all_embeddings]).astype('float32')
        
        # Build FAISS index using inner product similarity
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} documents")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents using FAISS similarity search with multimodal query support"""
        if not query or not self.index or not self.texts:
            return []
        
        try:
            # Generate query embedding using text embedding
            query_embedding = self.embedding_model.get_text_embeddings(
                texts=[query],
                batch_size=1,
                show_progress=False
            )
            # Convert to float32 for numpy compatibility
            query_embedding = query_embedding.float().cpu().numpy().astype('float32')
            
            # Search using FAISS
            distances, indices = self.index.search(query_embedding, min(top_k * 2, len(self.texts)))
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= len(self.filenames):
                    continue
                    
                filename = self.filenames[idx]
                content = self.documents[filename]
                image = self.imgs[filename]
                file_path = self.file_paths[filename]
                
                # Extract paper information from filename
                parts = filename.split('/')
                if len(parts) >= 2:
                    conference = parts[0]
                    paper_id = parts[1]
                    paper_key = f"{conference}/{paper_id}"
                    
                    # Determine file type
                    if '/images/' in filename:
                        file_type = "image"
                        content = self.ocr_image(file_path)
                        summary = "this is an image"
                    else:
                        file_type = "paper"
                        summary = content[:300] + "..." if len(content) > 300 else content
                    
                    results.append({
                        "filename": filename,
                        "file_path": file_path,
                        "file_type": file_type,
                        "score": float(distance),  # FAISS inner product score
                        "content": content,
                        "summary": summary,
                        "conference": conference,
                        "paper_id": paper_id,
                        "paper_key": paper_key
                    })
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []


@register_tool('FileSearchTool')
class FileSearchTool(BaseTool):
    name = "FileSearchTool"
    description = 'Search for relevant files in a document corpus using Ops-MM-embedding multimodal model. Supports text files and images with unified embedding space.'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'The search query to find relevant documents',
            'required': True
        },
        {
            'name': 'top_k',
            'type': 'integer', 
            'description': 'Number of top results to return (default: 5)',
            'required': False
        }
    ]
    
    def __init__(self):
        super().__init__()
        # Initialize search engine with corpus path
        corpus_path = "/share/project/xionglei/code/doc_parse/output"
        self.search_engine = FileSearchEngine(corpus_path)
    
    def call(self, params: str, **kwargs) -> str:
        """
        Search for files based on the query
        
        Args:
            params: JSON string containing 'query' and optional 'top_k'
            
        Returns:
            JSON string with search results including file paths and content
        """
        try:
            # Parse parameters
            if isinstance(params, str):
                params_dict = json.loads(params)
            else:
                params_dict = params

            #print(params_dict)
            query = params_dict.get('query', '')
            top_k = 3
            
            if not query:
                return json.dumps({
                    "error": "Query parameter is required"
                })
            
            # Perform search
            results = self.search_engine.search(query, top_k)
            
            if not results:
                return json.dumps({
                    "message": f"No documents found related to query: '{query}'",
                    "results": []
                })
            
            # Format results with file paths and content
            formatted_results = []
            for result in results:
                formatted_result = {
                    "filename": result["filename"],
                    "file_path": result["file_path"],
                    "file_type": result["file_type"],
                    "relevance_score": round(result["score"], 4),
                    "content": result["content"],
                    "summary": result["summary"]
                }
                
                formatted_results.append(formatted_result)
            
            all_content = "\n".join([result["content"] for result in formatted_results])
            return all_content
            
        except Exception as e:
            return f"Search failed: {str(e)}"


# Async wrapper for compatibility
async def file_search(query: str, top_k: int = 10) -> str:
    """Async wrapper for file search functionality"""
    tool = FileSearchTool()
    params = {"query": query, "top_k": top_k}
    return tool.call(params)


# Test the tool
if __name__ == "__main__":
    tool = FileSearchTool()
    
    # Test search
    result = tool.call('{"query": "hello", "top_k": 3}')
    print("Search result:")
    print(result)