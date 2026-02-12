"""
模型API集成模块
支持多种模型API的统一调用接口
"""

import requests
import json
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import openai
from huggingface_hub import InferenceClient


class BaseModelAPI(ABC):
    """模型API基类"""
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成回答"""
        pass
    
    @abstractmethod
    def generate_attribution_response(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """生成带归因的回答"""
        pass


class OpenAIAPI(BaseModelAPI):
    """OpenAI API封装"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成回答"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_attribution_response(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """生成带归因的回答"""
        attribution_prompt = f"""
        基于以下上下文回答问题，并为你的每个句子提供引用依据。

        上下文：
        {context}

        问题：{question}

        要求：
        1. 提供准确的回答
        2. 为每个句子标注引用的具体文本片段
        3. 按照JSON格式输出，包含句子和对应的引用

        输出格式：
        {{
            "sentences": [
                {{
                    "sentence": "句子内容",
                    "citations": ["引用文本1", "引用文本2"]
                }}
            ]
        }}
        """
        
        response = self.generate_response(attribution_prompt, **kwargs)
        
        # 尝试解析JSON响应
        try:
            parsed_response = json.loads(response)
            return parsed_response
        except json.JSONDecodeError:
            # 如果解析失败，返回原始响应
            return {
                "sentences": [{"sentence": response, "citations": []}]
            }


class HuggingFaceAPI(BaseModelAPI):
    """HuggingFace API封装"""
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = InferenceClient(token=api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成回答"""
        try:
            response = self.client.text_generation(
                prompt=prompt,
                model=self.model_name,
                max_new_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                return_full_text=False
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_attribution_response(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """生成带归因的回答"""
        attribution_prompt = f"""
        Context: {context}
        Question: {question}
        
        Please provide an answer with citations. Format your response as JSON with sentences and their corresponding citations.
        """
        
        response = self.generate_response(attribution_prompt, **kwargs)
        
        # 简化的响应处理
        return {
            "sentences": [{"sentence": response, "citations": []}]
        }


class CustomAPI(BaseModelAPI):
    """自定义API封装"""
    
    def __init__(self, api_key: str, model_name: str, api_url: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.api_url = api_url
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成回答"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": kwargs.get('temperature', 0.7),
                "max_tokens": kwargs.get('max_tokens', 1000)
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get('text', result.get('response', 'No response'))
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_attribution_response(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """生成带归因的回答"""
        attribution_prompt = f"""
        Context: {context}
        Question: {question}
        
        Provide answer with citations in JSON format.
        """
        
        response = self.generate_response(attribution_prompt, **kwargs)
        return {
            "sentences": [{"sentence": response, "citations": []}]
        }


class ModelAPIFactory:
    """模型API工厂类"""
    
    @staticmethod
    def create_api(api_type: str, api_key: str, model_name: str, **kwargs) -> BaseModelAPI:
        """创建对应类型的API实例"""
        if api_type == "OpenAI API":
            return OpenAIAPI(api_key, model_name, **kwargs)
        elif api_type == "HuggingFace API":
            return HuggingFaceAPI(api_key, model_name, **kwargs)
        elif api_type == "自定义API":
            api_url = kwargs.get('api_url', '')
            if not api_url:
                raise ValueError("自定义API需要提供api_url参数")
            return CustomAPI(api_key, model_name, api_url, **kwargs)
        else:
            raise ValueError(f"不支持的API类型: {api_type}")


# 评测指标计算
class AttributionMetrics:
    """归因评测指标计算"""
    
    @staticmethod
    def calculate_citation_precision(predicted_citations: List[str], 
                                   reference_citations: List[str]) -> float:
        """计算引用精度"""
        if not predicted_citations:
            return 0.0
        
        correct_citations = sum(1 for cite in predicted_citations 
                              if any(ref in cite or cite in ref for ref in reference_citations))
        return correct_citations / len(predicted_citations)
    
    @staticmethod
    def calculate_citation_recall(predicted_citations: List[str], 
                                reference_citations: List[str]) -> float:
        """计算引用召回率"""
        if not reference_citations:
            return 0.0
        
        found_references = sum(1 for ref in reference_citations 
                             if any(ref in cite or cite in ref for cite in predicted_citations))
        return found_references / len(reference_citations)
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def calculate_answer_similarity(predicted_answer: str, reference_answer: str) -> float:
        """计算答案相似度（简化版本）"""
        # 这里使用简单的词汇重叠计算相似度
        # 实际应用中可以使用更复杂的语义相似度计算
        pred_words = set(predicted_answer.lower().split())
        ref_words = set(reference_answer.lower().split())
        
        if not ref_words:
            return 0.0
        
        intersection = pred_words & ref_words
        return len(intersection) / len(ref_words)


def run_evaluation(model_api: BaseModelAPI, 
                   test_samples: List[Dict[str, Any]], 
                   progress_callback=None) -> Dict[str, float]:
    """运行模型评测"""
    
    total_precision = 0.0
    total_recall = 0.0
    total_similarity = 0.0
    valid_samples = 0
    
    for i, sample in enumerate(test_samples):
        if progress_callback:
            progress_callback(i / len(test_samples))
        
        try:
            # 获取样本数据
            question = sample.get('question', '')
            reference_answer = sample.get('answer', '')
            reference_citations = []
            
            # 提取参考引用
            model_answer_data = sample.get('sentence_level_citation', [])
            for sentence_data in model_answer_data:
                citations = sentence_data.get('citations', {})
                anchor_texts = citations.get('anchor_text', [])
                reference_citations.extend(anchor_texts)
            
            # 生成模型回答（这里简化处理，实际应该提供完整的上下文）
            context = " ".join(reference_citations[:3])  # 使用前3个引用作为上下文
            model_response = model_api.generate_attribution_response(question, context)
            
            # 提取预测的引用
            predicted_citations = []
            predicted_answer = ""
            
            for sentence in model_response.get('sentences', []):
                predicted_answer += sentence.get('sentence', '') + " "
                predicted_citations.extend(sentence.get('citations', []))
            
            # 计算指标
            precision = AttributionMetrics.calculate_citation_precision(
                predicted_citations, reference_citations
            )
            recall = AttributionMetrics.calculate_citation_recall(
                predicted_citations, reference_citations  
            )
            similarity = AttributionMetrics.calculate_answer_similarity(
                predicted_answer, reference_answer
            )
            
            total_precision += precision
            total_recall += recall
            total_similarity += similarity
            valid_samples += 1
            
        except Exception as e:
            print(f"处理样本 {i} 时出错: {str(e)}")
            continue
    
    if valid_samples == 0:
        return {
            "Citation Precision": 0.0,
            "Citation Recall": 0.0, 
            "F1 Score": 0.0,
            "Answer Similarity": 0.0
        }
    
    avg_precision = total_precision / valid_samples
    avg_recall = total_recall / valid_samples
    f1_score = AttributionMetrics.calculate_f1_score(avg_precision, avg_recall)
    avg_similarity = total_similarity / valid_samples
    
    return {
        "Citation Precision": avg_precision,
        "Citation Recall": avg_recall,
        "F1 Score": f1_score, 
        "Answer Similarity": avg_similarity
    }
