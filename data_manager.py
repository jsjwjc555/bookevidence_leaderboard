"""
数据管理模块
负责数据集的加载、处理和管理
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import random
from datetime import datetime
import sqlite3


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.datasets = {}
        self.dataset_stats = {}
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """加载所有数据集"""
        if not os.path.exists(self.data_dir):
            print(f"数据目录不存在: {self.data_dir}")
            return
        
        # 查找所有JSON文件
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_qao_v4.json'):
                domain = filename.replace('_qao_v4.json', '')
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    self.datasets[domain] = data
                    self._calculate_dataset_stats(domain, data)
                    print(f"成功加载 {domain} 数据集: {len(data)} 个样本")
                    
                except Exception as e:
                    print(f"加载 {filename} 失败: {str(e)}")
    
    def _calculate_dataset_stats(self, domain: str, data: Dict[str, Any]):
        """计算数据集统计信息"""
        if not data:
            return
        
        # 随机采样分析
        sample_keys = list(data.keys())
        sample_size = min(100, len(sample_keys))  # 最多分析100个样本
        sampled_keys = random.sample(sample_keys, sample_size)
        
        question_lengths = []
        answer_lengths = []
        citation_counts = []
        sentence_counts = []
        
        for key in sampled_keys:
            sample = data[key]
            
            # 问题长度
            question = sample.get('question', '')
            question_lengths.append(len(question))
            
            # 答案长度
            answer = sample.get('answer', '')
            answer_lengths.append(len(answer))
            
            # 归因信息
            model_answer = sample.get('sentence_level_citation', [])
            sentence_counts.append(len(model_answer))
            
            # 引用数量
            total_citations = 0
            for sentence_data in model_answer:
                citations = sentence_data.get('citations', {})
                anchor_texts = citations.get('anchor_text', [])
                total_citations += len(anchor_texts)
            citation_counts.append(total_citations)
        
        self.dataset_stats[domain] = {
            'total_samples': len(data),
            'avg_question_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            'avg_sentence_count': sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
            'avg_citation_count': sum(citation_counts) / len(citation_counts) if citation_counts else 0,
            'max_question_length': max(question_lengths) if question_lengths else 0,
            'max_answer_length': max(answer_lengths) if answer_lengths else 0,
        }
    
    def get_domains(self) -> List[str]:
        """获取所有领域"""
        return list(self.datasets.keys())
    
    def get_dataset(self, domain: str) -> Dict[str, Any]:
        """获取指定领域的数据集"""
        return self.datasets.get(domain, {})
    
    def get_sample_ids(self, domain: str) -> List[str]:
        """获取指定领域的样本ID列表"""
        dataset = self.get_dataset(domain)
        return list(dataset.keys())
    
    def get_sample(self, domain: str, sample_id: str) -> Dict[str, Any]:
        """获取指定样本"""
        dataset = self.get_dataset(domain)
        return dataset.get(sample_id, {})

    
    def get_random_samples(self, domain: str, count: int) -> List[Tuple[str, Dict[str, Any]]]:
        """随机获取指定数量的样本"""
        sample_ids = self.get_sample_ids(domain)
        if not sample_ids:
            return []
        
        selected_count = min(count, len(sample_ids))
        selected_ids = random.sample(sample_ids, selected_count)
        
        return [(id, self.get_sample(domain, id)) for id in selected_ids]
    
    def get_dataset_stats(self, domain: str) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return self.dataset_stats.get(domain, {})
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有数据集的统计信息"""
        return self.dataset_stats
    
    def search_samples(self, domain: str, keyword: str, max_results: int = 20) -> List[Tuple[str, Dict[str, Any]]]:
        """在指定领域中搜索包含关键词的样本"""
        dataset = self.get_dataset(domain)
        results = []
        
        for sample_id, sample_data in dataset.items():
            question = sample_data.get('question', '').lower()
            answer = sample_data.get('answer', '').lower()
            
            if keyword.lower() in question or keyword.lower() in answer:
                results.append((sample_id, sample_data))
                
                if len(results) >= max_results:
                    break
        
        return results


class LeaderboardManager:
    """排行榜管理器"""
    
    def __init__(self, db_path: str = "leaderboard.db"):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        if os.path.isabs(db_path):
            self.db_path = db_path
        else:
            self.db_path = os.path.abspath(os.path.join(base_dir, db_path))
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT,
                citation_precision REAL,
                citation_recall REAL,
                f1_score REAL,
                answer_similarity REAL,
                total_score REAL,
                test_domains TEXT,
                sample_count INTEGER,
                evaluation_date TEXT,
                config TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_evaluation_result(self, result: Dict[str, Any]) -> int:
        """保存评测结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算总分（可以根据需要调整权重）
        total_score = (
            result.get('Citation Precision', 0) * 0.3 +
            result.get('Citation Recall', 0) * 0.3 +
            result.get('F1 Score', 0) * 0.25 +
            result.get('Answer Similarity', 0) * 0.15
        )
        
        cursor.execute('''
            INSERT INTO evaluations (
                model_name, model_type, citation_precision, citation_recall,
                f1_score, answer_similarity, total_score, test_domains,
                sample_count, evaluation_date, config
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('model_name', ''),
            result.get('model_type', ''),
            result.get('Citation Precision', 0),
            result.get('Citation Recall', 0),
            result.get('F1 Score', 0),
            result.get('Answer Similarity', 0),
            total_score,
            json.dumps(result.get('test_domains', [])),
            result.get('sample_count', 0),
            datetime.now().isoformat(),
            json.dumps(result.get('config', {}))
        ))
        
        evaluation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return evaluation_id
    
    def get_leaderboard(self, limit: int = 50) -> pd.DataFrame:
        """获取排行榜数据"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                model_name,
                model_type,
                citation_precision,
                citation_recall,
                f1_score,
                answer_similarity,
                total_score,
                evaluation_date,
                sample_count
            FROM evaluations
            ORDER BY total_score DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[limit])
        conn.close()
        
        return df
    
    def get_model_history(self, model_name: str) -> pd.DataFrame:
        """获取指定模型的历史评测记录"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT *
            FROM evaluations
            WHERE model_name = ?
            ORDER BY evaluation_date DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[model_name])
        conn.close()
        
        return df
    
    def delete_evaluation(self, evaluation_id: int) -> bool:
        """删除评测记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM evaluations WHERE id = ?', (evaluation_id,))
        affected_rows = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return affected_rows > 0


class SampleAnalyzer:
    """样本分析器"""
    
    @staticmethod
    def analyze_citation_quality(sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析样本的引用质量"""
        model_answer = sample_data.get('sentence_level_citation', [])
        
        if not model_answer:
            return {
                'total_sentences': 0,
                'cited_sentences': 0,
                'total_citations': 0,
                'avg_citations_per_sentence': 0,
                'citation_coverage': 0
            }
        
        total_sentences = len(model_answer)
        cited_sentences = 0
        total_citations = 0
        citation_lengths = []
        
        for sentence_data in model_answer:
            citations = sentence_data.get('citations', {})
            anchor_texts = citations.get('anchor_text', [])
            
            if anchor_texts:
                cited_sentences += 1
                sentence_citations = len(anchor_texts)
                total_citations += sentence_citations
                
                # 分析引用文本长度
                for anchor_text in anchor_texts:
                    if isinstance(anchor_text, str):
                        citation_lengths.append(len(anchor_text))
        
        return {
            'total_sentences': total_sentences,
            'cited_sentences': cited_sentences,
            'total_citations': total_citations,
            'avg_citations_per_sentence': total_citations / total_sentences if total_sentences > 0 else 0,
            'citation_coverage': cited_sentences / total_sentences if total_sentences > 0 else 0,
            'avg_citation_length': sum(citation_lengths) / len(citation_lengths) if citation_lengths else 0,
            'max_citation_length': max(citation_lengths) if citation_lengths else 0
        }
    
    @staticmethod
    def extract_sample_features(sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取样本特征"""
        question = sample_data.get('question', '')
        answer = sample_data.get('answer', '')
        
        citation_analysis = SampleAnalyzer.analyze_citation_quality(sample_data)
        
        return {
            'question_length': len(question),
            'answer_length': len(answer),
            'question_word_count': len(question.split()),
            'answer_word_count': len(answer.split()),
            **citation_analysis
        }


def create_sample_leaderboard_data():
    """创建示例排行榜数据"""
    sample_data = [
        {
            'model_name': 'GPT-4',
            'model_type': 'OpenAI API',
            'Citation Precision': 0.876,
            'Citation Recall': 0.823,
            'F1 Score': 0.849,
            'Answer Similarity': 0.901,
            'test_domains': ['history', 'art', 'agriculture'],
            'sample_count': 150,
            'config': {'temperature': 0.7, 'max_tokens': 1000}
        },
        {
            'model_name': 'Claude-3',
            'model_type': 'Custom API',
            'Citation Precision': 0.854,
            'Citation Recall': 0.801,
            'F1 Score': 0.827,
            'Answer Similarity': 0.889,
            'test_domains': ['history', 'art'],
            'sample_count': 100,
            'config': {'temperature': 0.6, 'max_tokens': 1200}
        },
        {
            'model_name': 'Llama-3.1-70B',
            'model_type': 'HuggingFace API',
            'Citation Precision': 0.798,
            'Citation Recall': 0.765,
            'F1 Score': 0.781,
            'Answer Similarity': 0.834,
            'test_domains': ['history', 'agriculture'],
            'sample_count': 80,
            'config': {'temperature': 0.8, 'max_tokens': 800}
        }
    ]
    
    return sample_data
