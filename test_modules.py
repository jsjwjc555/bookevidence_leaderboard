#!/usr/bin/env python3
"""
模块测试脚本
测试各个功能模块是否正常工作
"""

import sys
import os
import json
import traceback
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def resolve_data_dir(default_path: str) -> str:
    raw_path = os.environ.get("WEB_LEADERBOARD_DATA_DIR_V4") or os.environ.get("WEB_LEADERBOARD_DATA_DIR")
    return os.path.abspath(raw_path or default_path)

def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        import streamlit as st
        print("✅ Streamlit 导入成功")
    except ImportError as e:
        print(f"❌ Streamlit 导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas 导入成功")
    except ImportError as e:
        print(f"❌ Pandas 导入失败: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly 导入成功")
    except ImportError as e:
        print(f"❌ Plotly 导入失败: {e}")
        return False
    
    try:
        from data_manager import DatasetManager, LeaderboardManager
        print("✅ 数据管理模块导入成功")
    except ImportError as e:
        print(f"❌ 数据管理模块导入失败: {e}")
        return False
    
    try:
        from model_api import ModelAPIFactory, AttributionMetrics
        print("✅ 模型API模块导入成功")
    except ImportError as e:
        print(f"❌ 模型API模块导入失败: {e}")
        return False
    
    return True

def test_data_manager():
    """测试数据管理器"""
    print("\n🧪 测试数据管理器...")
    
    try:
        from data_manager import DatasetManager, LeaderboardManager
        
        # 测试数据集管理器
        data_dir = resolve_data_dir("/Users/chengyihao/Documents/vscode-python/web_leaderboard/BookEvidenceQA_v4")
        
        if os.path.exists(data_dir):
            dataset_manager = DatasetManager(data_dir)
            domains = dataset_manager.get_domains()
            print(f"✅ 数据集管理器初始化成功，找到 {len(domains)} 个领域")
            
            if domains:
                test_domain = domains[0]
                sample_ids = dataset_manager.get_sample_ids(test_domain)
                print(f"✅ 领域 {test_domain} 包含 {len(sample_ids)} 个样本")
                
                if sample_ids:
                    test_sample = dataset_manager.get_sample(test_domain, sample_ids[0])
                    if test_sample:
                        print("✅ 样本数据获取成功")
                    else:
                        print("❌ 样本数据获取失败")
        else:
            print(f"⚠️  数据集目录不存在: {data_dir}")
        
        # 测试排行榜管理器
        leaderboard_manager = LeaderboardManager()
        print("✅ 排行榜管理器初始化成功")
        
        # 测试数据库操作
        test_result = {
            "model_name": "TestModel",
            "model_type": "Test API",
            "Citation Precision": 0.85,
            "Citation Recall": 0.78,
            "F1 Score": 0.81,
            "Answer Similarity": 0.88,
            "test_domains": ["test"],
            "sample_count": 10,
            "config": {"temperature": 0.7}
        }
        
        eval_id = leaderboard_manager.save_evaluation_result(test_result)
        print(f"✅ 测试评测结果保存成功，ID: {eval_id}")
        
        # 清理测试数据
        leaderboard_manager.delete_evaluation(eval_id)
        print("✅ 测试数据清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据管理器测试失败: {e}")
        traceback.print_exc()
        return False

def test_model_api():
    """测试模型API模块"""
    print("\n🧪 测试模型API模块...")
    
    try:
        from model_api import AttributionMetrics
        
        # 测试评测指标计算
        predicted_citations = ["引用1", "引用2", "引用3"]
        reference_citations = ["引用1", "引用4", "引用2"]
        
        precision = AttributionMetrics.calculate_citation_precision(
            predicted_citations, reference_citations
        )
        recall = AttributionMetrics.calculate_citation_recall(
            predicted_citations, reference_citations
        )
        f1 = AttributionMetrics.calculate_f1_score(precision, recall)
        
        print(f"✅ 指标计算成功: 精度={precision:.3f}, 召回={recall:.3f}, F1={f1:.3f}")
        
        # 测试答案相似度计算
        similarity = AttributionMetrics.calculate_answer_similarity(
            "这是一个测试答案", "这是测试答案内容"
        )
        print(f"✅ 相似度计算成功: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型API模块测试失败: {e}")
        traceback.print_exc()
        return False

def test_database():
    """测试数据库连接"""
    print("\n🧪 测试数据库连接...")
    
    try:
        import sqlite3
        
        # 测试SQLite连接
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # 创建测试表
        cursor.execute('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        ''')
        
        # 插入测试数据
        cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)", 
                      ("test", 1.23))
        
        # 查询数据
        cursor.execute("SELECT * FROM test_table")
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            print("✅ 数据库连接和操作正常")
            return True
        else:
            print("❌ 数据库操作失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")
        return False

def create_sample_data():
    """创建示例数据用于测试"""
    print("\n🧪 创建示例数据...")
    
    sample_data = {
        "test_sample_1": {
            "question": "这是一个测试问题？",
            "answer": "这是测试问题的参考答案。",
            "sentence_level_citation": [
                {
                    "sentence": "这是第一个句子的回答。",
                    "citations": {
                        "anchor_text": ["相关引用文本1"],
                        "prefix_text": ["这是引用的上下文"]
                    }
                },
                {
                    "sentence": "这是第二个句子的回答。",
                    "citations": {
                        "anchor_text": ["相关引用文本2", "相关引用文本3"],
                        "prefix_text": ["上下文1", "上下文2"]
                    }
                }
            ]
        }
    }
    
    # 保存示例数据
    test_file = os.path.abspath(os.path.join(BASE_DIR, "test_data.json"))
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 示例数据已保存到 {test_file}")
        
        # 清理测试文件
        os.remove(test_file)
        print("✅ 测试文件清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 示例数据创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧠 大模型归因分析平台 - 模块测试")
    print("=" * 50)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print()
    
    tests = [
        ("模块导入", test_imports),
        ("数据管理器", test_data_manager),
        ("模型API模块", test_model_api),
        ("数据库连接", test_database),
        ("示例数据", create_sample_data)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
        print("-" * 30)
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常运行。")
        print("\n🚀 启动应用命令:")
        print("   python enhanced_app.py")
        print("   或者")
        print("   streamlit run enhanced_app.py")
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        print("\n🔧 故障排除建议:")
        print("1. 检查依赖包是否正确安装: pip install -r requirements.txt")
        print("2. 检查数据集目录是否存在且包含正确格式的文件")
        print("3. 检查Python版本是否满足要求 (3.8+)")

if __name__ == "__main__":
    main()
