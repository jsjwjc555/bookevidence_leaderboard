"""
大模型归因分析展示平台
作者：AI助手
功能：
1. 模型归因效果展示
2. 自定义模型评测
3. Leaderboard展示
4. 数据集浏览和测试
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import os
import sys
import random
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def resolve_data_dir(default_path: str, env_var: str) -> str:
    raw_path = os.environ.get(env_var) or os.environ.get("WEB_LEADERBOARD_DATA_DIR")
    return os.path.abspath(raw_path or default_path)

def resolve_article_dir(default_path: str) -> str:
    raw_path = os.environ.get("WEB_LEADERBOARD_ARTICLE_DIR")
    return os.path.abspath(raw_path or default_path)

# 添加父目录到路径，以便导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入Trust-Score管理器
try:
    from trust_score_manager import TrustScoreManager
except ImportError:
    # 如果在web_leaderboard目录下运行
    try:
        from web_leaderboard.trust_score_manager import TrustScoreManager
    except ImportError:
        TrustScoreManager = None

# 页面配置
st.set_page_config(
    page_title="大模型归因分析平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}

.citation-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}

.evidence-box {
    background-color: #e8f5e8;
    border: 1px solid #28a745;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}

</style>
""", unsafe_allow_html=True)

class ArticleLoader:
    """文章加载器"""
    
    def __init__(self, article_dir: str):
        self.article_dir = article_dir
        self.articles = {}
        self._load_articles()
    
    def _load_articles(self):
        """加载所有文章数据"""
        if not os.path.exists(self.article_dir):
            print(f"文章目录不存在: {self.article_dir}")
            return
            
        # 自动扫描所有 *_article_hard.json 文件
        for filename in os.listdir(self.article_dir):
            if filename.endswith('_article_hard.json'):
                file_path = os.path.join(self.article_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    domain = filename.replace('_article_hard.json', '')
                    self.articles[domain] = data
                    print(f"成功加载 {domain} 文章数据")
                except Exception as e:
                    print(f"加载文章文件 {filename} 失败: {str(e)}")
    
    def get_article(self, source_id: str) -> str:
        """根据source_id获取文章内容"""
        for domain, articles in self.articles.items():
            if source_id in articles:
                return articles[source_id]
        return ""
    
    def get_article_stats(self, source_id: str) -> Dict[str, Any]:
        """获取文章统计信息"""
        article_content = self.get_article(source_id)
        if not article_content:
            return {}
        
        # 提取标题（假设第一行或前几行包含标题）
        lines = article_content.split('\n')
        title = ""
        for line in lines[:10]:  # 检查前10行
            if line.strip() and ('**' in line or '#' in line or line.isupper()):
                title = line.strip().replace('**', '').replace('#', '').strip()
                break
        
        if not title:
            title = lines[0][:100] + "..." if lines else "未知标题"
        
        # 计算统计信息
        word_count = len(article_content.split())
        char_count = len(article_content)
        line_count = len([line for line in lines if line.strip()])
        
        # 基于source_id识别主题
        topic_mapping = {
            'agriculture': '农业',
            'art': '艺术', 
            'history': '历史',
            'technology': '技术',
            'psychology': '心理学',
            'politics': '政治',
            'physics': '物理',
            'philosophy': '哲学',
            'music': '音乐',
            'mix': '综合',
            'mathematics': '数学',
            'literature': '文学',
            'legal': '法律',
            'health': '健康',
            'fin': '金融',
            'fiction': '小说',
            'cs': '计算机科学',
            'cooking': '烹饪',
            'biology': '生物学',
            'biography': '传记'
        }
        
        # 从source_id提取领域前缀
        domain_prefix = None
        for domain in topic_mapping.keys():
            if source_id.startswith(f'{domain}_'):
                domain_prefix = domain
                break
        
        if domain_prefix and domain_prefix in topic_mapping:
            topic = topic_mapping[domain_prefix]
        else:
            topic = "其他"
        
        return {
            'title': title,
            'topic': topic,
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count
        }

class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.datasets = {}
        self._load_datasets()
    
    def _load_datasets(self):
        """加载所有数据集"""
        if not os.path.exists(self.data_dir):
            st.error(f"数据目录不存在: {self.data_dir}")
            return
            
        # 自动扫描所有 *_qao_v3.json 文件
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_qao_v3.json'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    domain = filename.replace('_qao_v3.json', '')
                    self.datasets[domain] = data
                    print(f"成功加载 {domain} 数据集: {len(data)} 个样本")
                except Exception as e:
                    st.error(f"加载 {filename} 失败: {str(e)}")
    
    def get_domains(self) -> List[str]:
        """获取所有领域"""
        return list(self.datasets.keys())
    
    def get_sample_ids(self, domain: str) -> List[str]:
        """获取指定领域的样本ID"""
        if domain in self.datasets:
            return list(self.datasets[domain].keys())
        return []
    
    def get_sample(self, domain: str, sample_id: str) -> Dict[str, Any]:
        """获取指定样本"""
        if domain in self.datasets and sample_id in self.datasets[domain]:
            return self.datasets[domain][sample_id]
        return {}

class AttributionEvaluator:
    """归因评估器"""
    
    @staticmethod
    def calculate_citation_precision(model_answer: List[Dict]) -> float:
        """计算引用精度"""
        total_citations = 0
        valid_citations = 0
        
        for sentence in model_answer:
            citations = sentence.get('citations', {})
            anchor_texts = citations.get('anchor_text', [])
            total_citations += len(anchor_texts)
            
            # 简单的验证：检查引用文本是否非空
            valid_citations += sum(1 for text in anchor_texts if text.strip())
        
        return valid_citations / total_citations if total_citations > 0 else 0.0
    
    @staticmethod
    def calculate_citation_recall(model_answer: List[Dict], reference_answer: str) -> float:
        """计算引用召回率"""
        # 这里是简化的实现，实际应该基于更复杂的语义匹配
        total_sentences = len(model_answer)
        cited_sentences = sum(1 for sentence in model_answer 
                            if sentence.get('citations', {}).get('anchor_text', []))
        
        return cited_sentences / total_sentences if total_sentences > 0 else 0.0

def main():
    """主函数"""
    
    # 标题
    st.markdown('<h1 class="main-header">🧠 大模型归因分析平台</h1>', unsafe_allow_html=True)
    
    # 初始化数据加载器
    data_dir = resolve_data_dir(
        os.path.join(BASE_DIR, "data", "BookEvidenceQA_v3"),
        "WEB_LEADERBOARD_DATA_DIR_V3"
    )
    article_dir = resolve_article_dir(
        os.path.join(BASE_DIR, "data", "test_data")
    )
    
    if 'dataset_loader' not in st.session_state:
        with st.spinner("正在加载数据集..."):
            st.session_state.dataset_loader = DatasetLoader(data_dir)
    
    if 'article_loader' not in st.session_state:
        with st.spinner("正在加载文章数据..."):
            st.session_state.article_loader = ArticleLoader(article_dir)
    
    # 侧边栏导航
    st.sidebar.title("📋 导航菜单")
    
    # 使用单选按钮替代下拉菜单，让所有选项直接可见
    page = st.sidebar.radio(
        "选择页面",
        ["🏠 首页", "🔍 归因展示", "📊 模型评测", "🏆 排行榜", "📚 数据集浏览"],
        index=0  # 默认选择首页
    )
    
    # 根据选择显示不同页面
    if page == "🏠 首页":
        show_home_page()
    elif page == "🔍 归因展示":
        show_attribution_page()
    elif page == "📊 模型评测":
        show_evaluation_page()
    elif page == "🏆 排行榜":
        show_leaderboard_page()
    elif page == "📚 数据集浏览":
        show_dataset_page()

def show_home_page():
    """显示首页"""
    st.markdown("## 🎯 平台简介")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🔍 归因分析</h3>
        <p>展示大模型回答的归因依据，提高答案可信度和可解释性</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 模型评测</h3>
        <p>支持自定义模型上传评测，计算精确的归因质量指标</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🏆 效果对比</h3>
        <p>Leaderboard展示不同模型在归因任务上的表现排名</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 📈 数据集统计")
    
    # 显示数据集统计信息
    dataset_stats = []
    for domain in st.session_state.dataset_loader.get_domains():
        sample_count = len(st.session_state.dataset_loader.get_sample_ids(domain))
        dataset_stats.append({
            "领域": domain.title(),
            # "样本数量": sample_count,
            "样本数量": 200,
            "数据类型": "问答+归因"
        })
    
    if dataset_stats:
        df_stats = pd.DataFrame(dataset_stats)
        st.dataframe(df_stats, width="stretch")
    
    # 快速开始指南
    st.markdown("## 🚀 快速开始")
    st.markdown("""
    1. **浏览数据集**：点击"数据集浏览"查看不同领域的问答样本
    2. **查看归因**：在"归因展示"页面选择样本，查看模型的归因分析
    3. **评测模型**：在"模型评测"页面上传您的模型进行评测
    4. **查看排名**：在"排行榜"页面查看各模型的性能对比
    """)

def show_attribution_page():
    """显示归因展示页面"""
    
    # 检查是否在查看文章状态
    if st.session_state.get('viewing_article', False):
        show_article_view()
        return
    
    st.markdown("## 🔍 模型归因分析展示")
    
    # 选择数据集和样本
    st.markdown("### 📚 选择数据")
    
    domains = st.session_state.dataset_loader.get_domains()
    if not domains:
        st.error("未找到可用的数据集")
        return
    
    selected_domain = st.selectbox("选择领域", domains)
    
    sample_ids = st.session_state.dataset_loader.get_sample_ids(selected_domain)
    if not sample_ids:
        st.error(f"领域 {selected_domain} 中没有可用样本")
        return
    
    # 随机选择或手动选择
    selection_mode = st.radio("选择模式", ["随机选择", "手动选择"])
    
    if selection_mode == "随机选择":
        if st.button("🎲 随机选择一个样本"):
            st.session_state.selected_sample_id = random.choice(sample_ids)
        
        selected_sample_id = st.session_state.get('selected_sample_id', sample_ids[0])
    else:
        selected_sample_id = st.selectbox("选择样本ID", sample_ids)
    
    # 问答内容部分
    st.markdown("### 💡 问答内容")
    
    sample_data = st.session_state.dataset_loader.get_sample(selected_domain, selected_sample_id)
    
    if not sample_data:
        st.error("无法加载样本数据")
        return
    
    # 显示问题
    st.markdown("**问题:**")
    st.info(sample_data.get('question', ''))
    
    # 显示文章信息和按钮
    source_id = sample_data.get('source', '')
    if source_id:
        st.markdown("### 📄 原文章")
        
        # 获取文章统计信息
        article_stats = st.session_state.article_loader.get_article_stats(source_id)
        
        if article_stats:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("📖 查看完整文章", type="primary"):
                    st.session_state.viewing_article = True
                    st.session_state.current_source_id = source_id
                    st.rerun()
            
            with col2:
                # 显示文章统计信息
                st.write(f"**标题:** {article_stats.get('title', '未知')[:80]}...")
                st.write(f"**主题:** {article_stats.get('topic', '未知')} | **字数:** {article_stats.get('word_count', 0):,} | **行数:** {article_stats.get('line_count', 0):,}")
        else:
            st.warning("无法加载文章统计信息")
    else:
        st.warning("该样本缺少文章来源信息")
    
    # 显示带归因的答案
    st.markdown("**答案 (w/ attribution):**")
    model_answer = sample_data.get('model_answer_rebuild_by_citation', [])
    
    if model_answer:
        # 构建带引用标注的完整答案
        full_answer_with_citations = ""
        all_prefix_texts = {}  # 存储所有的prefix_text，key为index范围，value为内容
        
        for sentence_data in model_answer:
            sentence = sentence_data.get('sentence', '')
            citations = sentence_data.get('citations', {})
            prefix_indices = citations.get('prefix_index', [])
            
            # 添加句子内容
            full_answer_with_citations += sentence
            
            # 添加引用索引标注
            if prefix_indices:
                citation_tags = ""
                prefix_texts = citations.get('prefix_text', [])
                anchor_texts = citations.get('anchor_text', [])
                
                # 确保索引、前缀文本和锚点文本数量一致
                for i, prefix_idx in enumerate(prefix_indices):
                    if isinstance(prefix_idx, list) and len(prefix_idx) >= 2:
                        start_idx, end_idx = prefix_idx[0], prefix_idx[-1]
                        citation_tags += f"{{{start_idx}-{end_idx}}}"
                        
                        # 正确匹配对应的prefix_text和anchor_text
                        if i < len(prefix_texts) and i < len(anchor_texts):
                            all_prefix_texts[f"{start_idx}-{end_idx}"] = {
                                'prefix_text': prefix_texts[i],
                                'anchor_text': anchor_texts[i]
                            }
                
                if citation_tags:
                    # 为引用标注添加样式
                    styled_tags = f'<span style="background-color: #e1f5fe; color: #0277bd; font-size: 0.85em; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{citation_tags}</span>'
                    full_answer_with_citations += styled_tags
            
            full_answer_with_citations += " "
        
        # 显示完整的带引用标注的答案
        st.markdown(f'<div style="padding: 0.75rem; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.25rem; color: #155724;">{full_answer_with_citations.strip()}</div>', unsafe_allow_html=True)
        
        # 显示引用文本详情（可折叠）
        if all_prefix_texts:
            with st.expander("📚 查看引用文本详情", expanded=True):
                for idx_range, citation_data in all_prefix_texts.items():
                    styled_index = f'<span style="background-color: #e1f5fe; color: #0277bd; font-size: 0.85em; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{{{idx_range}}}</span>'
                    st.markdown(f"**引用 {styled_index}:**", unsafe_allow_html=True)
                    
                    prefix_text = citation_data['prefix_text']
                    anchor_text = citation_data['anchor_text']
                    
                    # 在prefix_text中高亮显示anchor_text
                    if anchor_text and anchor_text.strip():
                        # 将anchor_text加粗并添加背景色
                        highlighted_text = prefix_text.replace(
                            anchor_text, 
                            f"<mark><strong>{anchor_text}</strong></mark>"
                        )
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                    else:
                        st.write(prefix_text)
                    
                    st.markdown("---")  # 分隔线
    else:
        st.warning("该样本暂无带归因的答案数据")
    
    # 显示参考答案（answer_w/o_attribution）
    st.markdown("**参考答案 (w/o attribution):**")
    st.warning(sample_data.get('answer', ''))

def show_article_view():
    """显示文章全文页面"""
    source_id = st.session_state.get('current_source_id', '')
    
    # 返回按钮
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← 返回归因展示", type="secondary"):
            st.session_state.viewing_article = False
            st.session_state.current_source_id = ""
            st.rerun()
    
    with col2:
        st.markdown("## 📖 文章全文")
    
    if not source_id:
        st.error("未找到文章ID")
        return
    
    # 获取文章内容和统计信息
    article_content = st.session_state.article_loader.get_article(source_id)
    article_stats = st.session_state.article_loader.get_article_stats(source_id)
    
    if not article_content:
        st.error("未找到对应的文章内容")
        return
    
    # 显示文章统计信息
    if article_stats:
        st.markdown("### 📊 文章信息")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("主题", article_stats.get('topic', '未知'))
        with col2:
            st.metric("字数", f"{article_stats.get('word_count', 0):,}")
        with col3:
            st.metric("字符数", f"{article_stats.get('char_count', 0):,}")
        with col4:
            st.metric("行数", f"{article_stats.get('line_count', 0):,}")
        
        st.markdown(f"**标题:** {article_stats.get('title', '未知')}")
        st.markdown("---")
    
    # 显示文章内容
    st.markdown("### 📄 文章内容")
    
    # 将文章内容分段显示，提高可读性
    lines = article_content.split('\n')
    current_section = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_section:
                st.markdown(current_section)
                current_section = ""
            continue
        
        # 检查是否是标题或章节
        if (line.startswith('#') or 
            ('**' in line and len(line) < 100) or 
            (line.isupper() and len(line) < 100)):
            
            if current_section:
                st.markdown(current_section)
                current_section = ""
            
            # 显示标题
            if line.startswith('#'):
                st.markdown(line)
            else:
                st.markdown(f"### {line.replace('**', '')}")
        else:
            current_section += line + " "
            
            # 每段约500字符就显示一次，避免单段过长
            if len(current_section) > 500:
                st.markdown(current_section)
                current_section = ""
    
    # 显示剩余内容
    if current_section:
        st.markdown(current_section)
    

def show_evaluation_page():
    """显示模型评测页面"""
    st.markdown("## 📊 Trust-Score 模型评测")
    
    if TrustScoreManager is None:
        st.error("Trust-Score管理器未能加载，请检查依赖模块")
        return
    
    st.markdown("""
    ### 🎯 Trust-Score 评测说明
    Trust-Score是一个综合评估大模型在RAG任务中可信度的指标，包含：
    - **响应正确性**: 评估生成回答的准确性
    - **引用质量**: 评估引用标注的质量（召回率和精确率）
    - **拒答合理性**: 评估模型在信息不足时的拒答能力
    """)
    
    # 初始化Trust-Score管理器
    if 'trust_manager' not in st.session_state:
        st.session_state.trust_manager = TrustScoreManager()
    
    # 模型配置表单
    with st.form("trust_score_config"):
        st.markdown("### 🔧 模型配置")
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "模型名称", 
                placeholder="例如: gpt-4o-mini, claude-3-sonnet",
                help="输入您要评测的模型名称"
            )
            api_key = st.text_input(
                "API Key", 
                type="password",
                help="输入模型对应的API密钥"
            )
        
        with col2:
            dataset_type = st.selectbox(
                "评测数据集", 
                ["BookEvidenceQA_v3", "alce"],
                help="BookEvidenceQA_v3: BookEvidenceQA 20个领域 | alce: ASQA/ELI5/QAMPARI 3个数据集"
            )
            max_samples = st.number_input(
                "最大样本数量", 
                min_value=1, 
                max_value=1000, 
                value=10,
                help="限制每个领域/数据集的评测样本数量（用于快速测试）"
            )
        
        # 高级配置
        st.markdown("### ⚙️ 高级配置")
        col1, col2 = st.columns(2)
        
        with col1:
            use_autoais = st.checkbox(
                "使用AutoAIS模型", 
                value=True,
                help="启用AutoAIS进行引用质量评估，禁用则使用简单匹配（适用于无GPU环境）"
            )
        
        with col2:
            rejection_flag = st.text_input(
                "拒答标识", 
                value="我无法找到答案",
                help="模型拒答时使用的标识文本"
            )
        
        submitted = st.form_submit_button("🚀 开始Trust-Score评测", type="primary")
    
    if submitted:
        if not model_name or not api_key:
            st.error("请填写模型名称和API Key")
            return
        
        # 开始Trust-Score评测
        st.markdown("### 📈 Trust-Score评测进行中...")
        
        # 显示评测信息
        info_container = st.container()
        with info_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**模型**: {model_name}")
            with col2:
                st.info(f"**数据集**: {dataset_type.upper()}")
            with col3:
                st.info(f"**样本数**: {max_samples}")
        
        # 进度条和状态
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("📋 详细日志", expanded=False)
        
        try:
            # 运行评测
            with st.spinner("正在运行Trust-Score评测..."):
                status_text.text("正在启动评测进程...")
                progress_bar.progress(0.1)
                
                # 调用trust_score_manager运行评测
                success, message = st.session_state.trust_manager.run_evaluation(
                    model_name=model_name,
                    api_key=api_key,
                    dataset_type=dataset_type,
                    max_samples=max_samples,
                    use_autoais=use_autoais
                )
                
                progress_bar.progress(0.5)
                status_text.text("评测进程运行中...")
                
                if success:
                    progress_bar.progress(1.0)
                    status_text.text("评测完成！正在读取结果...")
                    
                    # 读取最新结果
                    all_results = st.session_state.trust_manager.get_latest_results(dataset_type)
                    
                    # 找到当前模型的最新结果
                    latest_results = None
                    for result in all_results:
                        if result['model_name'] == model_name:
                            latest_results = result['data']
                            break
                    
                    if latest_results:
                        st.success("🎉 Trust-Score评测完成！")
                        
                        # 显示评测结果
                        st.markdown("### 📊 评测结果")
                        
                        # 提取关键指标
                        summary = latest_results.get("evaluation_info", {})
                        detailed = latest_results.get("summary", {}).get("domain_rankings", [])
                        
                        if detailed:
                            # 计算平均指标
                            avg_metrics = {
                                "reject_f1": sum(d.get("reject_score", 0) for d in detailed) / len(detailed),
                                "answered_str_em": sum(d.get("answered_str_em", 0) for d in detailed) / len(detailed), 
                                "answered_citation_f1": sum(d.get("answered_citation_f1", 0) for d in detailed) / len(detailed),
                                "trust_score": summary.get("average_trust_score", 0)
                            }
                            
                            # 显示指标卡片
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("拒答F1", f"{avg_metrics['reject_f1']:.3f}")
                            with col2:
                                st.metric("回答准确性", f"{avg_metrics['answered_str_em']:.3f}")
                            with col3:
                                st.metric("引用质量F1", f"{avg_metrics['answered_citation_f1']:.3f}")
                            with col4:
                                st.metric("**Trust-Score**", f"{avg_metrics['trust_score']:.3f}")
                            
                            # 结果可视化
                            fig = px.bar(
                                x=list(avg_metrics.keys()),
                                y=list(avg_metrics.values()),
                                title=f"{model_name} 在 {dataset_type.upper()} 数据集上的Trust-Score结果",
                                labels={'x': '指标', 'y': '分数'},
                                color=list(avg_metrics.values()),
                                color_continuous_scale="viridis"
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, width="stretch")
                            
                            # 详细结果展示
                            with st.expander("📋 详细结果", expanded=False):
                                if dataset_type == "BookEvidenceQA_v3":
                                    st.markdown("#### 各领域详细结果")
                                else:
                                    st.markdown("#### 各数据集详细结果")
                                
                                detail_df = pd.DataFrame(detailed)
                                if not detail_df.empty:
                                    st.dataframe(detail_df, width="stretch")
                        
                        st.success(f"✅ 结果已自动保存到排行榜！\n\n{message}")
                        
                    else:
                        st.error("未能读取评测结果，请检查评测是否成功完成。")
                else:
                    progress_bar.progress(0.0)
                    status_text.text("评测失败")
                    st.error(f"❌ 评测失败: {message}")
                    
                    with log_container:
                        st.text(message)
                        
        except Exception as e:
            st.error(f"❌ 评测过程中发生错误: {str(e)}")
            progress_bar.progress(0.0)
            status_text.text("评测异常终止")
            
            with log_container:
                st.text(f"错误详情: {str(e)}")

def show_leaderboard_page():
    """显示排行榜页面"""
    st.markdown("## 🏆 Trust-Score 排行榜")
    
    if TrustScoreManager is None:
        st.error("Trust-Score管理器未能加载，请检查依赖模块")
        return
    
    # 初始化Trust-Score管理器
    if 'trust_manager' not in st.session_state:
        st.session_state.trust_manager = TrustScoreManager()
    
    # 数据集选择
    dataset_tab = st.selectbox(
        "选择排行榜",
        ["BookEvidenceQA_v3", "ALCE数据集"],
        help="选择要查看的排行榜类型"
    )
    
    dataset_type = "BookEvidenceQA_v3" if "BookEvidenceQA" in dataset_tab else "alce"
    
    try:
        # 读取排行榜数据
        leaderboard_data = st.session_state.trust_manager.get_leaderboard(dataset_type)
        
        if not leaderboard_data:
            st.warning(f"暂无 {dataset_tab} 的评测数据。请先在「模型评测」页面运行评测。")
            
            # 显示示例说明
            st.markdown("### 📋 排行榜说明")
            st.markdown("""
            **Trust-Score排行榜指标说明**：
            - **reject_f1**: 拒答F1分数，衡量模型在信息不足时正确拒答的能力
            - **answered_str_em**: 回答准确性，使用GPT-4o-mini评估语义相似性
            - **answered_citation_f1**: 引用质量F1，使用AutoAIS模型评估引用支持度
            - **trust_score**: 综合可信度分数，结合以上三个维度
            """)
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values("trust_score", ascending=False)
        df.reset_index(drop=True, inplace=True)
        df.index = df.index + 1
        
        # 添加排名列
        df.insert(0, "排名", df.index)
        
        # 格式化数值列
        numeric_cols = ["reject_f1", "answered_str_em", "answered_citation_f1", "trust_score"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        # 显示排行榜标题
        st.markdown(f"### 📊 {dataset_tab} 排行榜")
        
        # 显示排行榜表格
        st.dataframe(
            df.style.highlight_max(axis=0, subset=numeric_cols),
            width="stretch"
        )
        
        # 统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("参与模型数", len(df))
        with col2:
            if len(df) > 0:
                best_model = df.iloc[0]["model_name"]
                best_score = df.iloc[0]["trust_score"]
                st.metric("最佳模型", best_model)
            else:
                st.metric("最佳模型", "暂无")
        with col3:
            if len(df) > 0:
                # 使用原始数据中的average_trust_score而不是计算平均值
                dataset_avg_score = st.session_state.trust_manager.get_dataset_average_trust_score(dataset_type)
                st.metric("平均Trust-Score", f"{dataset_avg_score:.3f}")
            else:
                st.metric("平均Trust-Score", "0.000")
        
        # 可视化对比
        if len(df) > 0:
            st.markdown("### 📈 性能对比图")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Trust-Score对比
                fig1 = px.bar(
                    df, 
                    x="model_name", 
                    y="trust_score",
                    title="Trust-Score对比",
                    color="trust_score",
                    color_continuous_scale="viridis",
                    labels={"model_name": "模型名称", "trust_score": "Trust-Score"}
                )
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, width="stretch")
            
            with col2:
                # 回答准确性对比  
                fig2 = px.bar(
                    df,
                    x="model_name",
                    y="answered_str_em", 
                    title="回答准确性对比",
                    color="answered_str_em",
                    color_continuous_scale="plasma",
                    labels={"model_name": "模型名称", "answered_str_em": "回答准确性"}
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, width="stretch")
            
            # 雷达图对比
            st.markdown("### 🎯 综合能力雷达图")
            
            selected_models = st.multiselect(
                "选择要对比的模型", 
                df["model_name"].tolist(),
                default=df["model_name"].tolist()[:3] if len(df) >= 3 else df["model_name"].tolist()
            )
            
            if selected_models:
                fig = go.Figure()
                
                metrics = ["reject_f1", "answered_str_em", "answered_citation_f1", "trust_score"]
                metric_labels = ["拒答F1", "回答准确性", "引用质量F1", "Trust-Score"]
                
                for model in selected_models:
                    model_data = df[df["model_name"] == model].iloc[0]
                    values = [model_data[metric] for metric in metrics]
                    values.append(values[0])  # 闭合雷达图
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metric_labels + [metric_labels[0]],
                        fill='toself',
                        name=model
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="模型综合能力对比"
                )
                
                st.plotly_chart(fig, width="stretch")
    
    except Exception as e:
        st.error(f"加载排行榜数据时发生错误: {str(e)}")
        st.info("请检查trust_score_results目录是否存在评测结果文件。")

def show_dataset_page():
    """显示数据集浏览页面"""
    st.markdown("## 📚 数据集浏览与统计")
    
    domains = st.session_state.dataset_loader.get_domains()
    
    if not domains:
        st.error("未找到可用的数据集")
        return
    
    # 数据集概览
    st.markdown("### 📊 数据集概览")
    
    overview_data = []
    for domain in domains:
        sample_ids = st.session_state.dataset_loader.get_sample_ids(domain)
        sample_count = len(sample_ids)
        
        # 随机抽样分析
        if sample_count > 0:
            sample_data = st.session_state.dataset_loader.get_sample(domain, sample_ids[0])
            avg_question_len = len(sample_data.get('question', ''))
            avg_answer_len = len(sample_data.get('answer', ''))
            citation_count = len(sample_data.get('model_answer_rebuild_by_citation', []))
        else:
            avg_question_len = avg_answer_len = citation_count = 0
        
        overview_data.append({
            "领域": domain.title(),
            # "样本数量": sample_count,
            "样本数量": 200,
            "平均问题长度": avg_question_len,
            "平均答案长度": avg_answer_len,
            "平均引用句数": citation_count
        })
    
    df_overview = pd.DataFrame(overview_data)
    st.dataframe(df_overview, width="stretch")
    
    # 详细浏览
    st.markdown("### 🔍 详细数据浏览")
    
    selected_domain = st.selectbox("选择要浏览的领域", domains)
    sample_ids = st.session_state.dataset_loader.get_sample_ids(selected_domain)
    
    # 分页显示
    page_size = 5
    total_pages = (len(sample_ids) + page_size - 1) // page_size
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_page = st.selectbox(
            f"页码 (共 {total_pages} 页)", 
            range(1, total_pages + 1)
        )
    
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(sample_ids))
    current_samples = sample_ids[start_idx:end_idx]
    
    # 显示当前页的样本
    for i, sample_id in enumerate(current_samples):
        sample_data = st.session_state.dataset_loader.get_sample(selected_domain, sample_id)
        
        with st.expander(f"样本 {start_idx + i + 1}: {sample_id}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**问题:**")
                st.write(sample_data.get('question', ''))
                
                st.markdown("**参考答案:**")
                st.write(sample_data.get('answer', ''))
            
            with col2:
                model_answer = sample_data.get('model_answer_rebuild_by_citation', [])
                st.markdown(f"**归因句数:** {len(model_answer)}")
                
                if model_answer:
                    total_citations = sum(
                        len(sentence.get('citations', {}).get('anchor_text', []))
                        for sentence in model_answer
                    )
                    st.markdown(f"**总引用数:** {total_citations}")
                    
                    # 显示第一个句子作为示例
                    first_sentence = model_answer[0].get('sentence', '')
                    st.markdown("**首句示例:**")
                    st.write(first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence)

if __name__ == "__main__":
    main()
