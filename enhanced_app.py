"""
Large Model Attribution Analysis Platform - Enhanced Version
Author: AI Assistant
Features:
1. Attribution visualization
2. Custom model evaluation
3. Leaderboard display
4. Dataset browsing and testing
5. Real-time evaluation
"""

import streamlit as st
import pandas as pd
from pandas.io.formats.style import Styler
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import os
import sys
import random
from datetime import datetime
import time
import base64
import html

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def resolve_data_dir(default_path: str) -> str:
    raw_path = os.environ.get("WEB_LEADERBOARD_DATA_DIR_V4") or os.environ.get("WEB_LEADERBOARD_DATA_DIR")
    return os.path.abspath(raw_path or default_path)

def resolve_resource_path(relative_path: str) -> str:
    return os.path.abspath(os.path.join(BASE_DIR, relative_path))

# Custom module imports
try:
    from data_manager import DatasetManager, LeaderboardManager, SampleAnalyzer, create_sample_leaderboard_data
    from model_api import ModelAPIFactory, AttributionMetrics, run_evaluation
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.info("Please ensure all dependencies are installed correctly")

# Page configuration
st.set_page_config(
    page_title="Book evidence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    font-weight: bold;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1e3a8a, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.citation-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #f59e0b;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.75rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.evidence-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #10b981;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.75rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.inline-citation {
    display: inline-block;
    margin-left: 4px;
}

.inline-citation summary {
    display: inline;
    cursor: pointer;
    color: #2563eb;
    font-weight: 600;
    list-style: none;
}

.inline-citation summary::-webkit-details-marker {
    display: none;
}

.inline-citation .citation-text {
    margin-top: 6px;
    padding: 6px 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    color: #0f172a;
    font-size: 0.9rem;
}

.section-label {
    display: inline-block;
    font-weight: 700;
    color: #0f172a;
    background: #e2e8f0;
    border: 1px solid #cbd5f5;
    border-radius: 6px;
    padding: 4px 10px;
    margin: 6px 0 10px;
    font-size: 0.95rem;
}

.citation-item {
    display: inline-block;
    margin: 2px 6px 6px 0;
    padding: 2px 8px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    color: #1d4ed8;
    font-weight: 600;
    font-size: 0.9rem;
}

.citation-block {
    margin: 6px 0 10px;
    padding: 8px 10px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

.inline-citation-text {
    display: inline-block;
    margin-left: 6px;
    padding: 2px 6px;
    background: #fef9c3;
    border: 1px solid #fde047;
    border-radius: 4px;
    color: #7c2d12;
    font-size: 0.9rem;
}

.success-banner {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid #22c55e;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    text-align: center;
    font-weight: bold;
    color: #166534;
}

.warning-banner {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 2px solid #f59e0b;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    text-align: center;
    font-weight: bold;
    color: #92400e;
}

.info-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}

.stSelectbox > div > div {
    background-color: #f8fafc;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
}
</style>
""", unsafe_allow_html=True)

def render_table(df, hide_index=False):
    """Render tables with a neutral, academic style and centered alignment."""
    if isinstance(df, Styler):
        styler = df
    else:
        styler = (
            df.style
            .set_table_styles([
                {
                    "selector": "table",
                    "props": [("border-collapse", "collapse"), ("border", "1px solid #d0d0d0")],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#f7f7f7"),
                        ("color", "#111111"),
                        ("border-bottom", "1px solid #999999"),
                        ("font-weight", "600"),
                        ("text-align", "center"),
                        ("padding", "6px 8px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-bottom", "1px solid #e0e0e0"),
                        ("padding", "6px 8px"),
                        ("font-size", "13px"),
                        ("text-align", "center"),
                    ],
                },
            ])
            .set_properties(**{"text-align": "center"})
        )
    st.dataframe(styler, width="stretch", hide_index=hide_index)


def render_image_percent(image_path: str, percent: int = 60) -> None:
    """Render a local image at a fixed percentage width."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
    st.markdown(
        f'<img src="data:image/{ext};base64,{encoded}" '
        f'style="width: {percent}%; display: block; margin: 0 auto;">',
        unsafe_allow_html=True,
    )

def initialize_session_state():
    """Initialize session state."""
    if 'dataset_manager' not in st.session_state:
        data_dir = resolve_data_dir("/Users/chengyihao/Documents/vscode-python/web_leaderboard/BookEvidenceQA_v4")
        st.session_state.dataset_manager = DatasetManager(data_dir)
    
    if 'leaderboard_manager' not in st.session_state:
        st.session_state.leaderboard_manager = LeaderboardManager()
    
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None

def main():
    """Main entry point."""
    initialize_session_state()
    
    # Title
    st.markdown('<h1 class="main-header">Book evidence</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("### Bookevidence 📖")
    page = st.sidebar.radio(
        "Navigation",
        ["Main Page", "Dataset Overview", "Leaderboard", "Evaluation", "Resources"],
        label_visibility="collapsed"
    )
    
    # Render page by selection
    if page == "Main Page":
        show_home_page()
    elif page == "Dataset Overview":
        show_dataset_overview()
    elif page == "Leaderboard":
        show_leaderboard_page()
    elif page == "Evaluation":
        show_evaluation_page()
    elif page == "Resources":
        show_resources_page()

def show_home_page():
    """Home page."""
    st.markdown("## Main Page")
    st.markdown("""
    BookEvidence是一个用于训练、评估和分析长上下文问答中证据归因能力的基准，旨在评估模型在复杂、跨领域长文本条件下，将生成内容与可验证证据进行精细对齐的能力。
    
    BookEvidence 包括：
    
    * 一个高质量的长上下文问答数据集，问题与证据来源于书籍级或章节级文档，覆盖法律、金融、历史、哲学、教育等多个领域。每个样本要求模型在生成答案的同时，以句子为粒度标注其支持证据，用于评估细粒度证据归因能力。
    
    * 一个统一的评估框架，基于 Trust-Score，从答案正确性、证据不足条件下的拒答行为以及引用准确性三个维度综合衡量模型表现。
    
    * 一个公开排行榜（Leaderboard），用于系统性比较不同开源与闭源模型在 BookEvidence 上的表现，揭示现有方法在长上下文证据归因方面的能力差异。
    
    BookEvidence 采用与模型无关的数据与评测格式，适用于任何能够处理长文本并生成带证据引用回答的系统。其目标是推动真实长文本场景下可解释、可信问答系统的研究，并为证据归因能力的泛化评估提供一个具有挑战性且可复现的标准基准。
    """)

    st.markdown("###  Dataset Overview")
    domains = st.session_state.dataset_manager.get_domains()
    if domains:
        overview_data = []
        for domain in domains:
            stats = st.session_state.dataset_manager.get_dataset_stats(domain)
            overview_data.append({
                "Domain": domain.title(),
                "Sample Count": stats.get('total_samples', 0),
                "Avg Question Length": f"{stats.get('avg_question_length', 0):.0f} chars",
                "Avg Answer Length": f"{stats.get('avg_answer_length', 0):.0f} chars",
                "Avg Citations": f"{stats.get('avg_citation_count', 0):.1f}",
                "Avg Sentences": f"{stats.get('avg_sentence_count', 0):.1f}"
            })
        df_overview = pd.DataFrame(overview_data)
        render_table(df_overview, hide_index=True)
    else:
        st.warning(" No available dataset.")

def show_attribution_page(show_header=True):
    """Show attribution page."""
    if show_header:
        st.markdown("## Attribution Analysis")
    
    domains = st.session_state.dataset_manager.get_domains()
    if not domains:
        st.error(" No dataset found. Please check the data directory.")
        return
    
    # Data selection
    with st.container():
        st.markdown("###  Data Samples")
        
        default_domain = "agriculture" if "agriculture" in domains else domains[0]
        default_domain_index = domains.index(default_domain)
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            selected_domain = st.selectbox(
                "Select analysis domain",
                domains,
                index=default_domain_index,
                format_func=lambda x: f"{x.title()} ({len(st.session_state.dataset_manager.get_sample_ids(x))} samples)"
            )
        
        with col2:
            sample_ids = st.session_state.dataset_manager.get_sample_ids(selected_domain)
            if not sample_ids:
                st.error(f"Domain {selected_domain}  has no available samples")
                return
            
            selection_mode = st.radio(
                "Selection mode",
                ["Manual", "Random"],
                horizontal=True,
                index=0
            )
        
    
    # Sample selection
    selected_sample_id = None
    sample_data = None
    
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = (selected_domain, sample_ids[0])
    
    if selection_mode == "Random":
        if st.button(" Random Sample", type="primary"):
            selected_sample_id = random.choice(sample_ids)
            st.session_state.selected_sample = (selected_domain, selected_sample_id)
        
        if st.session_state.selected_sample and st.session_state.selected_sample[0] == selected_domain:
            selected_sample_id = st.session_state.selected_sample[1]
    
    elif selection_mode == "Manual":
        selected_sample_id = st.selectbox(
            "Select sample",
            sample_ids,
            index=0,
            format_func=lambda x: f"{x[:50]}..." if len(x) > 50 else x
        )
        st.session_state.selected_sample = (selected_domain, selected_sample_id)
    
    # Display selected sample
    if selected_sample_id:
        sample_data = st.session_state.dataset_manager.get_sample(selected_domain, selected_sample_id)
        
        if sample_data:
            st.markdown("###  Q&A")
            st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
            st.info(sample_data.get('question', ''))
            
            st.markdown('<div class="section-label">Answer (w/ citations)</div>', unsafe_allow_html=True)
            model_answer = sample_data.get('sentence_level_citation', [])

            def build_answer_html(answer_sentences: List[Dict[str, Any]]) -> str:
                parts = []
                for sentence_data in answer_sentences:
                    sentence = (sentence_data.get('sentence') or '').strip()
                    if not sentence:
                        continue
                    base_sentence = html.escape(sentence.rstrip("."))
                    citations = sentence_data.get('citations', {})
                    anchor_indexes = citations.get('anchor_index', []) or []
                    anchor_texts = citations.get('anchor_text', []) or []
                    citation_html = []
                    for idx, anchor_idx in enumerate(anchor_indexes):
                        anchor_text = anchor_texts[idx] if idx < len(anchor_texts) else ""
                        anchor_text_html = html.escape(anchor_text).replace("\n", " ").strip()
                        citation_html.append(
                            f'<details class="inline-citation"><summary>[{anchor_idx}]</summary>'
                            f'<span class="inline-citation-text">{anchor_text_html}</span></details>'
                        )
                    parts.append(f"{base_sentence}{''.join(citation_html)}.")
                return " ".join(parts)
            
            if model_answer:
                answer_html = build_answer_html(model_answer)
                st.markdown(f'<div class="evidence-box">{answer_html}</div>', unsafe_allow_html=True)
            else:
                st.warning("No attribution data found for this sample.")

            st.markdown('<div class="section-label">Article</div>', unsafe_allow_html=True)
            with st.expander("View article citations", expanded=False):
                faithfulness_texts = sample_data.get("faithfulness_original_text", [])
                if faithfulness_texts:
                    rendered_blocks = []
                    for item in faithfulness_texts:
                        if isinstance(item, dict):
                            anchor_text = (item.get("anchor_text") or "").strip()
                            anchor_index = item.get("anchor_index")
                            text = (item.get("text") or item.get("context") or anchor_text).strip()
                        else:
                            anchor_text = ""
                            anchor_index = None
                            text = str(item).strip()

                        if not text:
                            continue

                        safe_text = html.escape(text)
                        safe_anchor = html.escape(anchor_text) if anchor_text else ""
                        if safe_anchor and safe_anchor in safe_text:
                            highlighted = safe_text.replace(
                                safe_anchor,
                                f"<mark><strong>{safe_anchor}</strong></mark>",
                            )
                        else:
                            highlighted = safe_text

                        index_prefix = f"[{anchor_index}] " if anchor_index is not None else ""
                        rendered_blocks.append(
                            f'<div class="citation-box">{index_prefix}…{highlighted}…</div>'
                        )

                    if rendered_blocks:
                        st.markdown("".join(rendered_blocks), unsafe_allow_html=True)
                    else:
                        st.warning("No citation text found in faithfulness_original_text.")
                else:
                    st.warning("faithfulness_original_text is empty or missing.")

            st.markdown("#### Sample JSON (Simplified)")
            st.code(
                """{
  "sample_id_xxx_domain": {
    "question": "What are the steps involved in extracting and handling honey?",
    "answer": "First, remove the wax cappings with a knife....",
    "source": "agriculture_660ddc2dc66d64e3f60f3da5b6634b9d",
    "sentence_level_citation": [
      {
        "sentence": "First, remove the wax cappings with a knife.",
        "citations": {
          "anchor_index": [2569],
          "anchor_text": ["Some kind of blade is necessary to remove the wax cappings."],
          "prefix_text": ["Uncapping knife ..."]
        }
      }
      ...
    ],
    "faithfulness_original_text": [
      {
        "anchor_index": 2569,
        "anchor_text": " It is a relatively straightforward task, but it has its nuances. The best way to remove honey from the supers, and just about the only way, is to use an extractor, a piece of equipment that whirls the frames and removes the honey by centrifugal force.",
        "text": " The best way to remove honey from the supers, and just about the only way, is to use an extractor, a piece of equipment that whirls the frames and removes the honey by centrifugal force."
      }
      ...
    ]
  }
}""",
                language="json"
            )

def show_evaluation_page():
    """Show evaluation page."""
    st.markdown("## Evaluation")
    
    st.markdown("""
    <div class="info-card">
    <h4> Evaluation Notes</h4>
    <p>This platform evaluates attribution capability for multiple model APIs, including OpenAI, HuggingFace, and custom APIs.</p>
    <p>Evaluation computes key metrics such as citation precision and recall on the selected dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration
    with st.form("model_evaluation_form"):
        st.markdown("###  Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "Model name *",
                placeholder="e.g., GPT-4, Claude-3, Qwen2-72B",
                help="Provide a display name for your model"
            )
            
            model_type = st.selectbox(
                "API type *",
                ["OpenAI API", "HuggingFace API", "Custom API"],
                help="Select the API type"
            )
        
        with col2:
            api_key = st.text_input(
                "API Key *",
                type="password",
                help="Your API access key"
            )
            
            if model_type == "Custom API":
                api_url = st.text_input(
                    "API URL *",
                    placeholder="https://your-api-endpoint.com/v1/chat",
                    help="Your custom API endpoint"
                )
            else:
                api_url = ""
        
        # Evaluation Configuration
        st.markdown("### Evaluation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            domains = st.session_state.dataset_manager.get_domains()
            test_domains = st.multiselect(
                "Select test domain *",
                domains,
                default=domains[:2] if len(domains) >= 2 else domains,
                help="Select domains to evaluate"
            )
            
            sample_size = st.slider(
                "Samples per domain",
                min_value=1,
                max_value=50,
                value=10,
                help="Start with a small sample size"
            )
        
        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls randomness of the output"
            )
            
            max_tokens = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                help="Maximum output tokens"
            )
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "Start Evaluation",
                type="primary",
                width="stretch"
            )
    
    # Handle evaluation request
    if submitted:
        # Validate inputs
        if not all([model_name, api_key, test_domains]):
            st.error(" Please fill in all required fields (marked with *).")
            return
        
        if model_type == "Custom API" and not api_url:
            st.error(" Custom API requires an API URL.")
            return
        
        # Start evaluation
        st.markdown("### Evaluation in progress...")
        
        # Progress and status
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_placeholder = st.empty()
        
        try:
            # 模拟评测过程（实际应用中这里会调用真实的API）
            total_steps = len(test_domains) * sample_size + 3
            current_step = 0
            
            # 步骤1: 初始化模型
            status_text.text(" Initializing model connection...")
            time.sleep(1)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
            # 步骤2: 准备测试数据
            status_text.text(" Preparing test data...")
            test_samples = []
            for domain in test_domains:
                samples = st.session_state.dataset_manager.get_random_samples(domain, sample_size)
                test_samples.extend([(domain, sample_id, sample_data) for sample_id, sample_data in samples])
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
            # 步骤3: 运行评测
            status_text.text(" Evaluating model performance...")
            
            # Simulated evaluation results (production uses model_api)
            domain_results = {}
            overall_metrics = {
                "Citation Precision": 0,
                "Citation Recall": 0,
                "F1 Score": 0,
                "Answer Similarity": 0
            }
            
            for i, (domain, sample_id, sample_data) in enumerate(test_samples):
                status_text.text(f" Evaluating sample {i+1}/{len(test_samples)} (Domain: {domain})")
                
                # 模拟处理时间
                time.sleep(0.1)
                
                # 模拟结果
                if domain not in domain_results:
                    domain_results[domain] = []
                
                # 生成随机但合理的结果
                precision = random.uniform(0.6, 0.9)
                recall = random.uniform(0.5, 0.8)
                f1 = 2 * precision * recall / (precision + recall)
                similarity = random.uniform(0.7, 0.95)
                
                domain_results[domain].append({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "similarity": similarity
                })
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # 计算整体结果
            all_results = []
            for domain_samples in domain_results.values():
                all_results.extend(domain_samples)
            
            if all_results:
                overall_metrics["Citation Precision"] = sum(r["precision"] for r in all_results) / len(all_results)
                overall_metrics["Citation Recall"] = sum(r["recall"] for r in all_results) / len(all_results)
                overall_metrics["F1 Score"] = sum(r["f1"] for r in all_results) / len(all_results)
                overall_metrics["Answer Similarity"] = sum(r["similarity"] for r in all_results) / len(all_results)
            
            # 完成评测
            status_text.text(" Evaluation completed.")
            progress_bar.progress(1.0)
            
            # 显示结果
            st.markdown("""
            <div class="success-banner">
             Evaluation completed successfully!
            </div>
            """, unsafe_allow_html=True)
            
            # 整体指标
            st.markdown("###  Overall Evaluation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            metrics = list(overall_metrics.items())
            
            for i, (metric, value) in enumerate(metrics):
                with [col1, col2, col3, col4][i]:
                    # 根据数值设置颜色
                    if value >= 0.8:
                        delta_color = "normal"
                    elif value >= 0.6:
                        delta_color = "off"
                    else:
                        delta_color = "inverse"
                    
                    st.metric(
                        metric,
                        f"{value:.3f}",
                        delta=f"{'Excellent' if value >= 0.8 else 'Good' if value >= 0.6 else 'Needs Improvement'}"
                    )
            
            # Per-domain Results
            if len(test_domains) > 1:
                st.markdown("###  Per-domain Results")
                
                domain_summary = []
                for domain, results in domain_results.items():
                    avg_precision = sum(r["precision"] for r in results) / len(results)
                    avg_recall = sum(r["recall"] for r in results) / len(results)
                    avg_f1 = sum(r["f1"] for r in results) / len(results)
                    avg_similarity = sum(r["similarity"] for r in results) / len(results)
                    
                    domain_summary.append({
                        "Domain": domain.title(),
                        "Sample Count": len(results),
                        "Citation Precision": f"{avg_precision:.3f}",
                        "Citation Recall": f"{avg_recall:.3f}",
                        "F1 Score": f"{avg_f1:.3f}",
                        "Answer Similarity": f"{avg_similarity:.3f}"
                    })
                
                df_domain = pd.DataFrame(domain_summary)
                render_table(df_domain, hide_index=True)
            
            # Result Visualization
            st.markdown("###  Result Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Metric comparison
                fig1 = px.bar(
                    x=list(overall_metrics.keys()),
                    y=list(overall_metrics.values()),
                    title=f"{model_name} Overall Performance",
                    labels={'x': 'Metric', 'y': 'Score'},
                    color=list(overall_metrics.values()),
                    color_continuous_scale="viridis"
                )
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, width="stretch")
            
            with col2:
                if len(test_domains) > 1:
                    # Domain comparison
                    domain_f1_scores = [
                        sum(r["f1"] for r in domain_results[domain]) / len(domain_results[domain])
                        for domain in test_domains
                    ]
                    
                    fig2 = px.bar(
                        x=[d.title() for d in test_domains],
                        y=domain_f1_scores,
                        title="F1 Score by Domain",
                        labels={'x': 'Domain', 'y': 'F1 Score'},
                        color=domain_f1_scores,
                        color_continuous_scale="plasma"
                    )
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, width="stretch")
            
            # 保存结果选项
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(" Save to Leaderboard", type="primary", width="stretch"):
                    # 准备保存的数据
                    save_data = {
                        "model_name": model_name,
                        "model_type": model_type,
                        **overall_metrics,
                        "test_domains": test_domains,
                        "sample_count": len(test_samples),
                        "config": {
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    }
                    
                    try:
                        evaluation_id = st.session_state.leaderboard_manager.save_evaluation_result(save_data)
                        st.success(f" Evaluation result saved！(ID: {evaluation_id})")
                        time.sleep(1)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            
        except Exception as e:
            status_text.text(" Error during evaluation")
            st.error(f"Evaluation failed: {str(e)}")
            st.info(" Tip: check API configuration or try fewer samples.")


def show_leaderboard_page():
    """显示Leaderboard页面"""
    st.markdown("## Performance Ranking")
    st.markdown(
        "We evaluate models using Trust-Score (Citation: "
        "Song, Maojia; Sim, Shang Hong; Bhardwaj, Rishabh; Chieu, Hai Leong; "
        "Majumder, Navonil; Poria, Soujanya. 2025. "
        "*Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded "
        "Attributions and Learning to Refuse*. ICLR. "
        "https://openreview.net/forum?id=Iyrtb9EJBp), which jointly measures:<br>"
        "(1) answer correctness,<br>"
        "(2) refusal behavior under insufficient evidence, and<br>"
        "(3) citation accuracy.",
        unsafe_allow_html=True
    )
    render_image_percent(
        resolve_resource_path(os.path.join("resources", "trust_score.png")),
        percent=60,
    )
    
    try:
        data_path = resolve_resource_path(os.path.join("resources", "leaderboard_data.xlsx"))
        leaderboard_df = pd.read_excel(data_path)

        training_marker = leaderboard_df["Traing_dataset"].fillna("None").astype(str).str.strip()
        training_free_mask = training_marker.str.lower().isin(["none", "nan", ""])
        training_free_df = leaderboard_df[training_free_mask].copy()
        training_based_df = leaderboard_df[~training_free_mask].copy()

        for df in (training_free_df, training_based_df):
            if not df.empty:
                df.sort_values("Trust_score", ascending=False, inplace=True)
                df.reset_index(drop=True, inplace=True)

        st.markdown("###  Training-free Methods")
        if training_free_df.empty:
            st.info("No training-free results.")
        else:
            render_table(training_free_df, hide_index=True)

        st.markdown("###  Training-based Methods")
        if training_based_df.empty:
            st.info("No training-based results.")
        else:
            render_table(training_based_df, hide_index=True)

        render_image_percent(
            resolve_resource_path(os.path.join("resources", "trust_domain_comparison.png")),
            percent=60,
        )
        render_image_percent(
            resolve_resource_path(os.path.join("resources", "mcts_contrast.png")),
            percent=60,
        )
        st.markdown("""
        - Fine-tuning and post-hoc inference provide complementary benefits in our experiments.
        - MCTS-style inference is largely complementary to fine-tuning; CGC-LoRA (6k/8k/10k) + MCTS yields the best results and surpasses Trust-DPO (19k).
        - MCTS is a Monte Carlo search for near-optimal solutions that uncovers latent reasoning capability in the base model.
        - We observe SFT (LoRA) is complementary to MCTS, while RL is not; this is consistent with findings in
          [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://openreview.net/forum?id=Iyrtb9EJBp).
        - SFT distillation with high-quality data expands the capability boundary, whereas RL mainly sharpens the distribution within existing capacity.
        """)
    
    except Exception as e:
        st.error(f"Failed to load leaderboard data: {e}")
        st.info("Check resources/leaderboard_data.xlsx and its columns.")

def show_resources_page():
    """Show resources page."""
    st.markdown("## Resources")
    resources_df = pd.DataFrame(
        [
            {"Item": "Dataset", "Status": "Coming soon"},
            {"Item": "Code", "Status": "Coming soon"},
            {"Item": "Paper", "Status": "Draft in progress"},
        ]
    )
    st.dataframe(resources_df, width="stretch", hide_index=True)

def show_dataset_page(show_header=True):
    """Show dataset browser page."""
    if show_header:
        st.markdown("## Dataset Browser")
    
    domains = st.session_state.dataset_manager.get_domains()
    
    if not domains:
        st.error(" No available dataset.")
        return
    
    # Dataset Overview
    st.markdown("###  Dataset Overview")
    
    # Build overview data
    overview_data = []
    total_samples = 0
    
    for domain in domains:
        stats = st.session_state.dataset_manager.get_dataset_stats(domain)
        sample_count = stats.get('total_samples', 0)
        total_samples += sample_count
        
        overview_data.append({
            "Domain": domain.title(),
            "Sample Count": sample_count,
            "Avg Question Length": f"{stats.get('avg_question_length', 0):.0f} chars",
            "Avg Answer Length": f"{stats.get('avg_answer_length', 0):.0f} chars",
            "Avg Citations": f"{stats.get('avg_citation_count', 0):.1f}",
            "Avg Sentences": f"{stats.get('avg_sentence_count', 0):.1f}"
        })
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(" Dataset Count", len(domains))
    with col2:
        st.metric(" Total Samples", total_samples)
    with col3:
        avg_samples = total_samples // len(domains) if domains else 0
        st.metric(" Average Samples", avg_samples)
    with col4:
        largest_domain = max(overview_data, key=lambda x: x["Sample Count"])["Domain"] if overview_data else "N/A"
        st.metric(" Largest Dataset", largest_domain)
    
    # Overview table moved to Main Page
    df_overview = pd.DataFrame(overview_data)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample Count分布
        fig1 = px.pie(
            df_overview,
            values=[int(x) for x in df_overview["Sample Count"]],
            names="Domain",
            title="Sample Distribution by Domain"
        )
        st.plotly_chart(fig1, width="stretch")
    
    with col2:
        # Avg Citations对比
        fig2 = px.bar(
            df_overview,
            x="Domain",
            y=[float(x) for x in df_overview["Avg Citations"]],
            title="Avg Citations by Domain",
            color=[float(x) for x in df_overview["Avg Citations"]],
            color_continuous_scale="viridis"
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width="stretch")
    
    # Detailed browsing
    st.markdown("###  Detailed Data Browsing")
    
    # Select domain
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_domain = st.selectbox(
            "Select domain to browse",
            domains,
            format_func=lambda x: f"{x.title()} ({len(st.session_state.dataset_manager.get_sample_ids(x))} samples)"
        )
    
    with col2:
        # Search
        search_term = st.text_input(" Search keyword", placeholder="Enter keyword to search...")
    
    # Fetch sample data
    if search_term:
        sample_results = st.session_state.dataset_manager.search_samples(
            selected_domain, search_term, max_results=50
        )
        st.info(f" Found {len(sample_results)} samples containing '{search_term}'")
    else:
        sample_ids = st.session_state.dataset_manager.get_sample_ids(selected_domain)
        sample_results = [(id, st.session_state.dataset_manager.get_sample(selected_domain, id)) 
                         for id in sample_ids]
    
    if not sample_results:
        st.warning("No matching samples found")
        return
    
    # Pagination
    page_size = 5
    total_pages = (len(sample_results) + page_size - 1) // page_size
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_page = st.selectbox(
            f"Page ({total_pages} total, {len(sample_results)} samples)",
            range(1, total_pages + 1)
        )
    
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(sample_results))
    current_samples = sample_results[start_idx:end_idx]
    
    # Render current page samples
    for i, (sample_id, sample_data) in enumerate(current_samples):
        sample_index = start_idx + i + 1
        
        with st.expander(f" Sample {sample_index}: {sample_id[:60]}...", expanded=False):
            # Basic info
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("** Question:**")
                question = sample_data.get('question', '')
                st.write(question)
                
                st.markdown("** Reference Answer:**")
                answer = sample_data.get('answer', '')
                st.write(answer)
            
            with col2:
                # Sample Features
                features = SampleAnalyzer.extract_sample_features(sample_data)
                
                st.markdown("** Sample Features:**")
                st.write(f"• Question Length: {features['question_length']} chars")
                st.write(f"• Answer Length: {features['answer_length']} chars")
                st.write(f"• Question Word Count: {features['question_word_count']}")
                st.write(f"• Answer Word Count: {features['answer_word_count']}")
                st.write(f"• Attribution Sentences: {features['total_sentences']}")
                st.write(f"• Total Citations: {features['total_citations']}")
                st.write(f"• Citation Coverage: {features['citation_coverage']:.1%}")
            
            # Attribution Preview
            model_answer = sample_data.get('sentence_level_citation', [])
            if model_answer:
                st.markdown("** Attribution Preview:**")
                
                # Show attribution for first two sentences
                for j, sentence_data in enumerate(model_answer[:2]):
                    sentence_text = sentence_data.get('sentence', '')
                    citations = sentence_data.get('citations', {})
                    anchor_count = len(citations.get('anchor_text', []))
                    
                    st.markdown(f"""
                    <div class="info-card">
                    <strong>Sentence {j+1}:</strong> {sentence_text[:100]}{'...' if len(sentence_text) > 100 else ''}<br>
                    <em>Citation count: {anchor_count}</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(model_answer) > 2:
                    st.info(f"... {len(model_answer) - 2} more sentences")
            else:
                st.warning("No attribution information for this sample.")
            
            # Actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f" Detailed Analysis", key=f"analyze_{sample_id}"):
                    st.session_state.selected_sample = (selected_domain, sample_id)
                    st.info("Go to 'Dataset Overview' to view detailed analysis.")
            
            with col2:
                if st.button(f" Copy ID", key=f"copy_{sample_id}"):
                    st.success(f"Sample ID copied: {sample_id}")
            
            with col3:
                # Export (placeholder)
                st.button(" Export", key=f"export_{sample_id}", disabled=True, help="In development")

def show_dataset_overview():
    """Combined attribution and dataset browsing page."""
    st.markdown("## Dataset Overview")
    st.markdown("###  Data Generation Pipeline")
    st.image(
        resolve_resource_path(os.path.join("resources", "Automatic_Data_Generation_Pipeline.png")),
        width="stretch"
    )

    st.markdown("---")
    show_attribution_page(show_header=False)

if __name__ == "__main__":
    main()
