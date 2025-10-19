"""
Example Streamlit components for custom UI extensions
"""

import streamlit as st
from typing import Dict, List, Any

def metric_card(label: str, value: str, icon: str = "📊"):
    """Display a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <h4>{icon} {label}</h4>
        <p style="font-size: 24px; font-weight: bold;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

def document_card(title: str, score: float, source: str, snippet: str):
    """Display a styled document card"""
    st.markdown(f"""
    <div class="doc-box">
        <h5>{title}</h5>
        <p><strong>Score:</strong> {score:.3f} | <strong>Source:</strong> {source}</p>
        <p><em>"{snippet[:100]}..."</em></p>
    </div>
    """, unsafe_allow_html=True)

def cache_status_badge(is_hit: bool):
    """Display cache status badge"""
    if is_hit:
        st.markdown("""
        <div class="cache-hit">
        ✅ <b>Cache HIT</b> (0.01s)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cache-miss">
        ❌ <b>Cache MISS</b> (Full pipeline)
        </div>
        """, unsafe_allow_html=True)

def performance_table(timing: Dict[str, float]):
    """Display performance breakdown table"""
    import pandas as pd
    
    data = {
        'Stage': ['Retrieval', 'Reranking', 'Generation'],
        'Time (s)': [
            timing.get('retrieval_time', 0),
            timing.get('rerank_time', 0),
            timing.get('generation_time', 0)
        ]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def history_list(history: List[Dict[str, Any]]):
    """Display query history as interactive list"""
    st.subheader("Recent Queries")
    
    for i, entry in enumerate(reversed(history[-10:])):
        with st.expander(f"{i+1}. {entry['query'][:50]}..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Timestamp:** {entry['timestamp']}")
            
            with col2:
                cache_status = "✅ HIT" if entry['cache_hit'] else "❌ MISS"
                st.write(f"**Status:** {cache_status}")
            
            with col3:
                st.write(f"**Docs:** {entry['sources_count']}")
