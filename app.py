import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import io
import google.generativeai as genai
from modules.data_loader import render_data_loader
from modules.visualizations import render_visualizations
from modules.hypothesis_test import render_hypothesis_test
from modules.ai_assistant import render_ai_assistant

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatLab AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f17;
    --surface: #161924;
    --border: #252840;
    --accent: #6c63ff;
    --accent2: #ff6b6b;
    --text: #e8eaf0;
    --muted: #7a7d94;
    --green: #00d9a0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.5px;
}
h1 { color: var(--accent) !important; }
h2 { color: var(--text) !important; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
h3 { color: var(--accent) !important; }

/* Cards / containers */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 10px 0;
}
.metric-box {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    color: var(--accent);
}

/* Decision banners */
.decision-reject {
    background: rgba(255,107,107,0.12);
    border: 1px solid var(--accent2);
    border-left: 4px solid var(--accent2);
    border-radius: 8px;
    padding: 16px 20px;
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 16px;
}
.decision-fail {
    background: rgba(0,217,160,0.10);
    border: 1px solid var(--green);
    border-left: 4px solid var(--green);
    border-radius: 8px;
    padding: 16px 20px;
    color: var(--green);
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 16px;
}
.ai-response {
    background: linear-gradient(135deg, rgba(108,99,255,0.08), rgba(108,99,255,0.02));
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
}
.tag {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    border: 1px solid rgba(108,99,255,0.3);
}
/* Buttons */
.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #7d75ff;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(108,99,255,0.4);
}
/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: rgba(108,99,255,0.15) !important;
    border-radius: 8px;
}
/* Matplotlib dark theme */
</style>
""", unsafe_allow_html=True)

# ── Matplotlib global dark theme ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#161924',
    'axes.facecolor':   '#0d0f17',
    'axes.edgecolor':   '#252840',
    'axes.labelcolor':  '#e8eaf0',
    'axes.grid':        True,
    'grid.color':       '#252840',
    'grid.linewidth':   0.6,
    'xtick.color':      '#7a7d94',
    'ytick.color':      '#7a7d94',
    'text.color':       '#e8eaf0',
    'legend.facecolor': '#161924',
    'legend.edgecolor': '#252840',
})

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #252840; padding-bottom:16px; margin-bottom:24px;">
  <h1 style="margin:0; font-size:32px;">📊 StatLab <span style="color:#ff6b6b;">AI</span></h1>
  <p style="color:#7a7d94; margin:4px 0 0; font-family:'DM Sans'; font-size:14px;">
    Distribuciones · Pruebas de Hipótesis · Asistente Gemini
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 StatLab AI")
    st.markdown("---")
    page = st.radio(
        "Módulos",
        ["📂 Carga de Datos", "📈 Visualizaciones", "🧪 Prueba Z", "🤖 Asistente IA"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#7a7d94; font-family:'Space Mono',monospace;">
    Supuestos prueba Z:<br>
    • σ poblacional conocida<br>
    • n ≥ 30<br><br>
    API: Google Gemini
    </div>
    """, unsafe_allow_html=True)

# ── Route pages ─────────────────────────────────────────────────────────────────
if page == "📂 Carga de Datos":
    render_data_loader()
elif page == "📈 Visualizaciones":
    render_visualizations()
elif page == "🧪 Prueba Z":
    render_hypothesis_test()
elif page == "🤖 Asistente IA":
    render_ai_assistant()
