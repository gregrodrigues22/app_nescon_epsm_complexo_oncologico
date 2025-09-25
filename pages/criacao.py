# ---------------------------------------------------------------
# Set up
# ---------------------------------------------------------------
import os
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from scipy.stats import linregress
from plotly.subplots import make_subplots
from plotly.colors import sequential
import plotly.graph_objects as go

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------

# Esconde a lista padrão de páginas no topo da sidebar
st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("assets/logo.png", use_column_width=True)
    st.markdown("<div style='margin: 20px 0;'><hr style='border:none;border-top:1px solid #ccc;'/></div>", unsafe_allow_html=True)
    st.header("Menu")
    st.page_link("app.py", label="Complexo Oncológico", icon="📊")
    st.page_link("pages/criacao.py", label="Referência", icon="✅")
    st.markdown("<div style='margin: 20px 0;'><hr style='border:none;border-top:1px solid #ccc;'/></div>", unsafe_allow_html=True)

    if st.sidebar.button("🔄 Atualizar dados agora"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------
st.markdown("""
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Nescon EPSM</h1>
        <p style='color: white;'>Conheça a estrutura e equipe da Estação de Pesquisa de Sinais de Mercado</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Conteúdo
# ---------------------------------------------------------------
st.header("Em Contrução")