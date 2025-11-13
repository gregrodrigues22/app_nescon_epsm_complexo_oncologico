# ---------------------------------------------------------------
# Set up
# ---------------------------------------------------------------
import io 
import re
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import pandas as pd
import json, unicodedata
import os
from datetime import datetime
import pytz
import plotly.express as px
import hashlib
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ---------------------------------------------------------------
# Config da página
# ---------------------------------------------------------------
st.set_page_config(layout="wide", page_title="📊 Complexos Produtivos em Saúde")

# ---------------- Helpers para assets ----------------
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"

def first_existing(*relative_paths: str) -> Path | None:
    for rel in relative_paths:
        p = ASSETS / rel
        if p.exists():
            return p
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

# ---------------- Cabeçalho ----------------
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Complexos Produtivos da Saúde</h1>
        <p style='color: white;'>Explore os complexos produtivos assistenciais para organizar redes e decisões em saúde</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* Esconde a lista padrão de páginas no topo da sidebar */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- helper para evitar crash do st.page_link quando não é multipage ---
def safe_page_link(path: str, label: str, icon: str | None = None):
    try:
        if (APP_DIR / path).exists():
            st.page_link(path, label=label, icon=icon)
        else:
            st.button(label, icon=icon, disabled=True, help="Página não disponível neste app.")
    except Exception:
        st.button(label, icon=icon, disabled=True, help="Navegação multipage indisponível aqui.")

# ---------------- Sidebar (único) ----------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_container_width=True)
    else:
        st.warning(f"Logo não encontrada em {ASSETS}/logo.(png|jpg|jpeg|webp)")
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")

    # ---- Navegação por Complexo Produtivo ----
    with st.expander("Complexos Produtivos", expanded=True):
        safe_page_link("pages/complexo_oncologia.py",
                       label="Oncologia",
                       icon="🎗️")
        safe_page_link("pages/complexo_cardiovascular.py",
                       label="Cardiovascular",
                       icon="❤️")
        safe_page_link("pages/complexo_ortopedia_trauma.py",
                       label="Ortopedia e Trauma",
                       icon="🦴")
        safe_page_link("pages/complexo_obstetricia_neonatologia.py",
                       label="Obstetrícia e Neonatologia",
                       icon="🤰")
        safe_page_link("pages/complexo_neuro.py",
                       label="Neurologia/Neurocirurgia",
                       icon="🧠")
        safe_page_link("pages/complexo_nefrologia_trs.py",
                       label="Nefrologia e TRS",
                       icon="🧪")
        safe_page_link("pages/complexo_queimados.py",
                       label="Queimados",
                       icon="🔥")
        safe_page_link("pages/complexo_transplantes.py",
                       label="Transplantes",
                       icon="🫀")
        safe_page_link("pages/complexo_saude_mental.py",
                       label="Saúde Mental Especializada",
                       icon="🧩")
        safe_page_link("pages/complexo_reabilitacao.py",
                       label="Reabilitação",
                       icon="🦾")
        safe_page_link("pages/complexo_urg_emerg.py",
                       label="Urgência e Emergência",
                       icon="🚑")

with st.sidebar:
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Conecte-se")
    st.markdown("""
- 💼 [LinkedIn](https://www.linkedin.com/in/gregorio-healthdata/)
- ▶️ [YouTube](https://www.youtube.com/@Patients2Python)
- 📸 [Instagram](https://www.instagram.com/patients2python/)
- 🌐 [Site](https://patients2python.com.br/)
- 🐙 [GitHub](https://github.com/gregrodrigues22)
- 👥💬 [Comunidade](https://chat.whatsapp.com/CBn0GBRQie5B8aKppPigdd)
- 🤝💬 [WhatsApp](https://patients2python.sprinthub.site/r/whatsapp-olz)
- 🎓 [Escola](https://app.patients2python.com.br/browse)
    """, unsafe_allow_html=True)

# =========================
# Leitura de CSV (upload)
# =========================
@st.cache_data(show_spinner=False)
def _read_csv_smart(file, force_sep: str | None = None, dtype_map: dict | None = None) -> pd.DataFrame:
    """
    Lê CSV/TXT detectando separador quando possível.
    - force_sep: se informado, usa explicitamente (ex.: ';').
    - dtype_map: map de dtypes, ex.: {'id_pessoa':'string'}
    """
    dtype_map = dtype_map or {}
    if force_sep:
        return pd.read_csv(file, sep=force_sep, dtype=dtype_map)
    head = file.getvalue().splitlines()[0].decode("utf-8", errors="ignore")
    guess = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(io.BytesIO(file.getvalue()), sep=guess, dtype=dtype_map)

def schema_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "coluna": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "n_null": [df[c].isna().sum() for c in df.columns],
        "exemplo": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })

# =========================
# Área principal (Landing)
# =========================

st.subheader("🧭 Sobre este painel")
st.write(
    """
Este painel organiza a visão por **Complexos Produtivos em Saúde**, ajudando a enxergar
serviços especializados como partes de cadeias de valor clínico-assistenciais.

Use os cards abaixo para explorar cada complexo produtivo — por exemplo, 
**Oncologia**, **Cardiovascular**, **Ortopedia/Trauma**, **Urgência/Emergência** e outros.
Em cada um deles, você pode conectar serviços, habilitações, procedimentos e indicadores.
"""
)

# ---- componente de card com CTA ----
def card(title: str, desc: str, icon: str, page_path: str):
    with st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.caption(desc)

        page_file = (APP_DIR / page_path)

        # usamos SEMPRE st.button para manter o mesmo estilo visual
        clicked = st.button(
            f"Explorar {title}",
            icon=icon,
            key=f"btn_{page_path}",
            use_container_width=False,
        )

        if clicked:
            if page_file.exists():
                # navega para a página do complexo produtivo
                st.switch_page(page_path)
            else:
                st.warning("Página ainda não disponível para este complexo (em construção).")

# ---- definição dos complexos produtivos ----
complexos = [
    {
        "title": "Oncologia",
        "icon": "🎗️",
        "desc": "CACON/UNACON, radioterapia, quimioterapia e cirurgias oncológicas.",
        "page": "pages/complexo_oncologia.py",
    },
    {
        "title": "Cardiovascular",
        "icon": "❤️",
        "desc": "Hemodinâmica, cirurgias cardíacas, arritmias e UTI cardiológica.",
        "page": "pages/complexo_cardiovascular.py",
    },
    {
        "title": "Ortopedia e Traumatologia",
        "icon": "🦴",
        "desc": "Cirurgias de grande porte, próteses e reabilitação ortopédica.",
        "page": "pages/complexo_ortopedia_trauma.py",
    },
    {
        "title": "Obstetrícia e Neonatologia",
        "icon": "🤰",
        "desc": "Gestação de alto risco, UTI neonatal e cuidados perinatais.",
        "page": "pages/complexo_obstetricia_neonatologia.py",
    },
    {
        "title": "Neurologia e Neurocirurgia",
        "icon": "🧠",
        "desc": "Stroke, TCE, epilepsia e cirurgia funcional.",
        "page": "pages/complexo_neuro.py",
    },
    {
        "title": "Nefrologia e TRS",
        "icon": "🧪",
        "desc": "Hemodiálise, terapia renal substitutiva e transplante renal.",
        "page": "pages/complexo_nefrologia_trs.py",
    },
    {
        "title": "Queimados",
        "icon": "🔥",
        "desc": "Centros especializados em queimados e cuidados intensivos.",
        "page": "pages/complexo_queimados.py",
    },
    {
        "title": "Transplantes",
        "icon": "🫀",
        "desc": "Transplante de medula óssea e órgãos sólidos.",
        "page": "pages/complexo_transplantes.py",
    },
    {
        "title": "Saúde Mental Especializada",
        "icon": "🧩",
        "desc": "CAPS, internação psiquiátrica e reabilitação psicossocial.",
        "page": "pages/complexo_saude_mental.py",
    },
    {
        "title": "Reabilitação",
        "icon": "🦾",
        "desc": "CER, órteses/protóteses e reabilitação multiprofissional.",
        "page": "pages/complexo_reabilitacao.py",
    },
    {
        "title": "Urgência e Emergência",
        "icon": "🚑",
        "desc": "SAMU, portas de urgência e trauma.",
        "page": "pages/complexo_urg_emerg.py",
    },
]

# ---- layout dos cards em grade ----
cols = st.columns(3)
for i, comp in enumerate(complexos):
    col = cols[i % 3]
    with col:
        card(
            title=comp["title"],
            desc=comp["desc"],
            icon=comp["icon"],
            page_path=comp["page"],
        )

st.divider()
st.info(
    "Dica: use o menu lateral para navegar diretamente para um complexo produtivo específico. "
    "Cada complexo pode ter filtros, mapas de serviços e matrizes de indicadores próprios."
)