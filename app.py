# ---------------------------------------------------------------
# Set up
# ---------------------------------------------------------------
import io
import streamlit as st
from pathlib import Path
import pandas as pd

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

# ---------------- Esconder navegação padrão da sidebar ----------------
st.markdown(
    """
<style>
/* Esconde a lista padrão de páginas no topo da sidebar */
[data-testid="stSidebarNav"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# --- helper para evitar crash do st.page_link quando não é multipage ---
def safe_page_link(path: str, label: str, icon: str | None = None):
    """
    Cria link para página se o arquivo existir.
    Caso contrário, mostra botão desabilitado (em breve),
    sem quebrar o app.
    """
    full = APP_DIR / path
    try:
        if full.exists():
            st.page_link(path, label=label, icon=icon)
        else:
            st.button(label, icon=icon, disabled=True, help="Página não disponível neste app.")
    except Exception:
        st.button(label, icon=icon, disabled=True, help="Navegação multipage indisponível aqui.")


# ---------------- Sidebar (único) ----------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_column_width=True)

    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.subheader("Conecte-se")
    st.markdown(
        """
- 💼 [LinkedIn](https://www.linkedin.com/in/gregorio-healthdata/)
- ▶️ [YouTube](https://www.youtube.com/@Patients2Python)
- 📸 [Instagram](https://www.instagram.com/patients2python/)
- 🌐 [Site](https://patients2python.com.br/)
- 🐙 [GitHub](https://github.com/gregrodrigues22)
- 👥💬 [Comunidade](https://chat.whatsapp.com/CBn0GBRQie5B8aKppPigdd)
- 🤝💬 [WhatsApp](https://patients2python.sprinthub.site/r/whatsapp-olz)
- 🎓 [Escola](https://app.patients2python.com.br/browse)
        """,
        unsafe_allow_html=True,
    )

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

# =========================================================
# Abas customizadas (Introdução / Tutorial)
# =========================================================
def custom_tabs(tabs_list, default=0, cor="rgb(0,161,178)"):
    active_tab = st.radio("", tabs_list, index=default, horizontal=True)
    selected = tabs_list.index(active_tab) + 1

    st.markdown(
        f"""
        <style>
        div[role=radiogroup] {{
            border-bottom: 2px solid rgba(49, 51, 63, 0.1);
            flex-direction: row;
            gap: 2rem;
        }}
        div[role=radiogroup] > label > div:first-of-type {{
            display: none;
        }}
        div[role=radiogroup] label {{
            padding-bottom: 0.5em;
            border-radius: 0;
            position: relative;
            top: 3px;
            cursor: pointer;
        }}
        div[role=radiogroup] label p {{
            font-weight: 500;
        }}
        div[role=radiogroup] label:nth-child({selected}) {{
            border-bottom: 3px solid {cor};
        }}
        div[role=radiogroup] label:nth-child({selected}) p {{
            color: {cor};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    return active_tab


aba = custom_tabs(["📌 Introdução", "▶️ Tutorial", "📐 Metodologia"], cor="rgb(0,161,178)")

# =========================
# Helpers de CSV (mantidos)
# =========================
@st.cache_data(show_spinner=False)
def _read_csv_smart(file, force_sep: str | None = None, dtype_map: dict | None = None) -> pd.DataFrame:
    """
    Lê CSV/TXT detectando separador quando possível.
    - force_sep: se informado, usa explicitamente (ex.: ';').
    - dtype_map: map de dtypes, ex.: {'id_pessoa':'string'}
    """
    import io as _io

    dtype_map = dtype_map or {}
    if force_sep:
        return pd.read_csv(file, sep=force_sep, dtype=dtype_map)
    head = file.getvalue().splitlines()[0].decode("utf-8", errors="ignore")
    guess = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(_io.BytesIO(file.getvalue()), sep=guess, dtype=dtype_map)


def schema_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "coluna": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_null": [df[c].isna().sum() for c in df.columns],
            "exemplo": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
        }
    )

# =========================================================
# Definição dos complexos (usado na aba Introdução)
# =========================================================
def card(title: str, desc: str, icon: str, page_path: str):
    with st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.caption(desc)

        page_file = APP_DIR / page_path
        if page_file.exists():
            st.page_link(page_path, label=f"Explorar {title}", icon=icon)
        else:
            st.button(
                f"Explorar {title}",
                icon=icon,
                disabled=True,
                help="Página ainda não disponível para este complexo (em construção).",
            )


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

# =========================================================
# Conteúdo das abas
# =========================================================
if aba == "📌 Introdução":
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
        "Dica: você também pode usar o menu lateral para navegar diretamente para um complexo produtivo específico."
    )

elif aba == "▶️ Tutorial":
    st.subheader("▶️ Tutorial em vídeo")

    # Substitua pela URL real do seu vídeo:
    YOUTUBE_URL = "https://youtu.be/gaWaKHC9Yb4"

    st.markdown(
        """
Este vídeo mostra, passo a passo, como navegar pelos complexos produtivos,
usar os filtros e interpretar os principais elementos do painel.
        """
    )
    st.video(YOUTUBE_URL)

elif aba == "📐 Metodologia":
    st.subheader("📐 Metodologia")

    st.markdown(
        """
        ### **Metodologia dos Complexos Produtivos em Saúde**

        Este painel utiliza uma metodologia desenvolvida pelo NESCON/UFMG e pela Estação de Pesquisa
        de Sinais de Mercado (EPSM) para organizar serviços especializados dentro de **Complexos Produtivos
        em Saúde (CPS)**.

        A análise integra informações estruturais do CNES — como habilitações, serviços especializados, leitos,
        equipamentos e profissionais — com dados assistenciais do SIH/SUS e SIA/SUS, além de informações
        territoriais do IBGE.

        Cada complexo produtivo (como Oncologia, Cardiovascular, Obstetrícia & Neonatologia, Neurologia &
        Neurocirurgia, Urgência & Emergência, entre outros) é identificado a partir de **critérios técnicos
        específicos**, que incluem:

        - presença de serviços especializados essenciais;  
        - habilitações obrigatórias do CNES;  
        - equipamentos críticos;  
        - disponibilidade de equipe qualificada;  
        - estrutura mínima compatível com o nível de complexidade do complexo.

        A partir dessa classificação, o painel gera indicadores que ajudam a compreender a **distribuição
        territorial da capacidade instalada**, identificar **lacunas assistenciais** e apoiar o
        **planejamento da rede de atenção**.

        A metodologia completa pode ser baixada no link abaixo:
        """
    )

    # ============ PDF PARA DOWNLOAD ============
    pdf_path = "assets/arquivo_metodologia.pdf"  # coloque o caminho correto do seu PDF

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Baixar metodologia completa (PDF)",
            data=pdf_file,
            file_name="metodologia_complexos_produtivos.pdf",
            mime="application/pdf",
            use_container_width=True,
        )