from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
import plotly.graph_objects as go
from pathlib import Path

# Gráficos reutilizáveis
from src.graph import pareto_barh, bar_count
from src.graph import pie_standard, bar_yoy_trend
from src.graph import bar_total_por_grupo

# ==============================================================
# Config
# ==============================================================
try:
    st.set_page_config(layout="wide", page_title="📊 Painel Nescon EPSM")
except Exception:
    # set_page_config só pode ser chamado uma vez; se já foi no app principal, ignoramos aqui
    pass

PROJECT_ID = "escolap2p"

# ==============================================================
# BigQuery Client
# ==============================================================
def make_bq_client():
    return bigquery.Client(project=PROJECT_ID)

client = make_bq_client()

# ==============================================================
# Tabelas
# ==============================================================
TABLES = {
    "matriz": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_matriz_indicadores",
    "estabelecimentos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_estabelecimentos_applications",
    "servicos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_servicos_especializados_applications",
    "habilitacao": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_habilitacao_mart",
    "leitos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_leitos_applications",
    "equipamentos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_equipamentos_applications",
    "profissionais": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_profissionais_applications",
}

# ==============================================================
# FILTER SPEC por tabela (para abas genéricas)
# ==============================================================
FILTER_SPEC = {
    "equipamentos": {
        "equipamentos_ano": "year",
        "equipamentos_mes": "month",
        "equipamentos_sigla_uf": "str",
        "equipamentos_id_municipio": "str",
        "equipamentos_id_estabelecimento_cnes": "str",
        "ibge_no_municipio": "str",
        "ibge_no_regiao_saude": "int",
        "ibge_no_microrregiao": "str",
        "ibge_no_mesorregiao": "str",
        "ibge_no_uf": "str",
        "ibge_ivs": "str",
        "ibge_populacao_ibge_2021": "str",
        "equipamentos_id_equipamento": "str",
        "equipamentos_tipo_equipamento": "str",
        "estabelecimentos_nome_fantasia": "str",
        "estabelecimentos_tipo_novo_do_estabelecimento": "str",
        "estabelecimentos_tipo_do_estabelecimento": "str",
        "estabelecimentos_subtipo_do_estabelecimento": "str",
        "estabelecimentos_gestao": "str",
        "estabelecimentos_status_do_estabelecimento": "str",
        "estabelecimentos_convenio_sus": "str",
        "estabelecimentos_categoria_natureza_juridica": "str",
        "onco_cacon": "bool",
        "onco_unacon": "bool",
        "onco_radioterapia": "bool",
        "onco_quimioterapia": "bool",
        "habilitacao_agrupado_onco_cirurgica": "bool",
        "habilitacao_agrupado_transplantes_bancos": "bool",
        "habilitacao_agrupado_uti_adulto": "bool",
        "habilitacao_agrupado_uti_pediatrica": "bool",
        "habilitacao_agrupado_uti_neonatal": "bool",
        "habilitacao_agrupado_uti_coronariana": "bool",
        "habilitacao_agrupado_ucin": "bool",
        "habilitacao_agrupado_uti_queimados": "bool",
        "habilitacao_agrupado_saude_mental_caps_psiq": "bool",
        "habilitacao_agrupado_reabilitacao_cer": "bool",
        "habilitacao_agrupado_cardio_alta_complex": "bool",
        "habilitacao_agrupado_nutricao": "bool",
        "habilitacao_agrupado_odontologia_ceo": "bool",
    },
    "estabelecimentos": {
        "competencia": "str",
        "uf": "str",
        "codigo_do_municipio": "int",
        "municipio": "str",
        "cnes": "int",
        "nome_fantasia": "str",
        "tipo_novo_do_estabelecimento": "str",
        "tipo_do_estabelecimento": "str",
        "subtipo_do_estabelecimento": "str",
        "gestao": "str",
        "convenio_sus": "str",
        "categoria_natureza_juridica": "str",
        "status_do_estabelecimento": "str",
        "cod_ibge": "str",
        "cod_ibge_7": "str",
        "cod_regiao_saude": "str",
        "no_regiao_saude": "int",
        "no_municipio": "str",
        "cod_uf": "int",
        "sgl_uf": "str",
        "no_uf": "str",
        "cod_regiao": "int",
        "sgl_regiao": "str",
        "no_regiao": "str",
        "latitude": "str",
        "longitude": "str",
        "cod_mesorregiao": "int",
        "no_mesorregiao": "str",
        "cod_microrregiao": "int",
        "no_microrregiao": "str",
        "no_rm_ride_au": "str",
        "cod_rm_ride_au": "int",
        "ivs": "str",
        "cnes_1": "int",
        "onco_cacon": "bool",
        "onco_unacon": "bool",
        "onco_radioterapia": "bool",
        "onco_quimioterapia": "bool",
        "habilitacao_agrupado_onco_cirurgica": "bool",
        "habilitacao_agrupado_transplantes_bancos": "bool",
        "habilitacao_agrupado_uti_adulto": "bool",
        "habilitacao_agrupado_uti_pediatrica": "bool",
        "habilitacao_agrupado_uti_neonatal": "bool",
        "habilitacao_agrupado_uti_coronariana": "bool",
        "habilitacao_agrupado_ucin": "bool",
        "habilitacao_agrupado_uti_queimados": "bool",
        "habilitacao_agrupado_saude_mental_caps_psiq": "bool",
        "habilitacao_agrupado_reabilitacao_cer": "bool",
        "habilitacao_agrupado_cardio_alta_complex": "bool",
        "habilitacao_agrupado_nutricao": "bool",
        "habilitacao_agrupado_odontologia_ceo": "bool",
    },
    "habilitacao": {
        "habilitacao_ano": "year",
        "habilitacao_mes": "month",
        "habilitacao_sigla_uf": "str",
        "habilitacao_id_municipio": "str",
        "habilitacao_id_estabelecimento_cnes": "str",
        "ibge_no_municipio": "str",
        "ibge_no_regiao_saude": "int",
        "ibge_no_microrregiao": "str",
        "ibge_no_mesorregiao": "str",
        "ibge_no_uf": "str",
        "ibge_ivs": "str",
        "ibge_populacao_ibge_2021": "str",
        "habilitacao_quantidade_leitos": "int",
        "habilitacao_ano_competencia_inicial": "year",
        "habilitacao_mes_competencia_inicial": "month",
        "habilitacao_ano_competencia_final": "year",
        "habilitacao_mes_competencia_final": "month",
        "habilitacao_tipo_habilitacao": "str",
        "habilitacao_nivel_habilitacao": "str",
        "habilitacao_nivel_habilitacao_tipo": "str",
        "habilitacao_portaria": "str",
        "habilitacao_ano_portaria": "year",
        "habilitacao_mes_portaria": "month",
        "estabelecimentos_cnes": "int",
        "estabelecimentos_nome_fantasia": "str",
        "estabelecimentos_tipo_novo_do_estabelecimento": "str",
        "estabelecimentos_tipo_do_estabelecimento": "str",
        "estabelecimentos_subtipo_do_estabelecimento": "str",
        "estabelecimentos_gestao": "str",
        "estabelecimentos_status_do_estabelecimento": "str",
        "estabelecimentos_convenio_sus": "str",
        "estabelecimentos_categoria_natureza_juridica": "str",
        "referencia_habilitacao_co_habilitacao": "str",
        "referencia_habilitacao_no_habilitacao": "str",
        "referencia_habilitacao_no_categoria": "str",
        "referencia_habilitacao_ds_tag": "str",
    },
    "leitos": {
        "leitos_ano": "year",
        "leitos_mes": "month",
        "leitos_sigla_uf": "str",
        "leitos_id_municipio": "int",
        "leitos_id_estabelecimento_cnes": "str",
        "ibge_no_municipio": "str",
        "ibge_no_regiao_saude": "int",
        "ibge_no_microrregiao": "str",
        "ibge_no_mesorregiao": "str",
        "ibge_no_uf": "str",
        "ibge_ivs": "str",
        "ibge_populacao_ibge_2021": "str",
        "leitos_tipo_especialidade_leito": "str",
        "leitos_tipo_leito": "str",
        "leitos_tipo_leito_nome": "str",
        "estabelecimentos_cnes": "int",
        "estabelecimentos_nome_fantasia": "str",
        "estabelecimentos_tipo_novo_do_estabelecimento": "str",
        "estabelecimentos_tipo_do_estabelecimento": "str",
        "estabelecimentos_subtipo_do_estabelecimento": "str",
        "estabelecimentos_gestao": "str",
        "estabelecimentos_status_do_estabelecimento": "str",
        "estabelecimentos_convenio_sus": "str",
        "estabelecimentos_categoria_natureza_juridica": "str",
        "referencia_especialidade_codleito": "int",
        "referencia_especialidade_no_leito": "str",
        "cnes": "int",
        "onco_cacon": "bool",
        "onco_unacon": "bool",
        "onco_radioterapia": "bool",
        "onco_quimioterapia": "bool",
        "habilitacao_agrupado_onco_cirurgica": "bool",
        "habilitacao_agrupado_transplantes_bancos": "bool",
        "habilitacao_agrupado_uti_adulto": "bool",
        "habilitacao_agrupado_uti_pediatrica": "bool",
        "habilitacao_agrupado_uti_neonatal": "bool",
        "habilitacao_agrupado_uti_coronariana": "bool",
        "habilitacao_agrupado_ucin": "bool",
        "habilitacao_agrupado_uti_queimados": "bool",
        "habilitacao_agrupado_saude_mental_caps_psiq": "bool",
        "habilitacao_agrupado_reabilitacao_cer": "bool",
        "habilitacao_agrupado_cardio_alta_complex": "bool",
        "habilitacao_agrupado_nutricao": "bool",
        "habilitacao_agrupado_odontologia_ceo": "bool",
    },
    "profissionais": {
        "profissionais_ano": "year",
        "profissionais_mes": "month",
        "profissionais_sigla_uf": "str",
        "profissionais_id_municipio": "str",
        "profissionais_id_estabelecimento_cnes": "str",
        "ibge_no_municipio": "str",
        "ibge_no_regiao_saude": "int",
        "ibge_no_microrregiao": "str",
        "ibge_no_mesorregiao": "str",
        "ibge_no_uf": "str",
        "ibge_ivs": "str",
        "ibge_populacao_ibge_2021": "str",
        "profissionais_nome": "str",
        "profissionais_tipo_vinculo": "str",
        "profissionais_id_registro_conselho": "str",
        "profissionais_tipo_conselho": "str",
        "profissionais_cartao_nacional_saude": "str",
        "profissionais_cbo_2002": "str",
        "cbo_ocupacao": "str",
        "cbo_descricao": "str",
        "cbo_saude": "str",
        "cbo_regulamentacao": "str",
        "profissionais_indicador_estabelecimento_terceiro": "int",
        "profissionais_indicador_vinculo_contratado_sus": "int",
        "profissionais_indicador_vinculo_autonomo_sus": "int",
        "profissionais_indicador_vinculo_outros": "int",
        "profissionais_indicador_atende_sus": "int",
        "profissionais_indicador_atende_nao_sus": "int",
        "profissionais_carga_horaria_outros": "int",
        "profissionais_carga_horaria_hospitalar": "int",
        "profissionais_carga_horaria_ambulatorial": "int",
        "estabelecimentos_cnes": "int",
        "estabelecimentos_nome_fantasia": "str",
        "estabelecimentos_tipo_novo_do_estabelecimento": "str",
        "estabelecimentos_tipo_do_estabelecimento": "str",
        "estabelecimentos_subtipo_do_estabelecimento": "str",
        "estabelecimentos_gestao": "str",
        "estabelecimentos_status_do_estabelecimento": "str",
        "estabelecimentos_convenio_sus": "str",
        "estabelecimentos_categoria_natureza_juridica": "str",
        "profissionais_tipo_cbo": "str",
        "profissionais_tipo_ras": "str",
        "profissionais_tipo_especialidade": "str",
        "profissionais_tipo_grupo": "str",
        "profissional_vinculo_forma_contratacao_descricao": "str",
        "profissional_vinculo_forma_contratacao_empregador_descricao": "str",
        "profissional_vinculo_forma_contratacao_empregador_detalhamento_descricao": "str",
        "profissional_vinculo_forma_contratacao_empregador_final_codigo": "str",
        "profissional_vinculo_forma_contratacao_empregador_final_descricao": "str",
        "cnes": "int",
        "onco_cacon": "bool",
        "onco_unacon": "bool",
        "onco_radioterapia": "bool",
        "onco_quimioterapia": "bool",
        "habilitacao_agrupado_onco_cirurgica": "bool",
        "habilitacao_agrupado_transplantes_bancos": "bool",
        "habilitacao_agrupado_uti_adulto": "bool",
        "habilitacao_agrupado_uti_pediatrica": "bool",
        "habilitacao_agrupado_uti_neonatal": "bool",
        "habilitacao_agrupado_uti_coronariana": "bool",
        "habilitacao_agrupado_ucin": "bool",
        "habilitacao_agrupado_uti_queimados": "bool",
        "habilitacao_agrupado_saude_mental_caps_psiq": "bool",
        "habilitacao_agrupado_reabilitacao_cer": "bool",
        "habilitacao_agrupado_cardio_alta_complex": "bool",
        "habilitacao_agrupado_nutricao": "bool",
        "habilitacao_agrupado_odontologia_ceo": "bool",
    },
    "servicos": {
        "competencia": "str",
        "uf": "str",
        "codigo_do_municipio": "int",
        "municipio": "str",
        "cnes": "int",
        "nome_fantasia": "str",
        "tipo_novo_do_estabelecimento": "str",
        "tipo_do_estabelecimento": "str",
        "subtipo_do_estabelecimento": "str",
        "gestao": "str",
        "convenio_sus": "str",
        "categoria_natureza_juridica": "str",
        "status_do_estabelecimento": "str",
        "servico": "str",
        "servico_classificacao": "str",
        "servico_ambulatorial_sus": "str",
        "servico_ambulatorial_nao_sus": "str",
        "servico_hospitalar_sus": "str",
        "servico_hospitalar_nao_sus": "str",
        "servico_terceiro": "str",
        "cod_ibge": "str",
        "cod_ibge_7": "str",
        "cod_regiao_saude": "str",
        "no_regiao_saude": "int",
        "no_municipio": "str",
        "cod_uf": "int",
        "sgl_uf": "str",
        "no_uf": "str",
        "cod_regiao": "int",
        "sgl_regiao": "str",
        "no_regiao": "str",
        "cod_mesorregiao": "int",
        "no_mesorregiao": "str",
        "cod_microrregiao": "int",
        "no_microrregiao": "str",
        "no_rm_ride_au": "str",
        "cod_rm_ride_au": "int",
        "populacao_ibge_2013": "int",
        "populacao_ibge_2017": "int",
        "populacao_ibge_2021": "str",
        "ivs": "str",
        "onco_cacon": "bool",
        "onco_unacon": "bool",
        "onco_radioterapia": "bool",
        "onco_quimioterapia": "bool",
        "habilitacao_agrupado_onco_cirurgica": "bool",
        "habilitacao_agrupado_transplantes_bancos": "bool",
        "habilitacao_agrupado_uti_adulto": "bool",
        "habilitacao_agrupado_uti_pediatrica": "bool",
        "habilitacao_agrupado_uti_neonatal": "bool",
        "habilitacao_agrupado_uti_coronariana": "bool",
        "habilitacao_agrupado_ucin": "bool",
        "habilitacao_agrupado_uti_queimados": "bool",
        "habilitacao_agrupado_saude_mental_caps_psiq": "bool",
        "habilitacao_agrupado_reabilitacao_cer": "bool",
        "habilitacao_agrupado_cardio_alta_complex": "bool",
        "habilitacao_agrupado_nutricao": "bool",
        "habilitacao_agrupado_odontologia_ceo": "bool",
    },
}

# ==============================================================
# Cache de leitura
# ==============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_table(table_fqn: str) -> pd.DataFrame:
    return client.query(f"SELECT * FROM `{table_fqn}`").to_dataframe()

# ==============================================================
# Helpers de filtro genérico
# ==============================================================
def _opts(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    return sorted(s.unique().tolist())

def ui_range_numeric(df: pd.DataFrame, col: str, key: str, label: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    vmin, vmax = int(s.min()), int(s.max())
    if vmin == vmax:
        st.caption(f"{label}: {vmin} (apenas este valor disponível)")
        return (vmin, vmax)
    lo, hi = st.slider(label, min_value=vmin, max_value=vmax, value=(vmin, vmax), key=key)
    return (lo, hi)

def ui_multiselect(df: pd.DataFrame, col: str, key: str, label: str):
    return st.multiselect(label, options=_opts(df, col), key=key)

def ui_bool(df: pd.DataFrame, col: str, key: str, label: str):
    choice = st.select_slider(label, options=["(todos)", "True", "False"], value="(todos)", key=key)
    if choice == "(todos)":
        return None
    return True if choice == "True" else False

def apply_filters(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    dff = df.copy()
    for col, sel in selections.items():
        if sel is None:
            continue
        if isinstance(sel, tuple) and len(sel) == 2 and all(isinstance(v, (int, float)) for v in sel):
            lo, hi = sel
            dff = dff[pd.to_numeric(dff[col], errors="coerce").between(lo, hi)]
        elif isinstance(sel, list):
            if len(sel):
                dff = dff[dff[col].isin(sel)]
        elif isinstance(sel, bool):
            mask = (dff[col].astype("boolean") == sel).fillna(False)
            dff = dff[mask]
    return dff

def _draw_multis(df: pd.DataFrame, cols_list, selections, prefix, label_prefix):
    if not cols_list:
        return
    n = min(4, max(1, len(cols_list)))
    rows = (len(cols_list) + n - 1) // n
    idx = 0
    for _ in range(rows):
        cols = st.columns(n)
        for c in range(n):
            if idx >= len(cols_list):
                break
            colname = cols_list[idx]
            with cols[c]:
                selections[colname] = ui_multiselect(df, colname, f"{prefix}_{colname}", f"Filtrar {colname}")
            idx += 1

def render_filters(df: pd.DataFrame, spec: dict, prefix: str) -> pd.DataFrame:
    years  = [c for c, k in spec.items() if k == "year"  and c in df]
    months = [c for c, k in spec.items() if k == "month" and c in df]
    strs   = [c for c, k in spec.items() if k == "str"   and c in df]
    ints   = [c for c, k in spec.items() if k == "int"   and c in df]
    bools  = [c for c, k in spec.items() if k == "bool"  and c in df]

    selections: dict[str, object] = {}

    # Agrupados em expanders (funciona bem na sidebar)
    if years or months:
        with st.expander("⏱️ Período", expanded=True):
            cols = st.columns(max(1, len(years) + len(months)))
            i = 0
            for col in years:
                with cols[i]:
                    selections[col] = ui_range_numeric(df, col, f"{prefix}_{col}", f"Ano — {col}")
                i += 1
            for col in months:
                with cols[i]:
                    selections[col] = ui_range_numeric(df, col, f"{prefix}_{col}", f"Mês — {col}")
                i += 1

    with st.expander("🧭 Dimensões", expanded=False):
        _draw_multis(df, strs, selections, prefix, "Filtrar")
        _draw_multis(df, ints, selections, prefix, "Filtrar")

    if bools:
        with st.expander("✅ Filtros booleanos", expanded=False):
            cols = st.columns(min(4, len(bools)))
            for i, col in enumerate(bools):
                with cols[i % len(cols)]:
                    selections[col] = ui_bool(df, col, f"{prefix}_{col}", col)

    return apply_filters(df, selections)

def render_with_spinner_once(name: str, df: pd.DataFrame, spec: dict, prefix: str) -> pd.DataFrame:
    key = f"__spinner_done_{name}"
    if not st.session_state.get(key):
        with st.spinner("⏳ Carregando filtros..."):
            out = render_filters(df, spec, prefix)
        st.session_state[key] = True
        return out
    return render_filters(df, spec, prefix)

# Detecta se um valor mudou desde a última execução
def did_filters_change(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value
        return True
    if st.session_state[key] != value:
        st.session_state[key] = value
        return True
    return False

# Spinner "nulo" para quando não queremos mostrar nada
class DummySpinner:
    def __enter__(self): pass
    def __exit__(self, *args): pass

# Diretório raiz do app (onde está o app.py)
ROOT_DIR = Path(__file__).resolve().parent.parent

def safe_page_link(path: str, label: str, icon: str | None = None):
    full = ROOT_DIR / path
    try:
        if full.exists():
            st.page_link(path, label=label, icon=icon)
        else:
            st.button(label, icon=icon, disabled=True, help="Página em breve.")
    except Exception:
        st.button(label, icon=icon, disabled=True, help="Navegação multipage indisponível aqui.")

def fmt_num(n: float | int, decimals: int = 0) -> str:
    if n is None:
        return "-"
    if decimals == 0:
        return f"{int(round(n)):,}".replace(",", ".")
    s = f"{n:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

# =====================================================================
# Tabs customizadas (estilo aba, sem cara de radio)
# =====================================================================
def custom_tabs(tabs_list, default=0, cor="rgb(0,161,178)"):
    active_tab = st.radio("", tabs_list, index=default, horizontal=True)
    selected = tabs_list.index(active_tab) + 1

    st.markdown(
        f"""
        <style>
        div[role=radiogroup] {{
            border-bottom: 2px solid rgba(49, 51, 63, 0.1);
            flex-direction: row;
            gap: 0.8rem;              /* <<< estava 2rem */
        }}
        div[role=radiogroup] label {{
            padding: 0 0.6rem 0.5rem;  /* um pouco mais compacto */
            border-radius: 0;
            position: relative;
            top: 3px;
            cursor: pointer;
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

# ==============================================================
# Sidebar – navegação fixa (logo + menu)
# ==============================================================
st.markdown(
    "<style>[data-testid='stSidebarNav']{display:none;}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("assets/logo.png", use_column_width=True)

with st.sidebar:
    st.page_link("app.py", label="Voltar ao Menu Principal", icon="🏠")

# ==============================================================
# Cabeçalho
# ==============================================================
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Análise do Complexo Produtivo da Saúde Oncológica</h1>
        <p style='color: white;'>Explore os dados do Complexo Produtivo Oncológico para tomada de decisões.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Abas principais (agora usando custom_tabs)
# =====================================================================

tabs_labels = [
    "🧮 Indicadores",
    "🗂️ Estabelecimentos",
    "🗂️ Serviços",
    "✅ Habilitação",
    "🛏️ Leitos",
    "🧰 Equipamentos",
    "👩‍⚕️ Profissionais",
    "📋 Registros"
]

aba = custom_tabs(tabs_labels, cor="rgb(0,161,178)")

# =====================================================================
# Aba Principal
# =====================================================================

if aba == "🏠 Menu Principal":
    try:
        import streamlit as st
        st.switch_page("app.py")
    except Exception:
        st.info("Abra o arquivo `app.py` na barra lateral para voltar ao menu principal.")
    st.stop()

# =====================================================================
# 1) Matriz de Indicadores
# =====================================================================
if aba == "🧮 Indicadores":
    st.subheader("🧮 Matriz de Indicadores")

    df_matriz = load_table(TABLES["matriz"])

    # ----------------- Filtros na sidebar -----------------
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Matriz de Indicadores")
        st.caption("Use os agrupadores abaixo para refinar os resultados.")

        telas = sorted(df_matriz.get("tela", pd.Series(dtype=str)).dropna().unique().tolist())
        secoes = sorted(df_matriz.get("secao_tela", pd.Series(dtype=str)).dropna().unique().tolist())
        fontes = sorted(df_matriz.get("fonte_dados", pd.Series(dtype=str)).dropna().unique().tolist())
        tipos = sorted(df_matriz.get("tipo_indicador", pd.Series(dtype=str)).dropna().unique().tolist())

        with st.expander("Filtros principais", expanded=False):
            c1, c2, c3 = st.columns(3)
            tela_sel = st.multiselect(
                "Tela", options=telas, key="mat_telas",
                placeholder="(Todos. Filtros opcionais)",
            )
            secao_sel = st.multiselect(
                "Seção da tela", options=secoes, key="mat_secoes",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_sel = st.multiselect(
                "Tipo de indicador", options=tipos, key="mat_tipos",
                placeholder="(Todos. Filtros opcionais)",
            )
            fonte_sel = st.multiselect(
                "Fonte de dados", options=fontes, key="mat_fontes",
                placeholder="(Todos. Filtros opcionais)",
            )

    dfm = df_matriz.copy()
    if "tela" in dfm and tela_sel:
        dfm = dfm[dfm["tela"].isin(tela_sel)]
    if "secao_tela" in dfm and secao_sel:
        dfm = dfm[dfm["secao_tela"].isin(secao_sel)]
    if "tipo_indicador" in dfm and tipo_sel:
        dfm = dfm[dfm["tipo_indicador"].isin(tipo_sel)]
    if "fonte_dados" in dfm and fonte_sel:
        dfm = dfm[dfm["fonte_dados"].isin(fonte_sel)]

    st.info("📏 Grandes números: Visão rápida com filtros aplicados")
    c1, c2, c3 = st.columns(3)
    c1.metric("Indicadores", f"{len(dfm):,}".replace(",", "."))
    c2.metric(
        "Telas distintas",
        int(dfm["tela"].nunique() if "tela" in dfm else 0),
    )
    c3.metric(
        "Tipos de indicador",
        int(dfm["tipo_indicador"].nunique() if "tipo_indicador" in dfm else 0),
    )

    st.info("📊 Gráficos: Visuais para responder às perguntas segundo os filtros aplicados")

    if "tipo_indicador" in dfm and dfm["tipo_indicador"].notna().any():
        with st.expander("Quais tipos de indicadores serão analisados?", expanded=True):
            st.plotly_chart(
                bar_count(dfm, "tipo_indicador", "Distribuição por tipo de indicador"),
                use_container_width=True,
            )

    if "fonte_dados" in dfm and dfm["fonte_dados"].notna().any():
        with st.expander("Quais fontes de dados dos indicadores analisados?", expanded=True):
            st.plotly_chart(
                bar_count(dfm, "fonte_dados", "Distribuição por fonte de dados"),
                use_container_width=True,
            )

    st.info("📋 Tabelas: Veja detalhes da Ficha de cada indicador")

    tem_titulo = "titulo_indicador" in dfm and dfm["titulo_indicador"].notna().any()

    if tem_titulo:
        titulos = (
            dfm["titulo_indicador"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        sel = st.selectbox("Escolha o indicador", options=titulos, key="mat_sel_titulo")

        if sel:
            ordem_campos = [
                "tela",
                "secao_tela",
                "tipo_indicador",
                "titulo_indicador",
                "definicao_indicador",
                "proposito_indicador",
                "interpretacao",
                "utilizacao",
                "metodo_calculo",
                "unidade_medida",
                "periodo_referencia",
                "filtros",
                "fonte_dados",
                "limitacoes",
            ]
            campos = [c for c in ordem_campos if c in dfm.columns]

            linha = dfm[dfm["titulo_indicador"] == sel].iloc[0][campos].dropna()

            label_map = {
                "tela": "Tela",
                "secao_tela": "Seção da tela",
                "tipo_indicador": "Tipo de indicador",
                "titulo_indicador": "Título do indicador",
                "definicao_indicador": "Definição do indicador",
                "proposito_indicador": "Propósito do indicador",
                "interpretacao": "Interpretação",
                "utilizacao": "Utilização",
                "metodo_calculo": "Método de cálculo",
                "unidade_medida": "Unidade de medida",
                "periodo_referencia": "Período de referência",
                "filtros": "Filtros recomendados",
                "fonte_dados": "Fonte de dados",
                "limitacoes": "Limitações",
            }

            labels = [label_map.get(c, c) for c in linha.index]
            valores = linha.values.tolist()
            col_labels = ["<b>Campo</b>", "<b>Descrição</b>"]

            fig_tbl = go.Figure(
                data=[
                    go.Table(
                        columnwidth=[20, 80],
                        header=dict(
                            values=col_labels,
                            fill_color="#E8F0FF",
                            align="left",
                            font=dict(size=14, color="black"),
                            height=32,
                        ),
                        cells=dict(
                            values=[labels, valores],
                            align="left",
                            height=30,
                            fill_color="#FAFAFA",
                            font=dict(size=13),
                        ),
                    )
                ]
            )

            fig_tbl.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=min(900, 32 * (len(labels) + 2)),
            )

            st.plotly_chart(fig_tbl, use_container_width=True)
    else:
        st.info("Nenhum título de indicador disponível para detalhamento.")

# =====================================================================
# 2) Cadastro Estabelecimentos
# =====================================================================
elif aba == "🗂️ Estabelecimentos":
    st.subheader("🗂️ Estabelecimentos")

    # Carregar base detalhada (para filtros + tabela)
    # ---------------------------------------------------------
    with st.spinner("⏳ Carregando base de estabelecimentos..."):
        df_est = load_table(TABLES["estabelecimentos"]).copy()

    estab_table_id = TABLES["estabelecimentos"]

    # ---------------------------------------------------------
    # Helpers para BigQuery (agregações)
    # ---------------------------------------------------------
    from google.cloud import bigquery

    @st.cache_resource
    def get_bq_client():
        return bigquery.Client()

    def build_where_estab(
        comp_sel,
        reg_sel,
        uf_sel,
        meso_sel,
        micro_sel,
        reg_saude_sel,
        mun_sel,
        ivs_sel,
        tipo_sel,
        subtipo_sel,
        gestao_sel,
        convenio_sel,
        natureza_sel,
        status_sel,
        onco_cacon_sel,
        onco_unacon_sel,
        onco_radio_sel,
        onco_quimio_sel,
        hab_onco_cir_sel,
    ):
        """
        Monta WHERE dinâmico + parâmetros BigQuery
        para todos os filtros da aba Estabelecimentos.
        """
        clauses = ["1=1"]
        params: list[bigquery.QueryParameter] = []

        def add_in_array(col, param_name, values):
            if values:
                vals = [str(v) for v in values]
                clauses.append(f"CAST({col} AS STRING) IN UNNEST(@{param_name})")
                params.append(bigquery.ArrayQueryParameter(param_name, "STRING", vals))

        # Filtros categóricos
        add_in_array("competencia", "comp", comp_sel)
        add_in_array("no_regiao", "reg", reg_sel)
        add_in_array("no_uf", "uf", uf_sel)
        add_in_array("no_mesorregiao", "meso", meso_sel)
        add_in_array("no_microrregiao", "micro", micro_sel)
        add_in_array("cod_regiao_saude", "regsaude", reg_saude_sel)
        add_in_array("municipio", "mun", mun_sel)
        add_in_array("ivs", "ivs", ivs_sel)

        add_in_array("tipo_do_estabelecimento", "tipo", tipo_sel)
        add_in_array("subtipo_do_estabelecimento", "subtipo", subtipo_sel)
        add_in_array("gestao", "gestao", gestao_sel)
        add_in_array("convenio_sus", "convenio", convenio_sel)
        add_in_array("categoria_natureza_juridica", "natjur", natureza_sel)
        add_in_array("status_do_estabelecimento", "status", status_sel)

        # Booleanos oncológicos
        def add_bool(col, param_name, sel):
            if not sel:
                return
            vals = []
            if "Sim" in sel:
                vals.append(True)
            if "Não" in sel:
                vals.append(False)
            if vals:
                clauses.append(f"{col} IN UNNEST(@{param_name})")
                params.append(bigquery.ArrayQueryParameter(param_name, "BOOL", vals))

        add_bool("onco_cacon", "cacon", onco_cacon_sel)
        add_bool("onco_unacon", "unacon", onco_unacon_sel)
        add_bool("onco_radioterapia", "radio", onco_radio_sel)
        add_bool("onco_quimioterapia", "quimio", onco_quimio_sel)
        add_bool("habilitacao_agrupado_onco_cirurgica", "oncocir", hab_onco_cir_sel)

        where_sql = "WHERE " + " AND ".join(clauses)
        return where_sql, params

    def query_estab_kpis(where_sql: str, params, table_id: str):
        """
        KPIs direto do BQ:
          - total de estabelecimentos distintos
          - média de estabelecimentos por UF
          - média de estabelecimentos por Região de Saúde
          - estabelecimentos com CACON / UNACON / Radio / Quimio / Onco Cirúrgica
        """
        client = get_bq_client()
        sql = f"""
        WITH base AS (
          SELECT
            cnes,
            no_uf,
            cod_regiao_saude,
            SAFE_CAST(onco_cacon AS BOOL) AS onco_cacon,
            SAFE_CAST(onco_unacon AS BOOL) AS onco_unacon,
            SAFE_CAST(onco_radioterapia AS BOOL) AS onco_radioterapia,
            SAFE_CAST(onco_quimioterapia AS BOOL) AS onco_quimioterapia,
            SAFE_CAST(habilitacao_agrupado_onco_cirurgica AS BOOL)
              AS habilitacao_agrupado_onco_cirurgica
          FROM `{table_id}`
          {where_sql}
          GROUP BY
            cnes,
            no_uf,
            cod_regiao_saude,
            onco_cacon,
            onco_unacon,
            onco_radioterapia,
            onco_quimioterapia,
            habilitacao_agrupado_onco_cirurgica
        ),
        uf_agg AS (
          SELECT no_uf, COUNT(DISTINCT cnes) AS est_por_uf
          FROM base
          GROUP BY no_uf
        ),
        reg_agg AS (
          SELECT cod_regiao_saude, COUNT(DISTINCT cnes) AS est_por_regiao
          FROM base
          GROUP BY cod_regiao_saude
        )
        SELECT
          (SELECT COUNT(DISTINCT cnes) FROM base) AS tot_est,
          (SELECT AVG(est_por_uf) FROM uf_agg) AS media_por_uf,
          (SELECT AVG(est_por_regiao) FROM reg_agg) AS media_por_regiao,
          (SELECT COUNT(DISTINCT cnes) FROM base WHERE onco_cacon IS TRUE) AS n_cacon,
          (SELECT COUNT(DISTINCT cnes) FROM base WHERE onco_unacon IS TRUE) AS n_unacon,
          (SELECT COUNT(DISTINCT cnes) FROM base WHERE onco_radioterapia IS TRUE) AS n_radio,
          (SELECT COUNT(DISTINCT cnes) FROM base WHERE onco_quimioterapia IS TRUE) AS n_quimio,
          (SELECT COUNT(DISTINCT cnes)
             FROM base
            WHERE habilitacao_agrupado_onco_cirurgica IS TRUE) AS n_cir;
        """
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=params),
        )
        df = job.to_dataframe()
        return df.iloc[0]

    def query_estab_group(
        where_sql: str,
        params,
        table_id: str,
        col: str,
        null_label: str = "(não informado)",
        order_desc: bool = True,
    ) -> pd.DataFrame:
        """
        Agregado genérico: conta estabelecimentos distintos por coluna categórica.
        Retorna df com col (categ) e qtde.
        """
        client = get_bq_client()
        order_dir = "DESC" if order_desc else "ASC"
        sql = f"""
        SELECT
          COALESCE(CAST({col} AS STRING), @null_label) AS categoria,
          COUNT(DISTINCT cnes) AS qtde
        FROM `{table_id}`
        {where_sql}
        GROUP BY categoria
        ORDER BY qtde {order_dir}
        """
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=list(params) + [
                    bigquery.ScalarQueryParameter("null_label", "STRING", null_label)
                ]
            ),
        )
        return job.to_dataframe()

    # ---------------------------------------------------------
    # Filtros na sidebar
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Estabelecimentos")
        st.caption("Use os agrupadores abaixo para refinar o cadastro.")

        # Filtros de período
        with st.expander("Filtros de período", expanded=False):
            comp_sel = st.multiselect(
                "Competência",
                _opts(df_est, "competencia"),
                key="est_comp",
                placeholder="(Todos. Filtros opcionais)",
            )

        # Filtros de Território
        with st.expander("Filtros de Território", expanded=False):
            c1, c2, c3 = st.columns(3)
            reg_sel  = st.multiselect(
                "Região",
                _opts(df_est, "no_regiao"),
                key="est_reg",
                placeholder="(Todos. Filtros opcionais)",
            )
            uf_sel   = st.multiselect(
                "UF",
                _opts(df_est, "no_uf"),
                key="est_uf",
                placeholder="(Todos. Filtros opcionais)",
            )
            meso_sel = st.multiselect(
                "Mesorregião Geográfica",
                _opts(df_est, "no_mesorregiao"),
                key="est_meso",
                placeholder="(Todos. Filtros opcionais)",
            )
            micro_sel = st.multiselect(
                "Microrregião Geográfica",
                _opts(df_est, "no_microrregiao"),
                key="est_micro",
                placeholder="(Todos. Filtros opcionais)",
            )
            reg_saude_sel = st.multiselect(
                "Região de Saúde",
                _opts(df_est, "cod_regiao_saude"),
                key="est_regsaude",
                placeholder="(Todos. Filtros opcionais)",
            )
            mun_sel = st.multiselect(
                "Município",
                _opts(df_est, "municipio"),
                key="est_mun",
                placeholder="(Todos. Filtros opcionais)",
            )
            ivs_sel = st.multiselect(
                "Município IVS",
                _opts(df_est, "ivs"),
                key="est_ivs",
                placeholder="(Todos. Filtros opcionais)",
            )

        # Perfil do estabelecimento
        with st.expander("Filtros de Perfil do Estabelecimento", expanded=False):
            tipo_sel    = st.multiselect(
                "Tipo",
                _opts(df_est, "tipo_do_estabelecimento"),
                key="est_tipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            subtipo_sel = st.multiselect(
                "Subtipo",
                _opts(df_est, "subtipo_do_estabelecimento"),
                key="est_subtipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            gestao_sel  = st.multiselect(
                "Gestão",
                _opts(df_est, "gestao"),
                key="est_gestao",
                placeholder="(Todos. Filtros opcionais)",
            )
            convenio_sel = st.multiselect(
                "Convênio SUS",
                _opts(df_est, "convenio_sus"),
                key="est_convenio",
                placeholder="(Todos. Filtros opcionais)",
            )
            natureza_sel = st.multiselect(
                "Natureza Jurídica",
                _opts(df_est, "categoria_natureza_juridica"),
                key="est_natjur",
                placeholder="(Todos. Filtros opcionais)",
            )
            status_sel   = st.multiselect(
                "Status",
                _opts(df_est, "status_do_estabelecimento"),
                key="est_status",
                placeholder="(Todos. Filtros opcionais)",
            )

        # Habilitações oncológicas
        with st.expander("Fitros de Habilitações Oncológicas", expanded=False):
            onco_cacon_sel = st.multiselect(
                "CACON", ["Sim", "Não"],
                key="est_onco_cacon",
                placeholder="(Todos. Filtros opcionais)",
            )
            onco_unacon_sel = st.multiselect(
                "UNACON", ["Sim", "Não"],
                key="est_onco_unacon",
                placeholder="(Todos. Filtros opcionais)",
            )
            onco_radio_sel  = st.multiselect(
                "Radioterapia", ["Sim", "Não"],
                key="est_onco_radio",
                placeholder="(Todos. Filtros opcionais)",
            )
            onco_quimio_sel = st.multiselect(
                "Quimioterapia", ["Sim", "Não"],
                key="est_onco_quimio",
                placeholder="(Todos. Filtros opcionais)",
            )
            hab_onco_cir_sel = st.multiselect(
                "Onco Cirúrgica", ["Sim", "Não"],
                key="est_hab_onco_cir",
                placeholder="(Todos. Filtros opcionais)",
            )

    # ------------------------ Detectar mudanças ------------------------
    filter_values = {
        "comp": comp_sel,
        "reg": reg_sel,
        "uf": uf_sel,
        "meso": meso_sel,
        "micro": micro_sel,
        "regsaude": reg_saude_sel,
        "mun": mun_sel,
        "ivs": ivs_sel,
        "tipo": tipo_sel,
        "subtipo": subtipo_sel,
        "gestao": gestao_sel,
        "convenio": convenio_sel,
        "natjur": natureza_sel,
        "status": status_sel,
        "cacon": onco_cacon_sel,
        "unacon": onco_unacon_sel,
        "radio": onco_radio_sel,
        "quimio": onco_quimio_sel,
        "oncocir": hab_onco_cir_sel,
    }
    filters_changed = any(did_filters_change(k, v) for k, v in filter_values.items())
    spinner = st.spinner("⏳ Atualizando resultados…") if filters_changed else DummySpinner()

    # WHERE + parâmetros para todas as queries agregadas
    where_sql_est, bq_params_est = build_where_estab(
        comp_sel,
        reg_sel,
        uf_sel,
        meso_sel,
        micro_sel,
        reg_saude_sel,
        mun_sel,
        ivs_sel,
        tipo_sel,
        subtipo_sel,
        gestao_sel,
        convenio_sel,
        natureza_sel,
        status_sel,
        onco_cacon_sel,
        onco_unacon_sel,
        onco_radio_sel,
        onco_quimio_sel,
        hab_onco_cir_sel,
    )

    with spinner:
        # =====================================================
        # Base detalhada (apenas para TABELA)
        # =====================================================
        dfe = df_est.copy()

        def apply_multisel(df, col, sel):
            if sel and col in df:
                return df[df[col].isin(sel)]
            return df

        dfe = apply_multisel(dfe, "competencia", comp_sel)
        dfe = apply_multisel(dfe, "no_regiao", reg_sel)
        dfe = apply_multisel(dfe, "no_uf", uf_sel)
        dfe = apply_multisel(dfe, "no_mesorregiao", meso_sel)
        dfe = apply_multisel(dfe, "no_microrregiao", micro_sel)
        dfe = apply_multisel(dfe, "cod_regiao_saude", reg_saude_sel)
        dfe = apply_multisel(dfe, "municipio", mun_sel)
        dfe = apply_multisel(dfe, "ivs", ivs_sel)
        dfe = apply_multisel(dfe, "tipo_do_estabelecimento", tipo_sel)
        dfe = apply_multisel(dfe, "subtipo_do_estabelecimento", subtipo_sel)
        dfe = apply_multisel(dfe, "gestao", gestao_sel)
        dfe = apply_multisel(dfe, "convenio_sus", convenio_sel)
        dfe = apply_multisel(dfe, "categoria_natureza_juridica", natureza_sel)
        dfe = apply_multisel(dfe, "status_do_estabelecimento", status_sel)

        def apply_bool_local(df, col, sel):
            if not sel or col not in df:
                return df
            allowed = []
            if "Sim" in sel:
                allowed.append(True)
            if "Não" in sel:
                allowed.append(False)
            return df[df[col].astype("boolean").isin(allowed)]

        dfe = apply_bool_local(dfe, "onco_cacon", onco_cacon_sel)
        dfe = apply_bool_local(dfe, "onco_unacon", onco_unacon_sel)
        dfe = apply_bool_local(dfe, "onco_radioterapia", onco_radio_sel)
        dfe = apply_bool_local(dfe, "onco_quimioterapia", onco_quimio_sel)
        dfe = apply_bool_local(dfe, "habilitacao_agrupado_onco_cirurgica", hab_onco_cir_sel)

        # ===================== KPIs (via BigQuery) =====================
        st.info("📏 Grandes números: Visão rápida com filtros aplicados")

        kpis = query_estab_kpis(where_sql_est, bq_params_est, estab_table_id)

        tot_est = kpis["tot_est"] or 0
        media_por_uf = kpis["media_por_uf"] or 0
        media_por_regiao = kpis["media_por_regiao"] or 0
        n_cacon = kpis["n_cacon"] or 0
        n_unacon = kpis["n_unacon"] or 0
        n_radio = kpis["n_radio"] or 0
        n_quimio = kpis["n_quimio"] or 0
        n_cir = kpis["n_cir"] or 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estabelecimentos", fmt_num(tot_est))
        c2.metric("Média por UF", fmt_num(media_por_uf))
        c3.metric("Média por Reg. Saúde", fmt_num(media_por_regiao))
        c4.metric("CACON", fmt_num(n_cacon))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("UNACON", fmt_num(n_unacon))
        c6.metric("Radioterapia", fmt_num(n_radio))
        c7.metric("Quimioterapia", fmt_num(n_quimio))
        c8.metric("Onco Cirúrgica", fmt_num(n_cir))

        # ===================== Gráficos (todos agregados no BQ) =====================
        st.info("📊 Gráficos: Veja visuais gráficos com respostas segundo os filtros aplicados")

        # 1) Tipo de estabelecimento (barra horizontal tipo pareto)
        with st.expander("Por tipo de estabelecimento", expanded=True):
            df_tipo = query_estab_group(
                where_sql_est,
                bq_params_est,
                estab_table_id,
                col="tipo_do_estabelecimento",
                null_label="(não informado)",
            )
            if not df_tipo.empty:
                fig = bar_total_por_grupo(
                    df_tipo,
                    grupo_col="categoria",
                    valor_col="qtde",
                    titulo="Estab. por Tipo",
                    x_label="Qtde de estabelecimentos",
                    y_label="Tipo de estabelecimento",
                    orientation="h",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum estabelecimento encontrado para os filtros selecionados.")

        # 2) Tipo de gestão
        with st.expander("Por tipo de gestão", expanded=True):
            df_gestao = query_estab_group(
                where_sql_est,
                bq_params_est,
                estab_table_id,
                col="gestao",
                null_label="(não informado)",
            )
            if not df_gestao.empty:
                fig = bar_total_por_grupo(
                    df_gestao,
                    grupo_col="categoria",
                    valor_col="qtde",
                    titulo="Estab. por Gestão",
                    x_label="Qtde de estabelecimentos",
                    y_label="Gestão",
                    orientation="v",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum estabelecimento encontrado para os filtros selecionados.")

        # 3) Público x Privado (Natureza Jurídica)
        with st.expander("Por Público x Privado", expanded=True):
            df_natjur = query_estab_group(
                where_sql_est,
                bq_params_est,
                estab_table_id,
                col="categoria_natureza_juridica",
                null_label="(não informado)",
            )
            if not df_natjur.empty:
                fig = bar_total_por_grupo(
                    df_natjur,
                    grupo_col="categoria",
                    valor_col="qtde",
                    titulo="Estab. por Categoria de Natureza Jurídica",
                    x_label="Qtde de estabelecimentos",
                    y_label="Categoria",
                    orientation="v",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum estabelecimento encontrado para os filtros selecionados.")

        # 4) Convênio SUS (pizza)
        with st.expander("Por Convênio SUS", expanded=True):
            df_conv = query_estab_group(
                where_sql_est,
                bq_params_est,
                estab_table_id,
                col="convenio_sus",
                null_label="(não informado)",
            )
            if not df_conv.empty:
                fig = pie_standard(
                    df_conv,
                    names="categoria",
                    values="qtde",
                    title="Convênio SUS",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum estabelecimento encontrado para os filtros selecionados.")

        # ===================== Tabela descritiva dos estabelecimentos filtrados =====================
        st.info("📋 Tabela descritiva dos estabelecimentos filtrados")

        cols_desc_estab = [
            "cnes",
            "nome_fantasia",
            "municipio",
            "no_uf",
            "tipo_do_estabelecimento",
            "subtipo_do_estabelecimento",
            "gestao",
            "convenio_sus",
            "categoria_natureza_juridica",
        ]

        # apenas colunas existentes
        cols_ok_estab = [c for c in cols_desc_estab if c in dfe.columns]

        if cols_ok_estab:

            # ===================== Limite de linhas (5.000) =====================
            max_rows_display = 5000
            n_total = dfe.shape[0]

            if n_total > max_rows_display:
                st.warning(
                    f"A base filtrada possui {fmt_num(n_total)} linhas. "
                    f"Por desempenho, a tabela mostra apenas as primeiras {fmt_num(max_rows_display)} linhas. "
                    "O download também está limitado às mesmas linhas."
                )
            else:
                st.caption(f"A base filtrada possui {fmt_num(n_total)} linhas.")

            # dataframe limitado
            df_display = (
                dfe[cols_ok_estab]
                .sort_values(["no_uf", "municipio"])
                .head(max_rows_display)
            )

            # exibição
            st.dataframe(
                df_display,
                use_container_width=True,
                height=450,
            )

            # ===================== Download limitado =====================
            csv_estab = df_display.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Baixar CSV (até 5.000 linhas)",
                data=csv_estab,
                file_name="cadastro_estabelecimentos_filtrado.csv",
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.info("Não existem colunas suficientes para montar a tabela de estabelecimentos.")

# =====================================================================
# 3) Cadastro Serviços
# =====================================================================
elif aba == "🗂️ Serviços":
    st.subheader("🗂️ Serviços")

    # ---------------------------------------------------------
    # Carregar base para filtros/tabela
    # ---------------------------------------------------------
    with st.spinner("⏳ Carregando base de serviços..."):
        df_srv = load_table(TABLES["servicos"]).copy()

    # Helper local para opções dos filtros
    def _opts_srv(col: str):
        if col not in df_srv:
            return []
        return sorted(df_srv[col].dropna().unique())

    # =========================================================
    # SIDEBAR DE FILTROS
    # =========================================================
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Serviços especializados")
        st.caption("Use os agrupadores abaixo para refinar o cadastro.")

        # ---------------- Filtros de Período ----------------
        with st.expander("Filtros de Período", expanded=False):
            comp_sel = st.multiselect(
                "Competência",
                _opts_srv("competencia"),
                key="srv_comp",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ---------------- Filtros de Território --------------
        with st.expander("Filtros de Território", expanded=False):
            reg_sel = st.multiselect(
                "Região",
                _opts_srv("no_regiao"),
                key="srv_reg",
                placeholder="(Todos. Filtros opcionais)",
            )
            uf_sel = st.multiselect(
                "UF",
                _opts_srv("no_uf"),
                key="srv_uf",
                placeholder="(Todos. Filtros opcionais)",
            )
            meso_sel = st.multiselect(
                "Mesorregião Geográfica",
                _opts_srv("no_mesorregiao"),
                key="srv_meso",
                placeholder="(Todos. Filtros opcionais)",
            )
            micro_sel = st.multiselect(
                "Microrregião Geográfica",
                _opts_srv("no_microrregiao"),
                key="srv_micro",
                placeholder="(Todos. Filtros opcionais)",
            )
            reg_saude_sel = st.multiselect(
                "Região de Saúde",
                _opts_srv("cod_regiao_saude"),
                key="srv_regsaude",
                placeholder="(Todos. Filtros opcionais)",
            )
            mun_sel = st.multiselect(
                "Município",
                _opts_srv("municipio"),
                key="srv_mun",
                placeholder="(Todos. Filtros opcionais)",
            )
            ivs_sel = st.multiselect(
                "Município IVS",
                _opts_srv("ivs"),
                key="srv_ivs",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Perfil do estabelecimento -----------------
        with st.expander("Filtros de Perfil do Estabelecimento", expanded=False):
            tipo_sel = st.multiselect(
                "Tipo",
                _opts_srv("tipo_novo_do_estabelecimento"),
                key="srv_tipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            subtipo_sel = st.multiselect(
                "Subtipo",
                _opts_srv("subtipo_do_estabelecimento"),
                key="srv_subtipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            gestao_sel = st.multiselect(
                "Gestão",
                _opts_srv("gestao"),
                key="srv_gestao",
                placeholder="(Todos. Filtros opcionais)",
            )
            convenio_sel = st.multiselect(
                "Convênio SUS",
                _opts_srv("convenio_sus"),
                key="srv_convenio",
                placeholder="(Todos. Filtros opcionais)",
            )
            natureza_sel = st.multiselect(
                "Natureza Jurídica",
                _opts_srv("categoria_natureza_juridica"),
                key="srv_natjur",
                placeholder="(Todos. Filtros opcionais)",
            )
            status_sel = st.multiselect(
                "Status",
                _opts_srv("status_do_estabelecimento"),
                key="srv_status",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Habilitações oncológicas ------------------
        with st.expander("Filtros de Habilitações Oncológicas", expanded=False):
            onco_cacon_sel = st.multiselect(
                "CACON", ["Sim", "Não"],
                key="srv_onco_cacon",
                placeholder="(Todos. Filtros opcionais)",
            )
            onco_unacon_sel = st.multiselect(
                "UNACON", ["Sim", "Não"],
                key="srv_onco_unacon",
                placeholder="(Todos. Filtros opcionais)",
            )
            onco_radio_sel = st.multiselect(
                "Radioterapia", ["Sim", "Não"],
                key="srv_onco_radio",
                placeholder="(Todos. Filtros opcionais)",
            )
            onco_quimio_sel = st.multiselect(
                "Quimioterapia", ["Sim", "Não"],
                key="srv_onco_quimio",
                placeholder="(Todos. Filtros opcionais)",
            )
            hab_onco_cir_sel = st.multiselect(
                "Onco Cirúrgica", ["Sim", "Não"],
                key="srv_onco_cir",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Perfil do serviço especializado -----------
        with st.expander("Filtros de Serviços Especializados", expanded=False):
            servico_sel = st.multiselect(
                "Serviço especializado",
                _opts_srv("servico"),
                key="srv_servico",
                placeholder="(Todos. Filtros opcionais)",
            )
            servico_class_sel = st.multiselect(
                "Classificação",
                _opts_srv("servico_classificacao"),
                key="srv_servico_class",
                placeholder="(Todos. Filtros opcionais)",
            )
            amb_sus_sel = st.multiselect(
                "Ambulatorial SUS",
                _opts_srv("servico_ambulatorial_sus"),
                key="srv_amb_sus",
                placeholder="(Todos. Filtros opcionais)",
            )
            amb_nao_sus_sel = st.multiselect(
                "Ambulatorial não SUS",
                _opts_srv("servico_ambulatorial_nao_sus"),
                key="srv_amb_nao_sus",
                placeholder="(Todos. Filtros opcionais)",
            )
            hosp_sus_sel = st.multiselect(
                "Hospitalar SUS",
                _opts_srv("servico_hospitalar_sus"),
                key="srv_hosp_sus",
                placeholder="(Todos. Filtros opcionais)",
            )
            hosp_nao_sus_sel = st.multiselect(
                "Hospitalar não SUS",
                _opts_srv("servico_hospitalar_nao_sus"),
                key="srv_hosp_nao_sus",
                placeholder="(Todos. Filtros opcionais)",
            )
            terceiro_sel = st.multiselect(
                "Terceiro",
                _opts_srv("servico_terceiro"),
                key="srv_terceiro",
                placeholder="(Todos. Filtros opcionais)",
            )

    # =========================================================
    # Detectar mudanças de filtro (para spinner)
    # =========================================================
    filter_values = {
        "comp": comp_sel,
        "reg": reg_sel,
        "uf": uf_sel,
        "meso": meso_sel,
        "micro": micro_sel,
        "regsaude": reg_saude_sel,
        "mun": mun_sel,
        "ivs": ivs_sel,
        "tipo": tipo_sel,
        "subtipo": subtipo_sel,
        "gestao": gestao_sel,
        "convenio": convenio_sel,
        "natjur": natureza_sel,
        "status": status_sel,
        "cacon": onco_cacon_sel,
        "unacon": onco_unacon_sel,
        "radio": onco_radio_sel,
        "quimio": onco_quimio_sel,
        "oncocir": hab_onco_cir_sel,
        "serv": servico_sel,
        "serv_class": servico_class_sel,
        "amb_sus": amb_sus_sel,
        "amb_nao_sus": amb_nao_sus_sel,
        "hosp_sus": hosp_sus_sel,
        "hosp_nao_sus": hosp_nao_sus_sel,
        "terceiro": terceiro_sel,
    }
    filters_changed = any(
        did_filters_change(k, v) for k, v in filter_values.items()
    )
    spinner = (
        st.spinner("⏳ Atualizando resultados…")
        if filters_changed
        else DummySpinner()
    )

    # =========================================================
    # WHERE para as consultas agregadas no BigQuery
    # =========================================================
    def _cond_in(col: str, values):
        if not values:
            return ""
        esc = [str(v).replace("'", "''") for v in values]
        in_list = ", ".join(f"'{v}'" for v in esc)
        return f"{col} IN ({in_list})"

    def _cond_bool(col: str, values):
        if not values:
            return ""
        parts = []
        if "Sim" in values:
            parts.append(f"{col} = TRUE")
        if "Não" in values:
            parts.append(f"{col} = FALSE")
        if not parts:
            return ""
        return "(" + " OR ".join(parts) + ")"

    where_clauses = []

    for col, sel in [
        ("competencia", comp_sel),
        ("no_regiao", reg_sel),
        ("no_uf", uf_sel),
        ("no_mesorregiao", meso_sel),
        ("no_microrregiao", micro_sel),
        ("cod_regiao_saude", reg_saude_sel),
        ("municipio", mun_sel),
        ("ivs", ivs_sel),
        ("tipo_novo_do_estabelecimento", tipo_sel),
        ("subtipo_do_estabelecimento", subtipo_sel),
        ("gestao", gestao_sel),
        ("convenio_sus", convenio_sel),
        ("categoria_natureza_juridica", natureza_sel),
        ("status_do_estabelecimento", status_sel),
        ("servico", servico_sel),
        ("servico_classificacao", servico_class_sel),
        ("servico_ambulatorial_sus", amb_sus_sel),
        ("servico_ambulatorial_nao_sus", amb_nao_sus_sel),
        ("servico_hospitalar_sus", hosp_sus_sel),
        ("servico_hospitalar_nao_sus", hosp_nao_sus_sel),
        ("servico_terceiro", terceiro_sel),
    ]:
        c = _cond_in(col, sel)
        if c:
            where_clauses.append(c)

    for col, sel in [
        ("onco_cacon", onco_cacon_sel),
        ("onco_unacon", onco_unacon_sel),
        ("onco_radioterapia", onco_radio_sel),
        ("onco_quimioterapia", onco_quimio_sel),
        ("habilitacao_agrupado_onco_cirurgica", hab_onco_cir_sel),
    ]:
        c = _cond_bool(col, sel)
        if c:
            where_clauses.append(c)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # =========================================================
    # Função helper para rodar consultas no BigQuery
    # =========================================================
    from google.cloud import bigquery

    @st.cache_data(show_spinner=False)
    def run_bq_servicos(sql: str) -> pd.DataFrame:
        client = bigquery.Client()
        return client.query(sql).to_dataframe()

    bq_table = TABLES["servicos"]

    with spinner:
        # =====================================================
        # METRIC CARDS — agregados direto no BigQuery
        # =====================================================
        st.info("📏 Grandes números: Visão rápida com filtros aplicados")

        sql_kpis = f"""
        WITH base AS (
          SELECT
            cnes,
            no_uf,
            cod_regiao_saude,
            servico
          FROM `{bq_table}`
          {where_sql}
        ),
        por_uf AS (
          SELECT no_uf, COUNT(*) AS qtd
          FROM base
          GROUP BY no_uf
        ),
        por_reg_saude AS (
          SELECT cod_regiao_saude, COUNT(*) AS qtd
          FROM base
          GROUP BY cod_regiao_saude
        ),
        por_cnes AS (
          SELECT cnes, COUNT(*) AS qtd
          FROM base
          GROUP BY cnes
        )
        SELECT
          (SELECT COUNT(*) FROM base) AS total_servicos,
          (SELECT AVG(qtd) FROM por_uf) AS media_por_uf,
          (SELECT AVG(qtd) FROM por_reg_saude) AS media_por_reg_saude,
          (SELECT AVG(qtd) FROM por_cnes) AS media_por_estab
        """
        df_kpis = run_bq_servicos(sql_kpis)

        if df_kpis.empty:
            st.warning("Nenhum serviço encontrado com os filtros selecionados.")
            st.stop()

        kpi = df_kpis.iloc[0]
        total_srv = kpi.get("total_servicos", 0) or 0
        media_por_uf = kpi.get("media_por_uf", 0) or 0
        media_por_reg = kpi.get("media_por_reg_saude", 0) or 0
        media_por_estab = kpi.get("media_por_estab", 0) or 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de serviços", fmt_num(total_srv))
        col2.metric("Média por Estado", fmt_num(media_por_uf))
        col3.metric("Média por Reg. Saúde", fmt_num(media_por_reg))
        col4.metric("Média por Estabelecimento", fmt_num(media_por_estab))

        # =====================================================
        # GRÁFICOS — agregados direto no BigQuery
        # =====================================================
        st.info("📊 Gráficos — Resumo visual dos serviços filtrados")

        # --------- Número de serviços por tipo de serviço ----
        with st.expander("Número de serviços por Tipo de Serviço", expanded=True):
            sql_tipo = f"""
            SELECT
              servico,
              COUNT(*) AS qtde
            FROM `{bq_table}`
            {where_sql}
            GROUP BY servico
            ORDER BY qtde DESC
            """
            df_tipo = run_bq_servicos(sql_tipo)

            if not df_tipo.empty:
                fig_tipo = bar_total_por_grupo(
                    df_tipo,
                    grupo_col="servico",
                    valor_col="qtde",
                    titulo="Distribuição de Serviços por Tipo",
                    x_label="Qtde",
                    y_label="Serviço",
                    orientation="h",
                )
                st.plotly_chart(fig_tipo, use_container_width=True)
            else:
                st.info("Não há serviços para os filtros selecionados.")

        # --- Ambulatorial / Hospitalar – SUS x Não SUS -------
        with st.expander(
            "Serviços Ambulatoriais e Hospitalares – SUS/Não SUS",
            expanded=True,
        ):
            sql_ah = f"""
            SELECT
              SUM(CASE WHEN servico_ambulatorial_sus = 'Sim'
                       THEN 1 ELSE 0 END) AS servico_ambulatorial_sus,
              SUM(CASE WHEN servico_ambulatorial_nao_sus = 'Sim'
                       THEN 1 ELSE 0 END) AS servico_ambulatorial_nao_sus,
              SUM(CASE WHEN servico_hospitalar_sus = 'Sim'
                       THEN 1 ELSE 0 END) AS servico_hospitalar_sus,
              SUM(CASE WHEN servico_hospitalar_nao_sus = 'Sim'
                       THEN 1 ELSE 0 END) AS servico_hospitalar_nao_sus
            FROM `{bq_table}`
            {where_sql}
            """
            df_ah_raw = run_bq_servicos(sql_ah)

            if not df_ah_raw.empty:
                row = df_ah_raw.iloc[0]
                cols_ah = {
                    "servico_ambulatorial_sus": "Ambulatorial SUS",
                    "servico_ambulatorial_nao_sus": "Ambulatorial Não SUS",
                    "servico_hospitalar_sus": "Hospitalar SUS",
                    "servico_hospitalar_nao_sus": "Hospitalar Não SUS",
                }

                df_ah = pd.DataFrame(
                    {
                        "classificacao": [label for label in cols_ah.values()],
                        "qtd_servicos": [
                            int(row.get(col, 0) or 0) for col in cols_ah.keys()
                        ],
                    }
                )

                df_ah = df_ah[df_ah["qtd_servicos"] > 0]

                if not df_ah.empty:
                    fig_ah = bar_total_por_grupo(
                        df_ah,
                        grupo_col="classificacao",
                        valor_col="qtd_servicos",
                        titulo="Serviços por Classificação Amb/Hosp – SUS/Não SUS",
                        x_label="Qtde",
                        y_label="Classificação",
                        orientation="v",
                    )
                    st.plotly_chart(fig_ah, use_container_width=True)
                else:
                    st.info("Não há serviços Amb/Hosp com valor 'Sim' para os filtros.")
            else:
                st.info("Não há dados para Amb/Hosp com os filtros selecionados.")

        # =====================================================
        # TABELA DESCRITIVA (limitada a 5.000 linhas)
        # =====================================================
        st.info("📋 Tabela descritiva dos serviços filtrados")

        # Reaplicar filtros em pandas para a tabela / download
        dfs = df_srv.copy()

        def apply_multisel(df, col, sel):
            if sel and col in df:
                return df[df[col].isin(sel)]
            return df

        def apply_bool(df, col, sel):
            if not sel or col not in df:
                return df
            allowed = []
            if "Sim" in sel:
                allowed.append(True)
            if "Não" in sel:
                allowed.append(False)
            return df[df[col].astype("boolean").isin(allowed)]

        # Território / período
        dfs = apply_multisel(dfs, "competencia", comp_sel)
        dfs = apply_multisel(dfs, "no_regiao", reg_sel)
        dfs = apply_multisel(dfs, "no_uf", uf_sel)
        dfs = apply_multisel(dfs, "no_mesorregiao", meso_sel)
        dfs = apply_multisel(dfs, "no_microrregiao", micro_sel)
        dfs = apply_multisel(dfs, "cod_regiao_saude", reg_saude_sel)
        dfs = apply_multisel(dfs, "municipio", mun_sel)
        dfs = apply_multisel(dfs, "ivs", ivs_sel)

        # Perfil Estabelecimento
        dfs = apply_multisel(dfs, "tipo_novo_do_estabelecimento", tipo_sel)
        dfs = apply_multisel(dfs, "subtipo_do_estabelecimento", subtipo_sel)
        dfs = apply_multisel(dfs, "gestao", gestao_sel)
        dfs = apply_multisel(dfs, "convenio_sus", convenio_sel)
        dfs = apply_multisel(dfs, "categoria_natureza_juridica", natureza_sel)
        dfs = apply_multisel(dfs, "status_do_estabelecimento", status_sel)

        # Booleanos oncológicos
        dfs = apply_bool(dfs, "onco_cacon", onco_cacon_sel)
        dfs = apply_bool(dfs, "onco_unacon", onco_unacon_sel)
        dfs = apply_bool(dfs, "onco_radioterapia", onco_radio_sel)
        dfs = apply_bool(dfs, "onco_quimioterapia", onco_quimio_sel)
        dfs = apply_bool(dfs, "habilitacao_agrupado_onco_cirurgica", hab_onco_cir_sel)

        # Serviços
        dfs = apply_multisel(dfs, "servico", servico_sel)
        dfs = apply_multisel(dfs, "servico_classificacao", servico_class_sel)
        dfs = apply_multisel(dfs, "servico_ambulatorial_sus", amb_sus_sel)
        dfs = apply_multisel(dfs, "servico_ambulatorial_nao_sus", amb_nao_sus_sel)
        dfs = apply_multisel(dfs, "servico_hospitalar_sus", hosp_sus_sel)
        dfs = apply_multisel(dfs, "servico_hospitalar_nao_sus", hosp_nao_sus_sel)
        dfs = apply_multisel(dfs, "servico_terceiro", terceiro_sel)

        if dfs.empty:
            st.warning("Nenhum serviço encontrado para exibição na tabela.")
            st.stop()

        cols_desc = [
            "cnes",
            "nome_fantasia",
            "municipio",
            "no_uf",
            "servico",
            "servico_classificacao",
            "servico_terceiro",
            "servico_ambulatorial_sus",
            "servico_ambulatorial_nao_sus",
            "servico_hospitalar_sus",
            "servico_hospitalar_nao_sus",
        ]
        cols_ok = [c for c in cols_desc if c in dfs.columns]

        total_rows = dfs.shape[0]
        st.warning(
            f"A base filtrada possui {fmt_num(total_rows)} linhas. "
            "Por desempenho, a tabela mostra apenas as primeiras 5.000 linhas. "
            "O download também está limitado às mesmas linhas."
        )

        dfs_lim = dfs[cols_ok].head(5000)

        st.dataframe(
            dfs_lim,
            use_container_width=True,
            height=450,
        )

        csv = dfs_lim.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar CSV (até 5.000 linhas)",
            csv,
            file_name="cadastro_servicos_filtrado.csv",
            mime="text/csv",
            use_container_width=True,
        )

# =====================================================================
# 4) Cadastro Habilitações
# =====================================================================
elif aba == "✅ Habilitação":
    st.subheader("✅ Habilitação")

    # ---------------------------------------------------------
    # Carregar base para filtros / inicializar helpers
    # (spinner de página inteira, igual Serviços/Estabelecimentos)
    # ---------------------------------------------------------
    with st.spinner("⏳ Carregando base de habilitações..."):

        import pandas as pd
        from google.cloud import bigquery

        # Referência da tabela no BigQuery
        hab_table_id = TABLES["habilitacao"]

        # -----------------------------------------------------
        # Cliente BigQuery
        # -----------------------------------------------------
        @st.cache_resource
        def get_bq_client_hab():
            return bigquery.Client()

        # -----------------------------------------------------
        # WHERE dinâmico (para todas as queries do BQ)
        # -----------------------------------------------------
        def build_where_hab(
            ano_hab_sel,
            mes_hab_sel,
            ano_comp_ini_sel,
            mes_comp_ini_sel,
            ano_comp_fim_sel,
            mes_comp_fim_sel,
            ano_portaria_sel,
            mes_portaria_sel,
            uf_sel,
            reg_saude_sel,
            meso_sel,
            micro_sel,
            mun_sel,
            ivs_sel,
            tipo_novo_sel,
            subtipo_sel,
            gestao_sel,
            convenio_sel,
            nat_jur_sel,
            status_sel,
            nivel_tipo_sel,
            cat_hab_sel,
            no_hab_sel,
            tag_hab_sel,
        ):
            """
            Monta WHERE dinâmico + parâmetros BigQuery
            para todos os filtros da aba Habilitações.
            """
            clauses = ["1=1"]
            params = []

            def add_in_array(col: str, param_name: str, values):
                if values:
                    vals = [str(v) for v in values]
                    clauses.append(f"CAST({col} AS STRING) IN UNNEST(@{param_name})")
                    params.append(
                        bigquery.ArrayQueryParameter(param_name, "STRING", vals)
                    )

            # Período
            add_in_array("habilitacao_ano", "ano_hab", ano_hab_sel)
            add_in_array("habilitacao_mes", "mes_hab", mes_hab_sel)
            add_in_array(
                "habilitacao_ano_competencia_inicial",
                "ano_comp_ini",
                ano_comp_ini_sel,
            )
            add_in_array(
                "habilitacao_mes_competencia_inicial",
                "mes_comp_ini",
                mes_comp_ini_sel,
            )
            add_in_array(
                "habilitacao_ano_competencia_final",
                "ano_comp_fim",
                ano_comp_fim_sel,
            )
            add_in_array(
                "habilitacao_mes_competencia_final",
                "mes_comp_fim",
                mes_comp_fim_sel,
            )
            add_in_array(
                "habilitacao_ano_portaria",
                "ano_portaria",
                ano_portaria_sel,
            )
            add_in_array(
                "habilitacao_mes_portaria",
                "mes_portaria",
                mes_portaria_sel,
            )

            # Território
            add_in_array("ibge_no_uf", "uf", uf_sel)
            add_in_array("ibge_no_regiao_saude", "reg_saude", reg_saude_sel)
            add_in_array("ibge_no_mesorregiao", "meso", meso_sel)
            add_in_array("ibge_no_microrregiao", "micro", micro_sel)
            add_in_array("ibge_no_municipio", "mun", mun_sel)
            add_in_array("ibge_ivs", "ivs", ivs_sel)

            # Perfil do estabelecimento
            add_in_array(
                "estabelecimentos_tipo_novo_do_estabelecimento",
                "tipo_novo",
                tipo_novo_sel,
            )
            add_in_array(
                "estabelecimentos_subtipo_do_estabelecimento",
                "subtipo",
                subtipo_sel,
            )
            add_in_array(
                "estabelecimentos_gestao",
                "gestao",
                gestao_sel,
            )
            add_in_array(
                "estabelecimentos_convenio_sus",
                "convenio",
                convenio_sel,
            )
            add_in_array(
                "estabelecimentos_categoria_natureza_juridica",
                "nat_jur",
                nat_jur_sel,
            )
            add_in_array(
                "estabelecimentos_status_do_estabelecimento",
                "status_estab",
                status_sel,
            )

            # Habilitação
            add_in_array(
                "habilitacao_nivel_habilitacao_tipo",
                "nivel_tipo",
                nivel_tipo_sel,
            )
            add_in_array(
                "referencia_habilitacao_no_categoria",
                "cat_hab",
                cat_hab_sel,
            )
            add_in_array(
                "referencia_habilitacao_no_habilitacao",
                "no_hab",
                no_hab_sel,
            )
            add_in_array(
                "referencia_habilitacao_ds_tag",
                "tag_hab",
                tag_hab_sel,
            )

            where_sql = "WHERE " + " AND ".join(clauses)
            return where_sql, params

        # -----------------------------------------------------
        # KPIs (BigQuery)
        # -----------------------------------------------------
        def query_hab_kpis(where_sql: str, params, table_id: str):
            """
            KPIs direto do BigQuery:
              - total de habilitações (linhas)
              - média por UF
              - média por Região de Saúde
              - média por Estabelecimento
            """
            client = get_bq_client_hab()
            sql = f"""
            WITH base AS (
              SELECT
                habilitacao_id_estabelecimento_cnes,
                ibge_no_uf,
                ibge_no_regiao_saude
              FROM `{table_id}`
              {where_sql}
            ),
            uf_agg AS (
              SELECT ibge_no_uf, COUNT(*) AS hab_por_uf
              FROM base
              GROUP BY ibge_no_uf
            ),
            reg_agg AS (
              SELECT ibge_no_regiao_saude, COUNT(*) AS hab_por_reg
              FROM base
              GROUP BY ibge_no_regiao_saude
            ),
            estab_agg AS (
              SELECT habilitacao_id_estabelecimento_cnes, COUNT(*) AS hab_por_estab
              FROM base
              GROUP BY habilitacao_id_estabelecimento_cnes
            )
            SELECT
              (SELECT COUNT(*) FROM base) AS total_hab,
              (SELECT AVG(hab_por_uf) FROM uf_agg) AS media_uf,
              (SELECT AVG(hab_por_reg) FROM reg_agg) AS media_reg_saude,
              (SELECT AVG(hab_por_estab) FROM estab_agg) AS media_estab;
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            df = job.to_dataframe()
            return df.iloc[0]

        # -----------------------------------------------------
        # Agregado genérico (BigQuery)
        # -----------------------------------------------------
        def query_hab_group(
            where_sql: str,
            params,
            table_id: str,
            col: str,
            null_label: str = "(não informado)",
            order_desc: bool = True,
        ) -> pd.DataFrame:
            """
            Agregado genérico: conta habilitações por coluna categórica.
            Retorna df com 'categoria' e 'qtde'.
            """
            client = get_bq_client_hab()
            order_dir = "DESC" if order_desc else "ASC"
            sql = f"""
            SELECT
              COALESCE(CAST({col} AS STRING), @null_label) AS categoria,
              COUNT(*) AS qtde
            FROM `{table_id}`
            {where_sql}
            GROUP BY categoria
            ORDER BY qtde {order_dir}
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=list(params)
                    + [
                        bigquery.ScalarQueryParameter(
                            "null_label", "STRING", null_label
                        )
                    ]
                ),
            )
            return job.to_dataframe()

        # -----------------------------------------------------
        # Série temporal anual (BigQuery)
        # -----------------------------------------------------
        def query_hab_por_ano(where_sql: str, params, table_id: str) -> pd.DataFrame:
            """
            Retorna série temporal anual de habilitações.
            """
            client = get_bq_client_hab()
            sql = f"""
            SELECT
              SAFE_CAST(habilitacao_ano AS INT64) AS ano,
              COUNT(*) AS qtd_habilitacoes
            FROM `{table_id}`
            {where_sql}
            GROUP BY ano
            HAVING ano IS NOT NULL
            ORDER BY ano
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            return job.to_dataframe()

        # -----------------------------------------------------
        # Ativas vs Encerradas (BigQuery)
        # -----------------------------------------------------
        def query_hab_status(where_sql: str, params, table_id: str):
            """
            Calcula total de habilitações ativas e encerradas direto no BigQuery.
            Regra:
              - Ativa se ano_final=9999 e mes_final=9999
              - Ou se (ano_final*100 + mes_final) >= competência atual (ano*100+mes)
              - Caso contrário, encerrada
            """
            client = get_bq_client_hab()

            hoje = pd.Timestamp.today()
            comp_atual = hoje.year * 100 + hoje.month

            extra_params = list(params) + [
                bigquery.ScalarQueryParameter("comp_atual", "INT64", int(comp_atual))
            ]

            sql = f"""
            WITH base AS (
              SELECT
                SAFE_CAST(habilitacao_ano_competencia_final AS INT64) AS ano_final,
                SAFE_CAST(habilitacao_mes_competencia_final AS INT64) AS mes_final
              FROM `{table_id}`
              {where_sql}
            ),
            classif AS (
              SELECT
                CASE
                  WHEN ano_final = 9999 AND mes_final = 9999 THEN 'ativa'
                  WHEN ano_final IS NOT NULL AND mes_final IS NOT NULL
                       AND (ano_final * 100 + mes_final) >= @comp_atual
                    THEN 'ativa'
                  ELSE 'encerrada'
                END AS status
              FROM base
              WHERE ano_final IS NOT NULL AND mes_final IS NOT NULL
            )
            SELECT
              SUM(CASE WHEN status = 'ativa' THEN 1 ELSE 0 END) AS total_ativas,
              SUM(CASE WHEN status = 'encerrada' THEN 1 ELSE 0 END) AS total_encerradas
            FROM classif
            """

            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=extra_params),
            )
            df = job.to_dataframe()
            if df.empty:
                return 0, 0
            row = df.iloc[0]
            return int(row["total_ativas"] or 0), int(row["total_encerradas"] or 0)

        # -----------------------------------------------------
        # Tabela descritiva (BigQuery com LIMIT)
        # -----------------------------------------------------
        def query_hab_detalhe(
            where_sql: str,
            params,
            table_id: str,
            cols: list[str],
            limit_rows: int = 5000,
        ) -> pd.DataFrame:
            """
            Busca dados detalhados das habilitações com os filtros aplicados,
            limitado a `limit_rows` para exibição/CSV.
            """
            if not cols:
                return pd.DataFrame()

            client = get_bq_client_hab()
            cols_sql = ", ".join(cols)

            sql = f"""
            SELECT
              {cols_sql}
            FROM `{table_id}`
            {where_sql}
            LIMIT {limit_rows}
            """

            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            return job.to_dataframe()

        # -----------------------------------------------------
        # Opções dos filtros (DISTINCT no BigQuery)
        # -----------------------------------------------------
        @st.cache_data(show_spinner=False)
        def _opts_hab(col: str) -> list[str]:
            """
            Busca valores distintos de uma coluna na tabela de habilitações.
            Usado só para montar as opções dos filtros (sem aplicar WHERE).
            """
            client = get_bq_client_hab()
            sql = f"""
            SELECT DISTINCT CAST({col} AS STRING) AS val
            FROM `{hab_table_id}`
            WHERE {col} IS NOT NULL
            ORDER BY val
            """
            try:
                df = client.query(sql).to_dataframe()
                return df["val"].dropna().tolist()
            except Exception:
                # Em caso de erro (ex: coluna não existe), volta lista vazia
                return []

        # -----------------------------------------------------
        # Carregar todas as opções de filtros de uma vez
        # -----------------------------------------------------
        opts = {
            # Período
            "ano_hab": _opts_hab("habilitacao_ano"),
            "mes_hab": _opts_hab("habilitacao_mes"),
            "ano_comp_ini": _opts_hab("habilitacao_ano_competencia_inicial"),
            "mes_comp_ini": _opts_hab("habilitacao_mes_competencia_inicial"),
            "ano_comp_fim": _opts_hab("habilitacao_ano_competencia_final"),
            "mes_comp_fim": _opts_hab("habilitacao_mes_competencia_final"),
            "ano_portaria": _opts_hab("habilitacao_ano_portaria"),
            "mes_portaria": _opts_hab("habilitacao_mes_portaria"),

            # Território
            "uf": _opts_hab("ibge_no_uf"),
            "reg_saude": _opts_hab("ibge_no_regiao_saude"),
            "meso": _opts_hab("ibge_no_mesorregiao"),
            "micro": _opts_hab("ibge_no_microrregiao"),
            "mun": _opts_hab("ibge_no_municipio"),
            "ivs": _opts_hab("ibge_ivs"),

            # Perfil do estabelecimento
            "tipo_novo": _opts_hab("estabelecimentos_tipo_novo_do_estabelecimento"),
            "subtipo": _opts_hab("estabelecimentos_subtipo_do_estabelecimento"),
            "gestao": _opts_hab("estabelecimentos_gestao"),
            "convenio": _opts_hab("estabelecimentos_convenio_sus"),
            "nat_jur": _opts_hab("estabelecimentos_categoria_natureza_juridica"),
            "status": _opts_hab("estabelecimentos_status_do_estabelecimento"),

            # Habilitação
            "nivel_tipo": _opts_hab("habilitacao_nivel_habilitacao_tipo"),
            "cat_hab": _opts_hab("referencia_habilitacao_no_categoria"),
            "no_hab": _opts_hab("referencia_habilitacao_no_habilitacao"),
            "tag_hab": _opts_hab("referencia_habilitacao_ds_tag"),
        }

    # =========================================================
    # SIDEBAR DE FILTROS
    # =========================================================
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Habilitações")
        st.caption("Use os agrupadores abaixo para refinar o cadastro de habilitações.")

        # ------------------ Período -------------------------
        with st.expander("Filtros de Período", expanded=False):
            ano_hab_sel = st.multiselect(
                "Ano da habilitação",
                opts["ano_hab"],
                key="hab_ano",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_hab_sel = st.multiselect(
                "Mês da habilitação",
                opts["mes_hab"],
                key="hab_mes",
                placeholder="(Todos. Filtros opcionais)",
            )

            ano_comp_ini_sel = st.multiselect(
                "Ano competência inicial",
                opts["ano_comp_ini"],
                key="hab_ano_comp_ini",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_comp_ini_sel = st.multiselect(
                "Mês competência inicial",
                opts["mes_comp_ini"],
                key="hab_mes_comp_ini",
                placeholder="(Todos. Filtros opcionais)",
            )

            ano_comp_fim_sel = st.multiselect(
                "Ano competência final",
                opts["ano_comp_fim"],
                key="hab_ano_comp_fim",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_comp_fim_sel = st.multiselect(
                "Mês competência final",
                opts["mes_comp_fim"],
                key="hab_mes_comp_fim",
                placeholder="(Todos. Filtros opcionais)",
            )

            ano_portaria_sel = st.multiselect(
                "Ano da portaria",
                opts["ano_portaria"],
                key="hab_ano_portaria",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_portaria_sel = st.multiselect(
                "Mês da portaria",
                opts["mes_portaria"],
                key="hab_mes_portaria",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ------------------ Território -----------------------
        with st.expander("Filtros de Território", expanded=False):
            uf_sel = st.multiselect(
                "UF",
                opts["uf"],
                key="hab_uf",
                placeholder="(Todos. Filtros opcionais)",
            )
            reg_saude_sel = st.multiselect(
                "Região de Saúde",
                opts["reg_saude"],
                key="hab_reg_saude",
                placeholder="(Todos. Filtros opcionais)",
            )
            meso_sel = st.multiselect(
                "Mesorregião",
                opts["meso"],
                key="hab_meso",
                placeholder="(Todos. Filtros opcionais)",
            )
            micro_sel = st.multiselect(
                "Microrregião",
                opts["micro"],
                key="hab_micro",
                placeholder="(Todos. Filtros opcionais)",
            )
            mun_sel = st.multiselect(
                "Município",
                opts["mun"],
                key="hab_mun",
                placeholder="(Todos. Filtros opcionais)",
            )
            ivs_sel = st.multiselect(
                "Município IVS",
                opts["ivs"],
                key="hab_ivs",
                placeholder="(Todos. Filtros opcionais)",
            )

        # -------- Perfil do Estabelecimento ------------------
        with st.expander("Filtros de Perfil do Estabelecimento", expanded=False):
            tipo_novo_sel = st.multiselect(
                "Tipo",
                opts["tipo_novo"],
                key="hab_tipo_novo",
                placeholder="(Todos. Filtros opcionais)",
            )
            subtipo_sel = st.multiselect(
                "Subtipo",
                opts["subtipo"],
                key="hab_subtipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            gestao_sel = st.multiselect(
                "Gestão",
                opts["gestao"],
                key="hab_gestao",
                placeholder="(Todos. Filtros opcionais)",
            )
            convenio_sel = st.multiselect(
                "Convênio SUS",
                opts["convenio"],
                key="hab_convenio",
                placeholder="(Todos. Filtros opcionais)",
            )
            nat_jur_sel = st.multiselect(
                "Natureza jurídica",
                opts["nat_jur"],
                key="hab_natjur",
                placeholder="(Todos. Filtros opcionais)",
            )
            status_sel = st.multiselect(
                "Status",
                opts["status"],
                key="hab_status",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ------------- Filtros de Habilitação ----------------
        with st.expander("Filtros de Habilitação", expanded=False):
            nivel_tipo_sel = st.multiselect(
                "Nível/Tipo de habilitação",
                opts["nivel_tipo"],
                key="hab_nivel_tipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            cat_hab_sel = st.multiselect(
                "Categoria da habilitação",
                opts["cat_hab"],
                key="hab_categoria",
                placeholder="(Todos. Filtros opcionais)",
            )
            no_hab_sel = st.multiselect(
                "Descrição da habilitação",
                opts["no_hab"],
                key="hab_nome_hab",
                placeholder="(Todos. Filtros opcionais)",
            )
            tag_hab_sel = st.multiselect(
                "Tag (agrupador)",
                opts["tag_hab"],
                key="hab_tag",
                placeholder="(Todos. Filtros opcionais)",
            )

    # ------------------------ Detectar mudanças ------------------------
    filter_values = {
        "ano_hab": ano_hab_sel,
        "mes_hab": mes_hab_sel,
        "ano_comp_ini": ano_comp_ini_sel,
        "mes_comp_ini": mes_comp_ini_sel,
        "ano_comp_fim": ano_comp_fim_sel,
        "mes_comp_fim": mes_comp_fim_sel,
        "ano_portaria": ano_portaria_sel,
        "mes_portaria": mes_portaria_sel,
        "uf": uf_sel,
        "reg_saude": reg_saude_sel,
        "meso": meso_sel,
        "micro": micro_sel,
        "mun": mun_sel,
        "ivs": ivs_sel,
        "tipo_novo": tipo_novo_sel,
        "subtipo": subtipo_sel,
        "gestao": gestao_sel,
        "convenio": convenio_sel,
        "nat_jur": nat_jur_sel,
        "status": status_sel,
        "nivel_tipo": nivel_tipo_sel,
        "cat_hab": cat_hab_sel,
        "no_hab": no_hab_sel,
        "tag_hab": tag_hab_sel,
    }
    filters_changed = any(did_filters_change(k, v) for k, v in filter_values.items())
    spinner = st.spinner("⏳ Atualizando resultados…") if filters_changed else DummySpinner()

    # WHERE + parâmetros para todas as queries agregadas
    where_sql_hab, bq_params_hab = build_where_hab(
        ano_hab_sel,
        mes_hab_sel,
        ano_comp_ini_sel,
        mes_comp_ini_sel,
        ano_comp_fim_sel,
        mes_comp_fim_sel,
        ano_portaria_sel,
        mes_portaria_sel,
        uf_sel,
        reg_saude_sel,
        meso_sel,
        micro_sel,
        mun_sel,
        ivs_sel,
        tipo_novo_sel,
        subtipo_sel,
        gestao_sel,
        convenio_sel,
        nat_jur_sel,
        status_sel,
        nivel_tipo_sel,
        cat_hab_sel,
        no_hab_sel,
        tag_hab_sel,
    )

    with spinner:
        # =========================================================
        # METRIC CARDS – agregados direto no BigQuery
        # =========================================================
        st.info("📏 Grandes números: visão rápida das habilitações com os filtros aplicados")

        kpis_hab = query_hab_kpis(where_sql_hab, bq_params_hab, hab_table_id)

        total_hab = int(kpis_hab["total_hab"] or 0)
        media_uf = kpis_hab["media_uf"] or 0
        media_regsaude = kpis_hab["media_reg_saude"] or 0
        media_estab = kpis_hab["media_estab"] or 0

        if total_hab == 0:
            st.warning("Nenhuma habilitação encontrada com os filtros selecionados.")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de habilitações", fmt_num(total_hab))

        with col2:
            st.metric("Média de habilitações por UF", fmt_num(media_uf))

        with col3:
            st.metric("Média por Região de Saúde", fmt_num(media_regsaude))

        with col4:
            st.metric("Média por Estabelecimento", fmt_num(media_estab))

        # =========================================================
        # Habilitações ativas vs encerradas (BigQuery)
        # =========================================================
        total_ativas, total_encerradas = query_hab_status(
            where_sql_hab, bq_params_hab, hab_table_id
        )
        total_validas = total_ativas + total_encerradas

        perc_enc = (total_encerradas / total_validas * 100) if total_validas > 0 else 0
        perc_ati = (total_ativas / total_validas * 100) if total_validas > 0 else 0

        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Habilitações Ativas", fmt_num(total_ativas))
        with colB:
            st.metric("Habilitações Encerradas", fmt_num(total_encerradas))
        with colC:
            st.metric("% Encerradas", f"{perc_enc:.1f}%")
        with colD:
            st.metric("% Ativas", f"{perc_ati:.1f}%")

        # ============================================================
        # 📊 GRÁFICOS – todos com agregação no BigQuery
        # ============================================================
        st.info("📊 Gráficos — resumo visual das habilitações filtradas")

        # Evolução anual
        with st.expander("Evolução anual do número de habilitações", expanded=True):
            df_ano = query_hab_por_ano(where_sql_hab, bq_params_hab, hab_table_id)
            if not df_ano.empty:
                fig_ano = bar_yoy_trend(
                    df=df_ano,
                    x="ano",
                    y="qtd_habilitacoes",
                    title="Evolução anual do número de habilitações",
                    x_is_year=True,
                    fill_missing_years=True,
                    show_ma=True,
                    ma_window=3,
                    show_mean=True,
                    show_trend=True,
                    legend_pos="bottom",
                    y_label="Quantidade de habilitações",
                )
                st.plotly_chart(fig_ano, use_container_width=True)
            else:
                st.info("Não há dados suficientes para montar a série anual.")

        # Categoria (CNES)
        with st.expander("Habilitações por categoria (CNES)", expanded=True):
            df_cat = query_hab_group(
                where_sql_hab,
                bq_params_hab,
                hab_table_id,
                col="referencia_habilitacao_no_categoria",
                null_label="(não informado)",
            )
            if not df_cat.empty:
                fig_cat = pareto_barh(
                    df_cat,
                    "categoria",
                    None,
                    "Distribuição de habilitações por categoria",
                    "Qtde de habilitações",
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Coluna de categoria de habilitação não encontrada na base.")

        # Tipo de habilitação (pizza)
        with st.expander("Distribuição por Tipo de Habilitação (Pizza)", expanded=True):
            df_tipo = query_hab_group(
                where_sql_hab,
                bq_params_hab,
                hab_table_id,
                col="habilitacao_nivel_habilitacao_tipo",
                null_label="(não informado)",
            )
            if not df_tipo.empty:
                fig_pie = pie_standard(
                    df=df_tipo,
                    names="categoria",
                    values="qtde",
                    title="Distribuição de Habilitações por Tipo",
                    top_n=12,
                    others_label="Outros",
                    hole=0.35,
                    legend_pos="below_title",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info(
                    "A coluna `habilitacao_nivel_habilitacao_tipo` não está disponível na base ou está vazia."
                )

        # Habilitações por UF
        with st.expander("Habilitações por UF", expanded=True):
            df_uf = query_hab_group(
                where_sql_hab,
                bq_params_hab,
                hab_table_id,
                col="ibge_no_uf",
                null_label="(não informado)",
            )
            if not df_uf.empty:
                fig_uf = bar_total_por_grupo(
                    df_uf,
                    grupo_col="categoria",
                    valor_col="qtde",
                    titulo="Quantidade de habilitações por UF",
                    x_label="UF",
                    y_label="Qtde de habilitações",
                    orientation="v",
                    top_n=10,
                )
                st.plotly_chart(fig_uf, use_container_width=True)
            else:
                st.info("Coluna `ibge_no_uf` não encontrada para os filtros aplicados.")

        # ============================================================
        # 📋 TABELA DESCRITIVA — Habilitações (com limite de 5.000)
        # ============================================================
        st.info("📋 Tabela descritiva das habilitações filtradas")

        cols_desc = [
            "habilitacao_ano",
            "habilitacao_mes",
            "habilitacao_tipo_habilitacao",
            "referencia_habilitacao_no_categoria",
            "referencia_habilitacao_no_habilitacao",
            "referencia_habilitacao_ds_tag",
            "habilitacao_ano_competencia_inicial",
            "habilitacao_mes_competencia_inicial",
            "habilitacao_ano_competencia_final",
            "habilitacao_mes_competencia_final",
            "habilitacao_portaria",
            "habilitacao_data_portaria",
            "habilitacao_quantidade_leitos",
            "habilitacao_id_estabelecimento_cnes",
            "estabelecimentos_nome_fantasia",
            "estabelecimentos_tipo_novo_do_estabelecimento",
            "estabelecimentos_subtipo_do_estabelecimento",
            "estabelecimentos_gestao",
            "estabelecimentos_status_do_estabelecimento",
            "estabelecimentos_convenio_sus",
            "estabelecimentos_categoria_natureza_juridica",
            "ibge_no_municipio",
            "ibge_no_regiao_saude",
            "ibge_no_microrregiao",
            "ibge_no_mesorregiao",
            "ibge_no_uf",
            "ibge_ivs",
        ]

        cols_ok = cols_desc  # assumindo schema coerente com a base

        max_rows_display = 5000
        n_total = total_hab  # já veio do KPI

        if n_total > max_rows_display:
            st.warning(
                f"A base filtrada possui {fmt_num(n_total)} linhas. "
                f"Por desempenho, a tabela abaixo mostra apenas as primeiras {fmt_num(max_rows_display)} linhas. "
                "O download também está limitado às mesmas linhas."
            )
        else:
            st.caption(f"A base filtrada possui {fmt_num(n_total)} linhas.")

        df_display = query_hab_detalhe(
            where_sql_hab,
            bq_params_hab,
            hab_table_id,
            cols_ok,
            limit_rows=max_rows_display,
        )

        if df_display.empty:
            st.warning(
                "Não foi possível carregar a tabela detalhada de habilitações com os filtros aplicados."
            )
        else:
            st.dataframe(
                df_display,
                use_container_width=True,
                height=500,
            )

            csv = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Baixar CSV (até 5.000 linhas)",
                csv,
                "habilitacoes_filtradas.csv",
                "text/csv",
                use_container_width=True,
            )

# =====================================================================
# 4) Cadastro Leitos
# =====================================================================
elif aba == "🛏️ Leitos":
    st.subheader("🛏️ Leitos")

    # ---------------------------------------------------------
    # Inicialização: BigQuery + opções de filtros (com spinner)
    # ---------------------------------------------------------
    with st.spinner("⏳ Carregando base de leitos..."):

        import pandas as pd
        from google.cloud import bigquery

        lei_table_id = TABLES["leitos"]

        # ----------------------- Cliente BQ -------------------
        @st.cache_resource
        def get_bq_client_lei():
            return bigquery.Client()

        # ----------------------- WHERE dinâmico ---------------
        def build_where_lei(
            ano_sel,
            mes_sel,
            uf_sel,
            reg_saude_sel,
            meso_sel,
            micro_sel,
            mun_sel,
            ivs_sel,
            tipo_novo_sel,
            tipo_sel,
            subtipo_sel,
            gestao_sel,
            convenio_sel,
            natureza_sel,
            status_sel,
            # booleanos complexos / onco
            onco_cacon_sel,
            onco_unacon_sel,
            onco_radio_sel,
            onco_quimio_sel,
            hab_onco_cir_sel,
            uti_adulto_sel,
            uti_ped_sel,
            uti_neo_sel,
            uti_cor_sel,
            ucin_sel,
            uti_queim_sel,
            caps_psiq_sel,
            rehab_cer_sel,
            cardio_ac_sel,
            nutricao_sel,
            odonto_ceo_sel,
            # tipos de leito
            esp_sel,
            tipo_leito_sel,
            tipo_nome_sel,
            ref_no_leito_sel,
        ):
            """
            Monta WHERE dinâmico + parâmetros BigQuery
            para todos os filtros da aba Leitos.
            """
            clauses = ["1=1"]
            params: list[bigquery.QueryParameter] = []

            def add_in_array(col: str, param_name: str, values):
                if values:
                    vals = [str(v) for v in values]
                    clauses.append(f"CAST({col} AS STRING) IN UNNEST(@{param_name})")
                    params.append(
                        bigquery.ArrayQueryParameter(param_name, "STRING", vals)
                    )

            def add_bool(col: str, param_name: str, values):
                """
                values é lista com 'Sim'/'Não'.
                Convertemos para 1/0 e comparamos com SAFE_CAST(col AS INT64)
                para funcionar com bool/int.
                """
                if not values:
                    return
                allowed = []
                if "Sim" in values:
                    allowed.append(1)
                if "Não" in values:
                    allowed.append(0)
                if not allowed:
                    return
                clauses.append(
                    f"SAFE_CAST({col} AS INT64) IN UNNEST(@{param_name})"
                )
                params.append(
                    bigquery.ArrayQueryParameter(param_name, "INT64", allowed)
                )

            # Período / território
            add_in_array("leitos_ano", "ano", ano_sel)
            add_in_array("leitos_mes", "mes", mes_sel)
            add_in_array("ibge_no_uf", "uf", uf_sel)
            add_in_array("ibge_no_regiao_saude", "reg_saude", reg_saude_sel)
            add_in_array("ibge_no_mesorregiao", "meso", meso_sel)
            add_in_array("ibge_no_microrregiao", "micro", micro_sel)
            add_in_array("ibge_no_municipio", "mun", mun_sel)
            add_in_array("ibge_ivs", "ivs", ivs_sel)

            # Perfil Estabelecimento
            add_in_array(
                "estabelecimentos_tipo_novo_do_estabelecimento",
                "tipo_novo",
                tipo_novo_sel,
            )
            add_in_array(
                "estabelecimentos_tipo_do_estabelecimento",
                "tipo",
                tipo_sel,
            )
            add_in_array(
                "estabelecimentos_subtipo_do_estabelecimento",
                "subtipo",
                subtipo_sel,
            )
            add_in_array(
                "estabelecimentos_gestao",
                "gestao",
                gestao_sel,
            )
            add_in_array(
                "estabelecimentos_convenio_sus",
                "convenio",
                convenio_sel,
            )
            add_in_array(
                "estabelecimentos_categoria_natureza_juridica",
                "natureza",
                natureza_sel,
            )
            add_in_array(
                "estabelecimentos_status_do_estabelecimento",
                "status_estab",
                status_sel,
            )

            # Booleanos (onco / complexos / UTI / etc.)
            add_bool("onco_cacon", "onco_cacon", onco_cacon_sel)
            add_bool("onco_unacon", "onco_unacon", onco_unacon_sel)
            add_bool("onco_radioterapia", "onco_radio", onco_radio_sel)
            add_bool("onco_quimioterapia", "onco_quimio", onco_quimio_sel)
            add_bool(
                "habilitacao_agrupado_onco_cirurgica",
                "onco_cir",
                hab_onco_cir_sel,
            )

            add_bool("habilitacao_agrupado_uti_adulto", "uti_adulto", uti_adulto_sel)
            add_bool("habilitacao_agrupado_uti_pediatrica", "uti_ped", uti_ped_sel)
            add_bool("habilitacao_agrupado_uti_neonatal", "uti_neo", uti_neo_sel)
            add_bool(
                "habilitacao_agrupado_uti_coronariana",
                "uti_cor",
                uti_cor_sel,
            )
            add_bool("habilitacao_agrupado_ucin", "ucin", ucin_sel)
            add_bool(
                "habilitacao_agrupado_uti_queimados",
                "uti_queim",
                uti_queim_sel,
            )
            add_bool(
                "habilitacao_agrupado_saude_mental_caps_psiq",
                "caps_psiq",
                caps_psiq_sel,
            )
            add_bool(
                "habilitacao_agrupado_reabilitacao_cer",
                "rehab_cer",
                rehab_cer_sel,
            )
            add_bool(
                "habilitacao_agrupado_cardio_alta_complex",
                "cardio_ac",
                cardio_ac_sel,
            )
            add_bool(
                "habilitacao_agrupado_nutricao",
                "nutricao",
                nutricao_sel,
            )
            add_bool(
                "habilitacao_agrupado_odontologia_ceo",
                "odonto_ceo",
                odonto_ceo_sel,
            )

            # Tipos de leito
            add_in_array(
                "leitos_tipo_especialidade_leito",
                "esp",
                esp_sel,
            )
            add_in_array(
                "leitos_tipo_leito",
                "tipo_leito",
                tipo_leito_sel,
            )
            add_in_array(
                "leitos_tipo_leito_nome",
                "tipo_nome",
                tipo_nome_sel,
            )
            add_in_array(
                "referencia_especialidade_no_leito",
                "ref_no_leito",
                ref_no_leito_sel,
            )

            where_sql = "WHERE " + " AND ".join(clauses)
            return where_sql, params

        # ----------------------- KPIs -------------------------
        def query_lei_kpis(where_sql: str, params, table_id: str):
            """
            KPIs direto do BigQuery:
              - total_registros (linhas)
              - total_leitos (soma leitos_quantidade_total)
              - média de leitos por UF
              - média de leitos por Região de Saúde
              - média de leitos por Estabelecimento
            """
            client = get_bq_client_lei()
            sql = f"""
            WITH base AS (
              SELECT
                leitos_id_estabelecimento_cnes,
                ibge_no_uf,
                ibge_no_regiao_saude,
                COALESCE(CAST(leitos_quantidade_total AS FLOAT64), 0) AS qtd
              FROM `{table_id}`
              {where_sql}
            ),
            uf_agg AS (
              SELECT ibge_no_uf, SUM(qtd) AS leitos_uf
              FROM base
              GROUP BY ibge_no_uf
            ),
            reg_agg AS (
              SELECT ibge_no_regiao_saude, SUM(qtd) AS leitos_reg
              FROM base
              GROUP BY ibge_no_regiao_saude
            ),
            estab_agg AS (
              SELECT leitos_id_estabelecimento_cnes, SUM(qtd) AS leitos_estab
              FROM base
              GROUP BY leitos_id_estabelecimento_cnes
            )
            SELECT
              (SELECT COUNT(*) FROM base) AS total_registros,
              (SELECT SUM(qtd) FROM base) AS total_leitos,
              (SELECT AVG(leitos_uf) FROM uf_agg) AS media_uf,
              (SELECT AVG(leitos_reg) FROM reg_agg) AS media_reg_saude,
              (SELECT AVG(leitos_estab) FROM estab_agg) AS media_estab;
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            df = job.to_dataframe()
            return df.iloc[0] if not df.empty else None

        # ----------------------- Agregados genéricos ---------
        def query_lei_group_tipo_nome(where_sql: str, params, table_id: str):
            """
            Soma de leitos por leitos_tipo_leito_nome.
            """
            client = get_bq_client_lei()
            sql = f"""
            SELECT
              leitos_tipo_leito_nome,
              SUM(COALESCE(CAST(leitos_quantidade_total AS FLOAT64), 0)) AS qtd_leitos
            FROM `{table_id}`
            {where_sql}
            GROUP BY leitos_tipo_leito_nome
            HAVING leitos_tipo_leito_nome IS NOT NULL
            ORDER BY qtd_leitos DESC
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            return job.to_dataframe()

        def query_lei_sus_contrat_total(where_sql: str, params, table_id: str):
            """
            Totais de leitos SUS, contratados e total.
            """
            client = get_bq_client_lei()
            sql = f"""
            SELECT
              SUM(COALESCE(CAST(leitos_quantidade_sus AS FLOAT64), 0)) AS total_sus,
              SUM(COALESCE(CAST(leitos_quantidade_contratado AS FLOAT64), 0)) AS total_contratado,
              SUM(COALESCE(CAST(leitos_quantidade_total AS FLOAT64), 0)) AS total_total
            FROM `{table_id}`
            {where_sql}
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            df = job.to_dataframe()
            if df.empty:
                return 0, 0, 0
            row = df.iloc[0]
            return (
                float(row["total_sus"] or 0),
                float(row["total_contratado"] or 0),
                float(row["total_total"] or 0),
            )

        # ----------------------- Tabela detalhada ------------
        def query_lei_detalhe(
            where_sql: str,
            params,
            table_id: str,
            cols: list[str],
            limit_rows: int = 5000,
        ) -> pd.DataFrame:
            """
            Busca dados detalhados dos leitos com os filtros aplicados,
            limitado a `limit_rows` para exibição/CSV.
            """
            if not cols:
                return pd.DataFrame()
            client = get_bq_client_lei()
            cols_sql = ", ".join(cols)
            sql = f"""
            SELECT
              {cols_sql}
            FROM `{table_id}`
            {where_sql}
            LIMIT {limit_rows}
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            return job.to_dataframe()

        # ----------------------- Opções de filtros (DISTINCT) -
        @st.cache_data(show_spinner=False)
        def _opts_lei(col: str) -> list[str]:
            """
            Busca valores distintos de uma coluna na tabela de leitos.
            Usado só para montar as opções dos filtros (sem aplicar WHERE).
            """
            client = get_bq_client_lei()
            sql = f"""
            SELECT DISTINCT CAST({col} AS STRING) AS val
            FROM `{lei_table_id}`
            WHERE {col} IS NOT NULL
            ORDER BY val
            """
            try:
                df = client.query(sql).to_dataframe()
                return df["val"].dropna().tolist()
            except Exception:
                return []

        # ----------------------- Carregar todas as opções ----
        opts = {
            # Período
            "ano": _opts_lei("leitos_ano"),
            "mes": _opts_lei("leitos_mes"),

            # Território
            "uf": _opts_lei("ibge_no_uf"),
            "reg_saude": _opts_lei("ibge_no_regiao_saude"),
            "meso": _opts_lei("ibge_no_mesorregiao"),
            "micro": _opts_lei("ibge_no_microrregiao"),
            "mun": _opts_lei("ibge_no_municipio"),
            "ivs": _opts_lei("ibge_ivs"),

            # Perfil do estabelecimento
            "tipo_novo": _opts_lei("estabelecimentos_tipo_novo_do_estabelecimento"),
            "tipo": _opts_lei("estabelecimentos_tipo_do_estabelecimento"),
            "subtipo": _opts_lei("estabelecimentos_subtipo_do_estabelecimento"),
            "gestao": _opts_lei("estabelecimentos_gestao"),
            "convenio": _opts_lei("estabelecimentos_convenio_sus"),
            "natureza": _opts_lei("estabelecimentos_categoria_natureza_juridica"),
            "status": _opts_lei("estabelecimentos_status_do_estabelecimento"),

            # Tipos de leito
            "esp": _opts_lei("leitos_tipo_especialidade_leito"),
            "tipo_leito": _opts_lei("leitos_tipo_leito"),
            "tipo_nome": _opts_lei("leitos_tipo_leito_nome"),
            "ref_no_leito": _opts_lei("referencia_especialidade_no_leito"),
        }

    # =========================================================
    # SIDEBAR DE FILTROS
    # =========================================================
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Leitos hospitalares")
        st.caption("Use os agrupadores abaixo para refinar o cadastro de leitos.")

        # ----------------- Período --------------------
        with st.expander("Fitros de Período", expanded=False):
            ano_sel = st.multiselect(
                "Ano",
                opts["ano"],
                key="lei_ano",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_sel = st.multiselect(
                "Mês",
                opts["mes"],
                key="lei_mes",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ----------------- Território ------------------
        with st.expander("Fitros de Território", expanded=False):
            uf_sel = st.multiselect(
                "UF",
                opts["uf"],
                key="lei_uf",
                placeholder="(Todos. Filtros opcionais)",
            )
            reg_saude_sel = st.multiselect(
                "Região de Saúde",
                opts["reg_saude"],
                key="lei_regsaude",
                placeholder="(Todos. Filtros opcionais)",
            )
            meso_sel = st.multiselect(
                "Mesorregião",
                opts["meso"],
                key="lei_meso",
                placeholder="(Todos. Filtros opcionais)",
            )
            micro_sel = st.multiselect(
                "Microrregião",
                opts["micro"],
                key="lei_micro",
                placeholder="(Todos. Filtros opcionais)",
            )
            mun_sel = st.multiselect(
                "Município",
                opts["mun"],
                key="lei_mun",
                placeholder="(Todos. Filtros opcionais)",
            )
            ivs_sel = st.multiselect(
                "Município IVS",
                opts["ivs"],
                key="lei_ivs",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Perfil do Estabelecimento ----------
        with st.expander("Fitros de Perfil do Estabelecimento", expanded=False):
            tipo_novo_sel = st.multiselect(
                "Tipo (novo)",
                opts["tipo_novo"],
                key="lei_tipo_novo",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_sel = st.multiselect(
                "Tipo do estabelecimento",
                opts["tipo"],
                key="lei_tipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            subtipo_sel = st.multiselect(
                "Subtipo",
                opts["subtipo"],
                key="lei_subtipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            gestao_sel = st.multiselect(
                "Gestão",
                opts["gestao"],
                key="lei_gestao",
                placeholder="(Todos. Filtros opcionais)",
            )
            convenio_sel = st.multiselect(
                "Convênio SUS",
                opts["convenio"],
                key="lei_convenio",
                placeholder="(Todos. Filtros opcionais)",
            )
            natureza_sel = st.multiselect(
                "Natureza Jurídica",
                opts["natureza"],
                key="lei_natjur",
                placeholder="(Todos. Filtros opcionais)",
            )
            status_sel = st.multiselect(
                "Status",
                opts["status"],
                key="lei_status",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ------------- Complexos / Oncologia -----------
        with st.expander("Fitros de Complexos / Oncologia", expanded=False):
            def bool_multiselect(label, key):
                return st.multiselect(
                    label,
                    ["Sim", "Não"],
                    key=key,
                    placeholder="(Todos. Filtros opcionais)",
                )

            onco_cacon_sel = bool_multiselect("CACON", "lei_onco_cacon")
            onco_unacon_sel = bool_multiselect("UNACON", "lei_onco_unacon")
            onco_radio_sel = bool_multiselect("Radioterapia", "lei_onco_radio")
            onco_quimio_sel = bool_multiselect("Quimioterapia", "lei_onco_quimio")
            hab_onco_cir_sel = bool_multiselect("Onco Cirúrgica", "lei_onco_cir")

            uti_adulto_sel = bool_multiselect("UTI Adulto", "lei_uti_adulto")
            uti_ped_sel = bool_multiselect("UTI Pediátrica", "lei_uti_ped")
            uti_neo_sel = bool_multiselect("UTI Neonatal", "lei_uti_neo")
            uti_cor_sel = bool_multiselect("UTI Coronariana", "lei_uti_cor")
            ucin_sel = bool_multiselect("UCIN", "lei_ucin")
            uti_queim_sel = bool_multiselect("UTI Queimados", "lei_uti_queim")
            caps_psiq_sel = bool_multiselect(
                "Saúde Mental CAPS/Psiq", "lei_caps_psiq"
            )
            rehab_cer_sel = bool_multiselect("Reabilitação CER", "lei_rehab_cer")
            cardio_ac_sel = bool_multiselect(
                "Cardio Alta Complex.", "lei_cardio_ac"
            )
            nutricao_sel = bool_multiselect("Nutrição", "lei_nutricao")
            odonto_ceo_sel = bool_multiselect(
                "Odontologia CEO", "lei_odonto_ceo"
            )

        # ----------------- Tipos de Leito -----------------
        with st.expander("Fitros de Tipos de Leito", expanded=False):
            esp_sel = st.multiselect(
                "Especialidade do leito",
                opts["esp"],
                key="lei_esp",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_leito_sel = st.multiselect(
                "Tipo de leito (código)",
                opts["tipo_leito"],
                key="lei_tipo_leito",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_nome_sel = st.multiselect(
                "Tipo de leito (nome)",
                opts["tipo_nome"],
                key="lei_tipo_nome",
                placeholder="(Todos. Filtros opcionais)",
            )
            ref_no_leito_sel = st.multiselect(
                "Descrição CNES do leito",
                opts["ref_no_leito"],
                key="lei_no_leito",
                placeholder="(Todos. Filtros opcionais)",
            )

    # =========================================================
    # Detectar mudanças de filtros + WHERE/params
    # =========================================================
    filter_values = {
        "ano": ano_sel,
        "mes": mes_sel,
        "uf": uf_sel,
        "reg_saude": reg_saude_sel,
        "meso": meso_sel,
        "micro": micro_sel,
        "mun": mun_sel,
        "ivs": ivs_sel,
        "tipo_novo": tipo_novo_sel,
        "tipo": tipo_sel,
        "subtipo": subtipo_sel,
        "gestao": gestao_sel,
        "convenio": convenio_sel,
        "natureza": natureza_sel,
        "status": status_sel,
        "onco_cacon": onco_cacon_sel,
        "onco_unacon": onco_unacon_sel,
        "onco_radio": onco_radio_sel,
        "onco_quimio": onco_quimio_sel,
        "hab_onco_cir": hab_onco_cir_sel,
        "uti_adulto": uti_adulto_sel,
        "uti_ped": uti_ped_sel,
        "uti_neo": uti_neo_sel,
        "uti_cor": uti_cor_sel,
        "ucin": ucin_sel,
        "uti_queim": uti_queim_sel,
        "caps_psiq": caps_psiq_sel,
        "rehab_cer": rehab_cer_sel,
        "cardio_ac": cardio_ac_sel,
        "nutricao": nutricao_sel,
        "odonto_ceo": odonto_ceo_sel,
        "esp": esp_sel,
        "tipo_leito": tipo_leito_sel,
        "tipo_nome": tipo_nome_sel,
        "ref_no_leito": ref_no_leito_sel,
    }

    filters_changed = any(did_filters_change(k, v) for k, v in filter_values.items())
    spinner = st.spinner("⏳ Atualizando resultados…") if filters_changed else DummySpinner()

    where_sql_lei, bq_params_lei = build_where_lei(
        ano_sel,
        mes_sel,
        uf_sel,
        reg_saude_sel,
        meso_sel,
        micro_sel,
        mun_sel,
        ivs_sel,
        tipo_novo_sel,
        tipo_sel,
        subtipo_sel,
        gestao_sel,
        convenio_sel,
        natureza_sel,
        status_sel,
        onco_cacon_sel,
        onco_unacon_sel,
        onco_radio_sel,
        onco_quimio_sel,
        hab_onco_cir_sel,
        uti_adulto_sel,
        uti_ped_sel,
        uti_neo_sel,
        uti_cor_sel,
        ucin_sel,
        uti_queim_sel,
        caps_psiq_sel,
        rehab_cer_sel,
        cardio_ac_sel,
        nutricao_sel,
        odonto_ceo_sel,
        esp_sel,
        tipo_leito_sel,
        tipo_nome_sel,
        ref_no_leito_sel,
    )

    # =========================================================
    # Função helper para limitar categorias nos gráficos
    # =========================================================
    def col_top_n(df, col, top_n=40, outros_label="Outros"):
        if col not in df.columns:
            return df
        vc = df[col].value_counts(dropna=False)
        if len(vc) <= top_n:
            return df
        top_vals = set(vc.head(top_n).index)
        df2 = df.copy()
        df2[col] = df2[col].where(df2[col].isin(top_vals), outros_label)
        return df2

    # =========================================================
    # BLOCO PRINCIPAL COM SPINNER DE RESULTADOS
    # =========================================================
    with spinner:
        # ======================== KPIs =======================
        st.info("📏 Grandes números: visão rápida dos leitos filtrados")

        kpis_lei = query_lei_kpis(where_sql_lei, bq_params_lei, lei_table_id)

        if kpis_lei is None or (kpis_lei["total_registros"] or 0) == 0:
            st.warning("Nenhum leito encontrado com os filtros selecionados.")
            st.stop()

        total_reg = int(kpis_lei["total_registros"] or 0)
        total_leitos = float(kpis_lei["total_leitos"] or 0)
        media_uf = kpis_lei["media_uf"] or 0
        media_regsaude = kpis_lei["media_reg_saude"] or 0
        media_estab = kpis_lei["media_estab"] or 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de leitos (somados)", fmt_num(total_leitos))

        with col2:
            st.metric("Média de leitos por UF", fmt_num(media_uf))

        with col3:
            st.metric("Média de leitos por Reg. Saúde", fmt_num(media_regsaude))

        with col4:
            st.metric("Média de leitos por Estabelecimento", fmt_num(media_estab))

        # ======================== GRÁFICOS ====================
        st.info("📊 Gráficos — resumo visual dos leitos filtrados")

        # 1) Leitos por tipo de leito (nome)
        with st.expander("Leitos por tipo de leito (nome)", expanded=True):
            df_tipo = query_lei_group_tipo_nome(
                where_sql_lei, bq_params_lei, lei_table_id
            )
            if not df_tipo.empty:
                df_tipo = df_tipo.sort_values("qtd_leitos", ascending=False)
                df_tipo = col_top_n(df_tipo, "leitos_tipo_leito_nome", top_n=30)

                fig_tipo = bar_total_por_grupo(
                    df_tipo,
                    grupo_col="leitos_tipo_leito_nome",
                    valor_col="qtd_leitos",
                    titulo="Distribuição de leitos por tipo (nome)",
                    x_label="Quantidade de leitos",
                    y_label="Tipo de leito",
                    orientation="h",
                )
                st.plotly_chart(fig_tipo, use_container_width=True)
            else:
                st.info(
                    "Colunas de tipo de leito ou quantidade total não estão disponíveis."
                )

        # 2) Leitos SUS x Contratado x Total
        with st.expander("Leitos SUS, contratados e total", expanded=False):
            total_sus, total_cont, total_total = query_lei_sus_contrat_total(
                where_sql_lei, bq_params_lei, lei_table_id
            )

            if total_sus + total_cont + total_total > 0:
                df_sus = pd.DataFrame(
                    {
                        "categoria": [
                            "Leitos SUS",
                            "Leitos contratados",
                            "Leitos totais",
                        ],
                        "qtd": [total_sus, total_cont, total_total],
                    }
                )

                fig_sus = bar_total_por_grupo(
                    df_sus,
                    grupo_col="categoria",
                    valor_col="qtd",
                    titulo="Leitos SUS, contratados e total",
                    x_label="Categoria",
                    y_label="Quantidade de leitos",
                    orientation="v",
                )
                st.plotly_chart(fig_sus, use_container_width=True)
            else:
                st.info(
                    "Não há dados de quantidade de leitos SUS/contratados/total disponíveis para os filtros selecionados."
                )

        # ======================== TABELA DESCRITIVA ===========
        st.info("📋 Tabela descritiva dos leitos filtrados")

        cols_desc = [
            "leitos_ano",
            "leitos_mes",
            "ibge_no_municipio",
            "ibge_no_uf",
            "leitos_tipo_especialidade_leito",
            "leitos_tipo_leito",
            "leitos_tipo_leito_nome",
            "referencia_especialidade_no_leito",
            "leitos_quantidade_total",
            "leitos_quantidade_sus",
            "leitos_quantidade_contratado",
            "leitos_id_estabelecimento_cnes",
            "estabelecimentos_nome_fantasia",
            "estabelecimentos_tipo_novo_do_estabelecimento",
            "estabelecimentos_subtipo_do_estabelecimento",
            "estabelecimentos_gestao",
            "estabelecimentos_status_do_estabelecimento",
            "estabelecimentos_convenio_sus",
            "estabelecimentos_categoria_natureza_juridica",
            "onco_cacon",
            "onco_unacon",
            "onco_radioterapia",
            "onco_quimioterapia",
        ]

        cols_ok = cols_desc  # assumindo schema coerente

        max_rows_display = 5000
        n_total = total_reg

        if n_total > max_rows_display:
            st.warning(
                f"A base filtrada possui {fmt_num(n_total)} linhas. "
                f"Por desempenho, a tabela abaixo mostra apenas as primeiras {fmt_num(max_rows_display)} linhas. "
                "O download também está limitado às mesmas linhas."
            )
        else:
            st.caption(f"A base filtrada possui {fmt_num(n_total)} linhas.")

        df_display = query_lei_detalhe(
            where_sql_lei,
            bq_params_lei,
            lei_table_id,
            cols_ok,
            limit_rows=max_rows_display,
        )

        if df_display.empty:
            st.warning(
                "Não foi possível carregar a tabela detalhada de leitos com os filtros aplicados."
            )
        else:
            st.dataframe(df_display, use_container_width=True, height=500)

            csv = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Baixar CSV (até 5.000 linhas)",
                csv,
                "leitos_filtrados.csv",
                "text/csv",
                use_container_width=True,
            )

# =====================================================================
# X) Cadastro Equipamentos
# =====================================================================
elif aba == "🧰 Equipamentos":
    st.subheader("🧰 Equipamentos")

    # ---------------------------------------------------------
    # Inicialização: BigQuery + opções de filtros (com spinner)
    # ---------------------------------------------------------
    with st.spinner("⏳ Carregando base de equipamentos..."):

        import pandas as pd
        from google.cloud import bigquery

        eqp_table_id = TABLES["equipamentos"]

        # ----------------------- Cliente BQ -------------------
        @st.cache_resource
        def get_bq_client_eqp():
            return bigquery.Client()

        # ----------------------- WHERE dinâmico ---------------
        def build_where_eqp(
            ano_sel,
            mes_sel,
            uf_sel,
            reg_saude_sel,
            meso_sel,
            micro_sel,
            mun_sel,
            ivs_sel,
            tipo_novo_sel,
            tipo_sel,
            subtipo_sel,
            gestao_sel,
            convenio_sel,
            natureza_sel,
            status_sel,
            # booleanos complexos / onco
            onco_cacon_sel,
            onco_unacon_sel,
            onco_radio_sel,
            onco_quimio_sel,
            hab_onco_cir_sel,
            uti_adulto_sel,
            uti_ped_sel,
            uti_neo_sel,
            uti_cor_sel,
            ucin_sel,
            uti_queim_sel,
            caps_psiq_sel,
            rehab_cer_sel,
            cardio_ac_sel,
            nutricao_sel,
            odonto_ceo_sel,
            # tipos de equipamento
            tipo_eqp_sel,
            id_eqp_sel,
        ):
            """
            Monta WHERE dinâmico + parâmetros BigQuery
            para todos os filtros da aba Equipamentos.
            """
            clauses = ["1=1"]
            params: list[bigquery.QueryParameter] = []

            def add_in_array(col: str, param_name: str, values):
                if values:
                    vals = [str(v) for v in values]
                    clauses.append(f"CAST({col} AS STRING) IN UNNEST(@{param_name})")
                    params.append(
                        bigquery.ArrayQueryParameter(param_name, "STRING", vals)
                    )

            def add_bool(col: str, param_name: str, values):
                """
                values é lista com 'Sim'/'Não'.
                Convertemos para 1/0 e comparamos com SAFE_CAST(col AS INT64)
                para funcionar com bool/int.
                """
                if not values:
                    return
                allowed = []
                if "Sim" in values:
                    allowed.append(1)
                if "Não" in values:
                    allowed.append(0)
                if not allowed:
                    return
                clauses.append(
                    f"SAFE_CAST({col} AS INT64) IN UNNEST(@{param_name})"
                )
                params.append(
                    bigquery.ArrayQueryParameter(param_name, "INT64", allowed)
                )

            # Período / território
            add_in_array("equipamentos_ano", "ano", ano_sel)
            add_in_array("equipamentos_mes", "mes", mes_sel)
            add_in_array("ibge_no_uf", "uf", uf_sel)
            add_in_array("ibge_no_regiao_saude", "reg_saude", reg_saude_sel)
            add_in_array("ibge_no_mesorregiao", "meso", meso_sel)
            add_in_array("ibge_no_microrregiao", "micro", micro_sel)
            add_in_array("ibge_no_municipio", "mun", mun_sel)
            add_in_array("ibge_ivs", "ivs", ivs_sel)

            # Perfil Estabelecimento
            add_in_array(
                "estabelecimentos_tipo_novo_do_estabelecimento",
                "tipo_novo",
                tipo_novo_sel,
            )
            add_in_array(
                "estabelecimentos_tipo_do_estabelecimento",
                "tipo",
                tipo_sel,
            )
            add_in_array(
                "estabelecimentos_subtipo_do_estabelecimento",
                "subtipo",
                subtipo_sel,
            )
            add_in_array(
                "estabelecimentos_gestao",
                "gestao",
                gestao_sel,
            )
            add_in_array(
                "estabelecimentos_convenio_sus",
                "convenio",
                convenio_sel,
            )
            add_in_array(
                "estabelecimentos_categoria_natureza_juridica",
                "natureza",
                natureza_sel,
            )
            add_in_array(
                "estabelecimentos_status_do_estabelecimento",
                "status_estab",
                status_sel,
            )

            # Booleanos (onco / complexos / UTI / etc.)
            add_bool("onco_cacon", "onco_cacon", onco_cacon_sel)
            add_bool("onco_unacon", "onco_unacon", onco_unacon_sel)
            add_bool("onco_radioterapia", "onco_radio", onco_radio_sel)
            add_bool("onco_quimioterapia", "onco_quimio", onco_quimio_sel)
            add_bool(
                "habilitacao_agrupado_onco_cirurgica",
                "onco_cir",
                hab_onco_cir_sel,
            )

            add_bool("habilitacao_agrupado_uti_adulto", "uti_adulto", uti_adulto_sel)
            add_bool("habilitacao_agrupado_uti_pediatrica", "uti_ped", uti_ped_sel)
            add_bool("habilitacao_agrupado_uti_neonatal", "uti_neo", uti_neo_sel)
            add_bool(
                "habilitacao_agrupado_uti_coronariana",
                "uti_cor",
                uti_cor_sel,
            )
            add_bool("habilitacao_agrupado_ucin", "ucin", ucin_sel)
            add_bool(
                "habilitacao_agrupado_uti_queimados",
                "uti_queim",
                uti_queim_sel,
            )
            add_bool(
                "habilitacao_agrupado_saude_mental_caps_psiq",
                "caps_psiq",
                caps_psiq_sel,
            )
            add_bool(
                "habilitacao_agrupado_reabilitacao_cer",
                "rehab_cer",
                rehab_cer_sel,
            )
            add_bool(
                "habilitacao_agrupado_cardio_alta_complex",
                "cardio_ac",
                cardio_ac_sel,
            )
            add_bool(
                "habilitacao_agrupado_nutricao",
                "nutricao",
                nutricao_sel,
            )
            add_bool(
                "habilitacao_agrupado_odontologia_ceo",
                "odonto_ceo",
                odonto_ceo_sel,
            )

            # Tipos de equipamento
            add_in_array(
                "equipamentos_tipo_equipamento",
                "tipo_eqp",
                tipo_eqp_sel,
            )
            add_in_array(
                "equipamentos_id_equipamento",
                "id_eqp",
                id_eqp_sel,
            )

            where_sql = "WHERE " + " AND ".join(clauses)
            return where_sql, params

        # ----------------------- KPIs -------------------------
        def query_eqp_kpis(where_sql: str, params, table_id: str):
            """
            KPIs direto do BigQuery:
              - total_registros (linhas)
              - total_eqp (soma equipamentos_quantidade)
              - total_ativos (soma equipamentos_quantidade_ativos)
              - média de equipamentos por UF
              - média de equipamentos por Estabelecimento
            """
            client = get_bq_client_eqp()
            sql = f"""
            WITH base AS (
              SELECT
                equipamentos_id_estabelecimento_cnes,
                ibge_no_uf,
                COALESCE(CAST(equipamentos_quantidade AS FLOAT64), 0) AS qtd,
                COALESCE(CAST(equipamentos_quantidade_ativos AS FLOAT64), 0) AS qtd_ativos
              FROM `{table_id}`
              {where_sql}
            ),
            uf_agg AS (
              SELECT ibge_no_uf, SUM(qtd) AS eqp_uf
              FROM base
              GROUP BY ibge_no_uf
            ),
            estab_agg AS (
              SELECT equipamentos_id_estabelecimento_cnes, SUM(qtd) AS eqp_estab
              FROM base
              GROUP BY equipamentos_id_estabelecimento_cnes
            )
            SELECT
              (SELECT COUNT(*) FROM base) AS total_registros,
              (SELECT SUM(qtd) FROM base) AS total_eqp,
              (SELECT SUM(qtd_ativos) FROM base) AS total_ativos,
              (SELECT AVG(eqp_uf) FROM uf_agg) AS media_uf,
              (SELECT AVG(eqp_estab) FROM estab_agg) AS media_estab;
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            df = job.to_dataframe()
            return df.iloc[0] if not df.empty else None

        # ----------------------- Agregado por tipo ------------
        def query_eqp_group_tipo(where_sql: str, params, table_id: str):
            """
            Soma de equipamentos por equipamentos_tipo_equipamento.
            """
            client = get_bq_client_eqp()
            sql = f"""
            SELECT
              equipamentos_tipo_equipamento,
              SUM(COALESCE(CAST(equipamentos_quantidade AS FLOAT64), 0)) AS qtd_equipamentos
            FROM `{table_id}`
            {where_sql}
            GROUP BY equipamentos_tipo_equipamento
            HAVING equipamentos_tipo_equipamento IS NOT NULL
            ORDER BY qtd_equipamentos DESC
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            return job.to_dataframe()

        # ----------------------- Disponíveis x indisponíveis SUS
        def query_eqp_sus_disp(where_sql: str, params, table_id: str):
            """
            Totais de equipamentos disponíveis e indisponíveis para o SUS.
            Considera qualquer valor > 0 nos indicadores como "marcado".
            """
            client = get_bq_client_eqp()
            sql = f"""
            SELECT
              SUM(
                CASE WHEN SAFE_CAST(equipamentos_indicador_disponivel_sus AS FLOAT64) > 0
                     THEN COALESCE(CAST(equipamentos_quantidade AS FLOAT64), 0)
                     ELSE 0 END
              ) AS total_disp,
              SUM(
                CASE WHEN SAFE_CAST(equipamentos_indicador_indisponivel_sus AS FLOAT64) > 0
                     THEN COALESCE(CAST(equipamentos_quantidade AS FLOAT64), 0)
                     ELSE 0 END
              ) AS total_indisp
            FROM `{table_id}`
            {where_sql}
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            df = job.to_dataframe()
            if df.empty:
                return 0, 0
            row = df.iloc[0]
            return float(row["total_disp"] or 0), float(row["total_indisp"] or 0)

        # ----------------------- Tabela detalhada ------------
        def query_eqp_detalhe(
            where_sql: str,
            params,
            table_id: str,
            cols: list[str],
            limit_rows: int = 5000,
        ) -> pd.DataFrame:
            """
            Busca dados detalhados dos equipamentos com os filtros aplicados,
            limitado a `limit_rows` para exibição/CSV.
            """
            if not cols:
                return pd.DataFrame()
            client = get_bq_client_eqp()
            cols_sql = ", ".join(cols)
            sql = f"""
            SELECT
              {cols_sql}
            FROM `{table_id}`
            {where_sql}
            LIMIT {limit_rows}
            """
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(query_parameters=params),
            )
            return job.to_dataframe()

        # ----------------------- Opções de filtros (DISTINCT) -
        @st.cache_data(show_spinner=False)
        def _opts_eqp(col: str) -> list[str]:
            """
            Busca valores distintos de uma coluna na tabela de equipamentos.
            Usado só para montar as opções dos filtros (sem aplicar WHERE).
            """
            client = get_bq_client_eqp()
            sql = f"""
            SELECT DISTINCT CAST({col} AS STRING) AS val
            FROM `{eqp_table_id}`
            WHERE {col} IS NOT NULL
            ORDER BY val
            """
            try:
                df = client.query(sql).to_dataframe()
                return df["val"].dropna().tolist()
            except Exception:
                return []

        # ----------------------- Carregar todas as opções ----
        opts = {
            # Período
            "ano": _opts_eqp("equipamentos_ano"),
            "mes": _opts_eqp("equipamentos_mes"),

            # Território
            "uf": _opts_eqp("ibge_no_uf"),
            "reg_saude": _opts_eqp("ibge_no_regiao_saude"),
            "meso": _opts_eqp("ibge_no_mesorregiao"),
            "micro": _opts_eqp("ibge_no_microrregiao"),
            "mun": _opts_eqp("ibge_no_municipio"),
            "ivs": _opts_eqp("ibge_ivs"),

            # Perfil do estabelecimento
            "tipo_novo": _opts_eqp("estabelecimentos_tipo_novo_do_estabelecimento"),
            "tipo": _opts_eqp("estabelecimentos_tipo_do_estabelecimento"),
            "subtipo": _opts_eqp("estabelecimentos_subtipo_do_estabelecimento"),
            "gestao": _opts_eqp("estabelecimentos_gestao"),
            "convenio": _opts_eqp("estabelecimentos_convenio_sus"),
            "natureza": _opts_eqp("estabelecimentos_categoria_natureza_juridica"),
            "status": _opts_eqp("estabelecimentos_status_do_estabelecimento"),

            # Tipos de equipamento
            "tipo_eqp": _opts_eqp("equipamentos_tipo_equipamento"),
            "id_eqp": _opts_eqp("equipamentos_id_equipamento"),
        }

    # =========================================================
    # SIDEBAR DE FILTROS
    # =========================================================
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Equipamentos")
        st.caption("Use os agrupadores abaixo para refinar o cadastro de equipamentos.")

        # ----------------- Período --------------------
        with st.expander("Fitros de Período", expanded=False):
            ano_sel = st.multiselect(
                "Ano",
                opts["ano"],
                key="eqp_ano",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_sel = st.multiselect(
                "Mês",
                opts["mes"],
                key="eqp_mes",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ----------------- Território ------------------
        with st.expander("Fitros de Território", expanded=False):
            uf_sel = st.multiselect(
                "UF",
                opts["uf"],
                key="eqp_uf",
                placeholder="(Todos. Filtros opcionais)",
            )
            reg_saude_sel = st.multiselect(
                "Região de Saúde",
                opts["reg_saude"],
                key="eqp_regsaude",
                placeholder="(Todos. Filtros opcionais)",
            )
            meso_sel = st.multiselect(
                "Mesorregião",
                opts["meso"],
                key="eqp_meso",
                placeholder="(Todos. Filtros opcionais)",
            )
            micro_sel = st.multiselect(
                "Microrregião",
                opts["micro"],
                key="eqp_micro",
                placeholder="(Todos. Filtros opcionais)",
            )
            mun_sel = st.multiselect(
                "Município",
                opts["mun"],
                key="eqp_mun",
                placeholder="(Todos. Filtros opcionais)",
            )
            ivs_sel = st.multiselect(
                "Município IVS",
                opts["ivs"],
                key="eqp_ivs",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Perfil do Estabelecimento ----------
        with st.expander("Fitros de Perfil do Estabelecimento", expanded=False):
            tipo_novo_sel = st.multiselect(
                "Tipo (novo)",
                opts["tipo_novo"],
                key="eqp_tipo_novo",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_sel = st.multiselect(
                "Tipo do estabelecimento",
                opts["tipo"],
                key="eqp_tipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            subtipo_sel = st.multiselect(
                "Subtipo",
                opts["subtipo"],
                key="eqp_subtipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            gestao_sel = st.multiselect(
                "Gestão",
                opts["gestao"],
                key="eqp_gestao",
                placeholder="(Todos. Filtros opcionais)",
            )
            convenio_sel = st.multiselect(
                "Convênio SUS",
                opts["convenio"],
                key="eqp_convenio",
                placeholder="(Todos. Filtros opcionais)",
            )
            natureza_sel = st.multiselect(
                "Natureza Jurídica",
                opts["natureza"],
                key="eqp_natjur",
                placeholder="(Todos. Filtros opcionais)",
            )
            status_sel = st.multiselect(
                "Status",
                opts["status"],
                key="eqp_status",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ------------- Complexos / Oncologia -----------
        with st.expander("Fitros de Complexos / Oncologia", expanded=False):
            def bool_multiselect(label, key):
                return st.multiselect(
                    label,
                    ["Sim", "Não"],
                    key=key,
                    placeholder="(Todos. Filtros opcionais)",
                )

            onco_cacon_sel = bool_multiselect("CACON", "eqp_onco_cacon")
            onco_unacon_sel = bool_multiselect("UNACON", "eqp_onco_unacon")
            onco_radio_sel = bool_multiselect("Radioterapia", "eqp_onco_radio")
            onco_quimio_sel = bool_multiselect("Quimioterapia", "eqp_onco_quimio")
            hab_onco_cir_sel = bool_multiselect("Onco Cirúrgica", "eqp_onco_cir")

            uti_adulto_sel = bool_multiselect("UTI Adulto", "eqp_uti_adulto")
            uti_ped_sel = bool_multiselect("UTI Pediátrica", "eqp_uti_ped")
            uti_neo_sel = bool_multiselect("UTI Neonatal", "eqp_uti_neo")
            uti_cor_sel = bool_multiselect("UTI Coronariana", "eqp_uti_cor")
            ucin_sel = bool_multiselect("UCIN", "eqp_ucin")
            uti_queim_sel = bool_multiselect("UTI Queimados", "eqp_uti_queim")
            caps_psiq_sel = bool_multiselect(
                "Saúde Mental CAPS/Psiq", "eqp_caps_psiq"
            )
            rehab_cer_sel = bool_multiselect("Reabilitação CER", "eqp_rehab_cer")
            cardio_ac_sel = bool_multiselect(
                "Cardio Alta Complex.", "eqp_cardio_ac"
            )
            nutricao_sel = bool_multiselect("Nutrição", "eqp_nutricao")
            odonto_ceo_sel = bool_multiselect(
                "Odontologia CEO", "eqp_odonto_ceo"
            )

        # ----------------- Tipos de Equipamento -----------------
        with st.expander("Fitros de Tipos de Equipamento", expanded=False):
            tipo_eqp_sel = st.multiselect(
                "Tipo de equipamento",
                opts["tipo_eqp"],
                key="eqp_tipo_equipamento",
                placeholder="(Todos. Filtros opcionais)",
            )
            id_eqp_sel = st.multiselect(
                "Código do equipamento",
                opts["id_eqp"],
                key="eqp_id_equipamento",
                placeholder="(Todos. Filtros opcionais)",
            )

    # =========================================================
    # Detectar mudanças de filtros + WHERE/params
    # =========================================================
    filter_values = {
        "ano": ano_sel,
        "mes": mes_sel,
        "uf": uf_sel,
        "reg_saude": reg_saude_sel,
        "meso": meso_sel,
        "micro": micro_sel,
        "mun": mun_sel,
        "ivs": ivs_sel,
        "tipo_novo": tipo_novo_sel,
        "tipo": tipo_sel,
        "subtipo": subtipo_sel,
        "gestao": gestao_sel,
        "convenio": convenio_sel,
        "natureza": natureza_sel,
        "status": status_sel,
        "onco_cacon": onco_cacon_sel,
        "onco_unacon": onco_unacon_sel,
        "onco_radio": onco_radio_sel,
        "onco_quimio": onco_quimio_sel,
        "hab_onco_cir": hab_onco_cir_sel,
        "uti_adulto": uti_adulto_sel,
        "uti_ped": uti_ped_sel,
        "uti_neo": uti_neo_sel,
        "uti_cor": uti_cor_sel,
        "ucin": ucin_sel,
        "uti_queim": uti_queim_sel,
        "caps_psiq": caps_psiq_sel,
        "rehab_cer": rehab_cer_sel,
        "cardio_ac": cardio_ac_sel,
        "nutricao": nutricao_sel,
        "odonto_ceo": odonto_ceo_sel,
        "tipo_eqp": tipo_eqp_sel,
        "id_eqp": id_eqp_sel,
    }

    filters_changed = any(did_filters_change(k, v) for k, v in filter_values.items())
    spinner = st.spinner("⏳ Atualizando resultados…") if filters_changed else DummySpinner()

    where_sql_eqp, bq_params_eqp = build_where_eqp(
        ano_sel,
        mes_sel,
        uf_sel,
        reg_saude_sel,
        meso_sel,
        micro_sel,
        mun_sel,
        ivs_sel,
        tipo_novo_sel,
        tipo_sel,
        subtipo_sel,
        gestao_sel,
        convenio_sel,
        natureza_sel,
        status_sel,
        onco_cacon_sel,
        onco_unacon_sel,
        onco_radio_sel,
        onco_quimio_sel,
        hab_onco_cir_sel,
        uti_adulto_sel,
        uti_ped_sel,
        uti_neo_sel,
        uti_cor_sel,
        ucin_sel,
        uti_queim_sel,
        caps_psiq_sel,
        rehab_cer_sel,
        cardio_ac_sel,
        nutricao_sel,
        odonto_ceo_sel,
        tipo_eqp_sel,
        id_eqp_sel,
    )

    # =========================================================
    # Função helper para limitar categorias nos gráficos
    # =========================================================
    def col_top_n(df, col, top_n=40, outros_label="Outros"):
        if col not in df.columns:
            return df
        vc = df[col].value_counts(dropna=False)
        if len(vc) <= top_n:
            return df
        top_vals = set(vc.head(top_n).index)
        df2 = df.copy()
        df2[col] = df2[col].where(df2[col].isin(top_vals), outros_label)
        return df2

    # =========================================================
    # BLOCO PRINCIPAL COM SPINNER DE RESULTADOS
    # =========================================================
    with spinner:
        # ======================== KPIs =======================
        st.info("📏 Grandes números: visão rápida dos equipamentos filtrados")

        kpis_eqp = query_eqp_kpis(where_sql_eqp, bq_params_eqp, eqp_table_id)

        if kpis_eqp is None or (kpis_eqp["total_registros"] or 0) == 0:
            st.warning("Nenhum equipamento encontrado com os filtros selecionados.")
            st.stop()

        total_reg = int(kpis_eqp["total_registros"] or 0)
        total_eqp = float(kpis_eqp["total_eqp"] or 0)
        total_ativos = float(kpis_eqp["total_ativos"] or 0)
        media_uf = kpis_eqp["media_uf"] or 0
        media_estab = kpis_eqp["media_estab"] or 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de equipamentos (somados)", fmt_num(total_eqp))

        with col2:
            st.metric("Equipamentos ativos (somados)", fmt_num(total_ativos))

        with col3:
            st.metric("Média de equipamentos por UF", fmt_num(media_uf))

        with col4:
            st.metric("Média de equipamentos por Estabelecimento", fmt_num(media_estab))

        # ======================== GRÁFICOS ====================
        st.info("📊 Gráficos — resumo visual dos equipamentos filtrados")

        # 1) Equipamentos por tipo
        with st.expander("Equipamentos por tipo", expanded=True):
            df_tipo = query_eqp_group_tipo(where_sql_eqp, bq_params_eqp, eqp_table_id)
            if not df_tipo.empty:
                df_tipo = df_tipo.sort_values("qtd_equipamentos", ascending=False)
                df_tipo = col_top_n(
                    df_tipo, "equipamentos_tipo_equipamento", top_n=30
                )

                fig_tipo = bar_total_por_grupo(
                    df_tipo,
                    grupo_col="equipamentos_tipo_equipamento",
                    valor_col="qtd_equipamentos",
                    titulo="Distribuição de equipamentos por tipo",
                    x_label="Quantidade de equipamentos",
                    y_label="Tipo de equipamento",
                    orientation="h",
                )
                st.plotly_chart(fig_tipo, use_container_width=True)
            else:
                st.info(
                    "Colunas de tipo de equipamento ou quantidade não estão disponíveis."
                )

        # 2) Equipamentos disponíveis x indisponíveis SUS
        with st.expander(
            "Equipamentos disponíveis x indisponíveis para o SUS", expanded=False
        ):
            total_disp, total_indisp = query_eqp_sus_disp(
                where_sql_eqp, bq_params_eqp, eqp_table_id
            )

            if total_disp + total_indisp > 0:
                df_sus = pd.DataFrame(
                    {
                        "categoria": ["Disponíveis SUS", "Indisponíveis SUS"],
                        "qtd": [total_disp, total_indisp],
                    }
                )

                fig_sus = bar_total_por_grupo(
                    df_sus,
                    grupo_col="categoria",
                    valor_col="qtd",
                    titulo="Equipamentos disponíveis x indisponíveis para o SUS",
                    x_label="Categoria",
                    y_label="Quantidade de equipamentos",
                    orientation="v",
                )
                st.plotly_chart(fig_sus, use_container_width=True)
            else:
                st.info(
                    "Não há dados de indicador de disponibilidade SUS para os filtros selecionados."
                )

        # ======================== TABELA DESCRITIVA ===========
        st.info("📋 Tabela descritiva dos equipamentos filtrados")

        cols_desc = [
            "equipamentos_ano",
            "equipamentos_mes",
            "ibge_no_municipio",
            "ibge_no_uf",
            "equipamentos_id_equipamento",
            "equipamentos_tipo_equipamento",
            "equipamentos_quantidade",
            "equipamentos_quantidade_ativos",
            "equipamentos_indicador_disponivel_sus",
            "equipamentos_indicador_indisponivel_sus",
            "equipamentos_id_estabelecimento_cnes",
            "estabelecimentos_nome_fantasia",
            "estabelecimentos_tipo_novo_do_estabelecimento",
            "estabelecimentos_subtipo_do_estabelecimento",
            "estabelecimentos_gestao",
            "estabelecimentos_status_do_estabelecimento",
            "estabelecimentos_convenio_sus",
            "estabelecimentos_categoria_natureza_juridica",
            "onco_cacon",
            "onco_unacon",
            "onco_radioterapia",
            "onco_quimioterapia",
        ]

        cols_ok = cols_desc  # assumindo schema coerente com a tabela

        max_rows_display = 5000
        n_total = total_reg

        if n_total > max_rows_display:
            st.warning(
                f"A base filtrada possui {fmt_num(n_total)} linhas. "
                f"Por desempenho, a tabela abaixo mostra apenas as primeiras {fmt_num(max_rows_display)} linhas. "
                "O download também está limitado às mesmas linhas."
            )
        else:
            st.caption(f"A base filtrada possui {fmt_num(n_total)} linhas.")

        df_display = query_eqp_detalhe(
            where_sql_eqp,
            bq_params_eqp,
            eqp_table_id,
            cols_ok,
            limit_rows=max_rows_display,
        )

        if df_display.empty:
            st.warning(
                "Não foi possível carregar a tabela detalhada de equipamentos com os filtros aplicados."
            )
        else:
            st.dataframe(df_display, use_container_width=True, height=500)

            csv = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Baixar CSV (até 5.000 linhas)",
                csv,
                "equipamentos_filtrados.csv",
                "text/csv",
                use_container_width=True,
            )

# =====================================================================
# X) Cadastro Profissionais
# =====================================================================
elif aba == "👩‍⚕️ Profissionais":
    st.subheader("👩‍⚕️ Profissionais")

    # ---------------------------------------------------------
    # Carregar dados
    # ---------------------------------------------------------
    with st.spinner("⏳ Carregando base de profissionais..."):
        df_prof = load_table(TABLES["profissionais"]).copy()

    # Helper local para opções dos filtros
    def _opts_prof(col: str):
        if col not in df_prof:
            return []
        return sorted(df_prof[col].dropna().unique())

    # =========================================================
    # SIDEBAR DE FILTROS
    # =========================================================
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Profissionais de saúde")
        st.caption("Use os agrupadores abaixo para refinar a visão dos profissionais.")

        # ----------------- Período --------------------
        with st.expander("Fitros de Período", expanded=False):
            ano_sel = st.multiselect(
                "Ano",
                _opts_prof("profissionais_ano"),
                key="prof_ano",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_sel = st.multiselect(
                "Mês",
                _opts_prof("profissionais_mes"),
                key="prof_mes",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ----------------- Território ------------------
        with st.expander("Fitros de Território", expanded=False):
            uf_sel = st.multiselect(
                "UF",
                _opts_prof("ibge_no_uf"),
                key="prof_uf",
                placeholder="(Todos. Filtros opcionais)",
            )
            reg_saude_sel = st.multiselect(
                "Região de Saúde",
                _opts_prof("ibge_no_regiao_saude"),
                key="prof_regsaude",
                placeholder="(Todos. Filtros opcionais)",
            )
            meso_sel = st.multiselect(
                "Mesorregião",
                _opts_prof("ibge_no_mesorregiao"),
                key="prof_meso",
                placeholder="(Todos. Filtros opcionais)",
            )
            micro_sel = st.multiselect(
                "Microrregião",
                _opts_prof("ibge_no_microrregiao"),
                key="prof_micro",
                placeholder="(Todos. Filtros opcionais)",
            )
            mun_sel = st.multiselect(
                "Município",
                _opts_prof("ibge_no_municipio"),
                key="prof_mun",
                placeholder="(Todos. Filtros opcionais)",
            )
            ivs_sel = st.multiselect(
                "Município IVS",
                _opts_prof("ibge_ivs"),
                key="prof_ivs",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Perfil do Estabelecimento ----------
        with st.expander("Fitros de Perfil do Estabelecimento", expanded=False):
            tipo_novo_sel = st.multiselect(
                "Tipo (novo)",
                _opts_prof("estabelecimentos_tipo_novo_do_estabelecimento"),
                key="prof_tipo_novo",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_sel = st.multiselect(
                "Tipo do estabelecimento",
                _opts_prof("estabelecimentos_tipo_do_estabelecimento"),
                key="prof_tipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            subtipo_sel = st.multiselect(
                "Subtipo",
                _opts_prof("estabelecimentos_subtipo_do_estabelecimento"),
                key="prof_subtipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            gestao_sel = st.multiselect(
                "Gestão",
                _opts_prof("estabelecimentos_gestao"),
                key="prof_gestao",
                placeholder="(Todos. Filtros opcionais)",
            )
            convenio_sel = st.multiselect(
                "Convênio SUS",
                _opts_prof("estabelecimentos_convenio_sus"),
                key="prof_convenio",
                placeholder="(Todos. Filtros opcionais)",
            )
            natureza_sel = st.multiselect(
                "Natureza Jurídica",
                _opts_prof("estabelecimentos_categoria_natureza_juridica"),
                key="prof_natjur",
                placeholder="(Todos. Filtros opcionais)",
            )
            status_sel = st.multiselect(
                "Status",
                _opts_prof("estabelecimentos_status_do_estabelecimento"),
                key="prof_status",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Perfil Profissional ----------
        with st.expander("Filtros de Perfil Profissional", expanded=False):
            cbo_sel = st.multiselect(
                "Ocupação (CBO descrição)",
                _opts_prof("cbo_descricao"),
                key="prof_cbo_desc",
                placeholder="(Todos. Filtros opcionais)",
            )
            cbo_saude_sel = st.multiselect(
                "CBO Saúde",
                _opts_prof("cbo_saude"),
                key="prof_cbo_saude",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_cbo_sel = st.multiselect(
                "Tipo de CBO",
                _opts_prof("profissionais_tipo_cbo"),
                key="prof_tipo_cbo",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_ras_sel = st.multiselect(
                "Tipo RAS",
                _opts_prof("profissionais_tipo_ras"),
                key="prof_tipo_ras",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_esp_sel = st.multiselect(
                "Tipo de especialidade",
                _opts_prof("profissionais_tipo_especialidade"),
                key="prof_tipo_esp",
                placeholder="(Todos. Filtros opcionais)",
            )
            tipo_grupo_sel = st.multiselect(
                "Tipo de grupo",
                _opts_prof("profissionais_tipo_grupo"),
                key="prof_tipo_grupo",
                placeholder="(Todos. Filtros opcionais)",
            )
            cons_tipo_sel = st.multiselect(
                "Tipo de conselho",
                _opts_prof("profissionais_tipo_conselho"),
                key="prof_tipo_conselho",
                placeholder="(Todos. Filtros opcionais)",
            )
            vinculo_tipo_sel = st.multiselect(
                "Tipo de vínculo",
                _opts_prof("profissionais_tipo_vinculo"),
                key="prof_tipo_vinculo",
                placeholder="(Todos. Filtros opcionais)",
            )

        # --------- Vínculo / SUS ----------
        with st.expander("Filtros de Vínculo / SUS", expanded=False):
            def ind_multiselect(label, key):
                return st.multiselect(
                    label, ["Sim", "Não"], key=key,
                    placeholder="(Todos. Filtros opcionais)",
                )

            ind_terceiro_sel = ind_multiselect("Estabelecimento terceiro", "prof_terceiro")
            ind_contrat_sus_sel = ind_multiselect("Vínculo contratado SUS", "prof_contrat_sus")
            ind_autonomo_sus_sel = ind_multiselect("Vínculo autônomo SUS", "prof_autonomo_sus")
            ind_outros_sel = ind_multiselect("Vínculo outros", "prof_vinc_outros")
            ind_atende_sus_sel = ind_multiselect("Atende SUS", "prof_atende_sus")
            ind_atende_nao_sus_sel = ind_multiselect("Atende não SUS", "prof_atende_nao_sus")

        # ------------- Complexos / Oncologia -----------
        with st.expander("Filtros de Complexos / Oncologia", expanded=False):
            def bool_multiselect(label, key):
                return st.multiselect(
                    label, ["Sim", "Não"], key=key,
                    placeholder="(Todos. Filtros opcionais)",
                )

            onco_cacon_sel   = bool_multiselect("CACON",           "prof_onco_cacon")
            onco_unacon_sel  = bool_multiselect("UNACON",          "prof_onco_unacon")
            onco_radio_sel   = bool_multiselect("Radioterapia",    "prof_onco_radio")
            onco_quimio_sel  = bool_multiselect("Quimioterapia",   "prof_onco_quimio")
            hab_onco_cir_sel = bool_multiselect("Onco Cirúrgica",  "prof_onco_cir")

            uti_adulto_sel   = bool_multiselect("UTI Adulto",      "prof_uti_adulto")
            uti_ped_sel      = bool_multiselect("UTI Pediátrica",  "prof_uti_ped")
            uti_neo_sel      = bool_multiselect("UTI Neonatal",    "prof_uti_neo")
            uti_cor_sel      = bool_multiselect("UTI Coronariana", "prof_uti_cor")
            ucin_sel         = bool_multiselect("UCIN",            "prof_ucin")
            uti_queim_sel    = bool_multiselect("UTI Queimados",   "prof_uti_queim")
            caps_psiq_sel    = bool_multiselect("Saúde Mental CAPS/Psiq", "prof_caps_psiq")
            rehab_cer_sel    = bool_multiselect("Reabilitação CER",      "prof_rehab_cer")
            cardio_ac_sel    = bool_multiselect("Cardio Alta Complex.",  "prof_cardio_ac")
            nutricao_sel     = bool_multiselect("Nutrição",              "prof_nutricao")
            odonto_ceo_sel   = bool_multiselect("Odontologia CEO",       "prof_odonto_ceo")

    # =========================================================
    # Aplicação dos filtros
    # =========================================================
    dfp = df_prof.copy()

    def apply_multisel(df, col, sel):
        if sel and col in df:
            return df[df[col].isin(sel)]
        return df

    # Inteiros 1/0 -> booleanos "Sim/Não"
    def apply_indicator(df, col, sel):
        if not sel or col not in df:
            return df
        bool_series = df[col].fillna(0).astype(int) > 0
        allowed = []
        if "Sim" in sel:
            allowed.append(True)
        if "Não" in sel:
            allowed.append(False)
        mask = bool_series.isin(allowed)
        return df[mask]

    # Booleanos puros
    def apply_bool(df, col, sel):
        if not sel or col not in df:
            return df
        allowed = []
        if "Sim" in sel:
            allowed.append(True)
        if "Não" in sel:
            allowed.append(False)
        return df[df[col].astype("boolean").isin(allowed)]

    # Período / território
    dfp = apply_multisel(dfp, "profissionais_ano", ano_sel)
    dfp = apply_multisel(dfp, "profissionais_mes", mes_sel)
    dfp = apply_multisel(dfp, "ibge_no_uf",        uf_sel)
    dfp = apply_multisel(dfp, "ibge_no_regiao_saude", reg_saude_sel)
    dfp = apply_multisel(dfp, "ibge_no_mesorregiao",  meso_sel)
    dfp = apply_multisel(dfp, "ibge_no_microrregiao", micro_sel)
    dfp = apply_multisel(dfp, "ibge_no_municipio",    mun_sel)
    dfp = apply_multisel(dfp, "ibge_ivs",             ivs_sel)

    # Perfil Estabelecimento
    dfp = apply_multisel(dfp, "estabelecimentos_tipo_novo_do_estabelecimento", tipo_novo_sel)
    dfp = apply_multisel(dfp, "estabelecimentos_tipo_do_estabelecimento",      tipo_sel)
    dfp = apply_multisel(dfp, "estabelecimentos_subtipo_do_estabelecimento",   subtipo_sel)
    dfp = apply_multisel(dfp, "estabelecimentos_gestao",                       gestao_sel)
    dfp = apply_multisel(dfp, "estabelecimentos_convenio_sus",                 convenio_sel)
    dfp = apply_multisel(dfp, "estabelecimentos_categoria_natureza_juridica",  natureza_sel)
    dfp = apply_multisel(dfp, "estabelecimentos_status_do_estabelecimento",    status_sel)

    # Perfil Profissional
    dfp = apply_multisel(dfp, "cbo_descricao",                cbo_sel)
    dfp = apply_multisel(dfp, "cbo_saude",                    cbo_saude_sel)
    dfp = apply_multisel(dfp, "profissionais_tipo_cbo",       tipo_cbo_sel)
    dfp = apply_multisel(dfp, "profissionais_tipo_ras",       tipo_ras_sel)
    dfp = apply_multisel(dfp, "profissionais_tipo_especialidade", tipo_esp_sel)
    dfp = apply_multisel(dfp, "profissionais_tipo_grupo",     tipo_grupo_sel)
    dfp = apply_multisel(dfp, "profissionais_tipo_conselho",  cons_tipo_sel)
    dfp = apply_multisel(dfp, "profissionais_tipo_vinculo",   vinculo_tipo_sel)

    # Indicadores de vínculo / SUS (inteiros)
    dfp = apply_indicator(dfp, "profissionais_indicador_estabelecimento_terceiro",      ind_terceiro_sel)
    dfp = apply_indicator(dfp, "profissionais_indicador_vinculo_contratado_sus",        ind_contrat_sus_sel)
    dfp = apply_indicator(dfp, "profissionais_indicador_vinculo_autonomo_sus",          ind_autonomo_sus_sel)
    dfp = apply_indicator(dfp, "profissionais_indicador_vinculo_outros",                ind_outros_sel)
    dfp = apply_indicator(dfp, "profissionais_indicador_atende_sus",                    ind_atende_sus_sel)
    dfp = apply_indicator(dfp, "profissionais_indicador_atende_nao_sus",                ind_atende_nao_sus_sel)

    # Complexos / Onco
    dfp = apply_bool(dfp, "onco_cacon",                          onco_cacon_sel)
    dfp = apply_bool(dfp, "onco_unacon",                         onco_unacon_sel)
    dfp = apply_bool(dfp, "onco_radioterapia",                   onco_radio_sel)
    dfp = apply_bool(dfp, "onco_quimioterapia",                  onco_quimio_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_onco_cirurgica", hab_onco_cir_sel)

    dfp = apply_bool(dfp, "habilitacao_agrupado_uti_adulto",     uti_adulto_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_uti_pediatrica", uti_ped_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_uti_neonatal",   uti_neo_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_uti_coronariana",uti_cor_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_ucin",           ucin_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_uti_queimados",  uti_queim_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_saude_mental_caps_psiq", caps_psiq_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_reabilitacao_cer",        rehab_cer_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_cardio_alta_complex",     cardio_ac_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_nutricao",                nutricao_sel)
    dfp = apply_bool(dfp, "habilitacao_agrupado_odontologia_ceo",         odonto_ceo_sel)

    # Se depois dos filtros não sobrar nada, aborta o resto
    if dfp.empty:
        st.warning("Nenhum profissional encontrado com os filtros selecionados.")
        st.stop()

    # =========================================================
    # METRIC CARDS
    # =========================================================
    st.info("📏 Grandes números: visão rápida dos profissionais filtrados")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_vinc = dfp.shape[0]
        st.metric("Total de vínculos de profissionais", fmt_num(total_vinc))

    with col2:
        # Profissionais distintos por CNS ou registro conselho
        if "profissionais_cartao_nacional_saude" in dfp.columns:
            n_prof = dfp["profissionais_cartao_nacional_saude"].nunique()
        elif "profissionais_id_registro_conselho" in dfp.columns:
            n_prof = dfp["profissionais_id_registro_conselho"].nunique()
        else:
            n_prof = dfp.shape[0]
        st.metric("Profissionais distintos (aprox.)", fmt_num(n_prof))

    with col3:
        carga_cols = [
            c for c in [
                "profissionais_carga_horaria_outros",
                "profissionais_carga_horaria_hospitalar",
                "profissionais_carga_horaria_ambulatorial",
            ] if c in dfp.columns
        ]
        if carga_cols:
            carga_total = dfp[carga_cols].sum(axis=1).sum()
            st.metric("Carga horária total (somada)", fmt_num(carga_total))
        else:
            st.metric("Carga horária total (somada)", "-")

    with col4:
        if carga_cols and total_vinc > 0:
            carga_media = carga_total / total_vinc
            st.metric("Carga horária média por vínculo", f"{carga_media:,.1f}".replace(",", "."))
        else:
            st.metric("Carga horária média por vínculo", "-")

    # =========================================================
    # Função helper para limitar categorias nos gráficos
    # =========================================================
    def col_top_n(df, col, top_n=40, outros_label="Outros"):
        if col not in df.columns:
            return df
        vc = df[col].value_counts(dropna=False)
        if len(vc) <= top_n:
            return df
        top_vals = set(vc.head(top_n).index)
        df2 = df.copy()
        df2[col] = df2[col].where(df2[col].isin(top_vals), outros_label)
        return df2

    # ============================================================
    # 📊 GRÁFICOS
    # ============================================================
    st.info("📊 Gráficos — resumo visual dos profissionais filtrados")

    # 1) Profissionais por CBO (descrição)
    with st.expander("Profissionais por ocupação (CBO descrição)", expanded=True):
        if "cbo_descricao" in dfp.columns:
            df_cbo = (
                dfp.groupby("cbo_descricao")
                   .size()
                   .reset_index(name="qtd_profissionais")
            )
            df_cbo = df_cbo.sort_values("qtd_profissionais", ascending=False)
            df_cbo = col_top_n(df_cbo, "cbo_descricao", top_n=40)

            fig_cbo = bar_total_por_grupo(
                df_cbo,
                grupo_col="cbo_descricao",
                valor_col="qtd_profissionais",
                titulo="Distribuição de vínculos por ocupação (CBO)",
                x_label="Quantidade de vínculos",
                y_label="Ocupação (CBO)",
                orientation="h",
            )
            st.plotly_chart(fig_cbo, use_container_width=True)
        else:
            st.info("Coluna `cbo_descricao` não está disponível na base filtrada.")

    # 2) Profissionais por tipo RAS
    with st.expander("Distribuição por Tipo RAS", expanded=False):
        if "profissionais_tipo_ras" in dfp.columns:
            df_ras = (
                dfp.groupby("profissionais_tipo_ras")
                   .size()
                   .reset_index(name="qtd_profissionais")
            )
            df_ras = df_ras.sort_values("qtd_profissionais", ascending=False)

            fig_ras = bar_total_por_grupo(
                df_ras,
                grupo_col="profissionais_tipo_ras",
                valor_col="qtd_profissionais",
                titulo="Profissionais por tipo RAS",
                x_label="Quantidade de vínculos",
                y_label="Tipo RAS",
                orientation="v",
            )
            st.plotly_chart(fig_ras, use_container_width=True)
        else:
            st.info("Coluna `profissionais_tipo_ras` não está disponível na base filtrada.")

    # 3) Atende SUS x não SUS
    with st.expander("Atendimento SUS x não SUS", expanded=False):
        if "profissionais_indicador_atende_sus" in dfp.columns or "profissionais_indicador_atende_nao_sus" in dfp.columns:
            sus_mask  = dfp.get("profissionais_indicador_atende_sus", 0).fillna(0).astype(int)       > 0
            nao_sus_mask = dfp.get("profissionais_indicador_atende_nao_sus", 0).fillna(0).astype(int) > 0

            total_sus     = sus_mask.sum()
            total_nao_sus = nao_sus_mask.sum()

            df_sus = pd.DataFrame({
                "categoria": ["Atende SUS", "Atende não SUS"],
                "qtd": [total_sus, total_nao_sus],
            })

            fig_sus = bar_total_por_grupo(
                df_sus,
                grupo_col="categoria",
                valor_col="qtd",
                titulo="Distribuição de vínculos segundo atendimento SUS x não SUS",
                x_label="Quantidade de vínculos",
                y_label="Categoria",
                orientation="v",
            )
            st.plotly_chart(fig_sus, use_container_width=True)
        else:
            st.info("Colunas de indicador de atendimento SUS/Não SUS não estão disponíveis.")

    # ============================================================
    # 📋 TABELA DESCRITIVA — Profissionais (c/ limite de linhas)
    # ============================================================
    st.info("📋 Tabela descritiva dos profissionais filtrados")

    cols_desc = [
        "profissionais_ano",
        "profissionais_mes",
        "ibge_no_municipio",
        "ibge_no_uf",
        "profissionais_nome",
        "profissionais_tipo_conselho",
        "profissionais_id_registro_conselho",
        "profissionais_cartao_nacional_saude",
        "profissionais_tipo_vinculo",
        "cbo_ocupacao",
        "cbo_descricao",
        "profissionais_tipo_cbo",
        "profissionais_tipo_ras",
        "profissionais_tipo_especialidade",
        "profissionais_tipo_grupo",
        "profissionais_carga_horaria_outros",
        "profissionais_carga_horaria_hospitalar",
        "profissionais_carga_horaria_ambulatorial",
        "profissionais_indicador_estabelecimento_terceiro",
        "profissionais_indicador_vinculo_contratado_sus",
        "profissionais_indicador_vinculo_autonomo_sus",
        "profissionais_indicador_vinculo_outros",
        "profissionais_indicador_atende_sus",
        "profissionais_indicador_atende_nao_sus",
        "profissionais_id_estabelecimento_cnes",
        "estabelecimentos_nome_fantasia",
        "estabelecimentos_tipo_novo_do_estabelecimento",
        "estabelecimentos_subtipo_do_estabelecimento",
        "estabelecimentos_gestao",
        "estabelecimentos_status_do_estabelecimento",
        "estabelecimentos_convenio_sus",
        "estabelecimentos_categoria_natureza_juridica",
        "onco_cacon",
        "onco_unacon",
        "onco_radioterapia",
        "onco_quimioterapia",
    ]

    cols_ok = [c for c in cols_desc if c in dfp.columns]

    if cols_ok:
        max_rows_display = 5000
        n_total = dfp.shape[0]

        if n_total > max_rows_display:
            st.warning(
                f"A base filtrada possui {fmt_num(n_total)} linhas. "
                f"Por desempenho, a tabela abaixo mostra apenas as primeiras {fmt_num(max_rows_display)} linhas. "
                "Use o botão de download para obter o conjunto completo."
            )
        else:
            st.caption(f"A base filtrada possui {fmt_num(n_total)} linhas.")

        df_display = dfp[cols_ok].head(max_rows_display)

        st.dataframe(df_display, use_container_width=True, height=500)

        csv = dfp[cols_ok].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar CSV",
            csv,
            "profissionais_filtrados.csv",
            "text/csv",
        )
    else:
        st.info("Não existem colunas suficientes para montar a tabela de profissionais.")
