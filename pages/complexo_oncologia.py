from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
import plotly.graph_objects as go
from pathlib import Path

# Gráficos reutilizáveis
from src.graph import pareto_barh, bar_count
from src.graph import pie_standard
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
    "🧑 Profissionais",
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
    df_est = load_table(TABLES["estabelecimentos"]).copy()

    # ------------------------ Filtros na sidebar ------------------------
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Filtros — Estabelecimentos")
        st.caption("Use os agrupadores abaixo para refinar o cadastro.")

        # Filtros de período
        with st.expander("Filtros de período", expanded=False):
            comp_sel = st.multiselect("Competência", _opts(df_est, "competencia"),key="est_comp",placeholder="(Todos. Filtros opcionais)",)

        # Filtro de território
        with st.expander("Filtros de Território", expanded=False):
            c1, c2, c3 = st.columns(3)
            reg_sel  = st.multiselect("Região", _opts(df_est, "no_regiao"), key="est_reg",placeholder="(Todos. Filtros opcionais)",)
            uf_sel   = st.multiselect("UF", _opts(df_est, "no_uf"), key="est_uf",placeholder="(Todos. Filtros opcionais)",)
            meso_sel = st.multiselect("Mesorregião Geográfica", _opts(df_est, "no_mesorregiao"), key="est_meso",placeholder="(Todos. Filtros opcionais)",)
            micro_sel = st.multiselect("Microrregião Geográfica", _opts(df_est, "no_microrregiao"), key="est_micro",placeholder="(Todos. Filtros opcionais)",)
            reg_saude_sel = st.multiselect("Região de Saúde", _opts(df_est, "cod_regiao_saude"), key="est_regsaude",placeholder="(Todos. Filtros opcionais)",)
            mun_sel = st.multiselect("Município", _opts(df_est, "municipio"), key="est_mun",placeholder="(Todos. Filtros opcionais)",)
            ivs_sel = st.multiselect("Município IVS", _opts(df_est, "ivs"), key="est_ivs",placeholder="(Todos. Filtros opcionais)",)

        # Perfil do estabelecimento
        with st.expander("Filtros de Perfil do Estabelecimento", expanded=False):
            tipo_sel    = st.multiselect("Tipo", _opts(df_est, "tipo_do_estabelecimento"), key="est_tipo",placeholder="(Todos. Filtros opcionais)",)
            subtipo_sel = st.multiselect("Subtipo", _opts(df_est, "subtipo_do_estabelecimento"), key="est_subtipo",placeholder="(Todos. Filtros opcionais)",)
            gestao_sel  = st.multiselect("Gestão", _opts(df_est, "gestao"), key="est_gestao",placeholder="(Todos. Filtros opcionais)",)
            convenio_sel = st.multiselect("Convênio SUS", _opts(df_est, "convenio_sus"), key="est_convenio",placeholder="(Todos. Filtros opcionais)",)
            natureza_sel = st.multiselect("Natureza Jurídica", _opts(df_est, "categoria_natureza_juridica"), key="est_natjur",placeholder="(Todos. Filtros opcionais)",)
            status_sel   = st.multiselect("Status", _opts(df_est, "status_do_estabelecimento"), key="est_status",placeholder="(Todos. Filtros opcionais)",)

        # Habilitações oncológicas
        with st.expander("Fitros de Habilitações Oncológicas", expanded=False):
            onco_cacon_sel = st.multiselect("CACON", ["Sim","Não"], key="est_onco_cacon", placeholder="(Todos. Filtros opcionais)",)
            onco_unacon_sel = st.multiselect("UNACON", ["Sim","Não"], key="est_onco_unacon",placeholder="(Todos. Filtros opcionais)",)
            onco_radio_sel  = st.multiselect("Radioterapia", ["Sim","Não"], key="est_onco_radio",placeholder="(Todos. Filtros opcionais)",)
            onco_quimio_sel = st.multiselect("Quimioterapia", ["Sim","Não"], key="est_onco_quimio",placeholder="(Todos. Filtros opcionais)",)
            hab_onco_cir_sel = st.multiselect("Onco Cirúrgica", ["Sim","Não"], key="est_hab_onco_cir",placeholder="(Todos. Filtros opcionais)",)

    # ------------------------ Detectar Mudanças ------------------------
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

    with spinner:
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

        def apply_bool(df, col, sel):
            if not sel or col not in df:
                return df
            allowed = []
            if "Sim" in sel:  allowed.append(True)
            if "Não" in sel:  allowed.append(False)
            return df[df[col].astype("boolean").isin(allowed)]

        dfe = apply_bool(dfe, "onco_cacon", onco_cacon_sel)
        dfe = apply_bool(dfe, "onco_unacon", onco_unacon_sel)
        dfe = apply_bool(dfe, "onco_radioterapia", onco_radio_sel)
        dfe = apply_bool(dfe, "onco_quimioterapia", onco_quimio_sel)
        dfe = apply_bool(dfe, "habilitacao_agrupado_onco_cirurgica", hab_onco_cir_sel)

        # ===================== KPIs =====================
        st.info("📏 Grandes números: Visão rápida com filtros aplicados")

        tot_est = dfe["cnes"].nunique() if "cnes" in dfe else len(dfe)
        media_por_regiao = (
            dfe.groupby("cod_regiao_saude")["cnes"].nunique().mean()
            if "cod_regiao_saude" in dfe else 0
        )
        media_por_uf = (
            dfe.groupby("no_uf")["cnes"].nunique().mean()
            if "no_uf" in dfe else 0
        )

        def hab(col):
            return int(dfe[dfe[col].astype("boolean")]["cnes"].nunique()) if col in dfe else 0

        n_cacon = hab("onco_cacon")
        n_unacon = hab("onco_unacon")
        n_radio = hab("onco_radioterapia")
        n_quimio = hab("onco_quimioterapia")
        n_cir = hab("habilitacao_agrupado_onco_cirurgica")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estabelecimentos", fmt_num(tot_est))
        c2.metric("Média por UF", fmt_num(media_por_uf))  
        c3.metric("Média por Reg. Saúde", fmt_num(media_por_regiao)) 
        c4.metric("CACON", fmt_num(n_cacon))  

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("UNACON",  fmt_num(n_unacon))  
        c6.metric("Radioterapia",fmt_num(n_radio))   
        c7.metric("Quimioterapia",fmt_num(n_quimio))     
        c8.metric("Onco Cirúrgica", fmt_num(n_cir)) 

        # ===================== Gráficos =====================
        st.info("📊 Gráficos: Veja visuais gráficos com respostas segundo os filtros aplicados")

        with st.expander("Por tipo de estabelecimento", expanded=True):
            if "tipo_do_estabelecimento" in dfe:
                fig = pareto_barh(dfe, "tipo_do_estabelecimento", None, "Estab. por Tipo", "Qtde")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Por tipo de gestão", expanded=True):
            if "gestao" in dfe:
                fig = bar_count(dfe, "gestao", "Estab. por Gestão", 24)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Por Público x Privado", expanded=True):
            if "categoria_natureza_juridica" in dfe:
                fig = bar_count(dfe, "categoria_natureza_juridica", "Estab. por Categoria", 24)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Por Convênio SUS", expanded=True):
            if "convenio_sus" in dfe:
                df_conv = dfe.groupby("convenio_sus")["cnes"].nunique().reset_index(name="qtde")
                fig = pie_standard(df_conv, "convenio_sus", "qtde", "Convênio SUS")
                st.plotly_chart(fig, use_container_width=True)

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

        # só mantém colunas que existem em dfe
        cols_ok_estab = [c for c in cols_desc_estab if c in dfe.columns]

        if cols_ok_estab:
            st.dataframe(
                dfe[cols_ok_estab].sort_values(["no_uf", "municipio"]),
                use_container_width=True,
                height=450,
            )

            # botão de download no mesmo padrão da aba Serviços
            csv_estab = dfe[cols_ok_estab].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Baixar CSV",
                csv_estab,
                file_name="cadastro_estabelecimentos_filtrado.csv",
                mime="text/csv",
            )
        else:
            st.info("Não existem colunas suficientes para montar a tabela de estabelecimentos.")

# =====================================================================
# 3) Cadastro Serviços
# =====================================================================
elif aba == "🗂️ Serviços":
    st.subheader("🗂️ Serviços")

    # ---------------------------------------------------------
    # Carregar dados
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

        with st.expander("Fitros de Período", expanded=False):
            comp_sel = st.multiselect("Competência", _opts_srv("competencia"), key="srv_comp",placeholder="(Todos. Filtros opcionais)")

        with st.expander("Fitros de Território", expanded=False):
            reg_sel       = st.multiselect("Região",           _opts_srv("no_regiao"),         key="srv_reg",placeholder="(Todos. Filtros opcionais)")
            uf_sel        = st.multiselect("UF",               _opts_srv("no_uf"),             key="srv_uf",placeholder="(Todos. Filtros opcionais)")
            meso_sel      = st.multiselect("Mesorregião",      _opts_srv("no_mesorregiao"),    key="srv_meso",placeholder="(Todos. Filtros opcionais)")
            micro_sel     = st.multiselect("Microrregião",     _opts_srv("no_microrregiao"),   key="srv_micro",placeholder="(Todos. Filtros opcionais)")
            reg_saude_sel = st.multiselect("Região de Saúde",  _opts_srv("cod_regiao_saude"),  key="srv_regsaude",placeholder="(Todos. Filtros opcionais)")
            mun_sel       = st.multiselect("Município",        _opts_srv("municipio"),         key="srv_mun",placeholder="(Todos. Filtros opcionais)")
            ivs_sel       = st.multiselect("Município IVS",    _opts_srv("ivs"),               key="srv_ivs",placeholder="(Todos. Filtros opcionais)")

        with st.expander("Fitros de Perfil do Estabelecimento", expanded=False):
            tipo_sel     = st.multiselect("Tipo",              _opts_srv("tipo_novo_do_estabelecimento"), key="srv_tipo",placeholder="(Todos. Filtros opcionais)")
            subtipo_sel  = st.multiselect("Subtipo",           _opts_srv("subtipo_do_estabelecimento"),   key="srv_subtipo",placeholder="(Todos. Filtros opcionais)")
            gestao_sel   = st.multiselect("Gestão",            _opts_srv("gestao"),                        key="srv_gestao",placeholder="(Todos. Filtros opcionais)")
            convenio_sel = st.multiselect("Convênio SUS",      _opts_srv("convenio_sus"),                  key="srv_convenio",placeholder="(Todos. Filtros opcionais)")
            natureza_sel = st.multiselect("Natureza Jurídica", _opts_srv("categoria_natureza_juridica"),   key="srv_natjur",placeholder="(Todos. Filtros opcionais)")
            status_sel   = st.multiselect("Status",            _opts_srv("status_do_estabelecimento"),     key="srv_status",placeholder="(Todos. Filtros opcionais)")

        with st.expander("Fitros de Habilitações Oncológicas", expanded=False):
            def bool_multiselect(label, key):
                return st.multiselect(label, ["Sim", "Não"], key=key)
            onco_cacon_sel = st.multiselect("CACON", ["Sim","Não"], key="srv_onco_cacon", placeholder="(Todos. Filtros opcionais)",)
            onco_unacon_sel = st.multiselect("UNACON", ["Sim","Não"], key="srv_onco_unacon", placeholder="(Todos. Filtros opcionais)",)
            onco_radio_sel = st.multiselect("Radioterapia", ["Sim","Não"], key="srv_onco_radio", placeholder="(Todos. Filtros opcionais)",)
            onco_quimio_sel = st.multiselect("Quimioterapia", ["Sim","Não"], key="srv_onco_quimio", placeholder="(Todos. Filtros opcionais)",)
            hab_onco_cir_sel = st.multiselect("Onco Cirúrgica", ["Sim","Não"], key="srv_onco_cir", placeholder="(Todos. Filtros opcionais)",)

        with st.expander("Fitros de  Serviços especializados", expanded=False):
            servico_sel       = st.multiselect("Serviço especializado", _opts_srv("servico"),                    key="srv_servico",placeholder="(Todos. Filtros opcionais)")
            servico_class_sel = st.multiselect("Classificação",          _opts_srv("servico_classificacao"),     key="srv_servico_class",placeholder="(Todos. Filtros opcionais)")
            amb_sus_sel       = st.multiselect("Ambulatorial SUS",       _opts_srv("servico_ambulatorial_sus"),  key="srv_amb_sus",placeholder="(Todos. Filtros opcionais)")
            amb_nao_sus_sel   = st.multiselect("Ambulatorial não SUS",   _opts_srv("servico_ambulatorial_nao_sus"), key="srv_amb_nao_sus",placeholder="(Todos. Filtros opcionais)")
            hosp_sus_sel      = st.multiselect("Hospitalar SUS",         _opts_srv("servico_hospitalar_sus"),    key="srv_hosp_sus",placeholder="(Todos. Filtros opcionais)")
            hosp_nao_sus_sel  = st.multiselect("Hospitalar não SUS",     _opts_srv("servico_hospitalar_nao_sus"), key="srv_hosp_nao_sus",placeholder="(Todos. Filtros opcionais)")
            terceiro_sel      = st.multiselect("Terceiro",               _opts_srv("servico_terceiro"),          key="srv_terceiro",placeholder="(Todos. Filtros opcionais)")

    # =========================================================
    # Aplicação dos filtros
    # =========================================================
    dfs = df_srv.copy()

    def apply_multisel(df, col, sel):
        if sel and col in df:
            return df[df[col].isin(sel)]
        return df

    # Território / período
    dfs = apply_multisel(dfs, "competencia",          comp_sel)
    dfs = apply_multisel(dfs, "no_regiao",            reg_sel)
    dfs = apply_multisel(dfs, "no_uf",                uf_sel)
    dfs = apply_multisel(dfs, "no_mesorregiao",       meso_sel)
    dfs = apply_multisel(dfs, "no_microrregiao",      micro_sel)
    dfs = apply_multisel(dfs, "cod_regiao_saude",     reg_saude_sel)
    dfs = apply_multisel(dfs, "municipio",            mun_sel)
    dfs = apply_multisel(dfs, "ivs",                  ivs_sel)

    # Perfil Estabelecimento
    dfs = apply_multisel(dfs, "tipo_novo_do_estabelecimento",  tipo_sel)
    dfs = apply_multisel(dfs, "subtipo_do_estabelecimento",    subtipo_sel)
    dfs = apply_multisel(dfs, "gestao",                        gestao_sel)
    dfs = apply_multisel(dfs, "convenio_sus",                  convenio_sel)
    dfs = apply_multisel(dfs, "categoria_natureza_juridica",   natureza_sel)
    dfs = apply_multisel(dfs, "status_do_estabelecimento",     status_sel)

    # Booleanos oncológicos
    def apply_bool(df, col, sel):
        if not sel or col not in df:
            return df
        allowed = []
        if "Sim" in sel:
            allowed.append(True)
        if "Não" in sel:
            allowed.append(False)
        return df[df[col].astype("boolean").isin(allowed)]

    dfs = apply_bool(dfs, "onco_cacon",                           onco_cacon_sel)
    dfs = apply_bool(dfs, "onco_unacon",                          onco_unacon_sel)
    dfs = apply_bool(dfs, "onco_radioterapia",                    onco_radio_sel)
    dfs = apply_bool(dfs, "onco_quimioterapia",                   onco_quimio_sel)
    dfs = apply_bool(dfs, "habilitacao_agrupado_onco_cirurgica",  hab_onco_cir_sel)

    # Serviços
    dfs = apply_multisel(dfs, "servico",                          servico_sel)
    dfs = apply_multisel(dfs, "servico_classificacao",            servico_class_sel)
    dfs = apply_multisel(dfs, "servico_ambulatorial_sus",         amb_sus_sel)
    dfs = apply_multisel(dfs, "servico_ambulatorial_nao_sus",     amb_nao_sus_sel)
    dfs = apply_multisel(dfs, "servico_hospitalar_sus",           hosp_sus_sel)
    dfs = apply_multisel(dfs, "servico_hospitalar_nao_sus",       hosp_nao_sus_sel)
    dfs = apply_multisel(dfs, "servico_terceiro",                 terceiro_sel)

    # Se depois dos filtros não sobrar nada, aborta o resto
    if dfs.empty:
        st.warning("Nenhum serviço encontrado com os filtros selecionados.")
        st.stop()

    # =========================================================
    # METRIC CARDS
    # =========================================================
    st.info(f"📏 Grandes números: Visão rápida com filtros aplicados")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_srv = dfs.shape[0] if "servico" in dfs.columns else 0
        st.metric("Total de serviços distintos", fmt_num(total_srv))
    
    with col2:
        if "no_uf" in dfs.columns and "servico" in dfs.columns:
            mean_est = dfs.groupby("no_uf")["servico"].count().mean()
            st.metric("Média por Estado", fmt_num(mean_est))
        else:
            st.metric("Média por Estado", "-")

    with col3:
        if "cod_regiao_saude" in dfs.columns and "servico" in dfs.columns:
            mean_regsaude = dfs.groupby("cod_regiao_saude")["servico"].count().mean()
            st.metric("Média por Reg. Saúde", fmt_num(mean_regsaude))
        else:
            st.metric("Média por Região de Saúde", "-")

    with col4:
        if "cnes" in dfs.columns and "servico" in dfs.columns:
            mean_estab = dfs.groupby("cnes")["servico"].count().mean()
            st.metric("Média por Estabelecimento", fmt_num(mean_estab))
        else:
            st.metric("Média por Estabelecimento", "-")

    # =========================================================
    # Função helper para limitar categorias nos gráficos
    # =========================================================
    def col_top_n(df, col, top_n=40, outros_label="Outros"):
        """Agrupa coluna categórica em top_n + 'Outros' para usar no bar_count."""
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
    st.info("📊 Gráficos — Resumo visual dos serviços filtrados")

    with st.expander("Número de serviços por Tipo de Serviço", expanded=True):
        if "servico" in dfs.columns:
            fig_tipo = pareto_barh(dfs, "servico", None, "Distribuição de Serviços por Tipo", "Qtde")
            st.plotly_chart(fig_tipo, use_container_width=True)
        else:
            st.info("Coluna `servico` não existe.")
        
    # -----------------------------------------
    # 5) Ambulatorial / Hospitalar — SUS x Não SUS
    # -----------------------------------------
    with st.expander("Serviços Ambulatoriais e Hospitalares – SUS/Não SUS", expanded=True):

        cols_ah = {
            "servico_ambulatorial_sus":     "Ambulatorial SUS",
            "servico_ambulatorial_nao_sus": "Ambulatorial Não SUS",
            "servico_hospitalar_sus":       "Hospitalar SUS",
            "servico_hospitalar_nao_sus":   "Hospitalar Não SUS",
        }

        presentes = [c for c in cols_ah if c in dfs.columns]

        if presentes:
            df_ah = pd.DataFrame({
                "classificacao": [cols_ah[c] for c in presentes],
                "qtd_servicos": [(dfs[c] == "Sim").sum() for c in presentes],  # CORRETO
            })

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
            st.info("Não há colunas relacionadas a SUS/Não SUS.")


    # ============================================================
    # 📋 TABELA DESCRITIVA
    # ============================================================
    st.info("📋 Tabela descritiva dos serviços filtrados")

    cols_desc = [
        "cnes", "nome_fantasia", "municipio", "no_uf",
        "servico", "servico_classificacao", "servico_terceiro",
        "servico_ambulatorial_sus", "servico_ambulatorial_nao_sus",
        "servico_hospitalar_sus", "servico_hospitalar_nao_sus",
    ]

    cols_ok = [c for c in cols_desc if c in dfs.columns]

    if cols_ok:
        st.dataframe(dfs[cols_ok], use_container_width=True, height=450)

        csv = dfs[cols_ok].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar CSV",
            csv,
            "cadastro_servicos_filtrado.csv",
            "text/csv",
        )
    else:
        st.info("Não existem colunas suficientes para montar a tabela.")

# =====================================================================
# 4) Cadastro Habilitações
# =====================================================================
elif aba == "✅ Habilitação":
    st.subheader("✅ Habilitação")

    # ---------------------------------------------------------
    # Carregar dados
    # ---------------------------------------------------------
    with st.spinner("⏳ Carregando base de habilitações..."):
        df_hab = load_table(TABLES["habilitacao"]).copy()

    # Helper local para opções dos filtros
    def _opts_hab(col: str):
        if col not in df_hab:
            return []
        return sorted(df_hab[col].dropna().unique())

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
                _opts_hab("habilitacao_ano"),
                key="hab_ano",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_hab_sel = st.multiselect(
                "Mês da habilitação",
                _opts_hab("habilitacao_mes"),
                key="hab_mes",
                placeholder="(Todos. Filtros opcionais)",
            )

            ano_comp_ini_sel = st.multiselect(
                "Ano competência inicial",
                _opts_hab("habilitacao_ano_competencia_inicial"),
                key="hab_ano_comp_ini",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_comp_ini_sel = st.multiselect(
                "Mês competência inicial",
                _opts_hab("habilitacao_mes_competencia_inicial"),
                key="hab_mes_comp_ini",
                placeholder="(Todos. Filtros opcionais)",
            )

            ano_comp_fim_sel = st.multiselect(
                "Ano competência final",
                _opts_hab("habilitacao_ano_competencia_final"),
                key="hab_ano_comp_fim",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_comp_fim_sel = st.multiselect(
                "Mês competência final",
                _opts_hab("habilitacao_mes_competencia_final"),
                key="hab_mes_comp_fim",
                placeholder="(Todos. Filtros opcionais)",
            )

            ano_portaria_sel = st.multiselect(
                "Ano da portaria",
                _opts_hab("habilitacao_ano_portaria"),
                key="hab_ano_portaria",
                placeholder="(Todos. Filtros opcionais)",
            )
            mes_portaria_sel = st.multiselect(
                "Mês da portaria",
                _opts_hab("habilitacao_mes_portaria"),
                key="hab_mes_portaria",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ------------------ Território ----------------------- #Faltou Região Brasil
        with st.expander("Filtros de Território", expanded=False):
            uf_sel = st.multiselect(
                "UF",
                _opts_hab("ibge_no_uf"),
                key="hab_uf",
                placeholder="(Todos. Filtros opcionais)",
            )
            reg_saude_sel = st.multiselect(
                "Região de Saúde",
                _opts_hab("ibge_no_regiao_saude"),
                key="hab_reg_saude",
                placeholder="(Todos. Filtros opcionais)",
            )
            meso_sel = st.multiselect(
                "Mesorregião",
                _opts_hab("ibge_no_mesorregiao"),
                key="hab_meso",
                placeholder="(Todos. Filtros opcionais)",
            )
            micro_sel = st.multiselect(
                "Microrregião",
                _opts_hab("ibge_no_microrregiao"),
                key="hab_micro",
                placeholder="(Todos. Filtros opcionais)",
            )
            mun_sel = st.multiselect(
                "Município",
                _opts_hab("ibge_no_municipio"),
                key="hab_mun",
                placeholder="(Todos. Filtros opcionais)",
            )
            ivs_sel = st.multiselect(
                "Município IVS",
                _opts_hab("ibge_ivs"),
                key="hab_ivs",
                placeholder="(Todos. Filtros opcionais)",
            )

        # -------- Perfil do Estabelecimento ------------------
        with st.expander("Filtros de Perfil do Estabelecimento", expanded=False):
            tipo_novo_sel = st.multiselect(
                "Tipo",
                _opts_hab("estabelecimentos_tipo_novo_do_estabelecimento"),
                key="hab_tipo_novo",
                placeholder="(Todos. Filtros opcionais)",
            )
            subtipo_sel = st.multiselect(
                "Subtipo",
                _opts_hab("estabelecimentos_subtipo_do_estabelecimento"),
                key="hab_subtipo",
            )
            gestao_sel = st.multiselect(
                "Gestão",
                _opts_hab("estabelecimentos_gestao"),
                key="hab_gestao",
                placeholder="(Todos. Filtros opcionais)",
            )
            convenio_sel = st.multiselect(
                "Convênio SUS",
                _opts_hab("estabelecimentos_convenio_sus"),
                key="hab_convenio",
                placeholder="(Todos. Filtros opcionais)",
            )
            nat_jur_sel = st.multiselect(
                "Natureza jurídica",
                _opts_hab("estabelecimentos_categoria_natureza_juridica"),
                key="hab_natjur",
                placeholder="(Todos. Filtros opcionais)",
            )
            status_sel = st.multiselect(
                "Status",
                _opts_hab("estabelecimentos_status_do_estabelecimento"),
                key="hab_status",
                placeholder="(Todos. Filtros opcionais)",
            )

        # ------------- Filtros de Habilitação ----------------
        with st.expander("Filtros de Habilitação", expanded=False):
            nivel_tipo_sel = st.multiselect(
                "Nível/Tipo de habilitação",
                _opts_hab("habilitacao_nivel_habilitacao_tipo"),
                key="hab_nivel_tipo",
                placeholder="(Todos. Filtros opcionais)",
            )
            cat_hab_sel = st.multiselect(
                "Categoria da habilitação",
                _opts_hab("referencia_habilitacao_no_categoria"),
                key="hab_categoria",
                placeholder="(Todos. Filtros opcionais)",
            )
            no_hab_sel = st.multiselect(
                "Descrição da habilitação",
                _opts_hab("referencia_habilitacao_no_habilitacao"),
                key="hab_nome_hab",
                placeholder="(Todos. Filtros opcionais)",
            )
            tag_hab_sel = st.multiselect(
                "Tag (agrupador)",
                _opts_hab("referencia_habilitacao_ds_tag"),
                key="hab_tag",
                placeholder="(Todos. Filtros opcionais)",
            )

    # =========================================================
    # Aplicação dos filtros
    # =========================================================
    dfh = df_hab.copy()

    def apply_multisel(df, col, sel):
        if sel and col in df:
            return df[df[col].isin(sel)]
        return df

    # Período
    dfh = apply_multisel(dfh, "habilitacao_ano",                        ano_hab_sel)
    dfh = apply_multisel(dfh, "habilitacao_mes",                        mes_hab_sel)
    dfh = apply_multisel(dfh, "habilitacao_ano_competencia_inicial",    ano_comp_ini_sel)
    dfh = apply_multisel(dfh, "habilitacao_mes_competencia_inicial",    mes_comp_ini_sel)
    dfh = apply_multisel(dfh, "habilitacao_ano_competencia_final",      ano_comp_fim_sel)
    dfh = apply_multisel(dfh, "habilitacao_mes_competencia_final",      mes_comp_fim_sel)
    dfh = apply_multisel(dfh, "habilitacao_ano_portaria",               ano_portaria_sel)
    dfh = apply_multisel(dfh, "habilitacao_mes_portaria",               mes_portaria_sel)

    # Território
    dfh = apply_multisel(dfh, "ibge_no_uf",             uf_sel)
    dfh = apply_multisel(dfh, "ibge_no_regiao_saude",   reg_saude_sel)
    dfh = apply_multisel(dfh, "ibge_no_mesorregiao",    meso_sel)
    dfh = apply_multisel(dfh, "ibge_no_microrregiao",   micro_sel)
    dfh = apply_multisel(dfh, "ibge_no_municipio",      mun_sel)
    dfh = apply_multisel(dfh, "ibge_ivs",               ivs_sel)

    # Perfil do estabelecimento
    dfh = apply_multisel(dfh, "estabelecimentos_tipo_novo_do_estabelecimento", tipo_novo_sel)
    dfh = apply_multisel(dfh, "estabelecimentos_subtipo_do_estabelecimento",   subtipo_sel)
    dfh = apply_multisel(dfh, "estabelecimentos_gestao",                       gestao_sel)
    dfh = apply_multisel(dfh, "estabelecimentos_convenio_sus",                 convenio_sel)
    dfh = apply_multisel(dfh, "estabelecimentos_categoria_natureza_juridica",  nat_jur_sel)
    dfh = apply_multisel(dfh, "estabelecimentos_status_do_estabelecimento",    status_sel)

    # Habilitação
    dfh = apply_multisel(dfh, "habilitacao_nivel_habilitacao_tipo",    nivel_tipo_sel)
    dfh = apply_multisel(dfh, "referencia_habilitacao_no_categoria",   cat_hab_sel)
    dfh = apply_multisel(dfh, "referencia_habilitacao_no_habilitacao", no_hab_sel)
    dfh = apply_multisel(dfh, "referencia_habilitacao_ds_tag",         tag_hab_sel)

    # Se depois dos filtros não sobrar nada, aborta o resto
    if dfh.empty:
        st.warning("Nenhuma habilitação encontrada com os filtros selecionados.")
        st.stop()

    # =========================================================
    # METRIC CARDS
    # =========================================================
    st.info("📏 Grandes números: visão rápida das habilitações com os filtros aplicados")

    col1, col2, col3, col4 = st.columns(4)

    # 1) Total de habilitações (linhas da base filtrada)
    with col1:
        total_hab = dfh.shape[0]
        st.metric("Total de habilitações", fmt_num(total_hab))

    # 2) Média de habilitações por UF
    with col2:
        if "ibge_no_uf" in dfh.columns:
            # usa qualquer coluna não nula (pode ser a própria habilitação)
            mean_uf = dfh.groupby("ibge_no_uf")["referencia_habilitacao_no_habilitacao"].count().mean() \
                if "referencia_habilitacao_no_habilitacao" in dfh.columns else \
                dfh.groupby("ibge_no_uf").size().mean()
            st.metric("Média de habilitações por UF", fmt_num(mean_uf))
        else:
            st.metric("Média de habilitações por UF", "-")

    # 3) Média de habilitações por Região de Saúde
    with col3:
        if "ibge_no_regiao_saude" in dfh.columns:
            mean_regsaude = dfh.groupby("ibge_no_regiao_saude")["referencia_habilitacao_no_habilitacao"].count().mean() \
                if "referencia_habilitacao_no_habilitacao" in dfh.columns else \
                dfh.groupby("ibge_no_regiao_saude").size().mean()
            st.metric("Média por Região de Saúde", fmt_num(mean_regsaude))
        else:
            st.metric("Média por Região de Saúde", "-")

    # 4) Média de habilitações por estabelecimento
    with col4:
        if "habilitacao_id_estabelecimento_cnes" in dfh.columns:
            mean_estab = dfh.groupby("habilitacao_id_estabelecimento_cnes")["referencia_habilitacao_no_habilitacao"].count().mean() \
                if "referencia_habilitacao_no_habilitacao" in dfh.columns else \
                dfh.groupby("habilitacao_id_estabelecimento_cnes").size().mean()
            st.metric("Média por Estabelecimento", fmt_num(mean_estab))
        else:
            st.metric("Média por Estabelecimento", "-")

    # =========================================================
    # Habilitações ativas vs encerradas – usando 9999 e comparação com data atual
    # =========================================================

    # Garantir números
    for col in [
        "habilitacao_ano_competencia_final",
        "habilitacao_mes_competencia_final",
    ]:
        if col in dfh.columns:
            dfh[col] = pd.to_numeric(dfh[col], errors="coerce")

    ano_final = dfh["habilitacao_ano_competencia_final"]
    mes_final = dfh["habilitacao_mes_competencia_final"]

    # Data atual
    hoje = pd.Timestamp.today()
    comp_atual = hoje.year * 100 + hoje.month

    # Calcular competência final
    comp_final = ano_final * 100 + mes_final

    # Lógica:
    # 1) ano_final = 9999 e mes_final = 9999 → ATIVA
    ativa_mask = (ano_final == 9999) & (mes_final == 9999)

    # 2) senão → comparar com hoje
    ativa_mask |= (comp_final >= comp_atual)

    # Encerradas = tudo que não é ativo
    encerrada_mask = ~ativa_mask

    total_ativas = ativa_mask.sum()
    total_encerradas = encerrada_mask.sum()
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
    # 📊 GRÁFICOS – Habilitações
    # ============================================================
    st.info("📊 Gráficos — resumo visual das habilitações filtradas")

    # Helper para limitar categorias (top N + 'Outros')
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

    # ------------------------------------------------------------
    # 1) Distribuição por Categoria de Habilitação
    # ------------------------------------------------------------
    with st.expander("Habilitações por categoria (CNES)", expanded=True):
        col_cat = "referencia_habilitacao_no_categoria"
        if col_cat in dfh.columns:
            df_cat = col_top_n(dfh, col_cat, top_n=25)
            fig_cat = pareto_barh(
                df_cat,
                col_cat,
                None,
                "Distribuição de habilitações por categoria",
                "Qtde de habilitações",
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Coluna de categoria de habilitação não encontrada na base.")

    # ------------------------------------------------------------
    # 3) Habilitações por UF
    # ------------------------------------------------------------
    with st.expander("Habilitações por UF", expanded=True):
        if "ibge_no_uf" in dfh.columns:
            df_uf = (
                dfh.groupby("ibge_no_uf")
                .size()
                .reset_index(name="qtd_habilitacoes")
                .sort_values("qtd_habilitacoes", ascending=False)
            )

            fig_uf = bar_total_por_grupo(
                df_uf,
                grupo_col="ibge_no_uf",
                valor_col="qtd_habilitacoes",
                titulo="Quantidade de habilitações por UF",
                x_label="UF",
                y_label="Qtde de habilitações",
                orientation="v",
            )
            st.plotly_chart(fig_uf, use_container_width=True)
        else:
            st.info("Coluna `ibge_no_uf` não encontrada.")

    # ============================================================
    # 📋 TABELA DESCRITIVA — Habilitações (com limite de linhas)
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

    cols_ok = [c for c in cols_desc if c in dfh.columns]

    if cols_ok:
        # ---- Limite de linhas exibidas na tela ----
        max_rows_display = 5000  # ajuste se quiser
        n_total = dfh.shape[0]

        if n_total > max_rows_display:
            st.warning(
                f"A base filtrada possui {fmt_num(n_total)} linhas. "
                f"Por desempenho, a tabela abaixo mostra apenas as primeiras {fmt_num(max_rows_display)} linhas. "
                "Use o botão de download para obter o conjunto completo."
            )
        else:
            st.caption(f"A base filtrada possui {fmt_num(n_total)} linhas.")

        df_display = dfh[cols_ok].head(max_rows_display)

        st.dataframe(
            df_display,
            use_container_width=True,
            height=500,
        )

        # ---- Download CSV com TODAS as linhas filtradas ----
        csv = dfh[cols_ok].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar CSV das habilitações filtradas (completo)",
            data=csv,
            file_name="habilitacoes_filtradas.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.warning("Não existem colunas suficientes para montar a tabela de habilitações.")
