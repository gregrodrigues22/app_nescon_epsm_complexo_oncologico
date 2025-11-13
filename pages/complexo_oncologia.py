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
# FILTER SPEC por tabela
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
        "servicos_especializados_competencia": "str",
        "servicos_especializados_uf": "str",
        "servicos_especializados_codigo_do_municipio": "int",
        "servicos_especializados_municipio": "str",
        "servicos_especializados_cnes": "int",
        "servicos_especializados_nome_fantasia": "str",
        "servicos_especializados_tipo_novo_do_estabelecimento": "str",
        "servicos_especializados_tipo_do_estabelecimento": "str",
        "servicos_especializados_subtipo_do_estabelecimento": "str",
        "servicos_especializados_gestao": "str",
        "servicos_especializados_convenio_sus": "str",
        "servicos_especializados_categoria_natureza_juridica": "str",
        "servicos_especializados_status_do_estabelecimento": "str",
        "servicos_especializados_servico": "str",
        "servicos_especializados_servico_classificacao": "str",
        "servicos_especializados_servico_ambulatorial_sus": "str",
        "servicos_especializados_servico_ambulatorial_nao_sus": "str",
        "servicos_especializados_servico_hospitalar_sus": "str",
        "servicos_especializados_servico_hospitalar_nao_sus": "str",
        "servicos_especializados_servico_terceiro": "str",
        "ibge_no_municipio": "str",
        "ibge_no_regiao_saude": "int",
        "ibge_no_microrregiao": "str",
        "ibge_no_mesorregiao": "str",
        "ibge_no_uf": "str",
        "ibge_ivs": "str",
        "ibge_populacao_ibge_2021": "str",
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
}

# ==============================================================
# Cache de leitura
# ==============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_table(table_fqn: str) -> pd.DataFrame:
    return client.query(f"SELECT * FROM `{table_fqn}`").to_dataframe()

# ==============================================================
# Filtro Genérico
# ==============================================================
def _opts(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    return sorted(s.unique().tolist())

def ui_range_numeric(df: pd.DataFrame, col: str, key: str, label: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None

    vmin, vmax = int(s.min()), int(s.max())

    # Caso especial: só existe um valor na coluna
    if vmin == vmax:
        # Apenas informa o valor e não cria slider (evita erro)
        st.caption(f"{label}: {vmin} (apenas este valor disponível)")
        # Se quiser ainda aplicar o filtro, retornamos a tupla fixa
        return (vmin, vmax)

    lo, hi = st.slider(
        label,
        min_value=vmin,
        max_value=vmax,
        value=(vmin, vmax),
        key=key,
    )
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

    if years or months:
        st.markdown("### ⏱️ Período")
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

    st.markdown("### 🧭 Dimensões")
    _draw_multis(df, strs, selections, prefix, "Filtrar")
    _draw_multis(df, ints, selections, prefix, "Filtrar")

    if bools:
        st.markdown("### ✅ Filtros booleanos")
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

# Diretório raiz do app (onde está o app.py)
ROOT_DIR = Path(__file__).resolve().parent.parent

def safe_page_link(path: str, label: str, icon: str | None = None):
    """
    Cria link para página se o arquivo existir.
    Caso contrário, mostra botão desabilitado (em breve),
    sem quebrar o app.
    """
    full = ROOT_DIR / path
    try:
        if full.exists():
            st.page_link(path, label=label, icon=icon)
        else:
            st.button(label, icon=icon, disabled=True, help="Página em breve.")
    except Exception:
        st.button(label, icon=icon, disabled=True, help="Navegação multipage indisponível aqui.")

def fmt_num(n: float | int, decimals: int = 0) -> str:
    """Formata número em pt-BR: milhar com '.', decimal com ','."""
    if n is None:
        return "-"
    if decimals == 0:
        # inteiro com separador de milhar
        return f"{int(round(n)):,}".replace(",", ".")
    # exemplo: 1191.6 -> '1,191.6'
    s = f"{n:,.{decimals}f}"
    # troca vírgula por ponto e ponto por vírgula
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s
    
# ==============================================================
# Sidebar
# ==============================================================
st.markdown(
    "<style>[data-testid='stSidebarNav']{display:none;}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")

    # voltar ao menu principal
    safe_page_link("app.py", label="Menu Principal", icon="🏠")

    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.subheader("Complexos Produtivos")

    # este já existe (esta própria página)
    safe_page_link("pages/complexo_oncologia.py", label="Oncologia", icon="🎗️")

    # os demais ainda não existem -> aparecem como "em breve"
    safe_page_link("pages/complexo_cardiovascular.py", label="Cardiovascular", icon="❤️")
    safe_page_link("pages/complexo_ortopedia_trauma.py", label="Ortopedia e Traumatologia", icon="🦴")
    safe_page_link("pages/complexo_obstetricia_neonatologia.py", label="Obstetrícia e Neonatologia", icon="🤰")
    safe_page_link("pages/complexo_neuro.py", label="Neurologia e Neurocirurgia", icon="🧠")
    safe_page_link("pages/complexo_nefrologia_trs.py", label="Nefrologia e TRS", icon="🧪")
    safe_page_link("pages/complexo_queimados.py", label="Queimados", icon="🔥")
    safe_page_link("pages/complexo_transplantes.py", label="Transplantes", icon="🫀")
    safe_page_link("pages/complexo_saude_mental.py", label="Saúde Mental Especializada", icon="🧩")
    safe_page_link("pages/complexo_reabilitacao.py", label="Reabilitação", icon="🦾")
    safe_page_link("pages/complexo_urg_emerg.py", label="Urgência e Emergência", icon="🚑")

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

# ==============================================================
# Abas
# ==============================================================
abas = st.tabs(
    [
        "🧮 Matriz de Indicadores",
        "🗂️ Cadastro Estabelecimentos",
        "🗂️ Cadastro Serviços",
        "✅ Habilitação",
        "🛏️ Leitos",
        "🧰 Equipamentos",
        "🧑 Profissionais",
        "📋 Registros Hospitalares",
        "📐 Metodologia",
    ]
)

# ====================== 1) Matriz de Indicadores ======================
with abas[0]:
    st.subheader("🧮 Matriz de Indicadores")

    df_matriz = load_table(TABLES["matriz"])

    # ----------------- Filtros principais -----------------
    st.info("🔎 Filtros: Use os controles abaixo para refinar os resultados.")

    c1, c2, c3 = st.columns(3)

    telas = sorted(
        df_matriz.get("tela", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    secoes = sorted(
        df_matriz.get("secao_tela", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    fontes = sorted(
        df_matriz.get("fonte_dados", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    tipos = sorted(
        df_matriz.get("tipo_indicador", pd.Series(dtype=str)).dropna().unique().tolist()
    )

    tela_sel = c1.multiselect("Tela", options=telas, key="mat_telas", placeholder="Selecione a(s) opção(ões) desejadas")
    secao_sel = c2.multiselect("Seção da tela", options=secoes, key="mat_secoes", placeholder="Selecione a(s) opção(ões) desejadas")
    tipo_sel = c3.multiselect("Tipo de indicador", options=tipos, key="mat_tipos", placeholder="Selecione a(s) opção(ões) desejadas")

    fonte_sel = st.multiselect("Fonte de dados", options=fontes, key="mat_fontes",placeholder="Selecione a(s) opção(ões) desejadas")

    dfm = df_matriz.copy()

    if tela_sel:
        dfm = dfm[dfm["tela"].isin(tela_sel)]
    if secao_sel:
        dfm = dfm[dfm["secao_tela"].isin(secao_sel)]
    if tipo_sel:
        dfm = dfm[dfm["tipo_indicador"].isin(tipo_sel)]
    if fonte_sel:
        dfm = dfm[dfm["fonte_dados"].isin(fonte_sel)]

    # ----------------- Grandes números -----------------
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

    # ----------------- Gráficos -----------------
    st.info("📊 Gráficos: Visuais para responder às perguntas segundo os filtros aplicados")

    if "tipo_indicador" in dfm and dfm["tipo_indicador"].notna().any():     
        with st.expander("Quais tipos de indicadores serão analisados?", expanded=True):
            st.plotly_chart(
                bar_count(dfm, "tipo_indicador", "Distribuição por tipo de indicador"),
                use_container_width=True
            )

    if "fonte_dados" in dfm and dfm["fonte_dados"].notna().any():
        with st.expander("Quais fontes de dados dos indicadores analisados?", expanded=True):
            st.plotly_chart(
                bar_count(dfm, "fonte_dados", "Distribuição por fonte de dados"),
                use_container_width=True
            )

    # ----------------- Detalhe por indicador -----------------
    st.info("📋 Tabelas: Veja detalhes da Ficha de cada indicador")

    tem_titulo = (
        "titulo_indicador" in dfm
        and dfm["titulo_indicador"].notna().any()
    )

    if tem_titulo:
        titulos = (
            dfm["titulo_indicador"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        sel = st.selectbox(
            "Escolha o indicador",
            options=titulos,
            key="mat_sel_titulo",
        )

        if sel:
            # ordem lógica de campos
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

            linha = (
                dfm[dfm["titulo_indicador"] == sel]
                .iloc[0][campos]
                .dropna()
            )

            # rótulos bonitinhos
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
            col_widths = [0.20, 0.80]  # 30% / 70%

            fig_tbl = go.Figure(
                data=[
                    go.Table(
                        columnwidth=[w * 100 for w in col_widths],  # Define proporções
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

# ====================== 2) Cadastro Estabelecimentos ======================
with abas[1]:
    st.subheader("🗂️ Cadastro de Estabelecimentos")
    df_est = load_table(TABLES["estabelecimentos"]).copy()
    st.info("🔎 Filtros: Use os controles abaixo para refinar os resultados.")

    # ----------------- Filtros -----------------
    # Linha 1 – filtros principais
    c1, c2, c3 = st.columns(3)
    comp_sel = c1.multiselect("Competência", options=_opts(df_est, "competencia"), key="est_comp", placeholder="Selecione a(s) opção(ões) desejadas")
    uf_sel   = c2.multiselect("Estabelecimento - UF", options=_opts(df_est, "no_uf"), key="est_uf",placeholder="Selecione a(s) opção(ões) desejadas")
    reg_sel  = c3.multiselect("Estabelecimento - Região", options=_opts(df_est, "no_regiao"), key="est_reg",placeholder="Selecione a(s) opção(ões) desejadas")

    # Linha 2 – território
    c4, c5, c6 = st.columns(3)
    mun_sel        = c4.multiselect("Estabelecimento - Município", options=_opts(df_est, "municipio"), key="est_mun",placeholder="Selecione a(s) opção(ões) desejadas")
    reg_saude_sel  = c5.multiselect("Estabelecimento - Região de Saúde", options=_opts(df_est, "cod_regiao_saude"), key="est_regsaude",placeholder="Selecione a(s) opção(ões) desejadas")
    micro_sel      = c6.multiselect("Estabelecimento - Microrregião Geográfica", options=_opts(df_est, "no_microrregiao"), key="est_micro",placeholder="Selecione a(s) opção(ões) desejadas")

    # Linha 3 – perfil do estabelecimento
    c7, c8, c9 = st.columns(3)
    tipo_sel    = c7.multiselect("Estabelecimento - Tipo",    options=_opts(df_est, "tipo_do_estabelecimento"), key="est_tipo",placeholder="Selecione a(s) opção(ões) desejadas")
    subtipo_sel = c8.multiselect("Estabelecimento - Subtipo", options=_opts(df_est, "subtipo_do_estabelecimento"), key="est_subtipo",placeholder="Selecione a(s) opção(ões) desejadas")
    gestao_sel  = c9.multiselect("Estabelecimento - Gestão", options=_opts(df_est, "gestao"), key="est_gestao",placeholder="Selecione a(s) opção(ões) desejadas")

    c10, c11, c12 = st.columns(3)
    convenio_sel = c10.multiselect("Estabelecimento - Convênio SUS", options=_opts(df_est, "convenio_sus"), key="est_convenio",placeholder="Selecione a(s) opção(ões) desejadas")
    natureza_sel = c11.multiselect("Estabelecimento - Natureza jurídica", options=_opts(df_est, "categoria_natureza_juridica"), key="est_natjur",placeholder="Selecione a(s) opção(ões) desejadas")
    status_sel   = c12.multiselect("Estabelecimento - Status", options=_opts(df_est, "status_do_estabelecimento"), key="est_status",placeholder="Selecione a(s) opção(ões) desejadas")


    # Linha 4 – IVS
    ivs_sel = st.multiselect("Estabelecimento - Município IVS (Índice de Vulnerabilidade Social)", options=_opts(df_est, "ivs"), key="est_ivs",placeholder="Selecione a(s) opção(ões) desejadas")
    
    def bool_multiselect(label: str, key: str, placeholder: str | None = None):
        return st.multiselect(
            label,
            options=["Sim", "Não"],
            key=key,
            placeholder=placeholder,
        )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        onco_cacon_sel = bool_multiselect(
            "CACON",
            "est_onco_cacon",
            placeholder="Selecione a(s) opção(ões) desejadas"
        )

    with col2:
        onco_unacon_sel = bool_multiselect(
            "UNACON",
            "est_onco_unacon",
            placeholder="Selecione a(s) opção(ões) desejadas"
        )

    with col3:
        onco_radio_sel = bool_multiselect(
            "Radioterapia",
            "est_onco_radio",
            placeholder="Selecione a(s) opção(ões) desejadas"
        )

    with col4:
        onco_quimio_sel = bool_multiselect(
            "Quimioterapia",
            "est_onco_quimio",
            placeholder="Selecione a(s) opção(ões) desejadas"
        )

    with col5:
        hab_onco_cir_sel = bool_multiselect(
            "Onco Cirúrgica",
            "est_hab_onco_cir",
            placeholder="Selecione a(s) opção(ões) desejadas"
        )

    # ----------------- Aplicar filtros -----------------
    dfe = df_est.copy()

    def apply_multisel(df, col, sel):
        if sel and col in df:
            return df[df[col].isin(sel)]
        return df

    dfe = apply_multisel(dfe, "competencia",  comp_sel)
    dfe = apply_multisel(dfe, "no_uf",        uf_sel)
    dfe = apply_multisel(dfe, "no_regiao",    reg_sel)
    dfe = apply_multisel(dfe, "municipio",    mun_sel)
    dfe = apply_multisel(dfe, "cod_regiao_saude", reg_saude_sel)
    dfe = apply_multisel(dfe, "no_microrregiao",  micro_sel)

    dfe = apply_multisel(dfe, "tipo_do_estabelecimento",    tipo_sel)
    dfe = apply_multisel(dfe, "subtipo_do_estabelecimento", subtipo_sel)
    dfe = apply_multisel(dfe, "gestao",                     gestao_sel)
    dfe = apply_multisel(dfe, "convenio_sus",               convenio_sel)
    dfe = apply_multisel(dfe, "categoria_natureza_juridica", natureza_sel)
    dfe = apply_multisel(dfe, "status_do_estabelecimento",  status_sel)
    dfe = apply_multisel(dfe, "ivs",                        ivs_sel)

    def apply_bool_multisel(df, col, sel):
        if not sel or col not in df:
            return df
        bool_series = df[col].astype("boolean")
        allowed = []
        if "Sim" in sel:
            allowed.append(True)
        if "Não" in sel:
            allowed.append(False)
        return df[bool_series.isin(allowed)]

    dfe = apply_bool_multisel(dfe, "onco_cacon",                         onco_cacon_sel)
    dfe = apply_bool_multisel(dfe, "onco_unacon",                        onco_unacon_sel)
    dfe = apply_bool_multisel(dfe, "onco_radioterapia",                  onco_radio_sel)
    dfe = apply_bool_multisel(dfe, "onco_quimioterapia",                 onco_quimio_sel)
    dfe = apply_bool_multisel(dfe, "habilitacao_agrupado_onco_cirurgica", hab_onco_cir_sel)

    # ----------------- Grandes números -----------------
    st.info("📏 Grandes números: Visão rápida com filtros aplicados")

    # Total de estabelecimentos (CNES distintos)
    tot_est = dfe["cnes"].nunique() if "cnes" in dfe else len(dfe)

    # Médias por Região e por UF
    if "cod_regiao_saude" in dfe and "cnes" in dfe:
        reg_counts = dfe.dropna(subset=["cod_regiao_saude"]).groupby("cod_regiao_saude")["cnes"].nunique()
        media_por_regiao = float(reg_counts.mean()) if not reg_counts.empty else 0.0
    else:
        media_por_regiao = 0.0

    if "no_uf" in dfe and "cnes" in dfe:
        uf_counts = dfe.dropna(subset=["no_uf"]).groupby("no_uf")["cnes"].nunique()
        media_por_uf = float(uf_counts.mean()) if not uf_counts.empty else 0.0
    else:
        media_por_uf = 0.0

    def count_hab(col):
        if col not in dfe or "cnes" not in dfe:
            return 0
        s = dfe[col].astype("boolean")
        return int(dfe[s]["cnes"].nunique())

    n_cacon   = count_hab("onco_cacon")
    n_unacon  = count_hab("onco_unacon")
    n_radio   = count_hab("onco_radioterapia")
    n_quimio  = count_hab("onco_quimioterapia")
    n_onco_cir = count_hab("habilitacao_agrupado_onco_cirurgica")

    # Layout dos KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Número de estabelecimentos", fmt_num(tot_est, 0))
    c2.metric("Média de estabelecimentos por Região de Saúde",fmt_num(media_por_regiao, 1),)
    c3.metric("Média de estabelecimentos por UF",fmt_num(media_por_uf, 1),)
    c4.metric("Estab. habilitados CACON", fmt_num(n_cacon, 0))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Estab. habilitados UNACON", fmt_num(n_unacon, 0))
    c6.metric("Estab. habilitados Radioterapia", fmt_num(n_radio, 0))
    c7.metric("Estab. habilitados Quimioterapia", fmt_num(n_quimio, 0))
    c8.metric("Estab. habilitados Onco Cirúrgica", fmt_num(n_onco_cir, 0))

    # Gráficos = Tabela 1 CNES Serviços Especializados (última competência)

    st.info("📊 Gráficos — Tabela 1 = CNES Serviços Especializados (Última competência)")

    # 1) Número de estabelecimentos por Tipo de Estabelecimento
    with st.expander(
        "Número de estabelecimentos por Tipo de Estabelecimento", expanded=True
    ):
        if "tipo_do_estabelecimento" in dfe and dfe["tipo_do_estabelecimento"].notna().any():
            fig_tipo = pareto_barh(
                dfe,
                "tipo_do_estabelecimento",
                None,
                "Número de estabelecimentos por Tipo de Estabelecimento",
                "Qtde",
            )
            st.plotly_chart(fig_tipo, use_container_width=True)
        else:
            st.write("Nenhum dado disponível para este gráfico.")

    # 2) Número de estabelecimentos por Tipo de Gestão
    with st.expander(
        "Número de estabelecimentos por Tipo de Gestão", expanded=True
    ):
        if "gestao" in dfe and dfe["gestao"].notna().any():
            fig_gestao = bar_count(
                dfe,
                "gestao",
                "Número de estabelecimentos por Tipo de Gestão",
                24,
            )
            st.plotly_chart(fig_gestao, use_container_width=True)
        else:
            st.write("Nenhum dado disponível para este gráfico.")

    # 3) Número de estabelecimentos por Convênio SUS
    with st.expander(
        "Número de estabelecimentos por Convênio SUS", expanded=True
    ):
        if "convenio_sus" in dfe and dfe["convenio_sus"].notna().any():
            # exemplo de agregação – ajuste os nomes conforme sua base
            df_conv = (
                df_est
                .groupby("convenio_sus", dropna=False)["cnes"]   # ou a coluna de id do estabelecimento
                .nunique()
                .reset_index(name="qtde_estab")
            )

            fig_convenio = pie_standard(
                df_conv,
                names="convenio_sus",
                values="qtde_estab",
                title="Número de estabelecimentos por Convênio SUS",
                hole=0.35,                 # pizza com “buraco” (donut)
                top_n=None,                # se quiser limitar e agrupar em “Outros”, defina um número
                legend_pos="below_title",
                percent_digits=1,
                number_digits=0,
                thousands_sep=".",
            )

            st.plotly_chart(fig_convenio, use_container_width=True)
        else:
            st.write("Nenhum dado disponível para este gráfico.")

    # 4) Número de estabelecimentos por Categoria da Natureza Jurídica
    with st.expander(
        "Número de estabelecimentos por Categoria da Natureza Jurídica", expanded=True
    ):
        col_nat = "categoria_natureza_juridica"
        if col_nat in dfe and dfe[col_nat].notna().any():
            fig_nat = bar_count(
                dfe,
                col_nat,
                "Número de estabelecimentos por Categoria da Natureza Jurídica",
                28,
            )
            st.plotly_chart(fig_nat, use_container_width=True)
        else:
            st.write("Nenhum dado disponível para este gráfico.")

    # ==============================================================
    # Tabela descritiva do cadastro de cada estabelecimento
    # ==============================================================

    st.info(
        "📋 Tabela 1 — Tabela descritiva do cadastro de cada estabelecimento "
        "(CNES Serviços Especializados, última competência)"
    )

    # escolhe colunas mais importantes, se existirem
    cols_base = [
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
    cols_exist = [c for c in cols_base if c in dfe.columns]

    if cols_exist:
        st.dataframe(
            dfe[cols_exist].sort_values(["no_uf", "municipio", "nome_fantasia"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("Não há colunas de cadastro disponíveis para exibir a tabela.")

# ====================== 3) Cadastro Serviços ======================
with abas[2]:
    st.subheader("🗂️ Cadastro Serviços")

    df_srv = load_table(TABLES["servicos"])

    st.info("🔎 Filtros — Serviços especializados")
    dfs = render_with_spinner_once("servicos", df_srv, FILTER_SPEC["servicos"], prefix="srv")

    for cand in ["servicos_especializados_servico_classificacao", "servicos_especializados_servico"]:
        if cand in dfs and dfs[cand].notna().any():
            st.plotly_chart(
                bar_count(dfs, cand, f"Distribuição por {cand}"),
                use_container_width=True,
            )
            break

# ====================== 4) Habilitação ======================
with abas[3]:
    st.subheader("✅ Habilitação")

    df_hab = load_table(TABLES["habilitacao"])

    st.info("🔎 Filtros — Habilitação")
    dfh = render_with_spinner_once("habilitacao", df_hab, FILTER_SPEC["habilitacao"], prefix="hab")

    col = "referencia_habilitacao_no_habilitacao"
    if col in dfh and dfh[col].notna().any():
        st.plotly_chart(
            pareto_barh(dfh, col, None, "Habilitações — Pareto", "Qtde"),
            use_container_width=True,
        )

# ====================== 5) Leitos ======================
with abas[4]:
    st.subheader("🛏️ Leitos")

    df_lei = load_table(TABLES["leitos"])

    st.info("🔎 Filtros — Leitos")
    dfl = render_with_spinner_once("leitos", df_lei, FILTER_SPEC["leitos"], prefix="lei")

    if {"leitos_tipo_leito_nome", "leitos_quantidade_total"}.issubset(dfl.columns):
        st.plotly_chart(
            pareto_barh(
                dfl,
                "leitos_tipo_leito_nome",
                "leitos_quantidade_total",
                "Leitos por tipo (soma quantidade total)",
                "Qtde",
            ),
            use_container_width=True,
        )

# ====================== 6) Equipamentos ======================
with abas[5]:
    st.subheader("🧰 Equipamentos")

    df_eq = load_table(TABLES["equipamentos"])

    st.info("🔎 Filtros — Equipamentos")
    dfeq = render_with_spinner_once("equipamentos", df_eq, FILTER_SPEC["equipamentos"], prefix="eq")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(dfeq):,}".replace(",", "."))
    c2.empty()
    c3.empty()

    if "equipamentos_tipo_equipamento" in dfeq:
        st.plotly_chart(
            bar_count(
                dfeq,
                "equipamentos_tipo_equipamento",
                "Distribuição por tipo de equipamento",
                28,
            ),
            use_container_width=True,
        )

# ====================== 7) Profissionais ======================
with abas[6]:
    st.subheader("🧑 Profissionais")

    df_prof = load_table(TABLES["profissionais"])

    st.info("🔎 Filtros — Profissionais")
    dfp = render_with_spinner_once("profissionais", df_prof, FILTER_SPEC["profissionais"], prefix="pro")

    for cand in ["cbo_ocupacao", "cbo_descricao", "profissionais_tipo_cbo"]:
        if cand in dfp and dfp[cand].notna().any():
            st.plotly_chart(
                pareto_barh(
                    dfp,
                    cand,
                    None,
                    "Distribuição de profissionais por ocupação",
                    "Qtde",
                ),
                use_container_width=True,
            )
            break

# ====================== 8) Registros Hospitalares ======================
with abas[7]:
    st.subheader("📋 Registros Hospitalares")
    st.info("Conteúdo em construção.")

# ====================== 9) Metodologia ======================
with abas[8]:
    st.subheader("📐 Metodologia")
    st.markdown(
        """
        Esta aba resume a **metodologia de construção do Complexo Produtivo Oncológico**:

        - **Fontes de dados**: CNES (estabelecimentos, serviços especializados, leitos, equipamentos,
          profissionais), base da Matriz de Indicadores e integrações internas.
        - **Critérios de inclusão**:
            - Estabelecimentos com habilitação CACON/UNACON, serviços de radioterapia e quimioterapia,
              e leitos/serviços cirúrgicos oncológicos;
            - Vínculo SUS e situação de funcionamento ativa.
        - **Unidades de análise**: estabelecimento, município, região de saúde e UF.
        - **Transformações**:
            - Normalização de códigos CNES/IBGE;
            - Agregações por competência (ano/mês);
            - Criação de flags booleanas para componentes do complexo (onco_cacon, onco_unacon etc.).
        - **Matriz de Indicadores**:
            - Cada linha corresponde a um indicador com definição, fórmula, numerador, denominador,
              fonte de dados e nível de desagregação.
        - **Atualização**:
            - Periodicidade de atualização conforme carga das bases CNES e demais fontes.

        *(Texto meramente ilustrativo — você pode substituir aqui pela descrição oficial da metodologia.)*
        """
    )
