# app.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
import plotly.graph_objects as go

# Gráficos reutilizáveis
from src.graph import pareto_barh, bar_count

# ==============================================================
# Config
# ==============================================================
st.set_page_config(layout="wide", page_title="📊 Painel Nescon EPSM")
PROJECT_ID = "escolap2p"

# ==============================================================
# BigQuery Client
# ==============================================================
def make_bq_client():
    # Usa Application Default Credentials (Cloud Shell/VM com SA)
    return bigquery.Client(project=PROJECT_ID)

client = make_bq_client()

# ==============================================================
# Tabelas
# ==============================================================
TABLES = {
    "matriz": "escolap2p.cliente_epsm.complexo_oncologico_matriz_indicadores",
    "estabelecimentos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_estabelecimentos_applications",
    "servicos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_servicos_especializados_applications",
    "habilitacao": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_habilitacao_mart",
    "leitos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_leitos_applications",
    "equipamentos": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_equipamentos_applications",
    "profissionais": "escolap2p.basedosdados_cnes.complexo_oncologico_cnes_profissionais_applications",
}

# ==============================================================
# FILTER SPEC por tabela
#  - year / month → slider de faixa
#  - str / int    → multiselect (em colunas)
#  - bool         → selector (Todos/True/False)
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
        "onco_cacon": "bool", "onco_unacon": "bool",
        "onco_radioterapia": "bool", "onco_quimioterapia": "bool",
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
        "onco_cacon": "bool", "onco_unacon": "bool",
        "onco_radioterapia": "bool", "onco_quimioterapia": "bool",
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
        "onco_cacon": "bool", "onco_unacon": "bool",
        "onco_radioterapia": "bool", "onco_quimioterapia": "bool",
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
        "onco_cacon": "bool", "onco_unacon": "bool",
        "onco_radioterapia": "bool", "onco_quimioterapia": "bool",
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
    """Desenha multiselects em colunas (até 4 por linha)."""
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
    """Renderiza os filtros com seções em destaque."""
    years   = [c for c, k in spec.items() if k == "year"  and c in df]
    months  = [c for c, k in spec.items() if k == "month" and c in df]
    strs    = [c for c, k in spec.items() if k == "str"   and c in df]
    ints    = [c for c, k in spec.items() if k == "int"   and c in df]
    bools   = [c for c, k in spec.items() if k == "bool"  and c in df]

    selections: dict[str, object] = {}

    # Destaque: Período
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

    # Destaque: Dimensões (em colunas)
    st.markdown("### 🧭 Dimensões")
    _draw_multis(df, strs, selections, prefix, "Filtrar")
    _draw_multis(df, ints, selections, prefix, "Filtrar")

    # Destaque: Booleanos
    if bools:
        st.markdown("### ✅ Filtros booleanos")
        cols = st.columns(min(4, len(bools)))
        for i, col in enumerate(bools):
            with cols[i % len(cols)]:
                selections[col] = ui_bool(df, col, f"{prefix}_{col}", col)

    return apply_filters(df, selections)

def render_with_spinner_once(name: str, df: pd.DataFrame, spec: dict, prefix: str) -> pd.DataFrame:
    """
    Mostra spinner '⏳ Carregando filtros...' apenas na 1ª vez que a aba é aberta.
    Em execuções subsequentes, renderiza direto (sem spinner).
    """
    key = f"__spinner_done_{name}"
    if not st.session_state.get(key):
        with st.spinner("⏳ Carregando filtros..."):
            out = render_filters(df, spec, prefix)
        st.session_state[key] = True
        return out
    return render_filters(df, spec, prefix)

# ==============================================================
# Sidebar
# ==============================================================
st.markdown(
    "<style>[data-testid='stSidebarNav']{display:none;}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("<div style='margin: 20px 0;'><hr style='border:none;border-top:1px solid #ccc;'/></div>", unsafe_allow_html=True)
    st.header("Menu")
    st.page_link("app.py", label="Complexo Oncológico", icon="📊")
    st.page_link("pages/criacao.py", label="Referência", icon="✅")
    st.markdown("<div style='margin: 20px 0;'><hr style='border:none;border-top:1px solid #ccc;'/></div>", unsafe_allow_html=True)
    if st.button("🔄 Atualizar cache agora"):
        st.cache_data.clear()
        st.rerun()

# ==============================================================
# Cabeçalho
# ==============================================================
st.markdown("""
    <div style='background: linear-gradient(to right, #004e92, #000428); padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white;'>📊 Análise do Complexo Produtivo da Saúde Oncológica</h1>
        <p style='color: white;'>Explore os dados do Complexo Produtivo para tomada de decisões.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================
# Abas
# ==============================================================
abas = st.tabs([
    "🧮 Matriz de Indicadores",
    "🗂️ Cadastro Estabelecimentos",
    "🗂️ Cadastro Serviços",
    "✅ Habilitação",
    "🛏️ Leitos",
    "🧰 Equipamentos",
    "🧑 Profissionais",
    "📋 Registros Hospitalares",
])

# ====================== 1) Matriz de Indicadores ======================
with abas[0]:
    st.subheader("🧮 Matriz de Indicadores")

    df_matriz = load_table(TABLES["matriz"])
    if "Data" in df_matriz.columns:
        df_matriz["Data"] = pd.to_datetime(df_matriz["Data"], errors="coerce")
        df_matriz["Ano"] = df_matriz["Data"].dt.year

    st.info("🔎 Filtros")
    c1, c2 = st.columns(2)
    areas = sorted(df_matriz.get("Área do Indicador", pd.Series(dtype=str)).dropna().unique())
    fontes = sorted(df_matriz.get("Fonte de dados", pd.Series(dtype=str)).dropna().unique())
    area_sel = c1.multiselect("Área do Indicador", options=areas, key="mat_areas")
    fonte_sel = c2.multiselect("Fonte de dados", options=fontes, key="mat_fontes")

    dfm = df_matriz.copy()
    if area_sel: dfm = dfm[dfm["Área do Indicador"].isin(area_sel)]
    if fonte_sel: dfm = dfm[dfm["Fonte de dados"].isin(fonte_sel)]

    st.info("📏 Grandes números")
    c1, c2, c3 = st.columns(3)
    c1.metric("Indicadores", f"{len(dfm):,}".replace(",", "."))
    c2.metric("Áreas distintas", int(dfm["Área do Indicador"].nunique() if "Área do Indicador" in dfm else 0))
    c3.metric("Fontes distintas", int(dfm["Fonte de dados"].nunique() if "Fonte de dados" in dfm else 0))

    st.info("📊 Gráficos")
    st.plotly_chart(bar_count(dfm, "Área do Indicador", "Distribuição por Área do Indicador"), use_container_width=True)
    if "Fonte de dados" in dfm and dfm["Fonte de dados"].notna().any():
        st.plotly_chart(bar_count(dfm, "Fonte de dados", "Distribuição por Fonte de dados", 22), use_container_width=True)

    st.info("📋 Tabela (detalhe)")
    tem_titulo = "Título do indicador" in dfm and dfm["Título do indicador"].notna().any()
    if tem_titulo:
        titulos = dfm["Título do indicador"].dropna().unique().tolist()
        sel = st.selectbox("Escolha o indicador", options=titulos, key="mat_sel_titulo")
        if sel:
            linha = dfm[dfm["Título do indicador"] == sel].iloc[0].dropna()
            labels = linha.index.tolist()
            valores = linha.values.tolist()
            fig_tbl = go.Figure(data=[
                go.Table(
                    header=dict(values=["<b>Campo</b>", "<b>Valor</b>"], fill_color="#E8F0FF", align="left", height=30),
                    cells=dict(values=[labels, valores], align="left", height=26),
                )
            ])
            fig_tbl.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=min(720, 26*(len(labels)+2)))
            st.plotly_chart(fig_tbl, use_container_width=True)
    else:
        st.info("Nenhum título disponível para detalhamento.")

# ====================== 2) Cadastro Estabelecimentos ======================
with abas[1]:
    st.subheader("🗂️ Cadastro Estabelecimentos")
    df_est = load_table(TABLES["estabelecimentos"])

    st.info("🔎 Filtros — Estabelecimentos")
    dfe = render_with_spinner_once("estabelecimentos", df_est, FILTER_SPEC["estabelecimentos"], prefix="est")

    c1, c2, c3 = st.columns(3)
    tot = dfe["cnes"].nunique() if "cnes" in dfe else len(dfe)
    c1.metric("Estabelecimentos", f"{tot:,}".replace(",", "."))
    c2.metric("Municípios", int(dfe["municipio"].nunique() if "municipio" in dfe else 0))
    c3.metric("UF", int(dfe["uf"].nunique() if "uf" in dfe else 0))

    if "tipo_do_estabelecimento" in dfe:
        st.plotly_chart(
            pareto_barh(dfe, "tipo_do_estabelecimento", None, "Tipos de Estabelecimento — Pareto", "Qtde"),
            use_container_width=True
        )

# ====================== 3) Cadastro Serviços ======================
with abas[2]:
    st.subheader("🗂️ Cadastro Serviços")
    df_srv = load_table(TABLES["servicos"])

    st.info("🔎 Filtros — Serviços especializados")
    dfs = render_with_spinner_once("servicos", df_srv, FILTER_SPEC["servicos"], prefix="srv")

    for cand in ["servicos_especializados_servico_classificacao", "servicos_especializados_servico"]:
        if cand in dfs and dfs[cand].notna().any():
            st.plotly_chart(bar_count(dfs, cand, f"Distribuição por {cand}"), use_container_width=True)
            break

# ====================== 4) Habilitação ======================
with abas[3]:
    st.subheader("✅ Habilitação")
    df_hab = load_table(TABLES["habilitacao"])

    st.info("🔎 Filtros — Habilitação")
    dfh = render_with_spinner_once("habilitacao", df_hab, FILTER_SPEC["habilitacao"], prefix="hab")

    col = "referencia_habilitacao_no_habilitacao"
    if col in dfh and dfh[col].notna().any():
        st.plotly_chart(pareto_barh(dfh, col, None, "Habilitações — Pareto", "Qtde"), use_container_width=True)

# ====================== 5) Leitos ======================
with abas[4]:
    st.subheader("🛏️ Leitos")
    df_lei = load_table(TABLES["leitos"])

    st.info("🔎 Filtros — Leitos")
    dfl = render_with_spinner_once("leitos", df_lei, FILTER_SPEC["leitos"], prefix="lei")

    if {"leitos_tipo_leito_nome", "leitos_quantidade_total"}.issubset(dfl.columns):
        st.plotly_chart(
            pareto_barh(dfl, "leitos_tipo_leito_nome", "leitos_quantidade_total",
                        "Leitos por tipo (soma quantidade total)", "Qtde"),
            use_container_width=True
        )

# ====================== 6) Equipamentos ======================
with abas[5]:
    st.subheader("🧰 Equipamentos")
    df_eq = load_table(TABLES["equipamentos"])

    st.info("🔎 Filtros — Equipamentos")
    dfeq = render_with_spinner_once("equipamentos", df_eq, FILTER_SPEC["equipamentos"], prefix="eq")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(dfeq):,}".replace(",", "."))
    if "equipamentos_tipo_equipamento" in dfeq:
        st.plotly_chart(
            bar_count(dfeq, "equipamentos_tipo_equipamento", "Distribuição por tipo de equipamento", 28),
            use_container_width=True
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
                pareto_barh(dfp, cand, None, "Distribuição de profissionais por ocupação", "Qtde"),
                use_container_width=True
            )
            break

# ====================== 8) Registros Hospitalares ======================
with abas[7]:
    st.subheader("📋 Registros Hospitalares")
    st.info("Conteúdo em construção")
