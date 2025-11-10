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

# Configure a página o quanto antes (recomendação Streamlit)
st.set_page_config(layout="wide", page_title="📊 Painel Nescon EPSM")

# ---------------------------------------------------------------
# BigQuery via ADC (Service Account anexada ou Cloud Shell)
# ---------------------------------------------------------------
PROJECT_ID = "escolap2p"
QUERY_ONCO = """
    SELECT *
    FROM `escolap2p.cliente_epsm.complexo_oncologico_lista_cnes_staging`
"""

def make_bq_client():
    # Usa Application Default Credentials (ADC):
    # - Cloud Shell
    # - VM com Service Account anexada
    return bigquery.Client(project=PROJECT_ID)

client = make_bq_client()

# ---------------------------------------------------------------
# Aquisição de dados do BigQuery
# ---------------------------------------------------------------

# --- TABELA: Matriz de Indicadores ---
QUERY_MATRIZ = """
    SELECT
      * EXCEPT(`Priorização`, `Data`, `Sugestão`)
    FROM `escolap2p.cliente_epsm.complexo_oncologico_matriz_indicadores`
"""
@st.cache_data(ttl=1800)  # 30 min
def consultar_matriz_indicadores():
    df = client.query(QUERY_MATRIZ).to_dataframe()
    # Normaliza tipos e nomes úteis p/ filtros
    # (mantém os nomes originais p/ exibir, mas cria colunas auxiliares)
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df["Ano"] = df["Data"].dt.year
    if "Priorização" in df.columns:
        df["Priorizacao_num"] = pd.to_numeric(df["Priorização"], errors="coerce")
    # strings: strip
    for c in ["Fonte de dados","Área do Indicador","Título do indicador",
              "Definição do indicador","Propósito do indicador","Interpretação",
              "Usos","Método de cálculo","Unidade de medida","Frequência de mensuração",
              "Área de referência","Período de tempo de referência",
              "Nível de desagregação","Limitações","Sugestão"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    return df

# --- TABELA: Cadastro de estabelecimentos ---
QUERY_CADASTRO = """
SELECT
  *
FROM `escolap2p.cliente_epsm.complexo_oncologico_lista_cnes_staging`
WHERE TIPO = "CACON" OR TIPO = "UNACON"
"""

@st.cache_data(ttl=1800)
def consultar_cadastro():
    df = client.query(QUERY_CADASTRO).to_dataframe()

    # Normalizações úteis
    # numéricos
    for c in ["CNES", "CÓDIGO DO MUNICÍPIO", "municipio", "uf_1", "mesoregion",
              "microregion", "rgint", "rgi", "osm_relation_id",
              "lon", "lat", "pop_21", "area_k2", "municipio_cod_2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # boolean
    if "is_capital" in df.columns:
        df["is_capital"] = df["is_capital"].fillna(False).astype(bool)

    # strings: strip
    for c in ["UF","MUNICÍPIO","NOME FANTASIA","TIPO NOVO DO ESTABELECIMENTO",
              "TIPO DO ESTABELECIMENTO","SUBTIPO DO ESTABELECIMENTO","GESTÃO",
              "CONVÊNIO SUS","CATEGORIA NATUREZA JURÍDICA","SERVIÇO",
              "SERVIÇO CLASSIFICAÇÃO","STATUS DO ESTABELECIMENTO","name","no_accents",
              "slug_name","alternative_names"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # coluna auxiliar SUS binária (% SUS)
    if "CONVÊNIO SUS" in df.columns:
        df["conv_sus_bool"] = df["CONVÊNIO SUS"].str.upper().isin(["SIM", "S", "YES", "TRUE"])

    # coluna auxiliar Ativo
    if "STATUS DO ESTABELECIMENTO" in df.columns:
        df["ativo_bool"] = df["STATUS DO ESTABELECIMENTO"].str.upper().str.contains("ATIVO")

    # ---- Faixas de população (IBGE-like) ----
    # 0–20k | 20k–100k | 100k–500k | 500k–1M | 1M+
    if "pop_21" in df.columns:
        bins = [-np.inf, 20_000, 100_000, 500_000, 1_000_000, np.inf]
        labels = [
            "até 20 mil",
            "20–100 mil",
            "100–500 mil",
            "500 mil–1 milhão",
            "acima de 1 milhão",
        ]
        df["Faixa populacional"] = pd.cut(df["pop_21"], bins=bins, labels=labels, right=True)

    return df

def vspace(px: int = 12):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Funções de gráficos
# ---------------------------------------------------------------

# ================= Pareto Horizontal com eixo superior (%) =================
def pareto_barh(
    df: pd.DataFrame,
    cat_col: str,
    value_col: str | None = None,          # None => faz contagem
    title: str = "",
    colorbar_title: str = "",
    highlight_value: str | None = None,    # ex.: "Não identificado"
):
    # 1) agrega e ordena
    if value_col is None:
        df_agg = df.groupby(cat_col, dropna=False).size().reset_index(name="count")
    else:
        df_agg = df.groupby(cat_col, dropna=False)[value_col].sum().reset_index(name="count")

    dfp = df_agg.sort_values("count", ascending=False).reset_index(drop=True)
    dfp[cat_col] = dfp[cat_col].astype(str)
    total = dfp["count"].sum()
    dfp["pct"] = 100 * dfp["count"] / total
    dfp["cum_pct"] = dfp["pct"].cumsum()
    dfp["label_text"] = [f"{c:,} ({p:.1f}%)".replace(",", ".") for c, p in zip(dfp["count"], dfp["pct"])]
    cats = dfp[cat_col]

    # 2) barras horizontais
    fig = go.Figure(go.Bar(
        y=cats,
        x=dfp["count"],
        orientation="h",
        text=dfp["label_text"],
        textposition="outside",
        cliponaxis=False,
        name="Quantidade",
        marker=dict(
            color=dfp["count"],
            colorscale="Blues",
            colorbar=dict(title=colorbar_title or "Quantidade", x=0.90, xanchor="left")
        ),
        hovertemplate="<b>%{y}</b><br>Qtde: %{x}<extra></extra>",
    ))

    # 3) curva de Pareto no eixo superior (x2)
    fig.add_trace(go.Scatter(
        x=dfp["cum_pct"],
        y=cats,
        mode="lines+markers+text",
        line=dict(color="black", width=3, shape="spline"),
        marker=dict(size=8, color="white", line=dict(color="black", width=2)),
        text=[f"{v:.1f}%" for v in dfp["cum_pct"]],
        textposition="middle right",
        name="Acumulado (%)",
        xaxis="x2",
        hovertemplate="<b>%{y}</b><br>Acumulado: %{x:.1f}%<extra></extra>",
    ))

    # 4) layout / eixos
    xmax = float(dfp["count"].max()) * 1.45
    left_margin = max(200, int(min(420, cats.map(len).max() * 8)))  # margem para rótulos longos

    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=left_margin, r=160, t=110, b=90),  # +top, +bottom
        height=max(440, 26 * len(dfp) + 170),
        legend=dict(
            orientation="h",
            y=-0.28, yanchor="top",          # desce a legenda
            x=0.5, xanchor="center"
        ),
    )
    # domínio x + x2 (deixa um “gutter” à direita para a colorbar)
    x_domain = [0.0, 0.78]

    fig.update_layout(
        xaxis=dict(
            domain=x_domain,
            range=[0, xmax],
            title="Quantidade",
            title_standoff=18,                # respiro do título X
            showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        ),
        xaxis2=dict(
            overlaying="x", side="top",
            domain=x_domain,
            range=[0, 105],                   # vai além de 100% para respiro
            tickvals=[0, 20, 40, 60, 80, 100],
            ticksuffix="%",
            showgrid=False,
            title="Acumulado (%)",
            title_standoff=10
        ),
        yaxis=dict(autorange="reversed", title=""),
    )

    # 5) faixas A/B/C (80/95) + linhas guias
    th_A, th_B = 80, 95
    for (x0, x1, col) in [(0, th_A, "rgba(46, 204, 113, 0.18)"),
                          (th_A, th_B, "rgba(243, 156, 18, 0.18)"),
                          (th_B, 100, "rgba(231, 76, 60, 0.16)")]:
        fig.add_shape(type="rect", xref="x2", yref="paper",
                      x0=x0, x1=x1, y0=0, y1=1, fillcolor=col, line=dict(width=0), layer="below")
    for x in (th_A, th_B):
        fig.add_shape(type="line", xref="x2", yref="paper",
                      x0=x, x1=x, y0=0, y1=1, line=dict(color="gray", width=2, dash="dash"))
    fig.add_annotation(x=th_A/2, y=0.5, xref="x2", yref="paper",
                       text="<b>A</b>", showarrow=False, font=dict(size=20, color="rgba(0,0,0,0.6)"))
    fig.add_annotation(x=(th_A+th_B)/2, y=0.5, xref="x2", yref="paper",
                       text="<b>B</b>", showarrow=False, font=dict(size=20, color="rgba(0,0,0,0.6)"))
    fig.add_annotation(x=(th_B+100)/2, y=0.5, xref="x2", yref="paper",
                       text="<b>C</b>", showarrow=False, font=dict(size=20, color="rgba(0,0,0,0.6)"))

    # 6) destaque opcional de uma categoria
    if highlight_value is not None and highlight_value in set(dfp[cat_col]):
        colors = [("crimson" if v == highlight_value else c)
                  for v, c in zip(dfp[cat_col], dfp["count"])]
        fig.update_traces(selector=dict(type="bar"),
                          marker=dict(color=colors, colorscale="Blues",
                                      colorbar=dict(title=colorbar_title or "Quantidade",
                                                    x=0.90, xanchor="left")))
    # Evita cortes na curva/labels
    fig.update_traces(
        selector=dict(type="bar"),
        marker=dict(
            color=dfp["count"],
            colorscale="Blues",
            colorbar=dict(title=colorbar_title or "Quantidade", x=0.88, xanchor="left")  # antes ~0.90
        )
    )
    return fig

# ================= Mapa =================
def _normalize_lonlat(series: pd.Series) -> pd.Series:
    """
    Normaliza longitude/latitude caso venham inteiras escaladas (1e6, 1e7, 1e8, 1e9).
    Se já estiver em graus (<= 180), retorna como está.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s

    absmax = float(s.abs().max())
    if absmax <= 180:
        return s  # já está em graus

    # tenta dividir por fatores padrão até cair em [-180, 180]
    for f in (1e6, 1e7, 1e8, 1e9):
        if (s / f).abs().max() <= 180:
            return s / f

    # fallback: divide por 1e9
    return s / 1e9

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
        <h1 style='color: white;'>📊 Análise do Complexo Produtivo da Saúde Oncológica</h1>
        <p style='color: white;'>Explore os dados do Complexo Produtivo para tomada de decisões.</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Conteúdo
# ---------------------------------------------------------------
st.header("🎲 Tipos de Análise")

abas = st.tabs(["🧮 Matriz de Indicadores","🗂️ Cadastro", "✅ Habilitação", "🛏️ Leitos", "🧰 Equipamentos", "🧑 Profissionais", "📋 Registros Hospitalares"])

# ────────────────────────────────────────────────────────────────
#  ABA 1  –  Cadastro
# ────────────────────────────────────────────────────────────────
with abas[0]:
    vspace(12)
    st.subheader("🧮 Matriz de Indicadores")
    vspace(12)

    with st.spinner("Carregando matriz de indicadores..."):
        df_matriz = consultar_matriz_indicadores()

    if df_matriz.empty:
        st.warning("Não há registros na matriz de indicadores.")
        st.stop()

    # -------------------- FILTROS --------------------
    st.info("**🔎 Sessão de Filtros - Use os controles abaixo para refinar os resultados.**")

    colf1, colf2, colf3, colf4 = st.columns([1.1, 1.1, 1, 1])

    areas = sorted([a for a in df_matriz.get("Área do Indicador", pd.Series(dtype=str)).dropna().unique()])
    area_sel = colf1.multiselect("Área do Indicador", options=areas, default=[],
                                placeholder="Selecione...")

    fontes = sorted([f for f in df_matriz.get("Fonte de dados", pd.Series(dtype=str)).dropna().unique()])
    fonte_sel = colf2.multiselect("Fonte de dados", options=fontes, default=[],
                                placeholder="Selecione...")

    # Aplica filtros
    df_fil = df_matriz.copy()
    if area_sel:
        df_fil = df_fil[df_fil["Área do Indicador"].isin(area_sel)]
    if fonte_sel:
        df_fil = df_fil[df_fil["Fonte de dados"].isin(fonte_sel)]

    vspace(12)

    # -------------------- MÉTRICAS (big numbers) --------------------
    st.info("**📏 Sessão de Grandes Números - Visão rápida dos principais indicadores agregados com base nos filtros aplicados.**")
    c1, c2, c3 = st.columns(3)

    total_ind = len(df_fil)
    n_areas = df_fil["Área do Indicador"].nunique() if "Área do Indicador" in df_fil else 0
    n_fontes = df_fil["Fonte de dados"].nunique() if "Fonte de dados" in df_fil else 0

    c1.metric("Indicadores", f"{total_ind:,}".replace(",", "."))
    c2.metric("Áreas distintas", n_areas)
    c3.metric("Fontes distintas", n_fontes)
    
    vspace(12)
    # -------------------- GRÁFICO 1 --------------------
    st.info("📊 **Sessão de Gráficos - Visualizações para explorar distribuição, comparação entre grupos e padrões do conjunto filtrado.**")

    col_toggle, _ = st.columns([0.25, 0.75], gap="small")
    with col_toggle:
        expand_matriz = st.toggle("Abrir todos os gráficos", value=True, key="open_matriz")

    with st.expander("Como os indicadores se distribuem por área?", expanded=expand_matriz):
        if "Área do Indicador" in df_fil:
            cont = df_fil["Área do Indicador"].value_counts().reset_index()
            cont.columns = ["Área do Indicador", "Qtde"]
            if not cont.empty:
                def wrap_label(s: str, max_chars: int = 18) -> str:
                    s = str(s)
                    if len(s) <= max_chars: return s
                    words, lines, cur = s.split(), [], ""
                    for w in words:
                        if len(cur) + len(w) + 1 <= max_chars:
                            cur = (cur + " " + w).strip()
                        else:
                            lines.append(cur); cur = w
                    if cur: lines.append(cur)
                    return "<br>".join(lines)

                vals = cont["Qtde"]
                colorscale = "Blues"
                ymax = float(vals.max()) * 1.25
                fig = go.Figure(data=[
                    go.Bar(
                        x=cont["Área do Indicador"], y=vals,
                        text=vals, textposition="outside", cliponaxis=False,
                        marker=dict(color=vals, colorscale=colorscale, showscale=False),
                        hovertemplate="<b>%{x}</b><br>Quantidade: %{y}<extra></extra>",
                    )
                ])
                fig.update_layout(
                    title="Distribuição por Área do Indicador",
                    xaxis_title="", yaxis_title="Quantidade",
                    height=380, margin=dict(l=20, r=20, t=70, b=60),
                    uniformtext_minsize=10, uniformtext_mode="show",
                    paper_bgcolor="white",
                    plot_bgcolor="white"
                )
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=cont["Área do Indicador"],
                    ticktext=[wrap_label(x, 18) for x in cont["Área do Indicador"]],
                    tickangle=0, automargin=True
                )
                fig.update_yaxes(range=[0, ymax])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem dados para o filtro atual.")

    # -------------------- GRÁFICO 2 --------------------
    with st.expander("Quais as principais fontes de dados (ou frequência de mensuração)?", expanded=expand_matriz):
        def wrap_label2(s: str, max_chars: int = 22) -> str:
            s = str(s)
            if len(s) <= max_chars: return s
            words, lines, cur = s.split(), [], ""
            for w in words:
                if len(cur) + len(w) + 1 <= max_chars:
                    cur = (cur + " " + w).strip()
                else:
                    lines.append(cur); cur = w
            if cur: lines.append(cur)
            return "<br>".join(lines)

        campo = None
        if "Fonte de dados" in df_fil and df_fil["Fonte de dados"].notna().any():
            campo = "Fonte de dados"
        elif "Frequência de mensuração" in df_fil and df_fil["Frequência de mensuração"].notna().any():
            campo = "Frequência de mensuração"

        if campo:
            cont2 = df_fil[campo].fillna("—").value_counts().reset_index()
            cont2.columns = [campo, "Qtde"]
            cont2 = cont2.sort_values("Qtde", ascending=False)
            if not cont2.empty:
                vals = cont2["Qtde"]
                colorscale = "Blues"
                ymax = float(vals.max()) * 1.25
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=cont2[campo], y=vals,
                        text=vals, textposition="outside", cliponaxis=False,
                        marker=dict(color=vals, colorscale=colorscale, showscale=False),
                        hovertemplate=f"<b>%{{x}}</b><br>Quantidade: %{{y}}<extra></extra>",
                    )
                ])
                fig2.update_layout(
                    title=f"Distribuição por {campo}",
                    xaxis_title="", yaxis_title="Quantidade",
                    height=380, margin=dict(l=20, r=20, t=60, b=60),
                    uniformtext_minsize=10, uniformtext_mode="show",
                    paper_bgcolor="white",
                    plot_bgcolor="white"
                )
                fig2.update_xaxes(
                    tickmode="array",
                    tickvals=cont2[campo],
                    ticktext=[wrap_label2(x, 22) for x in cont2[campo]],
                    tickangle=0, automargin=True
                )
                fig2.update_yaxes(range=[0, ymax])
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Campo de fonte/frequência não disponível para o filtro atual.")

    # -------------------- TABELA --------------------
    st.info("**📋 Sessão de Tabelas - Detalhamento linha a linha dos registros; use para consulta, conferência e exportação.**")
    # Ordenação para escolher o indicador com mais prioridade primeiro
    ordem_cols = [c for c in [
        "Priorização","Data","Área do Indicador","Título do indicador","Definição do indicador",
        "Propósito do indicador","Interpretação","Usos","Método de cálculo","Unidade de medida",
        "Frequência de mensuração","Área de referência","Período de tempo de referência",
        "Nível de desagregação","Limitações","Fonte de dados","Sugestão","Priorizacao_num"
    ] if c in df_fil.columns]
    df_view = df_fil[ordem_cols].copy() if ordem_cols else df_fil.copy()
    if "Priorizacao_num" in df_view:
        df_view = df_view.sort_values(["Priorizacao_num"], ascending=[False])

    # helper para renderizar tabela transposta
    def render_plotly_transposed(row_series):
        labels = row_series.index.tolist()
        valores = row_series.values.tolist()
        fig_tbl = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["<b>Campo</b>", "<b>Valor</b>"],
                        fill_color="#E8F0FF",
                        align="left",
                        font=dict(size=12),
                        height=30
                    ),
                    cells=dict(
                        values=[labels, valores],
                        align="left",
                        height=26,
                        fill_color=[["#FFFFFF", "#F9FBFF"] * ((len(labels)//2)+1),
                                ["#FFFFFF", "#F9FBFF"] * ((len(valores)//2)+1)],
                    )
                )
            ]
        )
        fig_tbl.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=min(720, 26*(len(labels)+2)))
        st.plotly_chart(fig_tbl, use_container_width=True)

    # Selectbox para escolher o indicador (sempre transposto)
    tem_titulo = "Título do indicador" in df_view and df_view["Título do indicador"].notna().any()
    if tem_titulo:
        titulos = df_view["Título do indicador"].dropna().unique().tolist()
        # Índice padrão: o de maior prioridade (já ordenado acima)
        default_idx = 0
        sel = st.selectbox("Escolha o indicador", options=titulos, index=default_idx,
                        placeholder="Selecione um indicador...")
        if sel:
            linha = df_view[df_view["Título do indicador"] == sel].iloc[0].dropna()
            render_plotly_transposed(linha)
        else:
            st.info("Selecione um indicador para ver os detalhes.")
    else:
        if len(df_view):
            st.info("Coluna 'Título do indicador' não encontrada. Exibindo a primeira linha.")
            linha = df_view.iloc[0].dropna()
            render_plotly_transposed(linha)
        else:
            st.warning("Não há registros após os filtros.")

    
# ────────────────────────────────────────────────────────────────
#  ABA 1  –  Cadastro
# ────────────────────────────────────────────────────────────────
with abas[1]:
    vspace(12)
    st.subheader("🗂️ Cadastro")
    vspace(12)

    with st.spinner("Carregando cadastro de estabelecimentos..."):
        df_cad = consultar_cadastro()

    if df_cad.empty:
        st.warning("Base de cadastro vazia.")
        st.stop()

    # -------------------- FILTROS (topo) --------------------
    st.info("**🔎 Sessão de Filtros - Use os controles abaixo para refinar os resultados.**")
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    f5, f6 = st.columns([1, 1])

    # opções
    uf_opts  = sorted(df_cad["UF"].dropna().unique()) if "UF" in df_cad else []
    mun_opts = sorted(df_cad["MUNICÍPIO"].dropna().unique()) if "MUNICÍPIO" in df_cad else []
    st_opts  = sorted(df_cad["STATUS DO ESTABELECIMENTO"].dropna().unique()) if "STATUS DO ESTABELECIMENTO" in df_cad else []
    tipo_opts = sorted(df_cad["TIPO DO ESTABELECIMENTO"].dropna().unique()) if "TIPO DO ESTABELECIMENTO" in df_cad else []
    subtipo_opts = sorted(df_cad["SUBTIPO DO ESTABELECIMENTO"].dropna().unique()) if "SUBTIPO DO ESTABELECIMENTO" in df_cad else []
    gestao_opts = sorted(df_cad["GESTÃO"].dropna().unique()) if "GESTÃO" in df_cad else []
    sus_opts = sorted(df_cad["CONVÊNIO SUS"].dropna().unique()) if "CONVÊNIO SUS" in df_cad else []

    uf_sel  = f1.multiselect("UF", options=uf_opts, placeholder="Selecione...")
    mun_sel = f2.multiselect("Município", options=mun_opts, placeholder="Selecione...")
    status_sel = f3.multiselect("Status do estabelecimento", options=st_opts, placeholder="Selecione...")
    tipo_sel = f4.multiselect("Tipo do estabelecimento", options=tipo_opts, placeholder="Selecione...")
    subtipo_sel = f5.multiselect("Subtipo do estabelecimento", options=subtipo_opts, placeholder="Selecione...")
    gestao_sel = f6.multiselect("Gestão", options=gestao_opts, placeholder="Selecione...")
    sus_sel = st.multiselect("Convênio SUS", options=sus_opts, placeholder="Selecione...")

    # Faixa populacional (categorias)
    faixa_opts = sorted(df_cad["Faixa populacional"].dropna().unique()) if "Faixa populacional" in df_cad else []
    faixa_sel = st.multiselect("Faixa populacional do município", options=faixa_opts, placeholder="Selecione...")

    # aplica filtros
    df_cf = df_cad.copy()
    if uf_sel:
        df_cf = df_cf[df_cf["UF"].isin(uf_sel)]
    if mun_sel:
        df_cf = df_cf[df_cf["MUNICÍPIO"].isin(mun_sel)]
    if status_sel:
        df_cf = df_cf[df_cf["STATUS DO ESTABELECIMENTO"].isin(status_sel)]
    if tipo_sel:
        df_cf = df_cf[df_cf["TIPO DO ESTABELECIMENTO"].isin(tipo_sel)]
    if subtipo_sel:
        df_cf = df_cf[df_cf["SUBTIPO DO ESTABELECIMENTO"].isin(subtipo_sel)]
    if gestao_sel:
        df_cf = df_cf[df_cf["GESTÃO"].isin(gestao_sel)]
    if sus_sel:
        df_cf = df_cf[df_cf["CONVÊNIO SUS"].isin(sus_sel)]
    if faixa_sel and "Faixa populacional" in df_cf:
        df_cf = df_cf[df_cf["Faixa populacional"].isin(faixa_sel)]

    vspace(12)

    # -------------------- BIG NUMBERS --------------------
    st.info("**📏 Sessão de Grandes Números - Visão rápida dos principais indicadores agregados com base nos filtros aplicados.**")
    c1, c2, c3, c4 = st.columns(4)
    total_est = df_cf["CNES"].nunique() if "CNES" in df_cf else len(df_cf)
    pct_sus = (100 * df_cf["conv_sus_bool"].mean()) if "conv_sus_bool" in df_cf and len(df_cf) else 0
    pct_ativo = (100 * df_cf["ativo_bool"].mean()) if "ativo_bool" in df_cf and len(df_cf) else 0
    municipios_cov = df_cf["MUNICÍPIO"].nunique() if "MUNICÍPIO" in df_cf else 0

    c1.metric("Estabelecimentos", f"{total_est:,}".replace(",", "."))
    c2.metric("% com convênio SUS", f"{pct_sus:.1f}%")
    c3.metric("% ativos", f"{pct_ativo:.1f}%")
    c4.metric("Municípios cobertos", municipios_cov)

    vspace(12)

    st.info("📊 **Sessão de Gráficos - Visualizações para explorar distribuição, comparação entre grupos e padrões do conjunto filtrado.**")
    col_toggle_cad, _ = st.columns([0.25, 0.75], gap="small")
    with col_toggle_cad:
        expand_cadastro = st.toggle("Abrir todos os gráficos", value=True, key="open_cadastro")

    with st.expander("Onde estão os estabelecimentos no mapa?", expanded=True):
    # --- selecione as colunas que tiver no seu df filtrado ---
        col_lon = "lon"
        col_lat = "lat"
        col_city = "MUNICÍPIO"           # ajuste se o nome for outro
        col_tipo = "TIPO DO ESTABELECIMENTO"  # opcional para hover

        cols_exist = [c for c in [col_lon, col_lat] if c in df_cf.columns]
        if len(cols_exist) < 2:
            st.info("Não encontrei as colunas de coordenadas (lon/lat) no conjunto filtrado.")
        else:
            # prepara base de pontos
            df_map = df_cf[[col_lon, col_lat, col_city, col_tipo]].copy() if col_city in df_cf and col_tipo in df_cf \
                    else df_cf[[col_lon, col_lat]].copy()
            df_map[col_lon] = _normalize_lonlat(df_map[col_lon])
            df_map[col_lat] = _normalize_lonlat(df_map[col_lat])
            df_map = df_map.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_lon, col_lat])
            df_map = df_map[(df_map[col_lon].between(-180, 180)) & (df_map[col_lat].between(-90, 90))]

            if df_map.empty:
                st.info("Sem coordenadas válidas para exibir.")
            else:
                # opções de visualização
                modo = st.radio(
                    "Modo",
                    ["Pontos", "Agregado por município", "Mapa de calor"],
                    horizontal=True,
                    key="map_mode"
                )

                # centro/zoom
                c_lat = float(df_map[col_lat].mean())
                c_lon = float(df_map[col_lon].mean())

                if modo == "Pontos":
                    # cada linha = 1 ponto
                    hovertext = None
                    if col_city in df_map.columns and col_tipo in df_map.columns:
                        hovertext = df_map.apply(
                            lambda r: f"<b>{r.get(col_city,'')}</b><br>{r.get(col_tipo,'')}",
                            axis=1
                        )
                    fig = go.Figure(go.Scattermapbox(
                        lat=df_map[col_lat],
                        lon=df_map[col_lon],
                        mode="markers",
                        marker=dict(size=7, color="#1f77b4", opacity=0.6),
                        text=hovertext,
                        hovertemplate="%{text}<extra></extra>" if hovertext is not None else None,
                        name="Estabelecimentos"
                    ))
                    fig.update_layout(
                        mapbox=dict(style="open-street-map", center=dict(lat=c_lat, lon=c_lon), zoom=4.2),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=520
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif modo == "Agregado por município":
                    if col_city not in df_map.columns:
                        st.warning("Para agregar por município, a coluna de município precisa estar disponível.")
                    else:
                        g = (df_map.groupby(col_city)
                                    .agg(Qtde=(col_lon, "size"),
                                        lat=(col_lat, "mean"),
                                        lon=(col_lon, "mean"))
                                    .reset_index()
                                    .sort_values("Qtde", ascending=False))
                        # escala de tamanho
                        smin, smax = 8, 28
                        size = (g["Qtde"] - g["Qtde"].min()) / max(1, (g["Qtde"].max() - g["Qtde"].min()))
                        size = (size * (smax - smin) + smin).astype(float)

                        fig = go.Figure(go.Scattermapbox(
                            lat=g["lat"],
                            lon=g["lon"],
                            mode="markers",
                            marker=dict(size=size, color=g["Qtde"], colorscale="Blues", sizemode="diameter",
                                        colorbar=dict(title="Estabelecimentos")),
                            text=g.apply(lambda r: f"<b>{r[col_city]}</b><br>Estabelecimentos: {r['Qtde']}", axis=1),
                            hovertemplate="%{text}<extra></extra>",
                            name="Municípios"
                        ))
                        fig.update_layout(
                            mapbox=dict(style="open-street-map", center=dict(lat=c_lat, lon=c_lon), zoom=4.0),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=540
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:  # "Mapa de calor"
                    fig = go.Figure(go.Densitymapbox(
                        lat=df_map[col_lat],
                        lon=df_map[col_lon],
                        radius=18,                           # ajuste o raio se quiser
                        colorscale="Blues"
                    ))
                    fig.update_layout(
                        mapbox=dict(style="open-street-map", center=dict(lat=c_lat, lon=c_lon), zoom=4.2),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=520
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
    # -------------------- GRÁFICO 2 --------------------
    with st.expander("Quais municípios têm mais estabelecimentos?", expanded=expand_cadastro):
        col_mun = "MUNICÍPIO"
        if col_mun in df_cf and not df_cf.empty:
            # agrega
            top = (
                df_cf.groupby(col_mun).size()
                .reset_index(name="Qtde")
                .sort_values("Qtde", ascending=False)
                .head(15)
            )

            if not top.empty:
                # % em relação ao total do CONJUNTO FILTRADO (não só do top-15)
                total_filtrado = len(df_cf)
                top["Pct"] = (top["Qtde"] / total_filtrado * 100).round(1)
                top["Label"] = top.apply(lambda r: f"{r['Qtde']} ({r['Pct']:.1f}%)", axis=1)

                figm = go.Figure([
                    go.Bar(
                        x=top["Qtde"],
                        y=top[col_mun],
                        orientation="h",
                        text=top["Label"],                # << contagem + %
                        textposition="outside",
                        cliponaxis=False,
                        marker=dict(color=top["Qtde"], colorscale="Blues", showscale=False),
                        hovertemplate="<b>%{y}</b><br>Qtde: %{x}<br>Participação: %{customdata:.1f}%<extra></extra>",
                        customdata=top["Pct"],           # % no hover
                    )
                ])

                figm.update_layout(
                    title="TOP 15 Municípios por número de estabelecimentos",
                    xaxis_title="Quantidade",
                    yaxis_title="",
                    height=max(360, 26 * len(top)),
                    margin=dict(l=140, r=30, t=50, b=36),
                    uniformtext_minsize=10,
                    uniformtext_mode="show",
                )
                # ordem: maiores no topo
                figm.update_yaxes(
                    autorange="reversed",
                    categoryorder="array",
                    categoryarray=top[col_mun].tolist()
                )

                st.plotly_chart(figm, use_container_width=True)
            else:
                st.info("Sem dados para o filtro atual.")

    # -------------------- GRÁFICO 3 --------------------
    with st.expander("Como os estabelecimentos se distribuem por tamanho da população do município?", expanded=expand_cadastro):
        col_fp = "Faixa populacional"
        if col_fp in df_cf and df_cf[col_fp].notna().any():
            dist_fp = (
                df_cf[col_fp]
                .value_counts()
                .reindex(["até 20 mil","20–100 mil","100–500 mil","500 mil–1 milhão","acima de 1 milhão"])
                .dropna()
                .reset_index()
            )
            dist_fp.columns = [col_fp, "Qtde"]

            if not dist_fp.empty:
                total_fp = dist_fp["Qtde"].sum()
                dist_fp["Pct"] = (dist_fp["Qtde"] / total_fp * 100).round(1)
                dist_fp["Label"] = dist_fp.apply(lambda r: f"{r['Qtde']} ({r['Pct']:.1f}%)", axis=1)

                vals = dist_fp["Qtde"]
                ymax = float(vals.max()) * 1.25

                figp = go.Figure([
                    go.Bar(
                        x=dist_fp[col_fp],
                        y=vals,
                        text=dist_fp["Label"],          # << contagem + %
                        textposition="outside",
                        cliponaxis=False,
                        marker=dict(color=vals, colorscale="Blues", showscale=False),
                        hovertemplate="<b>%{x}</b><br>Qtde: %{y}<br>Participação: %{customdata:.1f}%<extra></extra>",
                        customdata=dist_fp["Pct"],     # % no hover
                    )
                ])
                figp.update_layout(
                    title="Distribuição por Faixa Populacional",
                    xaxis_title="",
                    yaxis_title="Quantidade",
                    height=420,
                    margin=dict(l=20, r=20, t=60, b=60),
                    uniformtext_minsize=10,
                    uniformtext_mode="show",
                )
                figp.update_yaxes(range=[0, ymax])
                st.plotly_chart(figp, use_container_width=True)
        else:
            st.info("Campo 'Faixa populacional' não disponível.")

    # -------------------- GRÁFICO 4 --------------------
    def wrap_label(s: str, max_chars: int = 22) -> str:
        s = str(s)
        if len(s) <= max_chars:
            return s
        words, lines, cur = s.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)
        return "<br>".join(lines)

    with st.expander("Quais são os tipos de estabelecimento mais comuns?", expanded=expand_cadastro):
        col = "TIPO DO ESTABELECIMENTO"
        if col in df_cf and df_cf[col].notna().any():
            fig_pareto = pareto_barh(
                df=df_cf,
                cat_col=col,
                value_col=None,                            # contagem
                title="Tipos de Estabelecimento — Pareto",
                colorbar_title="Estabelecimentos",
                highlight_value=None                      # ex.: "Não identificado"
            )
            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.info("Sem dados para o filtro atual.")
            
    # -------------------- TABELA TRANSPOSTA (detalhe por estabelecimento) --------------------
    st.info("**📋 Sessão de Tabelas - Detalhamento linha a linha dos registros; use para consulta, conferência e exportação.**")
    # escolha por CNES ou Nome Fantasia
    if "NOME FANTASIA" in df_cf and df_cf["NOME FANTASIA"].notna().any():
        nome_opts = df_cf.dropna(subset=["NOME FANTASIA"]).sort_values("NOME FANTASIA")
        # se tiver CNES, mostra junto para diferenciar
        if "CNES" in nome_opts:
            nome_opts["__rotulo__"] = nome_opts["NOME FANTASIA"].astype(str) + " — CNES " + nome_opts["CNES"].astype("Int64").astype(str)
            label_col = "__rotulo__"
        else:
            nome_opts["__rotulo__"] = nome_opts["NOME FANTASIA"]
            label_col = "__rotulo__"
        escolha = st.selectbox("Escolha o estabelecimento", options=nome_opts[label_col].tolist(), index=0)
        # recuperar linha
        linha_sel = nome_opts[nome_opts[label_col] == escolha].iloc[0].drop(labels=[label_col], errors="ignore")
    elif "CNES" in df_cf and df_cf["CNES"].notna().any():
        cnes_opts = df_cf["CNES"].dropna().astype(int).unique().tolist()
        escolha = st.selectbox("Escolha o CNES", options=sorted(cnes_opts), index=0)
        linha_sel = df_cf[df_cf["CNES"] == escolha].iloc[0]
    else:
        st.info("Não há campos para seleção de estabelecimento.")
        linha_sel = None

    if linha_sel is not None:
        # Remove auxiliares e NaN
        linha_series = linha_sel.drop(labels=[c for c in ["conv_sus_bool","ativo_bool"] if c in linha_sel.index], errors="ignore").dropna()
        labels = linha_series.index.tolist()
        valores = linha_series.values.tolist()

        fig_tbl = go.Figure(data=[
            go.Table(
                header=dict(
                    values=["<b>Campo</b>", "<b>Valor</b>"],
                    fill_color="#E8F0FF", align="left",
                    font=dict(size=12), height=30
                ),
                cells=dict(
                    values=[labels, valores],
                    align="left", height=26,
                    fill_color=[["#FFFFFF", "#F9FBFF"] * ((len(labels)//2)+1),
                               ["#FFFFFF", "#F9FBFF"] * ((len(valores)//2)+1)],
                )
            )
        ])
        fig_tbl.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=min(720, 26*(len(labels)+2)))
        st.plotly_chart(fig_tbl, use_container_width=True)

# ────────────────────────────────────────────────────────────────
#  ABA 2  –  Habilitação
# ────────────────────────────────────────────────────────────────
with abas[2]:
    st.subheader("✅ Habilitação")
    st.info("Conteúdo em construção previsto para 03/10/2025")

# ────────────────────────────────────────────────────────────────
#  ABA 3  –  Leitos
# ────────────────────────────────────────────────────────────────
with abas[3]:
    st.subheader("🛏️ Leitos")
    st.info("Conteúdo em construção previsto para 10/10/2025")

# ────────────────────────────────────────────────────────────────
#  ABA 4  –  Equipamentos
# ────────────────────────────────────────────────────────────────
with abas[4]:
    st.subheader("🧰 Equipamentos")
    st.info("Conteúdo em construção previsto para 17/10/2025")

# ────────────────────────────────────────────────────────────────
#  ABA 5  –  Profissionais
# ────────────────────────────────────────────────────────────────
with abas[5]:
    st.subheader("🧑 Profissionais")
    st.info("Conteúdo em construção previsto para 24/10/2025")

# ────────────────────────────────────────────────────────────────
#  ABA 6  –  Registros
# ────────────────────────────────────────────────────────────────
with abas[6]:
    st.subheader("📋 Registros Hospitalares")
    st.info("Conteúdo em construção previsto para 31/10/2025")
