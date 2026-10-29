"""VIGIAEDL: painel de ovitrampas (OVTs) e Estações Disseminadoras de Larvicida (EDLs) do Recife.

Rodar:  streamlit run app.py
Os dados vêm de data/processed (gerados por scripts/preparar_dados.py) ou da pasta
indicada na variável de ambiente VIGIAEDL_DADOS.
"""
from __future__ import annotations

import os
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from core import analise as an
from core import espacial as esp
from core.dados import carregar_processados, ler_instalacao_edl
from core.mapa import ZOOM_RECIFE, centro_do_mapa, construir_mapa, imagem_kde, legenda_calor_html
from core.visual import (AZUL, CINZA, CORES_CALOR, CSS, LARANJA, QUEBRAS, TINTA, cartao, fmt, limitacao)

NOMES_OBS = {"F": "fechada", "E": "extraviada", "R": "recusada", "D": "desocupada", "REC": "recuperada"}
PASTA = Path(os.environ.get("VIGIAEDL_DADOS", Path(__file__).parent / "data" / "processed"))

st.set_page_config(page_title="VIGIAEDL", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- dados
@st.cache_resource(show_spinner="Carregando os dados…")
def carregar(pasta: str):
    """cache_resource guarda o mesmo objeto para todos (sem cópia a cada clique). Por isso o app nunca altera estas tabelas."""
    leituras, ovt, edl, ds, rel = carregar_processados(pasta)
    instalacao = ler_instalacao_edl(pasta)
    leituras = esp.sinalizar_atipicos(leituras)
    return leituras, ovt, edl, ds, rel, instalacao


@st.cache_data(show_spinner=False)
def ovts_classificadas(_ovt, _edl, raio: float, _instalacao=None, chave_instalacao: int = 0):
    return an.classificar_ovts(_ovt, _edl, raio, instalacao=_instalacao)


@st.cache_data(show_spinner="Calculando o Kernel Density…")
def calcular_kde(_ovt, media_por_ovt: pd.Series, banda: float, modo: str):
    d = _ovt.merge(media_por_ovt.rename("media"), left_on="id_ovt", right_index=True)
    d = d.dropna(subset=["media", "x", "y"])
    if len(d) < 10:
        return None
    g = esp.kde_grade(d["x"], d["y"], d["media"], banda_m=banda, celula_m=max(50.0, banda / 4), modo=modo)
    d["kde"] = esp.kde_nos_pontos(g, d["x"], d["y"])
    limiar = esp.limiar_hotspot(g, 90)
    rgba, lim = imagem_kde(g)
    return {"rgba": rgba, "limites": lim, "limiar": limiar, "pontos": d[["id_ovt", "ds", "bairro", "kde"]]}


if not (PASTA / "leituras.csv").exists():
    st.error("Não encontrei os dados tratados.")
    st.markdown(
        "Gere as tabelas a partir da pasta enviada pela GEVACZ:\n\n"
        "```bash\npython scripts/preparar_dados.py --origem \"<pasta com INFORMAÇÕES - GEVACZ>\" --destino data/processed\n```\n"
        f"O painel procurou em `{PASTA}`. Para usar outra pasta, defina `VIGIAEDL_DADOS`.")
    st.stop()

leituras, ovt, edl, ds_geo, rel, instalacao = carregar(str(PASTA))
areas = ds_geo.set_index("ds")["area_km2"]


def layout(fig, altura=320, legenda=True):
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legenda, font=dict(color=TINTA),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(showgrid=False, linecolor="#D1D5DB")
    fig.update_yaxes(gridcolor="#EEF0F3", zeroline=False)
    return fig


# ----------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown('<div class="marca">VIGIA<span>EDL</span></div><div class="sub">Ovitrampas e eficácia das EDLs · Recife</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    anos_disp = sorted(int(a) for a in leituras["ano"].dropna().unique())
    ano_sel = st.selectbox("Ano", [str(a) for a in anos_disp[::-1]] + ["Todos os anos"])
    if ano_sel == "Todos os anos":
        anos, ciclos = None, None
    else:
        anos = int(ano_sel)
        mx = int(leituras.loc[leituras["ano"] == anos, "ciclo"].max())
        ciclos = st.slider("Ciclos", 1, mx, (1, mx)) if mx > 1 else (1, 1)
    ds_sel = st.selectbox("Distrito sanitário", ["Todos"] + [f"DS {i}" for i in sorted(areas.index)])
    ds_i = None if ds_sel == "Todos" else int(ds_sel.split()[1])
    raio = st.radio("Raio da EDL", [100, 200, 300, 500], index=1, horizontal=True, format_func=lambda v: f"{v} m")
    so_conferidas = st.checkbox("Só OVTs com posição conferida", value=True,
                                help="Tira as OVTs cuja posição foi reconstruída pela ordem do arquivo e não pôde ser conferida.")

    st.markdown("---")
    st.markdown("**Antes × depois da instalação**")
    tem_instalacao = instalacao is not None and len(instalacao) > 0
    corte_proprio = False
    if tem_instalacao:
        primeira = esp.data_corte_padrao(instalacao["data_instalacao"])
        mediana_inst = pd.to_datetime(instalacao["data_instalacao"]).sort_values().iloc[len(instalacao) // 2]
        st.caption(f"A GEVACZ informou a instalação de {len(instalacao)} EDLs, de {primeira:%d/%m/%Y} a "
                   f"{instalacao['data_instalacao'].max():%d/%m/%Y}.")
        usar_corte = st.checkbox("Comparar antes × depois", value=True)
        if usar_corte:
            metodo = st.radio("Como definir antes e depois",
                              ["Data da EDL de cada OVT (recomendado)", "Uma data única (menor data de instalação)"],
                              help="Com a primeira opção, cada OVT perto de uma EDL usa a data de instalação dessa EDL. "
                                   "As OVTs longe, que servem de comparação, usam a data abaixo. "
                                   "A segunda opção é a regra do plano de trabalho.")
            corte_proprio = metodo.startswith("Data da EDL")
            padrao_corte = mediana_inst if corte_proprio else primeira
            corte = st.date_input("Data de corte das OVTs longe" if corte_proprio else "Data de corte",
                                  value=padrao_corte.date(), format="DD/MM/YYYY")
        else:
            corte = None
    else:
        st.caption("A GEVACZ ainda não informou a data de instalação das EDLs. Uma data escolhida aqui é só uma hipótese.")
        usar_corte = st.checkbox("Testar uma data de corte", value=False)
        corte = st.date_input("Data de corte (hipótese)", value=pd.Timestamp("2025-01-01").date(), format="DD/MM/YYYY") if usar_corte else None
    corte = pd.Timestamp(corte) if corte is not None else None

lf = an.filtrar_leituras(leituras, anos=anos, ciclos=ciclos, ds=ds_i)            # período + distrito
lp = an.filtrar_leituras(leituras, anos=anos, ciclos=ciclos)                       # período, todos os distritos
v = lf.dropna(subset=["n_ovos"])
oc = ovts_classificadas(ovt, edl, float(raio), instalacao if tem_instalacao else None, len(instalacao) if tem_instalacao else 0)
periodo_txt = "todos os anos" if anos is None else f"{anos}, ciclos {ciclos[0]} a {ciclos[1]}"

c1, c2 = st.columns([3, 1])
c1.markdown('<div class="marca">VIGIA<span>EDL</span></div>', unsafe_allow_html=True)
c2.markdown('<div style="text-align:right"><span class="selo">Protótipo acadêmico · dados da GEVACZ</span></div>',
            unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Visão geral", "Ficha da OVT", "Histórico e eficácia"])


# ============================================================================ VISÃO GERAL
with tab1:
    st.markdown(f"## Visão geral do monitoramento")
    st.caption(f"Onde a densidade de ovos está mais alta e quanto tempo o dado leva para ficar disponível. "
               f"Período: {periodo_txt} · {ds_sel}.")
    if v.empty:
        st.warning("Não há leituras com contagem para esse filtro.")
    else:
        ult_ano, ult_ciclo = v.sort_values(["ano", "ciclo"])[["ano", "ciclo"]].iloc[-1]
        atual = v[(v["ano"] == ult_ano) & (v["ciclo"] == ult_ciclo)]
        anterior_chaves = v[(v["ano"] < ult_ano) | ((v["ano"] == ult_ano) & (v["ciclo"] < ult_ciclo))][["ano", "ciclo"]].drop_duplicates()
        delta_txt, delta_bom = "", None
        if len(anterior_chaves):
            pa, pc = anterior_chaves.sort_values(["ano", "ciclo"]).iloc[-1]
            ant = v[(v["ano"] == pa) & (v["ciclo"] == pc)]
            comuns = set(atual["id_ovt"]) & set(ant["id_ovt"])
            if comuns:
                a_m = atual[atual["id_ovt"].isin(comuns)]["n_ovos"].mean()
                b_m = ant[ant["id_ovt"].isin(comuns)]["n_ovos"].mean()
                var = (a_m - b_m) / b_m * 100 if b_m else np.nan
                delta_txt = f"{'+' if var > 0 else '−'}{fmt(abs(var), 1)}% vs ciclo {int(pc)} (mesmas {fmt(len(comuns))} OVTs)"
                delta_bom = var < 0
        pr = an.prazos_do_dado(lf)["digitacao"].dropna()

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(cartao("OVTs monitoradas", fmt(v["id_ovt"].nunique()),
                           nota=f"em {v['bairro'].nunique()} bairros e {v['ds'].nunique()} distritos sanitários"),
                    unsafe_allow_html=True)
        k2.markdown(cartao(f"Ovos por OVT · ciclo {int(ult_ciclo)}" + (f"/{int(ult_ano)}" if anos is None else ""),
                           fmt(atual["n_ovos"].mean(), 1), mini="média",
                           delta=delta_txt, delta_bom=delta_bom,
                           nota=f"Mediana {fmt(atual['n_ovos'].median())}. {fmt(atual['id_ovt'].nunique())} OVTs com leitura neste ciclo."),
                    unsafe_allow_html=True)
        k3.markdown(cartao("Da coleta até o dado digitado", (fmt(pr.median()) if len(pr) else "—"), mini="dias (mediana)" if len(pr) else "",
                           nota=(f"90% das leituras ficam digitadas em até {fmt(pr.quantile(.9))} dias." if len(pr)
                                 else "Sem datas de digitação neste período (só 2026 tem).")), unsafe_allow_html=True)
        base_card = an.base_por_ovt(lp if ds_i is None else lf, oc, apenas_validadas=so_conferidas)
        rc = esp.comparar_grupos(base_card)
        if rc["n_a"] and rc["n_b"]:
            k4.markdown(cartao(f"Perto × longe · {raio} m", f"{'+' if rc['diferenca_pct'] > 0 else '−'}{fmt(abs(rc['diferenca_pct']), 1)}%",
                               mini="perto vs longe",
                               nota=f"Mediana {fmt(rc['mediana_a'])} perto (n={fmt(rc['n_a'])}) e {fmt(rc['mediana_b'])} longe (n={fmt(rc['n_b'])}). "
                                    "É associação, não efeito: veja a aba de eficácia."), unsafe_allow_html=True)
        else:
            k4.markdown(cartao("Perto × longe", "—", nota="Poucas OVTs com posição neste filtro.", tracejado=True), unsafe_allow_html=True)

        st.write("")
        g1, g2 = st.columns([3, 2])
        with g1:
            st.markdown("**Ovos por OVT em cada ciclo**")
            ev = an.evolucao_por_ciclo(lf, por=None)
            multi = ev["ano"].nunique() > 1
            ev["x"] = ev.apply(lambda r: f"{int(r['ano'])}·C{int(r['ciclo'])}" if multi else f"C{int(r['ciclo'])}", axis=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ev["x"], y=ev["media"], name="Média", mode="lines+markers", line=dict(color=LARANJA, width=3)))
            fig.add_trace(go.Scatter(x=ev["x"], y=ev["mediana"], name="Mediana", mode="lines", line=dict(color=TINTA, width=2, dash="dash")))
            layout(fig, 330)
            fig.update_xaxes(type="category", tickangle=-45 if multi else 0, nticks=14)
            st.plotly_chart(fig, width="stretch")
            st.caption("Se a média fica acima da mediana, poucas OVTs com contagens muito altas estão puxando o valor para cima.")
        with g2:
            st.markdown("**Bairros com mais ovos por OVT**")
            minimo = 100 if len(v) > 2000 else 20
            rk = (v.groupby("bairro").agg(leituras=("n_ovos", "size"), media=("n_ovos", "mean"),
                                          ovts=("id_ovt", "nunique"), ds=("ds", "first")).query("leituras >= @minimo")
                  .sort_values("media", ascending=False).head(8).iloc[::-1])
            fig = go.Figure(go.Bar(x=rk["media"], y=[f"{b.title()} · DS {int(d)}" for b, d in zip(rk.index, rk["ds"])],
                                   orientation="h", marker_color=LARANJA, text=[fmt(x, 1) for x in rk["media"]], textposition="outside"))
            fig.add_vline(x=v["n_ovos"].mean(), line_dash="dash", line_color=TINTA)
            layout(fig, 330, legenda=False)
            fig.update_xaxes(range=[0, float(rk["media"].max()) * 1.18])
            fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Só bairros com {minimo} leituras ou mais. A linha tracejada é a média da cidade ({fmt(v['n_ovos'].mean(), 1)}).")

        # ---- mapa de calor distrito × ciclo
        st.markdown("**Média de ovos por OVT · distrito × ciclo**")
        ano_h = anos if anos is not None else int(v["ano"].max())
        vh = an.filtrar_leituras(leituras, anos=ano_h).dropna(subset=["n_ovos"])
        piv = vh.pivot_table(index="ds", columns="ciclo", values="n_ovos", aggfunc="mean").sort_index()
        zmin, zmax = 100, 350
        bordas = [zmin] + QUEBRAS + [zmax]
        escala = []
        for i, cor in enumerate(CORES_CALOR):
            escala += [[(bordas[i] - zmin) / (zmax - zmin), cor], [(bordas[i + 1] - zmin) / (zmax - zmin), cor]]
        fig = go.Figure(go.Heatmap(z=piv.values, x=[f"C{int(c)}" for c in piv.columns],
                                   y=[f"DS {int(d)} · {fmt(vh[vh['ds'] == d]['n_ovos'].mean(), 1)}" for d in piv.index],
                                   colorscale=escala, zmin=zmin, zmax=zmax, text=np.round(piv.values).astype("float"),
                                   texttemplate="%{text:.0f}", hovertemplate="%{y} · %{x}: %{z:.1f}<extra></extra>",
                                   xgap=2, ygap=2, colorbar=dict(title="ovos", tickvals=[125, 175, 225, 275, 325],
                                                                 ticktext=["<150", "150–199", "200–249", "250–299", "300+"])))
        layout(fig, 330, legenda=False)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Ano {ano_h}. O número depois do distrito é a média do ano.")

        # ---- distritos lado a lado
        st.markdown("**Distritos sanitários lado a lado**")
        ind = esp.indicadores_por_regiao(lp, areas)
        tds = pd.DataFrame({
            "Distrito": [f"DS {int(i)}" for i in ind.index], "OVTs": ind["n_ovts"].astype(int),
            "Ovos por OVT (média)": ind["ovos_por_ovt"], "Mediana": ind["mediana"],
            "IDO (média das positivas)": ind["ido"], "IPO (% positivas)": ind["ipo"],
            "OVTs por km²": ind["ovts_por_km2"], "Ovos por km² (por ciclo)": ind["ovos_por_km2"]})
        def _realce(linha):
            return ["background-color:#FEF3E7" if (ds_i is not None and linha["Distrito"] == f"DS {ds_i}") else "" for _ in linha]
        st.dataframe(tds.style.apply(_realce, axis=1).format({
            "Ovos por OVT (média)": "{:.1f}", "Mediana": "{:.0f}", "IDO (média das positivas)": "{:.1f}",
            "IPO (% positivas)": "{:.1f}", "OVTs por km²": "{:.1f}", "Ovos por km² (por ciclo)": "{:.0f}"}),
            width="stretch", hide_index=True)
        st.caption("Ovos por km² sobe também onde há mais OVTs por km²; por isso os dois aparecem juntos. "
                   "Para comparar distritos de tamanhos diferentes, o mais justo é ovos por OVT.")

        # ---- mapa
        st.markdown("---")
        st.markdown("### Mapa de OVTs, EDLs e hotspots")
        mc = st.columns([1.3, 1, 1, 1, 1])
        metricas = {"Ovos por OVT": "ovos_por_ovt", "IDO (média das OVTs positivas)": "ido",
                    "Ovos por km² (por ciclo)": "ovos_por_km2", "OVTs por km²": "ovts_por_km2"}
        nome_m = mc[0].selectbox("Colorir os distritos por", list(metricas))
        mostrar_kde = mc[1].checkbox("Kernel Density", value=True)
        mostrar_ovt = mc[2].checkbox("OVTs", value=True)
        mostrar_edl = mc[3].checkbox("EDLs", value=True)
        mostrar_raio = mc[4].checkbox("Raio das EDLs", value=True)
        if mostrar_kde:
            kc = st.columns([1.3, 1, 2])
            modo_nome = kc[0].selectbox("O Kernel Density mostra", ["Nível de infestação (média de ovos por OVT)",
                                                                  "Onde os ovos se concentram (ovos por km²)"])
            banda = kc[1].slider("Suavização (m)", 150, 800, 300, step=50)
            kc[2].caption("Nível de infestação não depende de quantas OVTs há no lugar. Concentração de ovos acende também onde há mais OVTs.")
            modo = "media" if modo_nome.startswith("Nível") else "soma"
        media_ovt = lp.dropna(subset=["n_ovos"]).groupby("id_ovt")["n_ovos"].mean()
        kde = calcular_kde(ovt, media_ovt, float(banda), modo) if mostrar_kde else None
        atip = set(lp.loc[lp["atipica"], "id_ovt"]) if "atipica" in lp else set()
        mapa = construir_mapa(ds_geo, ind[metricas[nome_m]], nome_m, ovt, media_ovt, edl, float(raio),
                              imagem=kde["rgba"] if kde else None, limites=kde["limites"] if kde else None,
                              mostrar_ovt=mostrar_ovt, mostrar_edl=mostrar_edl, mostrar_raio=mostrar_raio, destaques=atip)
        centro = centro_do_mapa(ds_geo)
        st_folium(mapa, height=640, use_container_width=True, returned_objects=[], key="mapa_geral",
                  center=centro, zoom=ZOOM_RECIFE)
        st.markdown(legenda_calor_html(), unsafe_allow_html=True)
        st.caption("Pontos com borda vermelha têm alguma leitura 10 vezes acima da mediana do bairro: vale conferir antes de tratar como foco.")

        if kde:
            pts = kde["pontos"]
            quentes = pts[pts["kde"] >= kde["limiar"]]
            st.markdown(f"**Hotspots:** {fmt(len(quentes))} OVTs ficam nos 10% mais intensos da superfície "
                        f"({'média de ovos por OVT' if modo == 'media' else 'ovos por km²'} ≥ {fmt(kde['limiar'], 1)}).")
            tq = (quentes.groupby("ds").size().reindex(sorted(areas.index), fill_value=0)
                  .rename("OVTs em hotspot").to_frame())
            tq["% das OVTs do distrito"] = (tq["OVTs em hotspot"] / pts.groupby("ds").size().reindex(tq.index) * 100).round(1)
            tq.index = [f"DS {i}" for i in tq.index]
            st.dataframe(tq.T, width="stretch")

        st.caption(f"Fonte: monitoramento de ovitrampas da GEVACZ · {fmt(len(v))} leituras com contagem em {periodo_txt}.")


# ============================================================================ FICHA DA OVT
with tab2:
    st.markdown("## Ficha da OVT")
    ids_ok = sorted(leituras["id_ovt"].unique())      # a ficha é histórica: lista todas, não só as do período
    atip_ids = lp[lp["atipica"]].groupby("id_ovt")["vezes_mediana"].max().sort_values(ascending=False) if "atipica" in lp else pd.Series(dtype=float)
    padrao = atip_ids.index[0] if len(atip_ids) else ids_ok[0]
    sel = st.selectbox("OVT", ids_ok, index=ids_ok.index(padrao), help="Digite para procurar. A lista começa na OVT mais fora do padrão do período.")
    h = leituras[leituras["id_ovt"] == sel].sort_values(["ano", "ciclo"])
    hv = h.dropna(subset=["n_ovos"])
    info = oc[oc["id_ovt"] == sel]
    bairro = h["bairro"].iloc[-1] if len(h) else "—"
    ds_o = int(h["ds"].dropna().iloc[-1]) if h["ds"].notna().any() else None
    n_bairro = int(ovt[ovt["bairro"] == bairro]["id_ovt"].nunique()) if len(ovt) else 0

    st.markdown(f"### OVT {sel} · {str(bairro).title()}")
    selos = [f"Distrito sanitário {ds_o}" if ds_o else "", f"{len(hv)} leituras com contagem"]
    selos.append("Sem coordenadas" if info.empty else ("Posição conferida" if bool(info["posicao_validada"].iloc[0]) else "Posição não conferida"))
    st.markdown(" ".join(f'<span class="selo">{s}</span>' for s in selos if s), unsafe_allow_html=True)
    f1, f2 = st.columns([1, 2])
    with f1:
        pos_txt = "—"
        if len(hv):
            medias = lp.dropna(subset=["n_ovos"]).groupby("id_ovt")["n_ovos"].mean()
            if sel in medias.index:
                pos_txt = f"{fmt(int((medias > medias[sel]).sum()) + 1)}ª de {fmt(len(medias))}"
        d_txt, tipo_txt = "—", "—"
        if not info.empty and pd.notna(info["dist_edl_m"].iloc[0]):
            d_txt = f"{fmt(info['dist_edl_m'].iloc[0])} m ({info['proximidade'].iloc[0]} no raio de {raio} m)"
            tipo_txt = str(info["tipo_edl_mais_proxima"].iloc[0])
        linhas = [("Bairro", str(bairro).title()), ("OVTs no bairro", fmt(n_bairro)),
                  ("Primeira leitura", f"{int(hv['ano'].iloc[0])} · ciclo {int(hv['ciclo'].iloc[0])}" if len(hv) else "—"),
                  ("Última leitura", f"{int(hv['ano'].iloc[-1])} · ciclo {int(hv['ciclo'].iloc[-1])}" if len(hv) else "—"),
                  ("Média de ovos (todas as leituras)", fmt(hv["n_ovos"].mean(), 1) if len(hv) else "—"),
                  (f"Ranking por média de ovos ({periodo_txt})", pos_txt), ("EDL mais próxima", d_txt), ("Tipo da EDL", tipo_txt)]
        st.markdown('<div class="cartao"><b>Cadastro</b>' + "".join(
            f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid #F3F4F6;padding:6px 0">'
            f'<span class="sub">{a}</span><b style="font-size:.9rem">{b}</b></div>' for a, b in linhas) + "</div>", unsafe_allow_html=True)
    with f2:
        st.markdown("**Ovos por leitura · OVT × mediana do bairro**")
        if len(hv):
            x = [f"{int(a)}·C{int(c)}" for a, c in zip(hv["ano"], hv["ciclo"])]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=hv["n_ovos"], name=f"OVT {sel}", mode="lines+markers", line=dict(color=LARANJA, width=3)))
            fig.add_trace(go.Scatter(x=x, y=hv["mediana_bairro"], name="Mediana do bairro", mode="lines", line=dict(color=TINTA, width=2, dash="dash")))
            layout(fig, 330)
            fig.update_xaxes(type="category", tickangle=-45, nticks=16)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Esta OVT não tem leituras com contagem.")

    a1, a2 = st.columns([1, 2])
    with a1:
        at = h[h["atipica"]] if "atipica" in h else h.iloc[0:0]
        if len(at):
            st.markdown('<div class="aviso"><b>Padrão que merece conferência</b><p>'
                        f'Em {len(at)} leituras, esta OVT registrou de {fmt(at["n_ovos"].min())} a {fmt(at["n_ovos"].max())} ovos, '
                        f'entre {fmt(at["vezes_mediana"].min(), 0)} e {fmt(at["vezes_mediana"].max(), 0)} vezes a mediana do bairro no mesmo ciclo.</p>'
                        '<p>Vale conferir a contagem da palheta e a digitação antes de tratar o ponto como foco real.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info"><b>Sem padrão fora do comum</b><p>Nenhuma leitura desta OVT passou de 10 vezes a mediana do bairro.</p></div>', unsafe_allow_html=True)
        if not info.empty:
            mini = folium.Map(location=(info["lat"].iloc[0], info["lon"].iloc[0]), zoom_start=15, tiles="OpenStreetMap")
            folium.Circle((info["lat"].iloc[0], info["lon"].iloc[0]), radius=float(raio), color=AZUL, weight=1, fill=True, fill_opacity=0.05).add_to(mini)
            folium.CircleMarker((info["lat"].iloc[0], info["lon"].iloc[0]), radius=6, color=TINTA, fill=True, fill_color=LARANJA, fill_opacity=1, tooltip=sel).add_to(mini)
            prox = info["edl_mais_proxima"].iloc[0]
            if pd.notna(prox):
                e_ = edl[edl["edl_id"] == prox].iloc[0]
                folium.CircleMarker((e_["lat"], e_["lon"]), radius=6, color=AZUL, fill=True, fill_color=AZUL, tooltip=f"EDL mais próxima · {e_['tipo']}").add_to(mini)
            st.markdown("**Localização**")
            st_folium(mini, height=240, use_container_width=True, returned_objects=[], key="mini_mapa",
                      center=(info["lat"].iloc[0], info["lon"].iloc[0]), zoom=15)
    with a2:
        st.markdown("**Últimas leituras**")
        ult = h.sort_values(["ano", "ciclo"], ascending=False).head(12).copy()
        tab = pd.DataFrame({"Ano": ult["ano"].astype(int), "Ciclo": ult["ciclo"].astype(int),
                            "Coleta": ult["data"].dt.strftime("%d/%m/%Y"), "Ovos": ult["n_ovos"],
                            "Mediana do bairro": ult["mediana_bairro"].round(0), "Vezes a mediana": ult["vezes_mediana"].round(1),
                            "Observação": ult["obs"].fillna("").astype(str).where(~ult["obs"].fillna("").astype(str).str.match(r"^\d{4}-\d{2}-\d{2}"), "")})
        est = (tab.style.apply(lambda r: ["background-color:#FEF3E7" if (pd.notna(r["Vezes a mediana"]) and r["Vezes a mediana"] >= 10) else "" for _ in r], axis=1)
               .format({"Ano": "{:d}", "Ciclo": "{:d}", "Ovos": "{:.0f}", "Mediana do bairro": "{:.0f}", "Vezes a mediana": "{:.1f}"}, na_rep="—"))
        st.dataframe(est, width="stretch", hide_index=True)
        st.caption("Linhas em destaque: 10 vezes a mediana do bairro ou mais. Ovos vazios: a OVT não foi lida (F fechado, E extraviado, R recusado, D desocupado, REC recuperada).")
        st.download_button("Exportar leituras (CSV)", h.drop(columns=[c for c in ["n_no_grupo"] if c in h]).to_csv(index=False).encode("utf-8"),
                           file_name=f"{sel}_leituras.csv", mime="text/csv")


# ============================================================================ HISTÓRICO E EFICÁCIA
with tab3:
    st.markdown("## Histórico e eficácia")
    st.caption("Como a densidade de ovos evoluiu em cada distrito, o que dá para dizer sobre as EDLs e o que os dados ainda não permitem afirmar.")

    st.markdown("**Evolução por distrito sanitário**")
    evd = an.evolucao_por_ciclo(lp, por="ds")
    evd["ordem"] = evd["ano"] * 100 + evd["ciclo"]
    ordens = sorted(evd["ordem"].unique())
    media_cidade = lp["n_ovos"].mean()
    fig = make_subplots(rows=2, cols=4, subplot_titles=[f"DS {i}" for i in sorted(areas.index)], shared_yaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.14)
    for k, d_ in enumerate(sorted(areas.index)):
        r_, c_ = divmod(k, 4)
        s = evd[evd["ds"] == d_].sort_values("ordem")
        fig.add_trace(go.Scatter(x=[ordens.index(o) for o in s["ordem"]], y=s["media"], mode="lines", line=dict(color=LARANJA, width=2),
                                 hovertemplate="DS %d · %%{y:.0f}<extra></extra>" % d_, showlegend=False), row=r_ + 1, col=c_ + 1)
        if len(s):
            pico = s.loc[s["media"].idxmax()]
            fig.add_trace(go.Scatter(x=[ordens.index(pico["ordem"])], y=[pico["media"]], mode="markers",
                                     marker=dict(color=LARANJA, size=8), showlegend=False, hoverinfo="skip"), row=r_ + 1, col=c_ + 1)
        fig.add_hline(y=media_cidade, line_dash="dot", line_color="#9CA3AF", row=r_ + 1, col=c_ + 1)
    layout(fig, 450, legenda=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(range=[max(0, float(evd["media"].min()) * 0.85), float(evd["media"].max()) * 1.08])
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Média de ovos por OVT em cada ciclo, na mesma escala. A linha pontilhada é a média da cidade ({fmt(media_cidade, 1)}). O ponto marca o pico.")

    # ------------------------------------------------------------------ ano a ano
    anos_todos = sorted(int(a) for a in leituras["ano"].dropna().unique())
    if len(anos_todos) > 1:
        limite = min(int(leituras.loc[leituras["ano"] == a, "ciclo"].max()) for a in anos_todos)
        aa = leituras[leituras["ciclo"] <= limite].dropna(subset=["n_ovos"])
        pv = aa.pivot_table(index="ds", columns="ano", values="n_ovos", aggfunc="mean").sort_index()
        st.markdown("**Ano a ano por distrito sanitário**")
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=[str(c) for c in pv.columns], y=[f"DS {int(d)}" for d in pv.index],
            colorscale=[[0, "#FDF0E3"], [0.5, "#E08A2E"], [1, "#7A3A0A"]], text=np.round(pv.values), texttemplate="%{text:.0f}",
            hovertemplate="%{y} · %{x}: %{z:.1f} ovos por OVT<extra></extra>", xgap=2, ygap=2, colorbar=dict(title="ovos")))
        layout(fig, 330, legenda=False)
        fig.update_xaxes(type="category")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Média de ovos por OVT nos ciclos 1 a {limite} de cada ano. O limite é o do ano com menos ciclos, "
                   "para comparar o mesmo período e não um ano inteiro com um ano pela metade.")

    # ------------------------------------------------------------------ eficácia
    st.markdown("---")
    st.markdown(f"### Eficácia das EDLs · perto × longe e antes × depois")
    corte_ativo = corte is not None
    base_leit = leituras if corte_ativo else lp                       # com corte, usa todos os anos
    if corte_ativo and anos is not None:
        st.caption("Com a data de corte ligada, a análise usa todos os anos, para ter leituras antes e depois.")
    base = an.base_por_ovt(base_leit, oc, corte=corte, apenas_validadas=so_conferidas, corte_proprio=corte_proprio)
    base_f = base if ds_i is None else base[base["ds"] == ds_i]
    res = an.eficacia(base_f)
    geral = res["geral"]

    e1, e2, e3 = st.columns([1, 1, 1])
    with e1:
        st.markdown(f"**Perto × longe da EDL** · raio de {raio} m")
        if geral["n_a"] and geral["n_b"]:
            fig = go.Figure(go.Bar(x=[f"Perto (até {raio} m)", f"Longe (mais de {raio} m)"], y=[geral["mediana_a"], geral["mediana_b"]],
                                   marker_color=[AZUL, "#9CA3AF"], text=[fmt(geral["mediana_a"], 1), fmt(geral["mediana_b"], 1)], textposition="outside"))
            layout(fig, 260, legenda=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Mediana de ovos por OVT. Perto: {fmt(geral['n_a'])} OVTs · longe: {fmt(geral['n_b'])} OVTs.")
        else:
            st.info("Poucas OVTs com posição neste filtro.")
    with e2:
        st.markdown("**Por faixa de distância até a EDL**")
        fx = an.por_faixa_de_distancia(base_f)
        if fx["n_ovts"].sum():
            fig = go.Figure(go.Bar(x=fx["faixa"].astype(str), y=fx["mediana"], marker_color=AZUL,
                                   text=[fmt(m, 0) if m == m else "" for m in fx["mediana"]], textposition="outside"))
            layout(fig, 260, legenda=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("Se a EDL reduz os ovos, a barra deveria subir conforme a OVT se afasta dela.")
    with e3:
        st.markdown("**Antes × depois da instalação**")
        did = res["did"]
        if did and not np.isnan(did["did"]):
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Perto da EDL", x=["Antes", "Depois"], y=[did["perto_antes"], did["perto_depois"]], marker_color=AZUL))
            fig.add_trace(go.Bar(name="Longe da EDL", x=["Antes", "Depois"], y=[did["longe_antes"], did["longe_depois"]], marker_color="#9CA3AF"))
            fig.update_layout(barmode="group")
            layout(fig, 260)
            st.plotly_chart(fig, width="stretch")
            regra = ("Perto: data da EDL de cada OVT. Longe" if corte_proprio else "Todas as OVTs") + f": corte em {corte:%d/%m/%Y}."
            st.caption(f"Mediana de ovos por OVT. {regra} A diferença das diferenças é {fmt(did['did'], 1)} ovos.")
        else:
            st.markdown('<div class="cartao tracejado"><div class="rotulo">Aguardando dados</div><div class="grande" style="font-size:1.3rem">Sem corte de data</div>'
                        '<div class="nota">Precisa da data de instalação de cada EDL e de leituras de antes dela. '
                        'Escolha uma data de corte na barra lateral para testar uma hipótese, ou use o modelo de planilha abaixo para pedir as datas à GEVACZ.</div></div>',
                        unsafe_allow_html=True)

    modelo = PASTA / "edl_instalacao_modelo.csv"
    if modelo.exists():
        st.download_button("Baixar modelo para a GEVACZ preencher (data de instalação das EDLs)", modelo.read_bytes(),
                           file_name="edl_instalacao_modelo.csv", mime="text/csv")

    # resumo em texto
    if geral["n_a"] and geral["n_b"]:
        sentido = "maior" if geral["diferenca_pct"] > 0 else "menor"
        sig = ("A diferença é maior do que o acaso explicaria" if (geral["p_valor"] == geral["p_valor"] and geral["p_valor"] < 0.05)
               else "A diferença não é maior do que o acaso explicaria")
        p_txt = f" (p = {fmt(geral['p_valor'], 3)})" if geral["p_valor"] == geral["p_valor"] else ""
        st.markdown(
            f'<div class="info"><b>Resumo em texto</b><p>Nas áreas a até {raio} m de uma EDL, a mediana de ovos por OVT foi '
            f'{fmt(abs(geral["diferenca_pct"]), 1)}% {sentido} do que nas mais distantes ({periodo_txt if not corte_ativo else "todos os anos"}). '
            f'{sig}{p_txt}.</p><p><b>Cuidado com a leitura:</b> a diferença mostra associação, não causa. '
            'As EDLs podem ter sido instaladas onde a infestação já era maior. Sem a data de instalação e sem leituras de antes, não dá para separar uma coisa da outra.</p></div>',
            unsafe_allow_html=True)
    elif geral["aviso"]:
        st.warning(geral["aviso"])

    st.markdown("**Perto × longe em cada distrito sanitário**")
    pd_ = res["por_ds"].copy()
    if len(pd_):
        tabela = pd.DataFrame({
            "Distrito": [f"DS {int(d)}" for d in pd_["ds"]], "OVTs perto": pd_["n_a"].astype(int), "OVTs longe": pd_["n_b"].astype(int),
            "Mediana perto": pd_["mediana_a"].round(1), "Mediana longe": pd_["mediana_b"].round(1),
            "Diferença (%)": pd_["diferenca_pct"].round(1), "p-valor": pd_["p_valor"].round(3)})
        st.dataframe(tabela, width="stretch", hide_index=True)
        st.caption("Oito comparações lado a lado: com tantos testes, alguns p pequenos aparecem por acaso. Vale olhar o padrão, não cada p isolado.")

    # ------------------------------------------------------------------ prazos
    st.markdown("---")
    pz = an.prazos_do_dado(lp)
    st.markdown("**Quanto tempo o dado leva para chegar**")
    if pz.dropna(how="all").empty:
        st.info("Sem datas de laboratório, digitação ou envio neste período (só 2026 tem).")
    else:
        nomes = {"leitura_lab": "Leitura no laboratório", "digitacao": "Digitação", "envio_ds": "Envio ao DS"}
        med = {n: pz[c].dropna().median() for c, n in nomes.items()}
        p90 = {n: pz[c].dropna().quantile(.9) for c, n in nomes.items()}
        fig = go.Figure()
        fig.add_trace(go.Bar(y=list(med), x=list(med.values()), orientation="h", name="Mediana", marker_color=AZUL))
        fig.add_trace(go.Bar(y=list(p90), x=list(p90.values()), orientation="h", name="90% prontos em até", marker_color="#BBD3F0"))
        fig.update_layout(barmode="group")
        layout(fig, 240)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        st.caption("Dias contados a partir da coleta. A data de envio ao DS só está registrada em parte dos ciclos.")

    # ------------------------------------------------------------------ limitações
    st.markdown("---")
    st.markdown("### Limitações dos dados")
    st.caption("O que o leitor precisa saber antes de tirar conclusões.")
    vt = leituras.dropna(subset=["n_ovos"])
    pos_pct = (vt["n_ovos"] > 0).mean() * 100
    sem_contagem = leituras["n_ovos"].isna().sum()
    cods = leituras.loc[leituras["n_ovos"].isna(), "obs_cod"].value_counts()
    anos_txt = ", ".join(str(a) for a in anos_disp)
    nv = rel.get("n_pontos_nao_validados", 0)
    itens = [
        ("1. Quase toda OVT é positiva", f"{fmt(pos_pct, 1)}% das leituras tiveram pelo menos 1 ovo. Por isso o painel mede densidade de ovos, não positividade."),
        ("2. Média acima da mediana", f"A média é {fmt(vt['n_ovos'].mean(), 1)} e a mediana {fmt(vt['n_ovos'].median())}. Poucas leituras muito altas puxam a média, então mostramos as duas."),
        ("3. Leituras sem contagem", f"{fmt(sem_contagem)} leituras ({fmt(sem_contagem / len(leituras) * 100, 1)}%) não têm número de ovos. Principais motivos: "
                                     + ", ".join(f"{NOMES_OBS.get(k, k)} ({fmt(n)})" for k, n in cods.head(4).items()) + ". Elas não entram como zero."),
        ("4. Anos disponíveis", f"Há leituras de {anos_txt}. A GEVACZ ainda vai informar se existem dados de 2018 em diante; sem eles não há comparação de antes."),
        ("5. Datas digitadas com ano errado", f"{fmt(rel.get('datas_corrigidas', 0))} leituras ({fmt(rel.get('datas_corrigidas', 0) / len(leituras) * 100, 1)}%) vieram com o ano da coleta digitado errado "
                                              "(ex.: dezembro de 2024 escrito como 2025). Corrigimos pelo ciclo e guardamos a data original. Vale a GEVACZ conferir."),
        ("6. Datas de instalação das EDLs", ("A GEVACZ informou datas para " + fmt(len(instalacao)) + " EDLs; as que ficaram sem data usam a data de corte das OVTs longe." if tem_instalacao
                                              else "Não vieram nos arquivos. Sem elas, o antes × depois só pode ser testado com uma data hipotética, e a comparação perto × longe é uma foto, não um efeito.")),
        ("7. Posição das OVTs", f"{fmt(nv)} OVTs do DS II tiveram o código reconstruído pela ordem do arquivo e a posição não pôde ser conferida; {fmt(rel.get('ovts_com_leitura_sem_ponto', 0))} OVTs com leitura não têm coordenada (a maioria são armadilhas antigas)."),
        ("8. Raio de 200 m", "É o valor do plano de trabalho. A GEVACZ ainda não confirmou o raio real de influência de uma EDL."),
        ("9. Cada ponto de EDL tem várias estações", "A distância é até o ponto cadastrado, não até cada estação. Não temos a posição individual de cada uma."),
    ]
    cols = st.columns(3)
    for i, (t_, x_) in enumerate(itens):
        cols[i % 3].markdown(limitacao(t_, x_), unsafe_allow_html=True)
        cols[i % 3].write("")
    st.caption("Fonte: monitoramento de ovitrampas da GEVACZ, tratado pelo pipeline do projeto (scripts/preparar_dados.py).")
