"""VIGIAEDL: painel de ovitrampas (OVTs) e Estações Disseminadoras de Larvicida (EDLs) do Recife.

Rodar:  streamlit run app.py
Os dados vêm de data/processed (gerados por scripts/preparar_dados.py) ou da pasta
indicada na variável de ambiente VIGIAEDL_DADOS.
"""
from __future__ import annotations

import html
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
from core import insights as ins
from core.dados import carregar_processados, ler_instalacao_edl
from core.mapa import ZOOM_RECIFE, centro_do_mapa, construir_mapa, imagem_kde, legenda_calor_html
from core.visual import (AZUL, CINZA, CORES_CALOR, CSS, LARANJA, QUEBRAS, TINTA, cartao, fmt, leitura, limitacao,
                         CONFIG_GRAFICO, LEGENDA_CORES_HTML, faixa_de_filtros, indice)

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


@st.cache_data(show_spinner="Calculando os focos que se repetem…")
def calcular_persistencia(_leituras, chave: int):
    per = ins.persistencia_de_hotspot(_leituras)
    return per, ins.hotspot_cronico_por_ds(per), ins.media_por_ovt_e_ano(_leituras)


if not (PASTA / "leituras.csv").exists():
    st.error("Não encontrei os dados tratados.")
    st.markdown(
        "Gere as tabelas a partir da pasta enviada pela GEVACZ:\n\n"
        "```bash\npython scripts/preparar_dados.py --origem \"<pasta com INFORMAÇÕES - GEVACZ>\" --destino data/processed\n```\n"
        f"O painel procurou em `{PASTA}`. Para usar outra pasta, defina `VIGIAEDL_DADOS`.")
    st.stop()

leituras, ovt, edl, ds_geo, rel, instalacao = carregar(str(PASTA))
areas = ds_geo.set_index("ds")["area_km2"]


def grafico(fig):
    """Todos os gráficos passam por aqui: mesma configuração, sem barra de ferramentas."""
    st.plotly_chart(fig, width="stretch", config=CONFIG_GRAFICO)


def layout(fig, altura=320, legenda=True):
    fig.update_layout(separators=",.", hoverlabel=dict(font_size=13), height=altura, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legenda, font=dict(color=TINTA),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(showgrid=False, linecolor="#D1D5DB")
    fig.update_yaxes(gridcolor="#EEF0F3", zeroline=False)
    return fig


def escolhas(linhas):
    """Quadro 'por que esse gráfico', para quem for apresentar o painel. linhas = [(pergunta, gráfico, por quê)]."""
    with st.expander("Por que escolhemos esses gráficos"):
        corpo = "| Pergunta | Gráfico | Por que esse tipo |\n|---|---|---|\n" + "\n".join(
            f"| {a} | {b} | {c} |" for a, b, c in linhas)
        st.markdown(corpo)


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

    tem_instalacao = instalacao is not None and len(instalacao) > 0
    st.markdown("---")
    st.markdown("**Como ler as cores**")
    st.markdown(LEGENDA_CORES_HTML, unsafe_allow_html=True)
    st.caption("Os controles do antes × depois ficam na aba Histórico e eficácia, perto do gráfico que eles mudam.")

def controles_antes_depois():
    """Controles do antes × depois. Chamada dentro da aba de eficácia. Retorna (corte, corte_proprio)."""
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


    return corte, corte_proprio


lf = an.filtrar_leituras(leituras, anos=anos, ciclos=ciclos, ds=ds_i)            # período + distrito
lp = an.filtrar_leituras(leituras, anos=anos, ciclos=ciclos)                       # período, todos os distritos
v = lf.dropna(subset=["n_ovos"])
oc = ovts_classificadas(ovt, edl, float(raio), instalacao if tem_instalacao else None, len(instalacao) if tem_instalacao else 0)
periodo_txt = "todos os anos" if anos is None else f"{anos}, ciclos {ciclos[0]} a {ciclos[1]}"

c1, c2 = st.columns([3, 1])
c1.markdown('<div class="marca">VIGIA<span>EDL</span></div>', unsafe_allow_html=True)
c2.markdown('<div style="text-align:right"><span class="selo">Protótipo acadêmico · dados da GEVACZ</span></div>',
            unsafe_allow_html=True)
st.markdown(faixa_de_filtros(["todos os anos" if anos is None else f"{anos}", "todos os ciclos" if ciclos is None else f"ciclos {ciclos[0]} a {ciclos[1]}",
                              ds_sel if ds_i is not None else "todos os distritos", f"raio da EDL {raio} m",
                              "só posição conferida" if so_conferidas else "inclui posição não conferida"]), unsafe_allow_html=True)
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

        ind = esp.indicadores_por_regiao(lp, areas)
        pts = ins.pontos_leitura_rapida(ind, v, minimo_bairro=100 if len(v) > 2000 else 20)
        if pts:
            st.markdown('<div class="leitura"><b>Leitura rápida</b>' + "".join(f"<p>• {html.escape(t)}</p>" for t in pts) + "</div>",
                        unsafe_allow_html=True)
        st.markdown(indice([("Evolução e bairros", "evolucao"), ("Distritos", "distritos"), ("Rankings por distrito", "ranking"), ("Ovos por área", "area"),
                            ("Mapa e hotspots", "mapa"), ("Distribuição dos ovos", "distribuicao")]), unsafe_allow_html=True)
        st.subheader("Evolução e bairros", anchor="evolucao")
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
            grafico(fig)
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
            grafico(fig)
            st.caption(f"Só bairros com {minimo} leituras ou mais. A linha tracejada é a média da cidade ({fmt(v['n_ovos'].mean(), 1)}).")

        # ---- mapa de calor distrito × ciclo
        st.subheader("Distritos sanitários", anchor="distritos")
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
        grafico(fig)
        st.caption(f"Ano {ano_h}. O número depois do distrito é a média do ano.")

        # ---- distritos lado a lado
        st.markdown("**Distritos sanitários lado a lado**")
        tds = pd.DataFrame({
            "Distrito": [f"DS {int(i)}" for i in ind.index], "OVTs": ind["n_ovts"].astype(int),
            "Ovos por OVT (média)": ind["ovos_por_ovt"], "Mediana": ind["mediana"],
            "IDO (média das positivas)": ind["ido"], "IPO (% positivas)": ind["ipo"],
            "OVTs por km²": ind["ovts_por_km2"], "Ovos por km² (por ciclo)": ind["ovos_por_km2"]})
        def _realce(linha):
            return ["background-color:#FEF3E7" if (ds_i is not None and linha["Distrito"] == f"DS {ds_i}") else "" for _ in linha]
        st.dataframe(tds.style.apply(_realce, axis=1).format({
            "OVTs": lambda x: fmt(x), "Ovos por OVT (média)": lambda x: fmt(x, 1), "Mediana": lambda x: fmt(x),
            "IDO (média das positivas)": lambda x: fmt(x, 1), "IPO (% positivas)": lambda x: fmt(x, 1),
            "OVTs por km²": lambda x: fmt(x, 1), "Ovos por km² (por ciclo)": lambda x: fmt(x)}),
            width="stretch", hide_index=True)
        st.caption("Ovos por km² sobe também onde há mais OVTs por km²; por isso os dois aparecem juntos. "
                   "Para comparar distritos de tamanhos diferentes, o mais justo é ovos por OVT.")

        # ---- rankings por distrito
        st.markdown("---")
        st.subheader("Rankings por distrito", anchor="ranking")
        st.caption("Escolha um distrito para ver só ele. Este quadro usa o período da barra lateral, mas ignora o filtro de distrito, para você poder trocar aqui.")
        opcoes_r = ["Cidade toda"] + [f"DS {i}" for i in sorted(areas.index)]
        esc_r = st.segmented_control("Ver o ranking de", opcoes_r, default="Cidade toda", key="rank_ds") or "Cidade toda"
        ds_r = None if esc_r == "Cidade toda" else int(esc_r.split()[1])
        vr = lp.dropna(subset=["n_ovos"])
        vrd = vr if ds_r is None else vr[vr["ds"] == ds_r]
        rk_b = (vrd.groupby("bairro").agg(leituras=("n_ovos", "size"), media=("n_ovos", "mean"), ovts=("id_ovt", "nunique"), ds=("ds", "first"))
                .query("leituras >= 20").sort_values("media", ascending=False).head(10))
        rk_o = (vrd.groupby("id_ovt").agg(leituras=("n_ovos", "size"), media=("n_ovos", "mean"), bairro=("bairro", "last"), ds=("ds", "last"),
                                          fora=("atipica", "sum")).query("leituras >= 3").sort_values("media", ascending=False).head(10))
        if len(rk_b) and len(rk_o):
            b1, o1 = rk_b.index[0], rk_o.index[0]
            frase = (f"{esc_r}: o bairro mais alto é {str(b1).title()} ({fmt(rk_b.iloc[0]['media'], 1)} ovos por OVT) e a OVT mais alta é {o1} "
                     f"({fmt(rk_o.iloc[0]['media'], 1)}, em {str(rk_o.iloc[0]['bairro']).title()}).")
            if rk_o.iloc[0]["fora"] > 0:
                frase += f" Essa OVT tem {fmt(rk_o.iloc[0]['fora'])} leituras fora do padrão, então vale conferir antes de chamar de foco."
            st.markdown(leitura(frase), unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Distritos, do mais ao menos infestado**")
            od = ind.dropna(subset=["ovos_por_ovt"]).sort_values("ovos_por_ovt")
            fig = go.Figure(go.Bar(x=od["ovos_por_ovt"], y=[f"DS {int(i)}" for i in od.index], orientation="h",
                                   marker_color=[LARANJA if (ds_r is None or i == ds_r) else "#E3B48A" for i in od.index],
                                   text=[fmt(x, 1) for x in od["ovos_por_ovt"]], textposition="outside"))
            layout(fig, 360, legenda=False)
            fig.update_xaxes(range=[0, float(od["ovos_por_ovt"].max()) * 1.15])
            fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
            grafico(fig)
            st.caption("Média de ovos por OVT no período. O distrito escolhido fica em laranja forte.")
        with r2:
            st.markdown(f"**Top 10 bairros · {esc_r}**")
            if len(rk_b):
                o = rk_b.iloc[::-1]
                fig = go.Figure(go.Bar(x=o["media"], y=[f"{b.title()} · DS {int(d)}" for b, d in zip(o.index, o["ds"])], orientation="h",
                                       marker_color=LARANJA, text=[fmt(x, 1) for x in o["media"]], textposition="outside",
                                       customdata=np.c_[o["ovts"], o["leituras"]],
                                       hovertemplate="%{y}<br>%{x:.1f} ovos por OVT<br>%{customdata[0]:.0f} OVTs · %{customdata[1]:.0f} leituras<extra></extra>"))
                layout(fig, 360, legenda=False)
                fig.update_xaxes(range=[0, float(o["media"].max()) * 1.15])
                fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
                grafico(fig)
                st.caption("Só bairros com 20 leituras ou mais no período.")
            else:
                st.info("Poucos bairros com leituras suficientes neste recorte.")
        r3, r4 = st.columns(2)
        with r3:
            st.markdown(f"**Top 10 OVTs · {esc_r}**")
            if len(rk_o):
                o = rk_o.iloc[::-1]
                fig = go.Figure(go.Bar(x=o["media"], y=[f"{i} · {str(b).title()}" for i, b in zip(o.index, o["bairro"])], orientation="h",
                                       marker_color=["#DC2626" if f > 0 else LARANJA for f in o["fora"]],
                                       text=[fmt(x, 0) for x in o["media"]], textposition="outside",
                                       customdata=np.c_[o["leituras"], o["fora"]],
                                       hovertemplate="%{y}<br>%{x:.0f} ovos em média<br>%{customdata[0]:.0f} leituras · %{customdata[1]:.0f} fora do padrão<extra></extra>"))
                layout(fig, 380, legenda=False)
                fig.update_xaxes(range=[0, float(o["media"].max()) * 1.15])
                fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
                grafico(fig)
                st.caption("Média de ovos da OVT no período (mínimo 3 leituras). Vermelho: a OVT tem alguma leitura 10× acima da mediana do bairro, confira antes de tratar como foco.")
        with r4:
            st.markdown(f"**OVTs que mais ficam no topo · {esc_r}**")
            per_r, _, _ = calcular_persistencia(leituras, len(leituras))
            pr_ = per_r if ds_r is None else per_r[per_r["ds"] == ds_r]
            pr_ = pr_.sort_values(["frac", "media"], ascending=False).head(10)
            if len(pr_):
                o = pr_.iloc[::-1]
                fig = go.Figure(go.Bar(x=o["frac"] * 100, y=[f"{i} · {str(b).title()}" for i, b in zip(o["id_ovt"], o["bairro"])], orientation="h",
                                       marker_color=LARANJA, text=[f"{int(k)} de {int(n)}" for k, n in zip(o["k"], o["n"])], textposition="outside",
                                       hovertemplate="%{y}<br>%{x:.0f}% das leituras entre os 10% mais altos da cidade<extra></extra>"))
                layout(fig, 380, legenda=False)
                fig.update_xaxes(range=[0, max(float(o["frac"].max()) * 100 * 1.25, 10)], ticksuffix="%")
                fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
                grafico(fig)
                st.caption("Em todos os anos: quantos ciclos a OVT ficou entre os 10% mais altos da cidade (OVTs com 15 leituras ou mais).")
            else:
                st.info("Sem OVTs com leituras suficientes para esse recorte.")

        # ---- ovos por área
        st.markdown("---")
        st.subheader("Ovos por área de cada distrito", anchor="area")
        ia = ind.dropna(subset=["ovos_por_km2"])
        if len(ia) >= 2:
            st.markdown(leitura(ins.texto_ovos_por_area(ind)), unsafe_allow_html=True)
            ga, gb = st.columns(2)
            with ga:
                st.markdown("**Ovos por km², por ciclo**")
                o = ia.sort_values("ovos_por_km2")
                fig = go.Figure(go.Bar(x=o["ovos_por_km2"], y=[f"DS {int(i)}" for i in o.index], orientation="h",
                                       marker_color=[LARANJA if i == ds_i else "#E3B48A" for i in o.index],
                                       text=[fmt(x, 0) for x in o["ovos_por_km2"]], textposition="outside",
                                       customdata=np.c_[o["area_km2"], o["n_ovts"]],
                                       hovertemplate="DS %{y}<br>%{x:,.0f} ovos por km²<br>%{customdata[0]:.1f} km² · %{customdata[1]:.0f} OVTs<extra></extra>"))
                layout(fig, 330, legenda=False)
                fig.update_xaxes(range=[0, float(o["ovos_por_km2"].max()) * 1.15])
                fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
                grafico(fig)
                st.caption("Total de ovos de um ciclo dividido pela área do distrito (média dos ciclos do período).")
            with gb:
                st.markdown("**Nível de infestação × cobertura de OVTs**")
                posicoes = ["top center", "bottom center", "middle right", "middle left"]
                ordem_x = ia["ovts_por_km2"].rank(method="first").astype(int) - 1
                fig = go.Figure(go.Scatter(x=ia["ovts_por_km2"], y=ia["ovos_por_ovt"], mode="markers+text",
                                           text=[f"DS {int(i)}" for i in ia.index], textposition=[posicoes[k % 4] for k in ordem_x],
                                           marker=dict(size=13, color=AZUL, line=dict(color="#FFFFFF", width=2)),
                                           hovertemplate="%{text}<br>%{x:.1f} OVTs por km²<br>%{y:.1f} ovos por OVT<extra></extra>"))
                fig.add_hline(y=float(v["n_ovos"].mean()), line_dash="dot", line_color="#9CA3AF")
                fig.add_vline(x=float(ia["n_ovts"].sum() / ia["area_km2"].sum()), line_dash="dot", line_color="#9CA3AF")
                layout(fig, 330, legenda=False)
                fig.update_xaxes(title="OVTs por km² (quanto o distrito é vigiado)")
                fig.update_yaxes(title="Ovos por OVT (quanto cada OVT pega)")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=40))
                grafico(fig)
                st.caption("As linhas pontilhadas são a média da cidade. Acima e à direita, o distrito tem OVT cheia e muita OVT por km².")

        # ---- mapa
        st.markdown("---")
        st.subheader("Mapa de OVTs, EDLs e hotspots", anchor="mapa")
        mc = st.columns([1.3, 1, 1, 1, 1])
        metricas = {"Ovos por OVT": "ovos_por_ovt", "IDO (média das OVTs positivas)": "ido",
                    "Ovos por km² (por ciclo)": "ovos_por_km2", "OVTs por km²": "ovts_por_km2"}
        nome_m = mc[0].selectbox("Colorir os distritos por", list(metricas))
        mostrar_kde = mc[1].checkbox("Kernel Density", value=True)
        modo_ovt = mc[2].selectbox("Pontos das OVTs", ["Só hotspots e fora do padrão", "Todas", "Nenhuma"],
                                   help="Com todas as OVTs, o mapa fica carregado. O padrão mostra só as que merecem atenção.")
        mostrar_ovt = modo_ovt != "Nenhuma"
        mostrar_edl = mc[3].checkbox("EDLs", value=True)
        mostrar_raio = mc[4].checkbox("Raio das EDLs", value=False)
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
        ids_ovt = None
        if modo_ovt.startswith("Só"):
            if kde:
                ids_ovt = set(kde["pontos"].loc[kde["pontos"]["kde"] >= kde["limiar"], "id_ovt"]) | atip
            else:
                ids_ovt = set(media_ovt[media_ovt >= media_ovt.quantile(0.9)].index) | atip
        mapa = construir_mapa(ds_geo, ind[metricas[nome_m]], nome_m, ovt, media_ovt, edl, float(raio),
                              imagem=kde["rgba"] if kde else None, limites=kde["limites"] if kde else None,
                              mostrar_ovt=mostrar_ovt, mostrar_edl=mostrar_edl, mostrar_raio=mostrar_raio, destaques=atip, ids_ovt=ids_ovt)
        centro = centro_do_mapa(ds_geo)
        st_folium(mapa, height=640, use_container_width=True, returned_objects=[], key="mapa_geral",
                  center=centro, zoom=ZOOM_RECIFE)
        st.markdown(legenda_calor_html(), unsafe_allow_html=True)
        st.caption("Pontos com borda vermelha têm alguma leitura 10 vezes acima da mediana do bairro: vale conferir antes de tratar como foco. Para ver todas as OVTs, troque \"Pontos das OVTs\".")

        if kde:
            pts = kde["pontos"]
            quentes = pts[pts["kde"] >= kde["limiar"]]
            st.markdown(f"**Hotspots:** {fmt(len(quentes))} OVTs ficam nos 10% mais intensos da superfície "
                        f"({'média de ovos por OVT' if modo == 'media' else 'ovos por km²'} ≥ {fmt(kde['limiar'], 1)}).")
            tq = (quentes.groupby("ds").size().reindex(sorted(areas.index), fill_value=0)
                  .rename("OVTs em hotspot").to_frame())
            tq["% das OVTs do distrito"] = (tq["OVTs em hotspot"] / pts.groupby("ds").size().reindex(tq.index) * 100).round(1).astype(float)
            tq.index = [f"DS {i}" for i in tq.index]
            tq_txt = pd.DataFrame({"OVTs em hotspot": tq["OVTs em hotspot"].map(lambda x: fmt(int(x))),
                                   "% das OVTs do distrito": tq["% das OVTs do distrito"].map(lambda x: fmt(x, 1))})
            st.dataframe(tq_txt.T, width="stretch")

        # ---- distribuição e concentração
        st.markdown("---")
        st.subheader("Como os ovos se distribuem", anchor="distribuicao")
        curva = ins.curva_de_concentracao(lf)
        st.markdown(leitura(ins.texto_distribuicao(v, curva)), unsafe_allow_html=True)
        dg1, dg2 = st.columns(2)
        pos = v[v["n_ovos"] > 0]["n_ovos"]
        with dg1:
            st.markdown("**Ovos por leitura**")
            if len(pos) > 20:
                bins = np.logspace(0, np.log10(max(float(pos.max()), 10.0)), 36)
                cont, borda = np.histogram(pos, bins=bins)
                centro = np.sqrt(borda[:-1] * borda[1:])
                fig = go.Figure(go.Bar(x=centro, y=cont, marker_color="#E3B48A", width=np.diff(borda) * 0.9,
                                       hovertemplate="de %{x:.0f} ovos: %{y} leituras<extra></extra>"))
                # em eixo logarítmico a linha usa o valor real, mas a anotação pede o log10 dele
                med_, mean_ = float(v["n_ovos"].median()), float(v["n_ovos"].mean())
                fig.add_vline(x=med_, line_color=TINTA, line_dash="dash")
                fig.add_vline(x=mean_, line_color=LARANJA, line_width=3)
                fig.add_annotation(x=float(np.log10(max(med_, 1))), y=1, yref="paper", text=f"mediana {fmt(med_)}", showarrow=False,
                                   xanchor="right", xshift=-4, yanchor="bottom")
                fig.add_annotation(x=float(np.log10(max(mean_, 1))), y=1, yref="paper", text=f"média {fmt(mean_, 0)}", showarrow=False,
                                   xanchor="left", xshift=4, yanchor="bottom", font=dict(color=LARANJA))
                layout(fig, 320, legenda=False)
                fig.update_xaxes(type="log", title="Ovos na leitura (escala logarítmica)")
                fig.update_yaxes(title="Leituras")
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=40))
                grafico(fig)
                st.caption("Só leituras com pelo menos 1 ovo. A cauda à direita é o que faz a média passar da mediana.")
        with dg2:
            st.markdown("**Quantos ovos estão nas OVTs mais infestadas**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="#9CA3AF", dash="dot"), name="Se todas tivessem igual", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=curva["pct_ovts"], y=curva["pct_ovos"], mode="lines", line=dict(color=LARANJA, width=3), name="OVTs, da mais para a menos infestada",
                                     hovertemplate="%{x:.0f}% das OVTs somam %{y:.0f}% dos ovos<extra></extra>"))
            for p_ in (10, 20):
                fig.add_trace(go.Scatter(x=[p_], y=[ins.parcela_dos_ovos(curva, p_)], mode="markers+text", showlegend=False,
                                         text=[f"{p_}% → {fmt(ins.parcela_dos_ovos(curva, p_), 0)}%"], textposition="middle right",
                                         marker=dict(size=9, color=LARANJA, line=dict(color="#FFFFFF", width=2)), hoverinfo="skip"))
            layout(fig, 320)
            fig.update_xaxes(title="% das OVTs (mais infestadas primeiro)", range=[0, 100])
            fig.update_yaxes(title="% dos ovos", range=[0, 100])
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=40))
            grafico(fig)
            st.caption("Quanto mais a curva se afasta da diagonal, mais os ovos se concentram em poucas OVTs.")

        escolhas([
            ("Os ovos estão subindo ou caindo?", "Linhas de média e mediana por ciclo",
             "Tempo no eixo horizontal é o jeito mais direto de ver tendência. Duas linhas mostram se poucos valores altos puxam a média."),
            ("Quais bairros estão pior?", "Barras horizontais ordenadas",
             "Nome de bairro é comprido e cabe melhor na horizontal. A ordem já responde quem é o pior."),
            ("Quais distritos, em quais ciclos?", "Mapa de calor distrito × ciclo",
             "Cruza duas coisas num lugar só. O número fica dentro da célula, então a leitura não depende só da cor (item de acessibilidade do backlog)."),
            ("Quem lidera em cada distrito?", "Barras de top 10 (bairros e OVTs) com seletor de distrito",
             "Ranking é o formato mais direto para priorizar. O seletor isola um distrito de cada vez, em vez de misturar os oito."),
            ("Distrito grande parece pior só por ser grande?", "Barras de ovos por km² e dispersão nível × cobertura",
             "Dividir pela área tira o efeito do tamanho. A dispersão separa quanto cada OVT pega de quantas OVTs há por km²."),
            ("Onde estão os hotspots?", "Mapa com Kernel Density",
             "Foi a troca combinada no plano de trabalho: o KDE suaviza os pontos e mostra regiões, em vez de pontos soltos."),
            ("A média engana?", "Histograma em escala logarítmica e curva de concentração",
             "A distribuição é muito torta. O log deixa a cauda visível e a curva mostra quanto dos ovos está em poucas OVTs."),
            ("O dado chega a tempo?", "Cartão com a mediana de dias até a digitação",
             "Ligado à hipótese do Drive: reduzir o tempo de análise do gestor. Um número só, sem gráfico, porque a pergunta é só essa."),
        ])
        st.caption(f"Fonte: monitoramento de ovitrampas da GEVACZ · {fmt(len(v))} leituras com contagem em {periodo_txt}.")


# ============================================================================ FICHA DA OVT
with tab2:
    st.markdown("## Ficha da OVT")
    st.markdown(indice([("Cadastro e histórico da OVT", "cadastro"), ("OVTs que mais fogem do padrão", "fora-do-padrao")]), unsafe_allow_html=True)
    ids_ok = sorted(leituras["id_ovt"].unique())      # a ficha é histórica: lista todas, não só as do período
    atip_ids = lp[lp["atipica"]].groupby("id_ovt")["vezes_mediana"].max().sort_values(ascending=False) if "atipica" in lp else pd.Series(dtype=float)
    padrao = atip_ids.index[0] if len(atip_ids) else ids_ok[0]
    if st.session_state.get("ovt_sel") not in ids_ok:
        st.session_state["ovt_sel"] = padrao
    sugestoes = list(atip_ids.index[:4])
    if "BVG6043" in ids_ok and "BVG6043" not in sugestoes:
        sugestoes.append("BVG6043")
    bc = st.columns([1.2] + [1] * len(sugestoes) + [3 - len(sugestoes) * 0.0])
    bc[0].markdown('<div class="sub" style="padding-top:8px">Casos para olhar</div>', unsafe_allow_html=True)
    for col_, o_ in zip(bc[1:], sugestoes):
        col_.button(o_, key=f"pick_{o_}", on_click=lambda o=o_: st.session_state.update(ovt_sel=o), width="stretch",
                    help="BVG6043 é o caso citado no plano de trabalho." if o_ == "BVG6043" else "Entre as OVTs com mais leituras fora do padrão no período.")
    sel = st.selectbox("OVT", ids_ok, key="ovt_sel", help="Digite para procurar. A lista começa na OVT mais fora do padrão do período.")
    h = leituras[leituras["id_ovt"] == sel].sort_values(["ano", "ciclo"])
    hv = h.dropna(subset=["n_ovos"])
    info = oc[oc["id_ovt"] == sel]
    bairro = h["bairro"].iloc[-1] if len(h) else "—"
    ds_o = int(h["ds"].dropna().iloc[-1]) if h["ds"].notna().any() else None
    n_bairro = int(ovt[ovt["bairro"] == bairro]["id_ovt"].nunique()) if len(ovt) else 0

    st.subheader(f"OVT {sel} · {str(bairro).title()}", anchor="cadastro")
    selos = [f"Distrito sanitário {ds_o}" if ds_o else "", f"{len(hv)} leituras com contagem"]
    selos.append("Sem coordenadas" if info.empty else ("Posição conferida" if bool(info["posicao_validada"].iloc[0]) else "Posição não conferida"))
    st.markdown(" ".join(f'<span class="selo">{s}</span>' for s in selos if s), unsafe_allow_html=True)
    f1, f2 = st.columns([1, 2])
    with f1:
        pos_txt = "—"
        if len(hv):
            medias = lp.dropna(subset=["n_ovos"]).groupby("id_ovt")["n_ovos"].mean()
            if sel in medias.index:
                pos_txt = f"{fmt(int((medias > medias[sel]).sum()) + 1)}ª de {fmt(len(medias))} · acima de {fmt(min((medias < medias[sel]).mean() * 100, 99.9), 1)}% das OVTs"
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
            if "atipica" in hv and hv["atipica"].any():
                ha = hv[hv["atipica"]]
                fig.add_trace(go.Scatter(x=[f"{int(a)}·C{int(c)}" for a, c in zip(ha["ano"], ha["ciclo"])], y=ha["n_ovos"], mode="markers",
                                         name="10× a mediana ou mais", marker=dict(size=11, color="#DC2626", line=dict(color="#FFFFFF", width=2)),
                                         hovertemplate="%{y:.0f} ovos · %{customdata:.1f}× a mediana<extra></extra>", customdata=ha["vezes_mediana"]))
            layout(fig, 330)
            fig.update_xaxes(type="category", tickangle=-45, nticks=16)
            grafico(fig)
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

    # ------------------------------------------------------------------ quem foge do padrão
    st.markdown("---")
    st.subheader("OVTs que mais fogem do padrão", anchor="fora-do-padrao")
    st.caption("Leitura marcada quando passa de 10 vezes a mediana do bairro no mesmo ciclo (com pelo menos 5 OVTs lidas no bairro).")
    if "atipica" in lf:
        fora = ins.ovts_mais_fora_do_padrao(lf, topo=10)
        efeito = ins.efeito_das_atipicas(lf)
        if len(fora):
            st.markdown(leitura(ins.texto_atipicas(lf, efeito)), unsafe_allow_html=True)
            q1, q2 = st.columns(2)
            with q1:
                st.markdown("**Leituras fora do padrão, por OVT**")
                o = fora.iloc[::-1]
                fig = go.Figure(go.Bar(x=o["leituras_atipicas"], y=[f"{i} · {b.title()}" for i, b in zip(o["id_ovt"], o["bairro"])], orientation="h",
                                       marker_color="#DC2626", text=o["leituras_atipicas"], textposition="outside",
                                       customdata=np.c_[o["maior_multiplo"], o["maior_valor"]],
                                       hovertemplate="%{y}<br>%{x} leituras · maior: %{customdata[1]:.0f} ovos (%{customdata[0]:.0f}× a mediana)<extra></extra>"))
                layout(fig, 340, legenda=False)
                fig.update_xaxes(title="Leituras com 10× a mediana ou mais", range=[0, float(o["leituras_atipicas"].max()) * 1.15])
                fig.update_layout(margin=dict(l=10, r=30, t=10, b=40))
                grafico(fig)
            with q2:
                st.markdown("**Quanto isso mexe na média do distrito**")
                e_ = efeito.sort_values("ds")
                fig = go.Figure(go.Bar(x=[f"DS {int(d)}" for d in e_["ds"]], y=e_["queda_pct"], marker_color="#E3B48A",
                                       text=[f"{fmt(x, 1)}%" for x in e_["queda_pct"]], textposition="outside",
                                       customdata=np.c_[e_["com"], e_["sem"], e_["n_atipicas"]],
                                       hovertemplate="%{x}<br>média %{customdata[0]:.1f} → %{customdata[1]:.1f} sem elas<br>%{customdata[2]:.0f} leituras<extra></extra>"))
                layout(fig, 340, legenda=False)
                fig.update_yaxes(title="Variação da média sem as leituras fora do padrão", ticksuffix="%", range=[float(e_["queda_pct"].min()) * 1.4, 0.5])
                grafico(fig)
            st.caption("Para olhar uma delas, escolha o código na lista da Ficha, lá no alto.")

    escolhas([
        ("Essa OVT é foco de verdade ou erro de digitação?", "Linha da OVT contra a mediana do bairro, com pontos vermelhos nas leituras 10× acima",
         "Comparar com as vizinhas no mesmo ciclo é o teste mais simples. A tabela ao lado mostra o mesmo em números."),
        ("Quais OVTs conferir primeiro?", "Barras com o número de leituras fora do padrão",
         "Ordenar por quantidade separa o problema recorrente do pico isolado. É o item 'reduzir erros e retrabalho' da ideação."),
        ("Isso muda a conclusão?", "Barras com a variação da média do distrito sem essas leituras",
         "Mostra o custo de não conferir: o quanto a média do distrito cai sem elas."),
    ])


# ============================================================================ HISTÓRICO E EFICÁCIA
with tab3:
    st.markdown("## Histórico e eficácia")
    st.markdown(indice([("Evolução por distrito", "evolucao-ds"), ("Ano a ano", "anos"), ("Focos que se repetem", "focos"),
                        ("Eficácia das EDLs", "eficacia"), ("Prazos do dado", "prazos"), ("Cuidados com os dados", "limitacoes")]), unsafe_allow_html=True)
    st.caption("Como a densidade de ovos evoluiu em cada distrito, o que dá para dizer sobre as EDLs e o que os dados ainda não permitem afirmar.")

    st.subheader("Evolução por distrito sanitário", anchor="evolucao-ds")
    evd = an.evolucao_por_ciclo(lp, por="ds")
    evd["ordem"] = evd["ano"] * 100 + evd["ciclo"]
    ordens = sorted(evd["ordem"].unique())
    media_cidade = lp["n_ovos"].mean()
    def _titulo_ds(d_):
        s_ = evd[evd["ds"] == d_]
        if s_.empty:
            return f"DS {d_}"
        p_ = s_.loc[s_["media"].idxmax()]
        rot = f"{int(p_['ano'])}·C{int(p_['ciclo'])}" if evd["ano"].nunique() > 1 else f"C{int(p_['ciclo'])}"
        return f"<b>DS {d_}</b> · média {fmt(lp[lp['ds'] == d_]['n_ovos'].mean(), 1)}<br><span style='font-size:11px;color:#6B7280'>pico de {fmt(p_['media'], 0)} em {rot}</span>"
    fig = make_subplots(rows=2, cols=4, subplot_titles=[_titulo_ds(i) for i in sorted(areas.index)], shared_yaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.2)
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
    layout(fig, 480, legenda=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(range=[max(0, float(evd["media"].min()) * 0.85), float(evd["media"].max()) * 1.08])
    grafico(fig)
    st.caption(f"Média de ovos por OVT em cada ciclo, na mesma escala. A linha pontilhada é a média da cidade ({fmt(media_cidade, 1)}). O ponto marca o pico.")

    # ------------------------------------------------------------------ ano a ano
    anos_todos = sorted(int(a) for a in leituras["ano"].dropna().unique())
    if len(anos_todos) > 1:
        limite = min(int(leituras.loc[leituras["ano"] == a, "ciclo"].max()) for a in anos_todos)
        aa = leituras[leituras["ciclo"] <= limite].dropna(subset=["n_ovos"])
        pv = aa.pivot_table(index="ds", columns="ano", values="n_ovos", aggfunc="mean").sort_index()
        st.subheader("Ano a ano", anchor="anos")
        st.markdown("**Média de ovos por OVT, distrito × ano**")
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=[str(c) for c in pv.columns], y=[f"DS {int(d)}" for d in pv.index],
            colorscale=[[0, "#FDF0E3"], [0.5, "#E08A2E"], [1, "#7A3A0A"]], text=np.round(pv.values), texttemplate="%{text:.0f}",
            hovertemplate="%{y} · %{x}: %{z:.1f} ovos por OVT<extra></extra>", xgap=2, ygap=2, colorbar=dict(title="ovos")))
        layout(fig, 330, legenda=False)
        fig.update_xaxes(type="category")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        grafico(fig)
        st.caption(f"Média de ovos por OVT nos ciclos 1 a {limite} de cada ano. O limite é o do ano com menos ciclos, "
                   "para comparar o mesmo período e não um ano inteiro com um ano pela metade.")

    # ------------------------------------------------------------------ sazonalidade
    lsaz = leituras if ds_i is None else leituras[leituras["ds"] == ds_i]
    saz = ins.sazonalidade(lsaz)
    if saz.shape[1] >= 1 and len(saz) > 1:
        st.markdown(f"**O ano em ciclos: média de ovos por OVT ({ds_sel})**")
        st.markdown(leitura(ins.texto_anos(saz)), unsafe_allow_html=True)
        rampa = ["#E8C9A8", "#E08A2E", "#B45309", "#7A3A0A"]
        fig = go.Figure()
        cols_a = list(saz.columns)
        n_a = len(cols_a)
        cores_a = rampa[-n_a:] if n_a <= 4 else [rampa[round(i / (n_a - 1) * 3)] for i in range(n_a)]
        for k, a in enumerate(cols_a):
            cor = cores_a[k]
            fig.add_trace(go.Scatter(x=saz.index, y=saz[a], name=str(int(a)), mode="lines+markers",
                                     line=dict(color=cor, width=3 if k == len(cols_a) - 1 else 2), marker=dict(size=6),
                                     hovertemplate=f"{int(a)} · ciclo %{{x}}: %{{y:.0f}}<extra></extra>"))
        layout(fig, 340)
        fig.update_xaxes(title="Ciclo", dtick=1, range=[0.5, float(saz.index.max()) + 0.5])
        fig.update_yaxes(title="Ovos por OVT (média)")
        grafico(fig)

    # ------------------------------------------------------------------ focos que se repetem
    st.markdown("---")
    st.subheader("Focos que se repetem", anchor="focos")
    st.caption("Uma OVT entre as 10% mais altas da cidade num ciclo pode ser sorte. Entre as mais altas em muitos ciclos, é foco.")
    per, cron, por_ano = calcular_persistencia(leituras, len(leituras))
    if len(per) and len(cron):
        n_cr = int((per["frac"] >= 0.3).sum())
        st.markdown(leitura(ins.texto_persistencia(por_ano, cron, n_cr, len(per))), unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("**OVTs no topo em pelo menos 30% dos ciclos**")
            cr = cron.sort_values("ds")
            media_c = cron["cronicas"].sum() / cron["n_ovts"].sum() * 100
            fig = go.Figure(go.Bar(x=[f"DS {int(d)}" for d in cr["ds"]], y=cr["pct"],
                                   marker_color=[LARANJA if d == ds_i else "#E3B48A" for d in cr["ds"]],
                                   text=[f"{fmt(x, 1)}%" for x in cr["pct"]], textposition="outside",
                                   customdata=np.c_[cr["cronicas"], cr["n_ovts"]],
                                   hovertemplate="%{x}: %{customdata[0]:.0f} de %{customdata[1]:.0f} OVTs<extra></extra>"))
            fig.add_hline(y=float(media_c), line_dash="dot", line_color="#9CA3AF",
                          annotation_text=f"cidade {fmt(media_c, 1)}%", annotation_position="top left")
            layout(fig, 320, legenda=False)
            fig.update_yaxes(title="% das OVTs do distrito", range=[0, float(cr["pct"].max()) * 1.2])
            grafico(fig)
            st.caption("Só OVTs com 15 leituras ou mais, em todos os anos. Se não houvesse foco fixo, o esperado seria perto de 10% ou menos em todo distrito.")
        with h2:
            anos_pa = sorted(por_ano.columns)
            if len(anos_pa) >= 2:
                a0, a1 = anos_pa[-2], anos_pa[-1]
                st.markdown(f"**A mesma OVT em {int(a0)} e em {int(a1)}**")
                d2 = por_ano[[a0, a1]].dropna()
                lim_ = float(max(d2.quantile(.995)))
                fig = go.Figure()
                fig.add_trace(go.Scattergl(x=d2[a0], y=d2[a1], mode="markers", marker=dict(size=5, color=AZUL, opacity=0.3),
                                           text=d2.index, hovertemplate="%{text}<br>" + f"{int(a0)}: " + "%{x:.0f}<br>" + f"{int(a1)}: " + "%{y:.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[0, lim_], y=[0, lim_], mode="lines", line=dict(color="#9CA3AF", dash="dot"), hoverinfo="skip"))
                layout(fig, 320, legenda=False)
                fig.update_xaxes(title=f"Média em {int(a0)}", range=[0, lim_])
                fig.update_yaxes(title=f"Média em {int(a1)}", range=[0, lim_])
                grafico(fig)
                st.caption("Cada ponto é uma OVT. Perto da diagonal, a OVT teve o mesmo nível nos dois anos.")
        topo_c = per[per["frac"] >= 0.3]
        if ds_i is not None:
            topo_c = topo_c[topo_c["ds"] == ds_i]
        topo_c = topo_c.sort_values(["frac", "media"], ascending=False).head(10)
        if len(topo_c):
            st.markdown("**As 10 OVTs que mais ficaram no topo**" + ("" if ds_i is None else f" · DS {ds_i}"))
            st.dataframe(pd.DataFrame({"OVT": topo_c["id_ovt"], "Bairro": topo_c["bairro"].str.title(), "DS": topo_c["ds"].astype("Int64"),
                                       "Ciclos no topo": topo_c["k"].astype(int), "Leituras": topo_c["n"].astype(int),
                                       "% dos ciclos no topo": (topo_c["frac"] * 100).round(0), "Média de ovos": topo_c["media"].round(0)}),
                         width="stretch", hide_index=True)
            st.caption("São as primeiras candidatas para uma conferência em campo. Algumas podem estar aqui por erro de contagem, veja a Ficha da OVT.")

    # ------------------------------------------------------------------ eficácia
    st.markdown("---")
    st.subheader("Eficácia das EDLs · perto × longe e antes × depois", anchor="eficacia")
    with st.expander("Ajustes do antes × depois (data de corte)"):
        corte, corte_proprio = controles_antes_depois()
    corte_ativo = corte is not None
    base_leit = leituras if corte_ativo else lp                       # com corte, usa todos os anos
    if corte_ativo and anos is not None:
        st.caption("Com a data de corte ligada, a análise usa todos os anos, para ter leituras antes e depois.")
    base = an.base_por_ovt(base_leit, oc, corte=corte, apenas_validadas=so_conferidas, corte_proprio=corte_proprio)
    base_f = base if ds_i is None else base[base["ds"] == ds_i]
    res = an.eficacia(base_f)
    geral = res["geral"]

    iv = ins.intervalo_da_diferenca(base)
    bairro_cmp = ins.comparar_dentro_do_bairro(base if ds_i is None else base_f)

    # a resposta primeiro, os gráficos depois
    if geral["n_a"] and geral["n_b"]:
        sig = ("A diferença é maior do que o acaso explicaria" if (geral["p_valor"] == geral["p_valor"] and geral["p_valor"] < 0.05)
               else "A diferença não é maior do que o acaso explicaria")
        p_txt = f" (p = {fmt(geral['p_valor'], 3)})" if geral["p_valor"] == geral["p_valor"] else ""
        st.markdown(leitura(ins.texto_eficacia(geral, bairro_cmp, iv, raio, f"{sig}{p_txt}.", ins.frase_tendencia(ins.tendencia_com_distancia(base_f))), "Resumindo"), unsafe_allow_html=True)
    elif geral["aviso"]:
        st.warning(geral["aviso"])


    e1, e2, e3 = st.columns([1, 1, 1])
    with e1:
        st.markdown(f"**Perto × longe da EDL** · raio de {raio} m")
        if geral["n_a"] and geral["n_b"]:
            fig = go.Figure(go.Bar(x=[f"Perto (até {raio} m)", f"Longe (mais de {raio} m)"], y=[geral["mediana_a"], geral["mediana_b"]],
                                   marker_color=[AZUL, "#9CA3AF"], text=[fmt(geral["mediana_a"], 1), fmt(geral["mediana_b"], 1)], textposition="outside"))
            layout(fig, 260, legenda=False)
            grafico(fig)
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
            grafico(fig)
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
            grafico(fig)
            regra = ("Perto: data da EDL de cada OVT. Longe" if corte_proprio else "Todas as OVTs") + f": corte em {corte:%d/%m/%Y}."
            st.caption(f"Mediana de ovos por OVT. {regra} A diferença das diferenças é {fmt(did['did'], 1)} ovos.")
        else:
            st.markdown('<div class="cartao tracejado"><div class="rotulo">Aguardando dados</div><div class="grande" style="font-size:1.3rem">Sem corte de data</div>'
                        '<div class="nota">Precisa da data de instalação de cada EDL e de leituras de antes dela. '
                        'Abra os ajustes do antes × depois, acima, para testar uma data como hipótese, ou use o modelo de planilha abaixo para pedir as datas à GEVACZ.</div></div>',
                        unsafe_allow_html=True)

    modelo = PASTA / "edl_instalacao_modelo.csv"
    if modelo.exists():
        st.download_button("Baixar modelo para a GEVACZ preencher (data de instalação das EDLs)", modelo.read_bytes(),
                           file_name="edl_instalacao_modelo.csv", mime="text/csv")

    # ---- intervalo por distrito e distribuição
    f1_, f2_ = st.columns(2)
    with f1_:
        st.markdown("**Diferença perto × longe em cada distrito, com intervalo**")
        ivp = iv.dropna(subset=["dif_pct"]).sort_values("ds", ascending=False)
        if len(ivp):
            fig = go.Figure()
            com_ic = ivp.dropna(subset=["ic_baixo"])
            fig.add_trace(go.Scatter(
                x=com_ic["dif_pct"], y=[f"DS {int(d)}" for d in com_ic["ds"]], mode="markers",
                marker=dict(size=10, color=AZUL, line=dict(color="#FFFFFF", width=2)),
                error_x=dict(type="data", symmetric=False, array=com_ic["ic_alto"] - com_ic["dif_pct"],
                             arrayminus=com_ic["dif_pct"] - com_ic["ic_baixo"], color="#6B7280", thickness=2, width=0),
                customdata=np.c_[com_ic["n_perto"], com_ic["n_longe"], com_ic["ic_baixo"], com_ic["ic_alto"]],
                hovertemplate="%{y}: %{x:+.1f}%<br>intervalo de %{customdata[2]:+.1f}% a %{customdata[3]:+.1f}%<br>"
                              "%{customdata[0]:.0f} OVTs perto · %{customdata[1]:.0f} longe<extra></extra>"))
            fig.add_vline(x=0, line_color=TINTA, line_width=1)
            layout(fig, 340, legenda=False)
            fig.update_xaxes(title="Mediana perto menos mediana longe (% da mediana longe)", zeroline=False, ticksuffix="%")
            fig.update_yaxes(gridcolor="#EEF0F3")
            grafico(fig)
            st.caption("À esquerda do zero, perto da EDL tem menos ovos. Quando a barra cruza o zero, não dá para separar a diferença do acaso. "
                       "Intervalo de 95% por bootstrap das OVTs.")
        else:
            st.info("Poucas OVTs com posição neste filtro.")
    with f2_:
        st.markdown("**Distribuição das OVTs: perto e longe**")
        if geral["n_a"] and geral["n_b"]:
            fig = go.Figure()
            for rot, grp, cor in ((f"Perto (até {raio} m)", "perto", AZUL), (f"Longe (mais de {raio} m)", "longe", "#9CA3AF")):
                fig.add_trace(go.Box(y=base_f.loc[base_f["proximidade"] == grp, "ovos_media"], name=rot, marker_color=cor,
                                     boxpoints=False, boxmean=True, line=dict(width=2)))
            layout(fig, 340, legenda=False)
            fig.update_yaxes(title="Média de ovos da OVT", range=[0, float(base_f["ovos_media"].quantile(0.97))])
            grafico(fig)
            st.caption("A caixa vai do 1º ao 3º quartil, a linha cheia é a mediana e a tracejada é a média. Cortei o eixo nos 3% mais altos para a caixa ficar legível.")

    st.markdown("**Perto × longe em cada distrito sanitário**")
    pd_ = res["por_ds"].copy()
    if len(pd_):
        tabela = pd.DataFrame({
            "Distrito": [f"DS {int(d)}" for d in pd_["ds"]], "OVTs perto": pd_["n_a"].astype(int), "OVTs longe": pd_["n_b"].astype(int),
            "Mediana perto": pd_["mediana_a"].round(1), "Mediana longe": pd_["mediana_b"].round(1),
            "Diferença (%)": pd_["diferenca_pct"].round(1), "p-valor": pd_["p_valor"].round(3)})
        st.dataframe(tabela.style.format({"Mediana perto": lambda x: fmt(x, 1), "Mediana longe": lambda x: fmt(x, 1),
                                          "Diferença (%)": lambda x: fmt(x, 1), "p-valor": lambda x: fmt(x, 3),
                                          "OVTs perto": lambda x: fmt(x), "OVTs longe": lambda x: fmt(x)}, na_rep="—"),
                     width="stretch", hide_index=True)
        st.caption("Oito comparações lado a lado: com tantos testes, alguns p pequenos aparecem por acaso. Vale olhar o padrão, não cada p isolado.")

    # ------------------------------------------------------------------ prazos
    st.markdown("---")
    pz = an.prazos_do_dado(lp)
    st.subheader("Quanto tempo o dado leva para chegar", anchor="prazos")
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
        grafico(fig)
        st.caption("Dias contados a partir da coleta. A data de envio ao DS só está registrada em parte dos ciclos.")

    # ------------------------------------------------------------------ limitações
    st.markdown("---")
    st.subheader("Cuidados com os dados", anchor="limitacoes")
    st.caption("O que olhar com desconfiança antes de concluir qualquer coisa.")
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
    escolhas([
        ("Cada distrito acompanha a cidade?", "Oito mini-gráficos de linha na mesma escala",
         "Escala igual deixa comparar a altura de um distrito com outro. A linha pontilhada é a cidade e o ponto marca o pico."),
        ("Um ano foi pior que o outro?", "Mapa de calor distrito × ano e linhas por ano",
         "O calor dá a visão geral. As linhas mostram em que ciclo cada ano subiu, e a sazonalidade aparece."),
        ("Os focos se repetem?", "Barras de OVTs que ficam no topo e dispersão da mesma OVT em dois anos",
         "Se a OVT alta num ano é alta no outro, o histórico serve para decidir onde agir. A diagonal mostra o quanto isso vale."),
        ("A EDL funciona?", "Barras perto × longe, faixas de distância, intervalo por distrito e antes × depois",
         "É a pergunta central do projeto. O intervalo evita confiar num p-valor isolado, e a comparação dentro do bairro tira o efeito de a EDL estar em área já mais infestada."),
        ("Quanto o dado demora?", "Barras horizontais com a mediana e o 90%",
         "Dois números por etapa, que é o que o gestor precisa para saber o quanto esperar."),
    ])
    st.caption("Fonte: monitoramento de ovitrampas da GEVACZ, tratado pelo pipeline do projeto (scripts/preparar_dados.py).")
