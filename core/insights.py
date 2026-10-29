"""Cálculos por trás dos gráficos de análise e os textos de leitura rápida.

Os textos saem dos números, então continuam valendo quando entrarem outros anos ou outro filtro.
Nada aqui altera as tabelas recebidas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from core.visual import fmt


# ----------------------------------------------------------------------------- distribuição e concentração
def curva_de_concentracao(leituras: pd.DataFrame) -> pd.DataFrame:
    """Curva de Lorenz dos ovos: com as OVTs da mais para a menos infestada, quanto dos ovos cada fatia soma.

    Usa o total de ovos de cada OVT no período. Retorna `pct_ovts` e `pct_ovos` (0 a 100).
    """
    tot = leituras.dropna(subset=["n_ovos"]).groupby("id_ovt")["n_ovos"].sum().sort_values(ascending=False)
    if tot.empty or tot.sum() == 0:
        return pd.DataFrame({"pct_ovts": [0.0, 100.0], "pct_ovos": [0.0, 100.0]})
    acum = tot.cumsum() / tot.sum() * 100
    pct = np.arange(1, len(tot) + 1) / len(tot) * 100
    return pd.DataFrame({"pct_ovts": np.r_[0, pct], "pct_ovos": np.r_[0, acum.to_numpy()]})


def parcela_dos_ovos(curva: pd.DataFrame, pct_ovts: float) -> float:
    """Quanto dos ovos (%) está nas `pct_ovts`% OVTs mais infestadas."""
    return float(np.interp(pct_ovts, curva["pct_ovts"], curva["pct_ovos"]))


def texto_distribuicao(v: pd.DataFrame, curva: pd.DataFrame) -> str:
    n = v["n_ovos"]
    if n.empty:
        return ""
    acima = (n > n.mean()).mean() * 100
    return (f"A média ({fmt(n.mean(), 1)}) fica bem acima da mediana ({fmt(n.median())}): só {fmt(acima, 0)}% das leituras passam da média. "
            f"As 10% de OVTs mais infestadas somam {fmt(parcela_dos_ovos(curva, 10), 0)}% dos ovos e as 20% mais infestadas, "
            f"{fmt(parcela_dos_ovos(curva, 20), 0)}%. Há concentração, mas ela está espalhada por muitas OVTs, não em meia dúzia.")


# ----------------------------------------------------------------------------- ovos por área
def texto_ovos_por_area(ind: pd.DataFrame) -> str:
    """Compara distritos pelo ovos por km² e mostra onde ele conta uma história diferente de ovos por OVT."""
    d = ind.dropna(subset=["ovos_por_km2", "ovos_por_ovt"])
    if len(d) < 2:
        return ""
    topo, base = d["ovos_por_km2"].idxmax(), d["ovos_por_km2"].idxmin()
    razao = d.loc[topo, "ovos_por_km2"] / d.loc[base, "ovos_por_km2"]
    t = (f"O DS {int(topo)} concentra mais ovos por km² ({fmt(d.loc[topo, 'ovos_por_km2'], 0)} por ciclo), "
         f"{fmt(razao, 1)} vezes o DS {int(base)} ({fmt(d.loc[base, 'ovos_por_km2'], 0)}). ")
    # distrito que muda de posição quando se olha a área em vez da OVT
    r_ovt = d["ovos_por_ovt"].rank(ascending=False)
    r_km2 = d["ovos_por_km2"].rank(ascending=False)
    salto = (r_ovt - r_km2)
    s = salto.abs().idxmax()
    if abs(salto[s]) >= 3:
        sentido = "sobe" if salto[s] > 0 else "cai"
        t += (f"O DS {int(s)} {sentido} da {int(r_ovt[s])}ª para a {int(r_km2[s])}ª posição quando a conta é por área, "
              f"porque tem {fmt(d.loc[s, 'ovts_por_km2'], 1)} OVTs por km². ")
    t += ("Ovos por km² mistura dois efeitos, quantos ovos cada OVT pega e quantas OVTs há por km². "
          "O gráfico ao lado separa os dois.")
    return t


# ----------------------------------------------------------------------------- sazonalidade e anos
def sazonalidade(leituras: pd.DataFrame) -> pd.DataFrame:
    """Média da cidade por ciclo (linhas) e ano (colunas)."""
    v = leituras.dropna(subset=["n_ovos"])
    return v.pivot_table(index="ciclo", columns="ano", values="n_ovos", aggfunc="mean").sort_index()


def texto_anos(saz: pd.DataFrame) -> str:
    """Compara os anos nos ciclos que todos têm e aponta os ciclos mais altos."""
    if saz.shape[1] < 2:
        return ""
    comum = saz.dropna()
    if comum.empty:
        return ""
    medias = comum.mean()
    maior, menor = medias.idxmax(), medias.idxmin()
    t = (f"Nos ciclos 1 a {int(comum.index.max())}, que existem em todos os anos, {int(maior)} foi o ano mais alto "
         f"({fmt(medias[maior], 1)} ovos por OVT) e {int(menor)} o mais baixo ({fmt(medias[menor], 1)}), "
         f"uma diferença de {fmt((medias[maior] / medias[menor] - 1) * 100, 0)}%. ")
    picos = []
    for a in saz.columns:
        s = saz[a].dropna()
        if len(s):
            picos.append(f"{int(a)}: ciclo {int(s.idxmax())}")
    t += "Ciclo de pico em cada ano, " + "; ".join(picos) + "."
    return t


# ----------------------------------------------------------------------------- hotspots que se repetem
def persistencia_de_hotspot(leituras: pd.DataFrame, quantil: float = 0.9, min_leituras: int = 15) -> pd.DataFrame:
    """Em quantos ciclos cada OVT ficou entre os 10% mais altos da cidade naquele ciclo.

    Uma OVT que aparece no topo de vez em quando é oscilação. Uma que aparece em boa parte dos ciclos é foco que se repete.
    Retorna id_ovt, ds, bairro, n (leituras), k (vezes no topo), frac (k/n) e media.
    """
    v = leituras.loc[leituras["n_ovos"].notna(), ["ano", "ciclo", "id_ovt", "ds", "bairro", "n_ovos"]].copy()   # só o necessário: poupa memória
    corte = v.groupby(["ano", "ciclo"])["n_ovos"].transform(lambda s: s.quantile(quantil))
    v["topo"] = v["n_ovos"] >= corte
    g = v.groupby("id_ovt")
    per = pd.DataFrame({"n": g.size(), "k": g["topo"].sum(), "media": g["n_ovos"].mean(),
                        "ds": g["ds"].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
                        "bairro": g["bairro"].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else "")})
    per = per[per["n"] >= min_leituras].copy()
    per["frac"] = per["k"] / per["n"]
    return per.reset_index()


def hotspot_cronico_por_ds(per: pd.DataFrame, limite: float = 0.3) -> pd.DataFrame:
    """% das OVTs de cada distrito que ficaram no topo em pelo menos `limite` dos ciclos."""
    if per.empty:
        return pd.DataFrame(columns=["ds", "n_ovts", "cronicas", "pct"])
    g = per.dropna(subset=["ds"]).groupby("ds")
    out = pd.DataFrame({"n_ovts": g.size(), "cronicas": g["frac"].apply(lambda s: int((s >= limite).sum()))})
    out["pct"] = out["cronicas"] / out["n_ovts"] * 100
    return out.reset_index()


def media_por_ovt_e_ano(leituras: pd.DataFrame, minimo: int = 8) -> pd.DataFrame:
    """Uma linha por OVT, uma coluna por ano, com a média de ovos (só anos com leituras suficientes)."""
    v = leituras.dropna(subset=["n_ovos"])
    g = v.groupby(["id_ovt", "ano"])["n_ovos"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= minimo]
    return g.pivot(index="id_ovt", columns="ano", values="mean")


def correlacao_entre_anos(por_ano: pd.DataFrame, a: int, b: int) -> dict:
    """Correlação de postos (Spearman) da média das OVTs entre dois anos."""
    d = por_ano[[a, b]].dropna() if {a, b} <= set(por_ano.columns) else pd.DataFrame()
    if len(d) < 20:
        return {"n": len(d), "rho": np.nan}
    rho = spearmanr(d[a], d[b]).statistic
    return {"n": len(d), "rho": float(rho)}


def texto_persistencia(por_ano: pd.DataFrame, cron_ds: pd.DataFrame, n_cronicas: int, n_total: int) -> str:
    anos = sorted(por_ano.columns)
    partes = []
    if len(anos) >= 2:
        c = correlacao_entre_anos(por_ano, anos[-2], anos[-1])
        if c["rho"] == c["rho"]:
            partes.append(f"A OVT que está alta num ano tende a estar alta no seguinte: a correlação entre {anos[-2]} e {anos[-1]} "
                          f"é {fmt(c['rho'], 2)} (em {fmt(c['n'])} OVTs). Isso quer dizer que os focos se repetem, "
                          "então o histórico ajuda a escolher onde agir.")
    if len(cron_ds):
        topo = cron_ds.sort_values("pct", ascending=False).iloc[0]
        media = n_cronicas / n_total * 100 if n_total else np.nan
        partes.append(f"O DS {int(topo['ds'])} tem {fmt(topo['pct'], 1)}% das OVTs no topo com frequência, contra {fmt(media, 1)}% na cidade toda.")
    return " ".join(partes)


# ----------------------------------------------------------------------------- leituras fora do padrão
def efeito_das_atipicas(leituras: pd.DataFrame) -> pd.DataFrame:
    """Média de ovos por distrito com e sem as leituras marcadas como atípicas."""
    v = leituras.dropna(subset=["n_ovos"])
    com = v.groupby("ds")["n_ovos"].mean()
    sem = v[~v["atipica"]].groupby("ds")["n_ovos"].mean()
    out = pd.DataFrame({"com": com, "sem": sem})
    out["queda_pct"] = (out["sem"] / out["com"] - 1) * 100
    out["n_atipicas"] = v[v["atipica"]].groupby("ds").size().reindex(out.index).fillna(0).astype(int)
    return out.reset_index()


def ovts_mais_fora_do_padrao(leituras: pd.DataFrame, topo: int = 10) -> pd.DataFrame:
    """OVTs com mais leituras atípicas, com o maior múltiplo da mediana do bairro."""
    a = leituras[leituras["atipica"]]
    if a.empty:
        return pd.DataFrame(columns=["id_ovt", "bairro", "ds", "leituras_atipicas", "maior_multiplo", "maior_valor"])
    g = a.groupby("id_ovt")
    out = pd.DataFrame({"bairro": g["bairro"].first(), "ds": g["ds"].first(), "leituras_atipicas": g.size(),
                        "maior_multiplo": g["vezes_mediana"].max(), "maior_valor": g["n_ovos"].max()})
    return out.sort_values(["leituras_atipicas", "maior_multiplo"], ascending=False).head(topo).reset_index()


def texto_atipicas(leituras: pd.DataFrame, ef: pd.DataFrame) -> str:
    v = leituras.dropna(subset=["n_ovos"])
    n = int(v["atipica"].sum())
    if not len(v):
        return ""
    t = (f"{fmt(n)} leituras ({fmt(n / len(v) * 100, 1)}%) passam de 10 vezes a mediana do bairro no mesmo ciclo. É pouco, "
         "mas puxa a média para cima. ")
    if len(ef):
        pior = ef.loc[ef["queda_pct"].idxmin()]
        t += (f"Tirando essas leituras, a média do DS {int(pior['ds'])} cai {fmt(abs(pior['queda_pct']), 1)}%, a maior queda entre os distritos. "
              "Elas não foram removidas do painel porque podem ser foco real, o certo é a GEVACZ conferir.")
    return t


# ----------------------------------------------------------------------------- eficácia: intervalo e comparação no bairro
def intervalo_da_diferenca(por_ovt: pd.DataFrame, regiao: str = "ds", n_boot: int = 1000, seed: int = 7,
                           minimo: int = 5) -> pd.DataFrame:
    """Diferença % entre a mediana das OVTs perto e longe, por distrito, com intervalo de 95% (bootstrap).

    O bootstrap sorteia OVTs com reposição dentro de cada grupo. Se o intervalo cruza o zero, a diferença
    pode ser só acaso. Distritos com menos de `minimo` OVTs em um dos grupos ficam sem intervalo.
    """
    rng = np.random.default_rng(seed)
    linhas = []
    for r, parte in por_ovt.groupby(regiao):
        A = parte.loc[parte["proximidade"] == "perto", "ovos_media"].dropna().to_numpy(float)
        B = parte.loc[parte["proximidade"] == "longe", "ovos_media"].dropna().to_numpy(float)
        lin = {regiao: r, "n_perto": len(A), "n_longe": len(B), "dif_pct": np.nan, "ic_baixo": np.nan, "ic_alto": np.nan}
        if len(A) and len(B) and np.median(B):
            lin["dif_pct"] = (np.median(A) - np.median(B)) / np.median(B) * 100
        if len(A) >= minimo and len(B) >= minimo:
            ia = rng.integers(0, len(A), (n_boot, len(A)))
            ib = rng.integers(0, len(B), (n_boot, len(B)))
            ma, mb = np.median(A[ia], axis=1), np.median(B[ib], axis=1)
            d = (ma - mb) / np.where(mb == 0, np.nan, mb) * 100
            lin["ic_baixo"], lin["ic_alto"] = np.nanpercentile(d, [2.5, 97.5])
        linhas.append(lin)
    return pd.DataFrame(linhas)


def comparar_dentro_do_bairro(por_ovt: pd.DataFrame, minimo_por_grupo: int = 3) -> dict:
    """Perto × longe só entre OVTs do mesmo bairro.

    Se as EDLs foram para bairros que já eram mais infestados, a comparação geral engana. Aqui cada bairro
    é comparado com ele mesmo. Precisa da coluna `bairro` em `por_ovt`.
    """
    if "bairro" not in por_ovt:
        return {"n_bairros": 0}
    difs = []
    for b, parte in por_ovt.groupby("bairro"):
        A = parte.loc[parte["proximidade"] == "perto", "ovos_media"].dropna()
        B = parte.loc[parte["proximidade"] == "longe", "ovos_media"].dropna()
        if len(A) >= minimo_por_grupo and len(B) >= minimo_por_grupo and B.median():
            difs.append((A.median() - B.median()) / B.median() * 100)
    res = {"n_bairros": len(difs), "mediana_dif_pct": np.nan, "pct_menor": np.nan, "p_valor": np.nan}
    if len(difs) >= 5:
        d = np.array(difs)
        res["mediana_dif_pct"] = float(np.median(d))
        res["pct_menor"] = float((d < 0).mean() * 100)
        if np.any(d != 0):
            res["p_valor"] = float(wilcoxon(d).pvalue)
    return res


def texto_eficacia(geral: dict, bairro: dict, intervalos: pd.DataFrame, raio: int, frase_teste: str = "", frase_dist: str = "") -> str:
    """Leitura das duas comparações, sempre dizendo que é associação."""
    partes = []
    if geral["n_a"] and geral["n_b"] and geral["diferenca_pct"] == geral["diferenca_pct"]:
        sentido = "menos" if geral["diferenca_pct"] < 0 else "mais"
        partes.append(f"Na cidade toda, as OVTs a até {raio} m de uma EDL têm {fmt(abs(geral['diferenca_pct']), 1)}% {sentido} ovos (mediana) do que as mais longe.")
        if frase_teste:
            partes.append(frase_teste)
    if bairro.get("n_bairros", 0) >= 5 and bairro["mediana_dif_pct"] == bairro["mediana_dif_pct"]:
        s = "menos" if bairro["mediana_dif_pct"] < 0 else "mais"
        partes.append(f"Comparando só dentro do mesmo bairro ({fmt(bairro['n_bairros'])} bairros), a diferença típica é de {fmt(abs(bairro['mediana_dif_pct']), 1)}% {s} perto da EDL, "
                      f"e {fmt(bairro['pct_menor'], 0)}% dos bairros têm menos ovos perto.")
    ic = intervalos.dropna(subset=["ic_baixo"])
    if len(ic):
        claro = ic[(ic["ic_baixo"] > 0) | (ic["ic_alto"] < 0)]
        if len(claro):
            nomes = ", ".join(f"DS {int(d)}" for d in claro["ds"])
            partes.append(f"Só em {len(claro)} de {len(ic)} distritos ({nomes}) o intervalo de confiança deixa de cruzar o zero; nos demais a diferença pode ser acaso.")
        else:
            partes.append(f"Em nenhum dos {len(ic)} distritos o intervalo de confiança deixa de cruzar o zero, então nada ali se distingue de acaso.")
    if frase_dist:
        partes.append(frase_dist)
    partes.append("Isso é associação, não efeito da EDL: ela pode ter sido instalada onde já havia mais (ou menos) ovos.")
    return " ".join(partes)


# ----------------------------------------------------------------------------- leitura rápida da visão geral
def pontos_leitura_rapida(ind: pd.DataFrame, v: pd.DataFrame, minimo_bairro: int = 100) -> list[str]:
    """Três ou quatro frases que resumem a tela inicial. `ind` vem de indicadores_por_regiao; `v` são as leituras com contagem."""
    out = []
    d = ind.dropna(subset=["ovos_por_ovt"])
    if len(d) >= 2:
        a, b = d["ovos_por_ovt"].idxmax(), d["ovos_por_ovt"].idxmin()
        out.append(f"Por OVT, o DS {int(a)} é o mais alto ({fmt(d.loc[a, 'ovos_por_ovt'], 1)} ovos) e o DS {int(b)} o mais baixo "
                   f"({fmt(d.loc[b, 'ovos_por_ovt'], 1)}), {fmt((d.loc[a, 'ovos_por_ovt'] / d.loc[b, 'ovos_por_ovt'] - 1) * 100, 0)}% de diferença.")
    k = ind.dropna(subset=["ovos_por_km2"]) if "ovos_por_km2" in ind else pd.DataFrame()
    if len(k) >= 2:
        t = k["ovos_por_km2"].idxmax()
        out.append(f"Por área, o DS {int(t)} lidera com {fmt(k.loc[t, 'ovos_por_km2'], 0)} ovos por km² a cada ciclo, "
                   f"e tem {fmt(k.loc[t, 'ovts_por_km2'], 1)} OVTs por km².")
    if len(v):
        rk = v.groupby("bairro")["n_ovos"].agg(["size", "mean"]).query("size >= @minimo_bairro").sort_values("mean", ascending=False)
        if len(rk):
            b = rk.index[0]
            out.append(f"O bairro com mais ovos por OVT é {str(b).title()} ({fmt(rk.loc[b, 'mean'], 1)}), "
                       f"contra {fmt(v['n_ovos'].mean(), 1)} na cidade.")
        if "atipica" in v and v["atipica"].any():
            out.append(f"{fmt(int(v['atipica'].sum()))} leituras passam de 10 vezes a mediana do bairro e merecem conferência antes de virar alarme.")
    return out


def tendencia_com_distancia(por_ovt: pd.DataFrame) -> dict:
    """Correlação de postos entre a distância até a EDL mais próxima e os ovos da OVT.

    Negativa: quanto mais longe da EDL, menos ovos. Se a EDL funcionasse sozinha, o esperado seria o contrário.
    """
    d = por_ovt.dropna(subset=["dist_edl_m", "ovos_media"])
    if len(d) < 30:
        return {"n": len(d), "rho": np.nan, "p_valor": np.nan}
    r = spearmanr(d["dist_edl_m"], d["ovos_media"])
    return {"n": len(d), "rho": float(r.statistic), "p_valor": float(r.pvalue)}


def frase_tendencia(t: dict) -> str:
    if not t or t["rho"] != t["rho"] or t["p_valor"] >= 0.05:
        return ""
    forca = "fraca" if abs(t["rho"]) < 0.3 else "moderada"
    if t["rho"] < 0:
        return (f"Olhando a distância como número, e não só perto ou longe, os ovos caem um pouco quando a OVT se afasta da EDL "
                f"(relação {forca}, correlação de {fmt(t['rho'], 2)}). Se a EDL funcionasse sozinha, o esperado era o contrário. "
                "O mais provável é que as EDLs estejam em áreas que já eram mais infestadas.")
    return (f"Olhando a distância como número, os ovos sobem um pouco quando a OVT se afasta da EDL (relação {forca}, correlação de {fmt(t['rho'], 2)}). "
            "Combina com a EDL ajudando, mas isso sozinho não prova nada.")
