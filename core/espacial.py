"""Regras espaciais e temporais da análise OVT × EDL.

Tudo aqui trabalha com coordenadas em METROS (EPSG:31985). Em graus, "200 m"
não significa nada e o resultado sai errado sem avisar.

Regras (ver README):
  * perto × longe: distância da OVT até a EDL mais próxima. "Perto" é até o raio,
    inclusive (distância <= raio). O padrão é 200 m e pode ser trocado.
  * antes × depois: a leitura é "depois" se a data de coleta for igual ou
    posterior ao corte. O corte padrão é a menor data de instalação das EDLs,
    quando a GEVACZ informar. Sem data, não há corte e a comparação fica desligada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

RAIO_PADRAO_M = 200.0
MIN_OVTS_COMPARACAO = 5      # abaixo disso o teste estatístico não é calculado
FATOR_ATIPICO = 10.0         # leitura >= 10 vezes a mediana do bairro no mesmo ciclo


# --------------------------------------------------------------------------
# Perto × longe
# --------------------------------------------------------------------------
def distancia_ate_edl_mais_proxima(x, y, edl_x, edl_y):
    """Distância (m) de cada OVT até a EDL mais próxima, e a posição dessa EDL.

    Devolve (distancias, posicoes). Sem coordenada na OVT, a distância é NaN e a
    posição é -1. Sem nenhuma EDL válida, tudo fica NaN / -1.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    edl_x, edl_y = np.asarray(edl_x, float), np.asarray(edl_y, float)
    dist = np.full(x.shape, np.nan)
    pos = np.full(x.shape, -1, dtype=int)

    edl_ok = np.flatnonzero(np.isfinite(edl_x) & np.isfinite(edl_y))
    ovt_ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if len(edl_ok) == 0 or len(ovt_ok) == 0:
        return dist, pos
    arvore = cKDTree(np.column_stack([edl_x[edl_ok], edl_y[edl_ok]]))
    d, i = arvore.query(np.column_stack([x[ovt_ok], y[ovt_ok]]))
    dist[ovt_ok] = d
    pos[ovt_ok] = edl_ok[i]
    return dist, pos


def classificar_proximidade(dist_m, raio_m: float = RAIO_PADRAO_M) -> np.ndarray:
    """'perto' se a distância for <= raio, 'longe' se maior, 'sem_posicao' se não houver distância."""
    d = np.asarray(dist_m, float)
    return np.where(np.isnan(d), "sem_posicao", np.where(d <= raio_m, "perto", "longe"))


def primeira_instalacao_no_raio(x, y, edl_x, edl_y, edl_datas, raio_m: float = RAIO_PADRAO_M):
    """Para cada OVT, a data mais antiga de instalação entre as EDLs dentro do raio.

    Devolve (datas, quantidade_de_edl_no_raio). Se nenhuma EDL do raio tem data
    conhecida, a data fica NaT. Serve para quando a GEVACZ mandar a data de
    cada EDL: a OVT passa a ter o seu próprio "antes" e "depois".
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    edl_x, edl_y = np.asarray(edl_x, float), np.asarray(edl_y, float)
    datas_edl = pd.to_datetime(pd.Series(edl_datas), errors="coerce").to_numpy()
    saida = np.full(x.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    contagem = np.zeros(x.shape, dtype=int)

    edl_ok = np.flatnonzero(np.isfinite(edl_x) & np.isfinite(edl_y))
    ovt_ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if len(edl_ok) == 0 or len(ovt_ok) == 0:
        return saida, contagem
    arvore = cKDTree(np.column_stack([edl_x[edl_ok], edl_y[edl_ok]]))
    vizinhos = arvore.query_ball_point(np.column_stack([x[ovt_ok], y[ovt_ok]]), r=raio_m)
    for k, lista in zip(ovt_ok, vizinhos):
        contagem[k] = len(lista)
        if lista:
            d = datas_edl[edl_ok[lista]]
            d = d[~np.isnat(d)]
            if len(d):
                saida[k] = d.min()
    return saida, contagem


# --------------------------------------------------------------------------
# Antes × depois
# --------------------------------------------------------------------------
def data_corte_padrao(datas_instalacao) -> pd.Timestamp | None:
    """A menor data de instalação conhecida (regra do Plano), ou None se não houver nenhuma."""
    d = pd.to_datetime(pd.Series(datas_instalacao), errors="coerce").dropna()
    return d.min() if len(d) else None


def classificar_periodo(datas, corte) -> pd.Series:
    """'antes' se a data for anterior ao corte, 'depois' se for igual ou posterior.

    Sem corte (None), tudo vira 'sem_corte'. Data ausente vira 'sem_data'.
    """
    d = pd.Series(pd.to_datetime(datas, errors="coerce"))
    if corte is None or pd.isna(corte):
        return pd.Series("sem_corte", index=d.index)
    corte = pd.Timestamp(corte)
    return pd.Series(np.where(d.isna(), "sem_data", np.where(d < corte, "antes", "depois")), index=d.index)


# --------------------------------------------------------------------------
# Resumos e comparações
# --------------------------------------------------------------------------
def resumo_por_ovt(leituras: pd.DataFrame, extra=()) -> pd.DataFrame:
    """Uma linha por OVT (e por grupo extra, como o período), usando só leituras com contagem.

    Tratar cada OVT uma vez evita contar a mesma armadilha várias vezes na
    comparação: uma OVT com 12 ciclos não pesa 12 vezes mais que uma com 1.
    """
    v = leituras.dropna(subset=["n_ovos"])
    chaves = ["id_ovt", *extra]
    out = v.groupby(chaves)["n_ovos"].agg(
        n_leituras="size", ovos_media="mean", ovos_mediana="median", ovos_max="max").reset_index()
    return out


def comparar_grupos(df: pd.DataFrame, grupo: str = "proximidade", valor: str = "ovos_media",
                    a: str = "perto", b: str = "longe", minimo: int = MIN_OVTS_COMPARACAO) -> dict:
    """Compara dois grupos de OVTs. Mostra medianas e, com OVTs suficientes, o teste de Mann-Whitney.

    O teste diz se a diferença é maior do que o acaso explicaria. Ele não diz
    que a EDL causou a diferença: as EDLs podem ter sido colocadas onde já havia mais ovos.
    """
    A = df.loc[df[grupo] == a, valor].dropna().to_numpy(float)
    B = df.loc[df[grupo] == b, valor].dropna().to_numpy(float)
    res = {
        "n_a": len(A), "n_b": len(B),
        "mediana_a": float(np.median(A)) if len(A) else np.nan,
        "mediana_b": float(np.median(B)) if len(B) else np.nan,
        "media_a": float(A.mean()) if len(A) else np.nan,
        "media_b": float(B.mean()) if len(B) else np.nan,
        "p_valor": np.nan, "efeito": np.nan, "aviso": None,
    }
    mb = res["mediana_b"]
    res["diferenca_pct"] = ((res["mediana_a"] - mb) / mb * 100) if (mb and not np.isnan(mb)) else np.nan
    if len(A) < minimo or len(B) < minimo:
        res["aviso"] = f"Poucas OVTs em um dos grupos (mínimo {minimo}); sem teste estatístico."
        return res
    u, p = mannwhitneyu(A, B, alternative="two-sided")
    res["p_valor"] = float(p)
    res["efeito"] = float(2 * u / (len(A) * len(B)) - 1)   # -1 a 1; 0 = sem diferença
    return res


def comparar_por_regiao(df: pd.DataFrame, regiao: str = "ds", **kw) -> pd.DataFrame:
    """A mesma comparação, uma linha por distrito sanitário."""
    linhas = []
    for r, parte in df.groupby(regiao):
        linhas.append({regiao: r, **comparar_grupos(parte, **kw)})
    return pd.DataFrame(linhas)


def diferenca_em_diferencas(df: pd.DataFrame, grupo: str = "proximidade", periodo: str = "periodo",
                            valor: str = "ovos_media") -> dict:
    """(perto_depois − perto_antes) − (longe_depois − longe_antes), usando medianas.

    Tira da conta o que mudou para todo mundo (chuva, estação do ano). Só faz
    sentido com um corte de data válido; sem as quatro células, devolve NaN.
    """
    med = df.groupby([grupo, periodo])[valor].median()
    try:
        pa, pd_, la, ld = (med[("perto", "antes")], med[("perto", "depois")],
                           med[("longe", "antes")], med[("longe", "depois")])
    except KeyError:
        return {"did": np.nan, "perto_antes": np.nan, "perto_depois": np.nan,
                "longe_antes": np.nan, "longe_depois": np.nan}
    return {"did": float((pd_ - pa) - (ld - la)), "perto_antes": float(pa), "perto_depois": float(pd_),
            "longe_antes": float(la), "longe_depois": float(ld)}


def indicadores_por_regiao(leituras: pd.DataFrame, areas_km2: pd.Series | None = None,
                           regiao: str = "ds") -> pd.DataFrame:
    """Indicadores por região (distrito sanitário por padrão).

    ovos_por_ovt : média de ovos nas leituras com contagem.
    ido          : média só nas OVTs positivas (é o que a GEVACZ chama de IDO).
    ipo          : % de leituras positivas.
    ovos_por_km2 : ovos capturados por km², por ciclo. Sobe também quando há mais
                   OVTs na região, por isso aparece junto de ovts_por_km2.
    """
    v = leituras.dropna(subset=["n_ovos"])
    g = v.groupby(regiao)
    res = pd.DataFrame({
        "n_ovts": g["id_ovt"].nunique(),
        "n_leituras": g.size(),
        "ovos_por_ovt": g["n_ovos"].mean(),
        "mediana": g["n_ovos"].median(),
        "ipo": g["n_ovos"].apply(lambda s: float((s > 0).mean() * 100)),
    })
    res["ido"] = v[v["n_ovos"] > 0].groupby(regiao)["n_ovos"].mean()
    if areas_km2 is not None:
        n_ciclos = v.groupby(regiao).apply(lambda d: len(d[["ano", "ciclo"]].drop_duplicates()), include_groups=False)
        total = g["n_ovos"].sum()
        a = areas_km2.reindex(res.index)
        res["area_km2"] = a
        res["ovts_por_km2"] = res["n_ovts"] / a
        res["ovos_por_km2"] = total / n_ciclos / a
    return res


def sinalizar_atipicos(leituras: pd.DataFrame, fator: float = FATOR_ATIPICO,
                       minimo_no_grupo: int = 5) -> pd.DataFrame:
    """Marca leituras muito acima da mediana do bairro no mesmo ciclo.

    Uma leitura 10 vezes maior que a mediana pode ser foco de verdade ou erro de
    digitação. A marca só pede conferência, não apaga nada.
    """
    df = leituras.copy()
    grupo = df.dropna(subset=["n_ovos"]).groupby(["ano", "ciclo", "bairro"])["n_ovos"]
    mediana = grupo.transform("median")
    tamanho = grupo.transform("size")
    df["mediana_bairro"] = mediana.reindex(df.index)
    df["n_no_grupo"] = tamanho.reindex(df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["vezes_mediana"] = df["n_ovos"] / df["mediana_bairro"].where(df["mediana_bairro"] > 0)
    df["atipica"] = (df["vezes_mediana"] >= fator) & (df["n_no_grupo"] >= minimo_no_grupo)
    return df


# --------------------------------------------------------------------------
# Kernel Density
# --------------------------------------------------------------------------
@dataclass
class Grade:
    """Grade regular em metros. valores[i, j] é a célula da linha i (sul→norte) e coluna j (oeste→leste)."""
    valores: np.ndarray
    x0: float
    y0: float
    celula_m: float

    @property
    def xs(self) -> np.ndarray:
        return self.x0 + (np.arange(self.valores.shape[1]) + 0.5) * self.celula_m

    @property
    def ys(self) -> np.ndarray:
        return self.y0 + (np.arange(self.valores.shape[0]) + 0.5) * self.celula_m


def kde_grade(x, y, pesos=None, banda_m: float = 300.0, celula_m: float = 100.0,
              modo: str = "soma", suporte_minimo: float = 0.05) -> Grade:
    """Kernel Density gaussiano das OVTs, ponderado pelo número de ovos.

    modo='soma'  : densidade de ovos, em ovos por km² (suavizada). Mostra onde os
                   ovos se concentram, mas também acende onde há mais OVTs.
    modo='media' : média de ovos por OVT ao redor de cada ponto, ponderada pela
                   distância. Não depende de quantas OVTs há no lugar.

    banda_m é o raio de suavização (desvio padrão do kernel). Células longe de
    qualquer OVT ficam NaN, para o mapa não inventar valor onde não há armadilha.
    `suporte_minimo` é a fração do kernel de uma OVT isolada que ainda conta como
    "perto de uma OVT": 0,05 corta a cerca de 2,4 bandas de distância.
    """
    if modo not in ("soma", "media"):
        raise ValueError("modo deve ser 'soma' ou 'media'")
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if pesos is None else np.asarray(pesos, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) == 0:
        raise ValueError("Sem pontos válidos para o Kernel Density.")

    margem = 3 * banda_m
    x0, y0 = x.min() - margem, y.min() - margem
    nx = int(np.ceil((x.max() + margem - x0) / celula_m))
    ny = int(np.ceil((y.max() + margem - y0) / celula_m))
    j = np.clip(((x - x0) / celula_m).astype(int), 0, nx - 1)
    i = np.clip(((y - y0) / celula_m).astype(int), 0, ny - 1)

    soma = np.zeros((ny, nx))
    cont = np.zeros((ny, nx))
    np.add.at(soma, (i, j), w)
    np.add.at(cont, (i, j), 1.0)
    sigma = banda_m / celula_m
    s = gaussian_filter(soma, sigma, mode="constant", truncate=4.0)
    c = gaussian_filter(cont, sigma, mode="constant", truncate=4.0)

    if modo == "soma":
        valores = s / (celula_m ** 2) * 1e6            # ovos por km²
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            valores = np.where(c > 0, s / c, np.nan)   # ovos por OVT
    pico_de_um_ponto = 1.0 / (2.0 * np.pi * sigma ** 2)      # c no centro de uma OVT isolada
    valores = np.where(c / pico_de_um_ponto >= suporte_minimo, valores, np.nan)
    return Grade(valores=valores, x0=x0, y0=y0, celula_m=celula_m)


def kde_nos_pontos(grade: Grade, x, y) -> np.ndarray:
    """Valor da grade em cada ponto (interpolação linear). Fora da grade ou sem suporte: NaN."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((grade.ys, grade.xs), grade.valores, bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([np.asarray(y, float), np.asarray(x, float)]))


def limiar_hotspot(grade: Grade, percentil: float = 90.0) -> float:
    """Valor acima do qual a célula conta como hotspot (percentil das células com suporte)."""
    v = grade.valores[np.isfinite(grade.valores)]
    return float(np.percentile(v, percentil)) if len(v) else np.nan


def grade_para_latlon(grade: Grade, passo_graus: float = 0.0008, crs_metros: int = 31985):
    """Reamostra a grade em metros para uma grade em graus, pronta para virar imagem no mapa.

    Devolve (imagem, (lat_min, lon_min, lat_max, lon_max)). A imagem tem a linha 0 ao norte.
    """
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator
    para_graus = Transformer.from_crs(crs_metros, 4326, always_xy=True)
    para_metros = Transformer.from_crs(4326, crs_metros, always_xy=True)
    xs, ys = grade.xs, grade.ys
    lons_c, lats_c = para_graus.transform([xs[0], xs[-1], xs[0], xs[-1]], [ys[0], ys[0], ys[-1], ys[-1]])
    lon_min, lon_max, lat_min, lat_max = min(lons_c), max(lons_c), min(lats_c), max(lats_c)
    lons = np.arange(lon_min, lon_max, passo_graus)
    lats = np.arange(lat_min, lat_max, passo_graus)
    LON, LAT = np.meshgrid(lons, lats)
    X, Y = para_metros.transform(LON, LAT)
    f = RegularGridInterpolator((ys, xs), grade.valores, bounds_error=False, fill_value=np.nan)
    img = f(np.column_stack([Y.ravel(), X.ravel()])).reshape(LON.shape)
    return img[::-1], (lat_min, lon_min, lat_max, lon_max)


def poligonos_de_nivel(grade: Grade, inferior: float, superior: float, crs_metros: int = 31985):
    """Região da grade com valor entre `inferior` e `superior`, como polígono em lat/lon (EPSG:4326).

    Usa curvas de nível (contourpy), então a borda sai lisa e vetorial, sem os quadradinhos de uma imagem.
    Células sem suporte (NaN) ficam de fora. Devolve uma geometria do shapely (possivelmente vazia).
    """
    from contourpy import FillType, contour_generator
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform, unary_union

    z = np.ma.masked_invalid(grade.valores)
    if z.mask.all():
        return Polygon()
    gen = contour_generator(x=grade.xs, y=grade.ys, z=z, fill_type=FillType.ChunkCombinedOffsetOffset)
    inf_, sup_ = (-1e18 if not np.isfinite(inferior) else inferior), (1e18 if not np.isfinite(superior) else superior)
    pts, offs, outer = gen.filled(inf_, sup_)
    polys = []
    for P, O, OO in zip(pts, offs, outer):
        if P is None:
            continue
        aneis = [P[O[i]:O[i + 1]] for i in range(len(O) - 1)]
        for a, b in zip(OO[:-1], OO[1:]):
            r = aneis[a:b]
            if len(r[0]) >= 4:
                pol = Polygon(r[0], [h for h in r[1:] if len(h) >= 4])
                polys.append(pol if pol.is_valid else pol.buffer(0))
    if not polys:
        return Polygon()
    geom = unary_union(polys)
    para_graus = Transformer.from_crs(crs_metros, 4326, always_xy=True)
    return transform(para_graus.transform, geom)
