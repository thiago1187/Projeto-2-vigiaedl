"""Junta leituras, OVTs e EDLs para o cruzamento OVT × EDL. O app e o script usam as mesmas funções."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import espacial as e


def classificar_ovts(ovt: pd.DataFrame, edl: pd.DataFrame, raio_m: float = e.RAIO_PADRAO_M,
                     instalacao: pd.DataFrame | None = None) -> pd.DataFrame:
    """Acrescenta à tabela de OVTs: distância e identificação da EDL mais próxima, e perto/longe.

    Com `instalacao` (edl_id, data_instalacao), acrescenta também `corte_edl`: a data de instalação mais
    antiga entre as EDLs dentro do raio da OVT, que é o "antes/depois" próprio daquela OVT.
    """
    d, pos = e.distancia_ate_edl_mais_proxima(ovt["x"], ovt["y"], edl["x"], edl["y"])
    out = ovt.copy()
    out["dist_edl_m"] = d
    out["edl_mais_proxima"] = np.where(pos >= 0, edl["edl_id"].to_numpy()[np.clip(pos, 0, None)], None)
    out["tipo_edl_mais_proxima"] = np.where(pos >= 0, edl["tipo"].to_numpy()[np.clip(pos, 0, None)], None)
    out["proximidade"] = e.classificar_proximidade(d, raio_m)
    if instalacao is not None and len(instalacao):
        datas_edl = edl["edl_id"].map(instalacao.drop_duplicates("edl_id").set_index("edl_id")["data_instalacao"])
        out["corte_edl"], out["n_edl_no_raio"] = e.primeira_instalacao_no_raio(
            ovt["x"], ovt["y"], edl["x"], edl["y"], datas_edl, raio_m)
    return out


def filtrar_leituras(leituras: pd.DataFrame, anos=None, ciclos=None, ds=None, desde=None, ate=None) -> pd.DataFrame:
    """Recorte por ano, intervalo de ciclos, distrito e datas. Ciclo só faz sentido junto de um ano."""
    m = pd.Series(True, index=leituras.index)
    if anos is not None:
        m &= leituras["ano"].isin(list(np.atleast_1d(anos)))
    if ciclos is not None:
        m &= leituras["ciclo"].between(ciclos[0], ciclos[1])
    if ds is not None and ds != "todos":
        m &= leituras["ds"] == ds
    if desde is not None:
        m &= leituras["data"] >= pd.Timestamp(desde)
    if ate is not None:
        m &= leituras["data"] <= pd.Timestamp(ate)
    return leituras[m]


def base_por_ovt(leituras: pd.DataFrame, ovt_class: pd.DataFrame, corte=None,
                 apenas_validadas: bool = False, minimo_leituras: int = 3,
                 corte_proprio: bool = False) -> pd.DataFrame:
    """Uma linha por OVT (e por período, se houver corte) com a média de ovos e o grupo perto/longe.

    OVTs sem coordenada ficam de fora (não dá para dizer se estão perto ou longe).
    `apenas_validadas` tira as OVTs cuja posição foi reconstruída e não pôde ser conferida.
    `corte_proprio`: as OVTs "perto" usam a data de instalação da própria EDL (coluna `corte_edl`);
    as "longe" (e as perto sem data conhecida) usam `corte`.
    """
    cols = ["id_ovt", "proximidade", "dist_edl_m", "posicao_validada"]
    if corte_proprio and "corte_edl" in ovt_class:
        cols.append("corte_edl")
    l = leituras.merge(ovt_class[cols], on="id_ovt", how="inner")
    l = l[l["proximidade"].isin(["perto", "longe"])]
    if apenas_validadas:
        l = l[l["posicao_validada"]]
    if corte_proprio and "corte_edl" in l:
        c = l["corte_edl"].where(l["proximidade"] == "perto").fillna(pd.Timestamp(corte) if corte is not None else pd.NaT)
        d = pd.to_datetime(l["data"], errors="coerce")
        periodo = np.where(d.isna(), "sem_data", np.where(c.isna(), "sem_corte", np.where(d < c, "antes", "depois")))
        l = l.assign(periodo=periodo)
    else:
        l = l.assign(periodo=e.classificar_periodo(l["data"], corte).to_numpy())
    por_ovt = e.resumo_por_ovt(l, extra=["ds", "proximidade", "dist_edl_m", "periodo"])
    if "bairro" in leituras:
        por_ovt["bairro"] = por_ovt["id_ovt"].map(leituras.dropna(subset=["bairro"]).groupby("id_ovt")["bairro"].last())
    return por_ovt[por_ovt["n_leituras"] >= minimo_leituras].reset_index(drop=True)


FAIXAS_M = (0, 100, 200, 300, 500, np.inf)


def por_faixa_de_distancia(por_ovt: pd.DataFrame, faixas=FAIXAS_M) -> pd.DataFrame:
    """Ovos por OVT conforme a distância até a EDL mais próxima.

    Se a EDL reduz os ovos, a mediana deveria subir conforme a OVT se afasta
    dela. Cada faixa inclui o limite de cima (100 m entra em "até 100 m"), igual à regra de "perto".
    """
    rotulos = [f"até {int(faixas[1])} m"] + [
        (f"{int(a)}–{int(b)} m" if np.isfinite(b) else f"mais de {int(a)} m")
        for a, b in zip(faixas[1:-1], faixas[2:])]
    faixa = pd.cut(por_ovt["dist_edl_m"], bins=list(faixas), labels=rotulos, right=True, include_lowest=True)
    return (por_ovt.groupby(faixa, observed=False)["ovos_media"]
            .agg(n_ovts="size", mediana="median", media="mean").reset_index()
            .rename(columns={"dist_edl_m": "faixa"}))


def eficacia(por_ovt: pd.DataFrame) -> dict:
    """Resultado das comparações: geral, por distrito e (com corte) antes × depois."""
    geral = e.comparar_grupos(por_ovt)
    por_ds = e.comparar_por_regiao(por_ovt)
    tem_periodo = set(por_ovt["periodo"].unique()) >= {"antes", "depois"}
    did = e.diferenca_em_diferencas(por_ovt[por_ovt["periodo"].isin(["antes", "depois"])]) if tem_periodo else None
    return {"geral": geral, "por_ds": por_ds, "did": did}


def evolucao_por_ciclo(leituras: pd.DataFrame, por: str = "ds") -> pd.DataFrame:
    """Média e mediana de ovos por OVT em cada ciclo, por distrito (ou geral, com por=None)."""
    v = leituras.dropna(subset=["n_ovos"])
    chaves = ["ano", "ciclo"] + ([por] if por else [])
    return (v.groupby(chaves)["n_ovos"].agg(media="mean", mediana="median", n="size").reset_index())


def prazos_do_dado(leituras: pd.DataFrame) -> pd.DataFrame:
    """Dias entre a coleta e cada etapa (leitura no laboratório, digitação, envio ao DS)."""
    l = leituras.dropna(subset=["data"])
    out = pd.DataFrame({
        "leitura_lab": (l["dt_leitura"] - l["data"]).dt.days,
        "digitacao": (l["dt_dig"] - l["data"]).dt.days,
        "envio_ds": (l["dt_envio_ds"] - l["data"]).dt.days,
    })
    return out[(out >= 0).all(axis=1) | out.notna().any(axis=1)]
