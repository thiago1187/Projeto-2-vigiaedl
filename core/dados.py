"""Leitura e limpeza dos dados enviados pela GEVACZ.

Entrada: a pasta "INFORMAÇÕES - GEVACZ" (shapefiles de OVT e EDL, planilhas de
ovos de 2024/2025/2026, polígonos dos distritos sanitários).
Saída: tabelas limpas, em CSV/GeoJSON, que o app lê (ver preparar_tudo).

Problemas reais encontrados nos dados e tratados aqui (ver README):
  * Os pontos das EDLs declaram o CRS EPSG:32000, mas o .prj diz SIRGAS UTM 25S.
  * No DS II, os shapefiles de OVT perderam o ID (Name = "0").
  * IDs repetidos entre shapefiles (DS_VII.shp repete os bairros do DS VII).
  * Coordenadas sujas na planilha de georreferenciamento (sem sinal, com espaço).
"""
from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

CRS_GEO = 4326
CRS_METROS = 31985  # SIRGAS 2000 / UTM 25S, em metros. Serve para o Recife.

ROMANO = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
PADRAO_ID = re.compile(r"^[A-Z]{2,5}\d{3,5}$")

# Limites aproximados do Recife, só para detectar coordenada fora do lugar.
LAT_MIN, LAT_MAX = -8.20, -7.90
LON_MIN, LON_MAX = -35.10, -34.80

COLUNAS_LEITURA = [
    "ano", "ciclo", "ds", "bairro", "id_ovt", "qt", "data", "n_ovos", "obs", "obs_cod",
    "dt_entrega_apoio", "dt_entrega_lab", "dt_leitura", "dt_dig", "dt_envio_ds",
]


# --------------------------------------------------------------------------
# Utilidades de texto
# --------------------------------------------------------------------------
def slug(texto) -> str:
    """'ÁGUA FRIA' -> 'agua_fria'. Serve para casar nomes de bairro e de arquivo."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def normalizar_id(texto) -> str:
    """'REC 1001', 'rec/1001' e 'REC-1001' viram 'REC1001'."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    return re.sub(r"[\s/\-_.]+", "", str(texto)).upper()


def ds_para_int(valor) -> int | None:
    """'VI', 'DS 6', 'DS-VI', 6 e '6' viram 6."""
    s = re.sub(r"[\s\-_.]", "", str(valor).upper()).replace("DS", "")
    if s.isdigit():
        return int(s)
    return ROMANO.get(s)


def _achar(base: str | Path, padrao: str) -> str:
    achados = sorted(glob.glob(os.path.join(str(base), padrao)))
    achados = [a for a in achados if "~$" not in os.path.basename(a)]
    if not achados:
        raise FileNotFoundError(f"Nada em {base!s} para o padrão {padrao!r}")
    return achados[0]


def achar_pasta_base(origem: str | Path) -> Path:
    """Aceita a pasta 'INFORMAÇÕES - GEVACZ' ou a pasta que a contém."""
    origem = Path(origem)
    if (origem / "BASE Distritos sanitários shapefile").exists() or glob.glob(str(origem / "BASE Dis*")):
        return origem
    for sub in origem.iterdir():
        if sub.is_dir() and glob.glob(str(sub / "BASE Dis*")):
            return sub
    raise FileNotFoundError(f"Não achei a pasta de dados da GEVACZ em {origem!s}")


def _ler_shp(caminho: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(caminho)
    if g.crs is None:
        g = g.set_crs(CRS_GEO)
    return g.to_crs(CRS_GEO)


def _com_metros(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Acrescenta lat, lon (graus) e x, y (metros, EPSG:31985)."""
    g = g.to_crs(CRS_GEO).copy()
    g["lon"], g["lat"] = g.geometry.x, g.geometry.y
    m = g.to_crs(CRS_METROS)
    g["x"], g["y"] = m.geometry.x.values, m.geometry.y.values
    return g


# --------------------------------------------------------------------------
# Leituras de ovos
# --------------------------------------------------------------------------
def _codigo_obs(obs) -> str | float:
    """F, E, R, D ou REC (recuperada, segundo a legenda das planilhas)."""
    if obs is None or (isinstance(obs, float) and np.isnan(obs)):
        return np.nan
    s = str(obs).strip().upper()
    if s.startswith("REC"):
        return "REC"
    return s[0] if s and s[0] in "FERD" else np.nan


def ler_ovos_2026(base: str | Path) -> pd.DataFrame:
    """Planilha geral de 2026: uma linha por OVT por ciclo (já em formato longo)."""
    df = pd.read_excel(_achar(base, "OVITRAMPAS 2026/*.xlsx"), sheet_name="CICLO 1")
    df.columns = [str(c).split("\n")[0].strip() for c in df.columns]
    out = pd.DataFrame({
        "ano": pd.to_numeric(df["Ano"], errors="coerce"),
        "ciclo": pd.to_numeric(df["CICLOS"], errors="coerce"),
        "ds": df["DS"].map(ds_para_int),
        "bairro": df["BAIRRO"].astype(str).str.strip(),
        "id_ovt": df["ID_OVT"].map(normalizar_id),
        "qt": pd.to_numeric(df["QT"], errors="coerce"),
        "data": pd.to_datetime(df["DT_COLETA"], errors="coerce"),
        "n_ovos": pd.to_numeric(df["N_OVOS"], errors="coerce"),
        "obs": df["OBS"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan}),
        "dt_entrega_apoio": pd.to_datetime(df["DT_ENTREGA_APOIO"], errors="coerce"),
        "dt_entrega_lab": pd.to_datetime(df["DT_ENTREGA_LAB"], errors="coerce"),
        "dt_leitura": pd.to_datetime(df["DT_LEITURA"], errors="coerce"),
        "dt_dig": pd.to_datetime(df["DT_DIG"], errors="coerce"),
        "dt_envio_ds": pd.to_datetime(df["DT_ENVIO_DS"], errors="coerce"),
    })
    out["obs_cod"] = out["obs"].map(_codigo_obs)
    out = out[out["id_ovt"].str.match(PADRAO_ID)].copy()
    out.loc[out["n_ovos"] < 0, "n_ovos"] = np.nan
    return out[COLUNAS_LEITURA].reset_index(drop=True)


def _norm_cel(v) -> str:
    return "" if pd.isna(v) else re.sub(r"[^A-Z0-9]", "", str(v).upper())


def parse_aba_historica(bruto: pd.DataFrame, ano: int, ds: int, aba: str) -> pd.DataFrame:
    """Lê uma aba de bairro das planilhas de 2024/2025.

    Layout: uma linha por OVT; a partir de uma coluna, cada ciclo ocupa três
    colunas (DATA, NÚMERO OVOS, OBS). Acha o cabeçalho procurando "ID. OVT".
    """
    vazio = pd.DataFrame(columns=COLUNAS_LEITURA)
    linha_cab = None
    for i in range(min(30, len(bruto))):
        if any(_norm_cel(v).startswith("IDOVT") for v in bruto.iloc[i]):
            linha_cab = i
            break
    if linha_cab is None:
        return vazio
    cab = bruto.iloc[linha_cab]
    col_id = next(j for j, v in enumerate(cab) if _norm_cel(v).startswith("IDOVT"))
    col_qt = next((j for j, v in enumerate(cab) if _norm_cel(v) == "QT"), None)
    rotulos = bruto.iloc[linha_cab - 1] if linha_cab > 0 else pd.Series(dtype=object)

    ciclos = []  # (numero, col_data, col_ovos, col_obs)
    for j, v in enumerate(cab):
        if _norm_cel(v) == "DATA" and j + 1 < len(cab) and "OVOS" in _norm_cel(cab.iloc[j + 1]):
            rot = str(rotulos.iloc[j]) if j < len(rotulos) else ""
            m = re.search(r"(\d+)", rot)
            num = int(m.group(1)) if (m and "CICLO" in rot.upper()) else len(ciclos) + 1
            col_obs = j + 2 if (j + 2 < len(cab) and _norm_cel(cab.iloc[j + 2]).startswith("OBS")) else None
            ciclos.append((num, j, j + 1, col_obs))

    linhas = []
    for i in range(linha_cab + 1, len(bruto)):
        id_ovt = normalizar_id(bruto.iat[i, col_id])
        if not PADRAO_ID.match(id_ovt):
            continue
        qt = bruto.iat[i, col_qt] if col_qt is not None else np.nan
        for num, cd, co, cobs in ciclos:
            data, ovos = bruto.iat[i, cd], bruto.iat[i, co]
            obs = bruto.iat[i, cobs] if cobs is not None else np.nan
            if pd.isna(data) and pd.isna(ovos) and pd.isna(obs):
                continue
            n = pd.to_numeric(ovos, errors="coerce")
            if pd.isna(n) and pd.notna(ovos) and pd.isna(obs):
                obs = ovos  # texto como "F" digitado na coluna de ovos
            linhas.append((ano, num, ds, aba.strip(), id_ovt, qt, data, n, obs))
    if not linhas:
        return vazio
    df = pd.DataFrame(linhas, columns=["ano", "ciclo", "ds", "bairro", "id_ovt", "qt", "data", "n_ovos", "obs"])
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["qt"] = pd.to_numeric(df["qt"], errors="coerce")
    df["obs"] = df["obs"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df["obs_cod"] = df["obs"].map(_codigo_obs)
    df.loc[df["n_ovos"] < 0, "n_ovos"] = np.nan
    for c in ["dt_entrega_apoio", "dt_entrega_lab", "dt_leitura", "dt_dig", "dt_envio_ds"]:
        df[c] = pd.NaT
    return df[COLUNAS_LEITURA]


def ler_ovos_historico(base: str | Path, ano: int) -> pd.DataFrame:
    """Planilhas por distrito de 2024 ou 2025 (uma aba por bairro)."""
    partes = []
    for arq in sorted(glob.glob(os.path.join(str(base), f"OVITRAMPAS {ano}", "Planilha OVT DS-*.xls*"))):
        m = re.search(r"DS-?\s*([IVX]+)", os.path.basename(arq))
        ds = ds_para_int(m.group(1)) if m else None
        xl = pd.ExcelFile(arq)
        for aba in xl.sheet_names:
            p = parse_aba_historica(xl.parse(aba, header=None), ano, ds, aba)
            if len(p):
                partes.append(p)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=COLUNAS_LEITURA)


# --------------------------------------------------------------------------
# Distritos sanitários e EDLs
# --------------------------------------------------------------------------
def ler_ds(base: str | Path) -> gpd.GeoDataFrame:
    """Os 8 distritos, com área recalculada em metros (a base vem em EPSG:3857)."""
    g = gpd.read_file(_achar(base, "BASE Dis*/BASE_RECIFE_DISTRITOSt.shp"))
    m = g.to_crs(CRS_METROS)
    out = gpd.GeoDataFrame({
        "ds": g["ds"].map(ds_para_int),
        "area_km2": (m.geometry.area / 1e6).round(2).values,
        "area_km2_oficial": pd.to_numeric(g["área_km2"], errors="coerce").values,
        "pop_2017": pd.to_numeric(g["pop_2017"], errors="coerce").values,
    }, geometry=g.to_crs(CRS_GEO).geometry.values, crs=CRS_GEO)
    out["nome"] = "DS " + out["ds"].astype(str)
    return out.sort_values("ds").reset_index(drop=True)


def ler_edl(base: str | Path) -> gpd.GeoDataFrame:
    """Pontos das EDLs. O CRS vem rotulado errado (EPSG:32000); o .prj diz SIRGAS UTM 25S."""
    partes = []
    for p in sorted(glob.glob(os.path.join(str(base), "ESTA*", "DS *", "*.shp"))):
        ds = int(re.search(r"DS\s*(\d+)", p).group(1))
        g = gpd.read_file(p).set_crs(CRS_METROS, allow_override=True)
        g["ds"] = ds
        partes.append(g)
    g = pd.concat(partes, ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=CRS_METROS)
    g = g[~g.geometry.is_empty & g.geometry.notna()].copy()
    g["_wkb"] = g.geometry.to_wkb()
    g = g.drop_duplicates(subset=["ds", "_wkb"]).drop(columns="_wkb")
    g = _com_metros(g)
    fora = ~g["lat"].between(LAT_MIN, LAT_MAX) | ~g["lon"].between(LON_MIN, LON_MAX)
    if fora.any():
        raise ValueError(f"{int(fora.sum())} EDLs caíram fora do Recife; confira o CRS.")
    g["edl_id"] = "DS" + g["ds"].astype(str) + "-" + g["Id"].astype(str)
    g["bairro"] = g["Bairro"].astype(str).str.strip().str.title()
    g["tipo"] = g["PE_ou_PA"].astype(str).str.strip().str.capitalize()
    # Nome fantasia e endereço de estabelecimentos ficam de fora de propósito.
    return g[["edl_id", "ds", "bairro", "tipo", "lat", "lon", "x", "y", "geometry"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Pontos das OVTs
# --------------------------------------------------------------------------
def limpar_coordenada(valor) -> float:
    """'8. 0194858' e '-8,0194858' viram 8.0194858 (o sinal é tratado fora)."""
    s = re.sub(r"\s+", "", str(valor)).replace(",", ".")
    try:
        return abs(float(s))
    except ValueError:
        return np.nan


def ler_georreferenciamento(base: str | Path) -> pd.DataFrame:
    """Planilha de coordenadas por OVT. Está suja; serve só para conferir os shapefiles."""
    xl = pd.ExcelFile(_achar(base, "Georreferenciamento*.xlsx"))
    partes = []
    for aba in xl.sheet_names:
        bruto = xl.parse(aba, header=None)
        linha = next((i for i in range(min(10, len(bruto)))
                      if any(_norm_cel(v) == "IDOVT" for v in bruto.iloc[i])), None)
        if linha is None:
            continue
        df = bruto.iloc[linha + 1:].copy()
        df.columns = [_norm_cel(c) for c in bruto.iloc[linha]]
        df = df[["IDOVT", "LATITUDE", "LONGITUDE"]].dropna(subset=["IDOVT"])
        df["id_ovt"] = df["IDOVT"].map(normalizar_id)
        df["lat"] = -df["LATITUDE"].map(limpar_coordenada)
        df["lon"] = -df["LONGITUDE"].map(limpar_coordenada)
        partes.append(df[["id_ovt", "lat", "lon"]])
    return pd.concat(partes, ignore_index=True)


def ids_esperados_por_bairro(leituras_2026: pd.DataFrame) -> dict[tuple[int, str], list[str]]:
    """(ds, bairro_slug) -> IDs em ordem numérica, a partir da planilha de 2026."""
    u = leituras_2026[["ds", "bairro", "id_ovt"]].drop_duplicates()
    u["num"] = u["id_ovt"].str.extract(r"(\d+)$")[0].astype(float)
    res = {}
    for (ds, b), grupo in u.groupby(["ds", u["bairro"].map(slug)]):
        res[(int(ds), b)] = grupo.sort_values("num")["id_ovt"].tolist()
    return res


def reconstruir_ids_por_ordem(n_pontos: int, ids: list[str]) -> list[str] | None:
    """Atribui os IDs aos pontos pela ordem do arquivo. Só vale se as contagens forem iguais."""
    return list(ids) if len(ids) == n_pontos else None


def _corrigir_id_extra_zero(id_ovt: str, conhecidos: set[str]) -> str:
    """'ENC20001' -> 'ENC2001', mas só se o ID corrigido existir na planilha."""
    if id_ovt in conhecidos:
        return id_ovt
    m = re.match(r"^([A-Z]+)20(\d{3})$", id_ovt)
    if m and (cand := f"{m.group(1)}2{m.group(2)}") in conhecidos:
        return cand
    return id_ovt


def ler_ovt_pontos(base: str | Path, leituras_2026: pd.DataFrame, ds_poligonos: gpd.GeoDataFrame,
                   limite_validacao_m: float = 50.0) -> tuple[pd.DataFrame, dict]:
    """Pontos das OVTs, sem repetição, com IDs reconstruídos onde o arquivo perdeu o nome."""
    esperados = ids_esperados_por_bairro(leituras_2026)
    conhecidos = set(leituras_2026["id_ovt"])
    georref = ler_georreferenciamento(base)
    registros, relatorio = [], {"arquivos_sem_id_recuperavel": []}

    for p in sorted(glob.glob(os.path.join(str(base), "ATUALIZA*", "DS *", "*.shp"))):
        ds = int(re.search(r"DS\s*(\d+)", p).group(1))
        arq = os.path.basename(p)[:-4]
        g = _ler_shp(p)
        g = g[~g.geometry.is_empty & g.geometry.notna() & (g.geom_type == "Point")].reset_index(drop=True)
        coluna = "Name" if "Name" in g.columns else ("OVITRAMPA" if "OVITRAMPA" in g.columns else None)
        bruto = g[coluna].astype(str).str.strip() if coluna else pd.Series([""] * len(g))
        bairro_arq = (g["FolderPath"].astype(str).str.strip() if "FolderPath" in g.columns
                      else pd.Series([arq.replace("_", " ").upper()] * len(g)))

        if not bruto.map(normalizar_id).str.match(PADRAO_ID).any():   # nenhum nome é um ID ("0", "Placemark")
            lista = esperados.get((ds, slug(arq)))
            if lista is None:   # o arquivo pode estar na pasta de outro distrito (ex.: Torrões)
                candidatos = [v for (_, s), v in esperados.items() if s == slug(arq)]
                lista = candidatos[0] if len(candidatos) == 1 else []
            ids = reconstruir_ids_por_ordem(len(g), lista)
            if ids is None:
                relatorio["arquivos_sem_id_recuperavel"].append(f"DS {ds} / {arq} ({len(g)} pontos)")
                continue
            id_final, origem = pd.Series(ids), "ordem_do_arquivo"
        else:
            id_final = bruto.map(normalizar_id).map(lambda s: _corrigir_id_extra_zero(s, conhecidos))
            origem = "nome_no_arquivo"

        parte = pd.DataFrame({"id_ovt": id_final.values, "ds": ds, "arquivo": arq,
                              "bairro_arquivo": bairro_arq.values, "origem_id": origem,
                              "geometry": g.geometry.values})
        registros.append(parte)

    bruto = gpd.GeoDataFrame(pd.concat(registros, ignore_index=True), geometry="geometry", crs=CRS_GEO)
    bruto = _com_metros(bruto)
    relatorio["pontos_lidos"] = len(bruto)
    # Ponto fora do Recife (ex.: coordenada 0,0) não serve para medir distância: sai e fica no relatório.
    fora = ~bruto["lat"].between(LAT_MIN, LAT_MAX) | ~bruto["lon"].between(LON_MIN, LON_MAX)
    relatorio["pontos_fora_do_recife"] = sorted(bruto.loc[fora, "id_ovt"].astype(str).tolist())
    bruto = bruto[~fora].copy()
    validos = bruto["id_ovt"].str.match(PADRAO_ID)
    relatorio["pontos_com_id_invalido"] = int((~validos).sum())
    bruto = bruto[validos].copy()

    # Duplicados: copias idênticas somem; posições em conflito (>1 m) ficam marcadas.
    bruto["_n"] = bruto.groupby("id_ovt")["id_ovt"].transform("size")
    ref = bruto.groupby("id_ovt")[["x", "y"]].transform("first")
    bruto["_dist_primeira"] = np.hypot(bruto["x"] - ref["x"], bruto["y"] - ref["y"])
    conflito = bruto[bruto["_n"] > 1].groupby("id_ovt")["_dist_primeira"].max().gt(1.0)
    ids_conflito = set(conflito[conflito].index)
    relatorio["ids_repetidos"] = int((bruto.groupby("id_ovt").size() > 1).sum())
    relatorio["ids_com_posicao_conflitante"] = sorted(ids_conflito)
    pontos = bruto.drop_duplicates(subset="id_ovt", keep="first").copy()
    pontos["conflito"] = pontos["id_ovt"].isin(ids_conflito)

    # Validação das posições reconstruídas contra a planilha de georreferenciamento.
    pontos["posicao_validada"] = pontos["origem_id"].eq("nome_no_arquivo")
    georref_i = georref.drop_duplicates("id_ovt").set_index("id_ovt")
    rec = pontos[pontos["origem_id"] == "ordem_do_arquivo"].copy()
    if len(rec):
        c = georref_i.reindex(rec["id_ovt"])
        gg = gpd.GeoSeries(gpd.points_from_xy(c["lon"].values, c["lat"].values), crs=CRS_GEO).to_crs(CRS_METROS)
        rec["dist_planilha_m"] = np.hypot(rec["x"].values - gg.x.values, rec["y"].values - gg.y.values)
        por_arq = rec.groupby("arquivo")["dist_planilha_m"].agg(["median", "count"])
        # "mediana" calculada só onde a planilha tem coordenada; sem coordenada, não valida.
        ok = por_arq[(por_arq["count"] >= 3) & (por_arq["median"] < limite_validacao_m)].index
        pontos.loc[pontos["arquivo"].isin(ok) & (pontos["origem_id"] == "ordem_do_arquivo"), "posicao_validada"] = True
        relatorio["reconstruidos_validados"] = sorted(ok)
        relatorio["reconstruidos_nao_validados"] = sorted(set(rec["arquivo"]) - set(ok))
        relatorio["n_pontos_nao_validados"] = int(
            pontos[(pontos["origem_id"] == "ordem_do_arquivo") & ~pontos["posicao_validada"]].shape[0])
    pontos.loc[pontos["conflito"], "posicao_validada"] = False

    # DS oficial: o da planilha de leituras. O da pasta do shapefile só entra se o ID não tiver leitura.
    ds_planilha = leituras_2026.dropna(subset=["ds"]).groupby("id_ovt")["ds"].agg(lambda s: int(s.mode().iloc[0]))
    ds_pasta = pontos["ds"].copy()
    pontos["ds"] = pontos["id_ovt"].map(ds_planilha).fillna(ds_pasta).astype(int)
    relatorio["pontos_em_pasta_de_outro_ds"] = int((pontos["ds"] != ds_pasta).sum())

    # DS pela posição: confere o DS oficial contra o polígono onde o ponto caiu.
    pts = gpd.GeoDataFrame(pontos[["id_ovt"]], geometry=gpd.points_from_xy(pontos["lon"], pontos["lat"]), crs=CRS_GEO)
    j = gpd.sjoin(pts, ds_poligonos[["ds", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    pontos["ds_poligono"] = j["ds"].reindex(pontos.index).values
    pontos["ds_confere"] = pontos["ds_poligono"].eq(pontos["ds"])
    relatorio["pontos_fora_do_ds_oficial"] = int((~pontos["ds_confere"]).sum())

    # Nome do bairro: o da planilha de 2026 quando existir.
    bairro26 = (leituras_2026.drop_duplicates("id_ovt").set_index("id_ovt")["bairro"])
    pontos["bairro"] = pontos["id_ovt"].map(bairro26).fillna(pontos["bairro_arquivo"])
    cols = ["id_ovt", "ds", "bairro", "lat", "lon", "x", "y", "origem_id", "posicao_validada",
            "conflito", "ds_poligono", "ds_confere"]
    return pontos[cols].reset_index(drop=True), relatorio


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------
def preparar_tudo(origem: str | Path, destino: str | Path) -> dict:
    """Lê a pasta da GEVACZ e grava as tabelas limpas em `destino`."""
    base = achar_pasta_base(origem)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    anos = descobrir_anos(base)
    por_ano = {ano: ler_ovos_ano(base, ano) for ano in anos}
    leituras = pd.concat(list(por_ano.values()), ignore_index=True)
    # O cadastro de IDs e bairros vem do ano mais recente que tem todos os dados.
    ref = por_ano[max(por_ano, key=lambda a: por_ano[a]["id_ovt"].nunique())]

    ds = ler_ds(base)
    edl = ler_edl(base)
    ovt, rel = ler_ovt_pontos(base, ref, ds)

    # Bairro das leituras antigas: usa o do cadastro de 2026 quando o ID existe lá.
    bairro_por_id = ovt.set_index("id_ovt")["bairro"]
    leituras["bairro"] = leituras["id_ovt"].map(bairro_por_id).fillna(leituras["bairro"])

    leituras.to_csv(destino / "leituras.csv", index=False)
    ovt.to_csv(destino / "ovt_pontos.csv", index=False)
    edl.drop(columns="geometry").to_csv(destino / "edl_pontos.csv", index=False)
    ds.to_file(destino / "ds.geojson", driver="GeoJSON")
    # Modelo para a GEVACZ preencher com a data de instalação de cada EDL.
    modelo = edl[["edl_id", "ds", "bairro", "tipo"]].assign(data_instalacao="")
    modelo.to_csv(destino / "edl_instalacao_modelo.csv", index=False)

    rel.update({
        "anos_com_leituras": anos,
        "edl_instalacao_preenchida": (destino / "edl_instalacao.csv").exists(),
        "leituras_por_ano": leituras.groupby("ano").size().astype(int).to_dict(),
        "leituras_com_ovos": int(leituras["n_ovos"].notna().sum()),
        "ovts_com_leitura_sem_ponto": int((~leituras["id_ovt"].drop_duplicates().isin(ovt["id_ovt"])).sum()),
        "ovts_com_leitura": int(leituras["id_ovt"].nunique()),
        "ovts_com_ponto": int(len(ovt)),
        "edl_pontos": int(len(edl)),
        "ds_area_km2_calculada": dict(zip(ds["ds"], ds["area_km2"])),
        "ds_area_km2_oficial": dict(zip(ds["ds"], ds["area_km2_oficial"])),
    })
    (destino / "relatorio_qualidade.json").write_text(
        json.dumps(rel, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    return rel


def descobrir_anos(base: str | Path) -> list[int]:
    """Anos com pasta 'OVITRAMPAS AAAA'. Quando chegarem 2018 a 2023, entram sozinhos."""
    anos = []
    for d in glob.glob(os.path.join(str(base), "OVITRAMPAS *")):
        m = re.search(r"OVITRAMPAS (\d{4})$", d.rstrip("/"))
        if m and os.path.isdir(d):
            anos.append(int(m.group(1)))
    return sorted(anos)


def ler_ovos_ano(base: str | Path, ano: int) -> pd.DataFrame:
    """Usa as planilhas por distrito quando existem; senão, a planilha geral em formato longo."""
    df = ler_ovos_historico(base, ano)
    if len(df):
        return df
    if glob.glob(os.path.join(str(base), f"OVITRAMPAS {ano}", "*.xlsx")):
        return ler_ovos_2026(base) if ano == 2026 else df
    return df


def ler_instalacao_edl(pasta: str | Path) -> pd.DataFrame | None:
    """Lê edl_instalacao.csv (edl_id, data_instalacao) se a GEVACZ já mandou. Senão, None."""
    arq = Path(pasta) / "edl_instalacao.csv"
    if not arq.exists():
        return None
    df = pd.read_csv(arq)
    df["data_instalacao"] = pd.to_datetime(df["data_instalacao"], errors="coerce", dayfirst=True)
    return df.dropna(subset=["data_instalacao"])[["edl_id", "data_instalacao"]]


def carregar_processados(pasta: str | Path):
    """Lê o que preparar_tudo gravou. É isto que o app usa."""
    pasta = Path(pasta)
    leituras = pd.read_csv(pasta / "leituras.csv", low_memory=False)
    # Converte uma a uma: coluna que vem toda vazia (as datas de laboratório só existem em 2026)
    # seria lida como texto e quebraria as contas de prazo.
    for c in ["data", "dt_entrega_apoio", "dt_entrega_lab", "dt_leitura", "dt_dig", "dt_envio_ds"]:
        leituras[c] = pd.to_datetime(leituras[c], errors="coerce")
    ovt = pd.read_csv(pasta / "ovt_pontos.csv")
    edl = pd.read_csv(pasta / "edl_pontos.csv")
    ds = gpd.read_file(pasta / "ds.geojson")
    rel = json.loads((pasta / "relatorio_qualidade.json").read_text(encoding="utf-8"))
    return leituras, ovt, edl, ds, rel
