"""Testes do cruzamento (junção de leituras, OVTs e EDLs)."""
import numpy as np
import pandas as pd
import pytest

from core import analise as a


def _edl():
    return pd.DataFrame({"edl_id": ["E1"], "tipo": ["Outros"], "x": [0.0], "y": [0.0]})


def _ovt():
    # Distâncias até a EDL: 50, 100, 150, 200, 250, 600 m, e uma OVT sem coordenada.
    return pd.DataFrame({
        "id_ovt": list("ABCDEFG"), "ds": 1, "posicao_validada": [True] * 5 + [False, True],
        "x": [50.0, 100.0, 150.0, 200.0, 250.0, 600.0, np.nan], "y": [0.0] * 6 + [np.nan]})


def test_classificar_ovts_perto_longe_e_sem_posicao():
    oc = a.classificar_ovts(_ovt(), _edl(), raio_m=200)
    assert list(oc["proximidade"]) == ["perto"] * 4 + ["longe"] * 2 + ["sem_posicao"]
    assert oc.loc[0, "edl_mais_proxima"] == "E1" and oc.loc[0, "dist_edl_m"] == pytest.approx(50.0)
    assert pd.isna(oc.loc[6, "edl_mais_proxima"])        # sem posição, sem EDL associada


def _leituras():
    linhas = []
    for id_ovt, base in zip("ABCDEFG", [10, 10, 10, 10, 100, 100, 100]):
        for ciclo in (1, 2, 3):
            linhas.append({"id_ovt": id_ovt, "ds": 1, "ano": 2026, "ciclo": ciclo,
                           "data": pd.Timestamp("2026-01-01") + pd.Timedelta(days=30 * ciclo), "n_ovos": base})
    return pd.DataFrame(linhas)


def test_base_por_ovt_tira_quem_nao_tem_coordenada():
    oc = a.classificar_ovts(_ovt(), _edl(), 200)
    base = a.base_por_ovt(_leituras(), oc)
    assert set(base["id_ovt"]) == set("ABCDEF")            # G não entra
    assert set(base["proximidade"]) == {"perto", "longe"}


def test_apenas_validadas_tira_a_ovt_nao_conferida():
    oc = a.classificar_ovts(_ovt(), _edl(), 200)
    assert "F" in set(a.base_por_ovt(_leituras(), oc)["id_ovt"])
    assert "F" not in set(a.base_por_ovt(_leituras(), oc, apenas_validadas=True)["id_ovt"])


def test_minimo_de_leituras_por_ovt():
    oc = a.classificar_ovts(_ovt(), _edl(), 200)
    l = _leituras()
    l = l[~((l.id_ovt == "A") & (l.ciclo > 1))]            # A fica com 1 leitura
    assert "A" not in set(a.base_por_ovt(l, oc, minimo_leituras=3)["id_ovt"])
    assert "A" in set(a.base_por_ovt(l, oc, minimo_leituras=1)["id_ovt"])


def test_eficacia_sem_corte_nao_calcula_antes_depois():
    oc = a.classificar_ovts(_ovt(), _edl(), 200)
    r = a.eficacia(a.base_por_ovt(_leituras(), oc))
    assert r["did"] is None
    assert r["geral"]["mediana_a"] == 10 and r["geral"]["mediana_b"] == 100    # perto 10, longe 100


def test_eficacia_com_corte_calcula_diferenca_em_diferencas():
    oc = a.classificar_ovts(_ovt(), _edl(), 200)
    l = _leituras()
    depois = l["ciclo"] >= 2
    l.loc[depois & l["id_ovt"].isin(list("ABCD")), "n_ovos"] = 5      # perto cai de 10 para 5 depois do corte
    base = a.base_por_ovt(l, oc, corte="2026-02-15", minimo_leituras=1)
    r = a.eficacia(base)
    assert r["did"]["perto_antes"] == 10 and r["did"]["perto_depois"] == 5
    assert r["did"]["longe_antes"] == 100 and r["did"]["longe_depois"] == 100
    assert r["did"]["did"] == pytest.approx(-5.0)


def test_faixas_de_distancia_incluem_o_limite_de_cima():
    oc = a.classificar_ovts(_ovt(), _edl(), 200)
    f = a.por_faixa_de_distancia(a.base_por_ovt(_leituras(), oc))
    assert list(f["faixa"]) == ["até 100 m", "100–200 m", "200–300 m", "300–500 m", "mais de 500 m"]
    assert list(f["n_ovts"]) == [2, 2, 1, 0, 1]     # 50 e 100 | 150 e 200 | 250 | nenhuma | 600


def test_filtrar_leituras():
    l = _leituras()
    assert len(a.filtrar_leituras(l, ciclos=(2, 3))) == 14
    assert len(a.filtrar_leituras(l, anos=2025)) == 0
    assert len(a.filtrar_leituras(l, ds=2)) == 0 and len(a.filtrar_leituras(l, ds="todos")) == len(l)
