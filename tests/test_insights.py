"""Testes dos cálculos de análise (core/insights.py) com dados pequenos e conhecidos."""
import numpy as np
import pandas as pd

from core import insights as ins


def _leituras(valores_por_ovt, ano=2025, ciclos=None):
    linhas = []
    for i, vals in enumerate(valores_por_ovt):
        for c, n in enumerate(vals, start=1):
            linhas.append({"ano": ano, "ciclo": c, "id_ovt": f"O{i}", "ds": 1 + i % 2, "bairro": "B", "n_ovos": n, "atipica": False})
    return pd.DataFrame(linhas)


def test_concentracao_com_ovos_iguais_segue_a_diagonal():
    l = _leituras([[10, 10]] * 10)
    c = ins.curva_de_concentracao(l)
    assert abs(ins.parcela_dos_ovos(c, 50) - 50) < 1e-6


def test_concentracao_com_uma_ovt_dominante():
    l = _leituras([[1000, 1000]] + [[1, 1]] * 9)
    c = ins.curva_de_concentracao(l)
    assert ins.parcela_dos_ovos(c, 10) > 99


def test_concentracao_sem_dados_nao_quebra():
    c = ins.curva_de_concentracao(pd.DataFrame({"id_ovt": [], "n_ovos": []}))
    assert list(c["pct_ovts"]) == [0.0, 100.0]


def test_persistencia_separa_foco_fixo_de_oscilacao():
    # O0 é sempre a mais alta; as outras 19 variam sem padrão.
    rng = np.random.default_rng(1)
    dados = [[500] * 20] + [list(rng.integers(10, 100, 20)) for _ in range(19)]
    per = ins.persistencia_de_hotspot(_leituras(dados), min_leituras=15)
    assert per.set_index("id_ovt").loc["O0", "frac"] == 1.0
    assert per[per["id_ovt"] != "O0"]["frac"].max() < 0.5


def test_persistencia_ignora_ovt_com_poucas_leituras():
    per = ins.persistencia_de_hotspot(_leituras([[1, 2, 3]] * 5), min_leituras=15)
    assert per.empty


def test_cronicas_por_distrito():
    per = pd.DataFrame({"id_ovt": list("abcd"), "ds": [1, 1, 2, 2], "frac": [0.5, 0.0, 0.1, 0.0]})
    out = ins.hotspot_cronico_por_ds(per).set_index("ds")
    assert out.loc[1, "cronicas"] == 1 and out.loc[2, "cronicas"] == 0
    assert out.loc[1, "pct"] == 50


def test_correlacao_entre_anos_perfeita_e_curta():
    a = pd.DataFrame({2024: np.arange(50.0), 2025: np.arange(50.0) * 2})
    assert abs(ins.correlacao_entre_anos(a, 2024, 2025)["rho"] - 1) < 1e-9
    assert np.isnan(ins.correlacao_entre_anos(a.head(5), 2024, 2025)["rho"])


def test_efeito_das_atipicas_reduz_a_media():
    l = _leituras([[100, 100]] * 3)
    l.loc[0, "n_ovos"] = 1000
    l["atipica"] = l["n_ovos"] >= 1000
    ef = ins.efeito_das_atipicas(l).set_index("ds")
    assert (ef["queda_pct"] <= 0).all() and ef["queda_pct"].min() < -10
    assert ef["n_atipicas"].sum() == 1


def test_intervalo_cruza_zero_quando_nao_ha_diferenca():
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({"ds": 1, "proximidade": ["perto"] * n + ["longe"] * n, "ovos_media": rng.normal(200, 30, 2 * n)})
    r = ins.intervalo_da_diferenca(df).iloc[0]
    assert r["ic_baixo"] < 0 < r["ic_alto"]


def test_intervalo_nao_cruza_zero_quando_ha_diferenca_grande():
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({"ds": 1, "proximidade": ["perto"] * n + ["longe"] * n,
                       "ovos_media": np.r_[rng.normal(100, 10, n), rng.normal(200, 10, n)]})
    r = ins.intervalo_da_diferenca(df).iloc[0]
    assert r["ic_alto"] < 0 and r["dif_pct"] < -40


def test_intervalo_com_poucas_ovts_fica_sem_intervalo():
    df = pd.DataFrame({"ds": 1, "proximidade": ["perto"] * 2 + ["longe"] * 2, "ovos_media": [1.0, 2.0, 3.0, 4.0]})
    assert np.isnan(ins.intervalo_da_diferenca(df).iloc[0]["ic_baixo"])


def test_comparacao_dentro_do_bairro_desfaz_efeito_de_bairro():
    # Bairros ricos têm EDL; dentro de cada bairro, perto e longe são iguais.
    linhas = []
    for b, base in [("A", 300), ("B", 100), ("C", 310), ("D", 110), ("E", 290), ("F", 105), ("G", 295)]:
        for k in range(6):
            linhas.append({"bairro": b, "proximidade": "perto" if k < 3 else "longe", "ovos_media": base + k % 3})
    r = ins.comparar_dentro_do_bairro(pd.DataFrame(linhas))
    assert r["n_bairros"] == 7 and abs(r["mediana_dif_pct"]) < 1


def test_comparacao_dentro_do_bairro_sem_coluna():
    assert ins.comparar_dentro_do_bairro(pd.DataFrame({"proximidade": ["perto"], "ovos_media": [1.0]}))["n_bairros"] == 0


def test_texto_ovos_por_area_cita_maior_e_menor():
    ind = pd.DataFrame({"ovos_por_ovt": [200, 220, 180], "ovos_por_km2": [3000, 6000, 1000], "ovts_por_km2": [15, 27, 5]}, index=[1, 2, 3])
    t = ins.texto_ovos_por_area(ind)
    assert "DS 2" in t and "DS 3" in t and "6,0 vezes" in t


def test_sazonalidade_e_texto_de_anos():
    l = pd.concat([_leituras([[100] * 4], ano=2024), _leituras([[150] * 4], ano=2025)])
    saz = ins.sazonalidade(l)
    assert saz.shape == (4, 2)
    assert "2025" in ins.texto_anos(saz)


def test_leitura_rapida_nao_quebra_sem_atipica():
    ind = pd.DataFrame({"ovos_por_ovt": [200, 220], "ovos_por_km2": [3000, 6000], "ovts_por_km2": [15, 27]}, index=[1, 2])
    v = _leituras([[100, 200]] * 3).drop(columns="atipica")
    assert len(ins.pontos_leitura_rapida(ind, v, minimo_bairro=2)) >= 3


def test_tendencia_com_distancia_negativa_e_frase():
    rng = np.random.default_rng(5)
    dist = rng.uniform(0, 1000, 300)
    df = pd.DataFrame({"dist_edl_m": dist, "ovos_media": 300 - dist * 0.1 + rng.normal(0, 10, 300)})
    t = ins.tendencia_com_distancia(df)
    assert t["rho"] < -0.5 and t["p_valor"] < 0.05
    assert "caem" in ins.frase_tendencia(t)


def test_tendencia_sem_dados_suficientes_nao_gera_frase():
    t = ins.tendencia_com_distancia(pd.DataFrame({"dist_edl_m": [1.0, 2.0], "ovos_media": [3.0, 4.0]}))
    assert ins.frase_tendencia(t) == ""
