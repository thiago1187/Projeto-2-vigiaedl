"""Testes das regras de perto × longe, antes × depois e Kernel Density.

Os dados são pequenos e inventados de propósito: dá para conferir cada resposta de cabeça.
"""
import numpy as np
import pandas as pd
import pytest

from core import espacial as e


# ---------------------------------------------------------------- perto × longe
def test_distancia_pitagoras():
    # OVT a 3 m para leste e 4 m para o norte da EDL: distância 5 m.
    d, pos = e.distancia_ate_edl_mais_proxima([3.0], [4.0], [0.0], [0.0])
    assert d[0] == pytest.approx(5.0)
    assert pos[0] == 0


def test_escolhe_a_edl_mais_proxima():
    d, pos = e.distancia_ate_edl_mais_proxima([90.0], [0.0], [0.0, 100.0, 500.0], [0.0, 0.0, 0.0])
    assert pos[0] == 1 and d[0] == pytest.approx(10.0)


def test_raio_e_inclusivo():
    d = [199.99, 200.0, 200.01]
    assert list(e.classificar_proximidade(d, 200)) == ["perto", "perto", "longe"]


def test_raio_pode_ser_trocado():
    assert list(e.classificar_proximidade([150.0], 100)) == ["longe"]
    assert list(e.classificar_proximidade([150.0], 300)) == ["perto"]


def test_ovt_sem_coordenada_nao_vira_longe():
    d, pos = e.distancia_ate_edl_mais_proxima([np.nan, 10.0], [np.nan, 0.0], [0.0], [0.0])
    assert np.isnan(d[0]) and pos[0] == -1
    assert list(e.classificar_proximidade(d)) == ["sem_posicao", "perto"]


def test_sem_nenhuma_edl():
    d, pos = e.distancia_ate_edl_mais_proxima([1.0, 2.0], [1.0, 2.0], [], [])
    assert np.isnan(d).all() and (pos == -1).all()


def test_edl_com_coordenada_faltando_e_ignorada():
    d, pos = e.distancia_ate_edl_mais_proxima([10.0], [0.0], [np.nan, 0.0], [np.nan, 0.0])
    assert pos[0] == 1 and d[0] == pytest.approx(10.0)


# ---------------------------------------------------- data de instalação por EDL
def test_primeira_instalacao_pega_a_mais_antiga_dentro_do_raio():
    datas, n = e.primeira_instalacao_no_raio(
        x=[0.0, 1000.0], y=[0.0, 0.0],
        edl_x=[50.0, 120.0, 5000.0], edl_y=[0.0, 0.0, 0.0],
        edl_datas=["2024-06-01", "2024-02-10", "2020-01-01"], raio_m=200)
    assert pd.Timestamp(datas[0]) == pd.Timestamp("2024-02-10")   # a de 2020 está longe demais
    assert n[0] == 2
    assert np.isnat(datas[1]) and n[1] == 0


def test_primeira_instalacao_ignora_edl_sem_data_mas_conta_ela():
    datas, n = e.primeira_instalacao_no_raio([0.0], [0.0], [10.0, 20.0], [0.0, 0.0], [None, "2025-03-01"], 200)
    assert pd.Timestamp(datas[0]) == pd.Timestamp("2025-03-01") and n[0] == 2


def test_primeira_instalacao_raio_inclusivo():
    datas, n = e.primeira_instalacao_no_raio([0.0], [0.0], [200.0], [0.0], ["2024-01-01"], 200)
    assert n[0] == 1


# ------------------------------------------------------------- antes × depois
def test_periodo_antes_e_depois_com_a_data_do_corte_contando_como_depois():
    r = e.classificar_periodo(["2024-01-01", "2024-03-14", "2024-03-15", "2025-01-01"], "2024-03-15")
    assert list(r) == ["antes", "antes", "depois", "depois"]


def test_periodo_sem_data_e_sem_corte():
    assert list(e.classificar_periodo([pd.NaT, "2024-01-01"], "2024-06-01")) == ["sem_data", "antes"]
    assert list(e.classificar_periodo(["2024-01-01"], None)) == ["sem_corte"]


def test_corte_padrao_e_a_menor_data():
    assert e.data_corte_padrao(["2024-05-01", "2024-02-10", None]) == pd.Timestamp("2024-02-10")
    assert e.data_corte_padrao([None, pd.NaT]) is None
    assert e.data_corte_padrao([]) is None


# --------------------------------------------------------------- resumos e testes
def _leituras(valores, ovt="A"):
    return pd.DataFrame({"id_ovt": ovt, "n_ovos": valores})


def test_resumo_por_ovt_ignora_leitura_sem_contagem_e_conta_zero():
    r = e.resumo_por_ovt(_leituras([0, 100, np.nan, 200])).iloc[0]
    assert r["n_leituras"] == 3                      # o NaN não conta como leitura
    assert r["ovos_media"] == pytest.approx(100.0)   # (0 + 100 + 200) / 3
    assert r["ovos_mediana"] == 100


def test_comparar_grupos_com_diferenca_clara():
    df = pd.DataFrame({
        "proximidade": ["perto"] * 10 + ["longe"] * 10,
        "ovos_media": list(range(10, 20)) + list(range(100, 110)),
    })
    r = e.comparar_grupos(df)
    assert r["n_a"] == 10 and r["n_b"] == 10
    assert r["mediana_a"] == 14.5 and r["mediana_b"] == 104.5
    assert r["diferenca_pct"] == pytest.approx((14.5 - 104.5) / 104.5 * 100)
    assert r["p_valor"] < 0.001
    assert r["efeito"] == pytest.approx(-1.0)        # todo "perto" abaixo de todo "longe"
    assert r["aviso"] is None


def test_comparar_grupos_com_poucas_ovts_nao_calcula_p():
    df = pd.DataFrame({"proximidade": ["perto"] * 2 + ["longe"] * 8, "ovos_media": [1, 2] + list(range(8))})
    r = e.comparar_grupos(df)
    assert np.isnan(r["p_valor"]) and r["aviso"]


def test_comparar_grupos_sem_diferenca():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"proximidade": ["perto"] * 40 + ["longe"] * 40, "ovos_media": rng.normal(100, 10, 80)})
    assert e.comparar_grupos(df)["p_valor"] > 0.05


def test_comparar_por_regiao_devolve_uma_linha_por_ds():
    df = pd.DataFrame({"ds": [1] * 12 + [2] * 12,
                       "proximidade": (["perto"] * 6 + ["longe"] * 6) * 2,
                       "ovos_media": np.arange(24.0)})
    r = e.comparar_por_regiao(df)
    assert list(r["ds"]) == [1, 2] and (r["n_a"] == 6).all()


def test_diferenca_em_diferencas():
    # perto: 100 -> 60 (caiu 40). longe: 100 -> 90 (caiu 10). Efeito atribuível: -30.
    df = pd.DataFrame({
        "proximidade": ["perto", "perto", "longe", "longe"],
        "periodo": ["antes", "depois", "antes", "depois"],
        "ovos_media": [100.0, 60.0, 100.0, 90.0],
    })
    r = e.diferenca_em_diferencas(df)
    assert r["did"] == pytest.approx(-30.0)


def test_diferenca_em_diferencas_sem_as_quatro_celulas():
    df = pd.DataFrame({"proximidade": ["perto"], "periodo": ["antes"], "ovos_media": [1.0]})
    assert np.isnan(e.diferenca_em_diferencas(df)["did"])


def test_indicadores_por_regiao():
    # DS 1: ciclo 1 com ovos 0, 100, 200 (3 OVTs); ciclo 2 com 100, 100, NaN. Área 2 km².
    leit = pd.DataFrame({
        "ds": 1, "ano": 2026, "ciclo": [1, 1, 1, 2, 2, 2],
        "id_ovt": ["a", "b", "c", "a", "b", "c"], "n_ovos": [0, 100, 200, 100, 100, np.nan]})
    r = e.indicadores_por_regiao(leit, pd.Series({1: 2.0})).loc[1]
    assert r["n_ovts"] == 3 and r["n_leituras"] == 5
    assert r["ovos_por_ovt"] == pytest.approx(100.0)       # 500 / 5
    assert r["ido"] == pytest.approx(125.0)                # (100+200+100+100) / 4 positivas
    assert r["ipo"] == pytest.approx(80.0)                 # 4 de 5
    assert r["ovos_por_km2"] == pytest.approx(500 / 2 / 2.0)   # 500 ovos, 2 ciclos, 2 km²
    assert r["ovts_por_km2"] == pytest.approx(1.5)


def test_atipicos_marca_o_valor_dez_vezes_acima_da_mediana():
    leit = pd.DataFrame({"ano": 2026, "ciclo": 7, "bairro": "Boa Viagem",
                         "id_ovt": list("abcdefg"), "n_ovos": [100, 90, 110, 95, 105, 100, 1500]})
    r = e.sinalizar_atipicos(leit)
    assert list(r["atipica"]) == [False] * 6 + [True]
    assert r.loc[6, "vezes_mediana"] == pytest.approx(15.0)


def test_atipicos_nao_marca_em_grupo_pequeno():
    leit = pd.DataFrame({"ano": 2026, "ciclo": 1, "bairro": "X", "id_ovt": list("abc"), "n_ovos": [10, 10, 500]})
    assert not e.sinalizar_atipicos(leit)["atipica"].any()


# ------------------------------------------------------------------ Kernel Density
def test_kde_soma_conserva_a_massa():
    # A soma da grade (ovos por km² × área da célula em km²) tem de devolver o total de ovos.
    x, y, w = np.array([1000.0, 1500.0, 3000.0]), np.array([1000.0, 1200.0, 2500.0]), np.array([100.0, 50.0, 200.0])
    g = e.kde_grade(x, y, w, banda_m=300, celula_m=50, modo="soma", suporte_minimo=0)
    total = np.nansum(g.valores) * (g.celula_m ** 2) / 1e6
    assert total == pytest.approx(350.0, rel=0.01)


def test_kde_pico_fica_no_ponto():
    g = e.kde_grade([5000.0], [8000.0], [100.0], banda_m=300, celula_m=50, modo="soma")
    i, j = np.unravel_index(np.nanargmax(g.valores), g.valores.shape)
    assert abs(g.xs[j] - 5000) <= 50 and abs(g.ys[i] - 8000) <= 50


def test_kde_media_de_pesos_iguais_devolve_esse_peso():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 3000, 60), rng.uniform(0, 3000, 60)
    g = e.kde_grade(x, y, np.full(60, 120.0), banda_m=300, celula_m=100, modo="media")
    assert np.nanmax(np.abs(g.valores - 120.0)) < 1e-6


def test_kde_media_nao_depende_de_quantas_ovts_ha():
    # Duas regiões com o mesmo nível (100 ovos por OVT); uma tem 5 vezes mais OVTs.
    rng = np.random.default_rng(2)
    xa, ya = rng.uniform(0, 1000, 50), rng.uniform(0, 1000, 50)
    xb, yb = rng.uniform(5000, 6000, 10), rng.uniform(0, 1000, 10)
    x, y, w = np.r_[xa, xb], np.r_[ya, yb], np.full(60, 100.0)
    soma = e.kde_grade(x, y, w, banda_m=300, celula_m=100, modo="soma")
    media = e.kde_grade(x, y, w, banda_m=300, celula_m=100, modo="media")
    va, vb = e.kde_nos_pontos(soma, [500.0, 5500.0], [500.0, 500.0])
    ma, mb = e.kde_nos_pontos(media, [500.0, 5500.0], [500.0, 500.0])
    assert va > 3 * vb                        # "soma" acende onde há mais armadilhas
    assert ma == pytest.approx(mb, rel=1e-6)  # "media" não


def test_kde_celulas_longe_das_ovts_ficam_sem_valor():
    g = e.kde_grade([0.0, 100.0], [0.0, 100.0], [1.0, 1.0], banda_m=100, celula_m=50)
    assert np.isnan(g.valores[0, 0]) or np.isfinite(g.valores).any()
    assert np.isnan(g.valores).any()


def test_kde_modo_invalido_e_sem_pontos():
    with pytest.raises(ValueError):
        e.kde_grade([0.0], [0.0], modo="outro")
    with pytest.raises(ValueError):
        e.kde_grade([np.nan], [np.nan])


def test_limiar_hotspot():
    g = e.kde_grade([0.0, 400.0, 800.0], [0.0, 0.0, 0.0], [10.0, 500.0, 10.0], banda_m=150, celula_m=50, modo="media")
    t = e.limiar_hotspot(g, 90)
    assert np.nanmin(g.valores) <= t <= np.nanmax(g.valores)


def test_grade_para_latlon_mantem_a_posicao():
    from pyproj import Transformer
    lon, lat = -34.880, -8.050
    x, y = Transformer.from_crs(4326, 31985, always_xy=True).transform(lon, lat)
    g = e.kde_grade([x], [y], [100.0], banda_m=300, celula_m=50, modo="soma")
    img, (lat0, lon0, lat1, lon1) = e.grade_para_latlon(g, passo_graus=0.0005)
    i, j = np.unravel_index(np.nanargmax(img), img.shape)
    lat_pico = lat1 - (i + 0.5) * (lat1 - lat0) / img.shape[0]     # a linha 0 é o norte
    lon_pico = lon0 + (j + 0.5) * (lon1 - lon0) / img.shape[1]
    assert abs(lat_pico - lat) < 0.002 and abs(lon_pico - lon) < 0.002   # ~200 m


def test_poligonos_de_nivel_cobre_so_o_pico_e_respeita_buracos():
    import numpy as np
    from shapely.geometry import Point
    xs = np.arange(0, 3000, 100.0)
    X, Y = np.meshgrid(xs, xs)
    z = np.exp(-(((X - 1500) ** 2 + (Y - 1500) ** 2) / (2 * 500 ** 2))) * 100
    g = e.Grade(valores=z, x0=0.0, y0=0.0, celula_m=100.0)
    alto = e.poligonos_de_nivel(g, 60, np.inf)
    baixo = e.poligonos_de_nivel(g, -np.inf, 60)
    assert not alto.is_empty and not baixo.is_empty
    assert alto.area < baixo.area                      # o pico é menor que o resto
    assert alto.area > 0 and abs(alto.area + baixo.area - e.poligonos_de_nivel(g, -np.inf, np.inf).area) < 1e-12 + 1e-6 * baixo.area


def test_poligonos_de_nivel_sem_valor_valido_devolve_vazio():
    import numpy as np
    g = e.Grade(valores=np.full((5, 5), np.nan), x0=0.0, y0=0.0, celula_m=100.0)
    assert e.poligonos_de_nivel(g, 0, 1).is_empty
