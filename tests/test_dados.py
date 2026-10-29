"""Testes da leitura e limpeza dos dados. Os casos vêm de problemas reais dos arquivos da GEVACZ."""
import numpy as np
import pandas as pd
import pytest

from core import dados as d


def test_normalizar_id():
    assert d.normalizar_id("REC 1001") == "REC1001"
    assert d.normalizar_id("rec/1001") == "REC1001"
    assert d.normalizar_id(" ENC-2001 ") == "ENC2001"
    assert d.normalizar_id(None) == "" and d.normalizar_id(float("nan")) == ""


def test_ds_para_int():
    assert [d.ds_para_int(v) for v in ["I", "VI", "DS 6", "DS-VIII", 3, "4"]] == [1, 6, 6, 8, 3, 4]
    assert d.ds_para_int("XX") is None


def test_slug_casa_bairro_e_arquivo():
    assert d.slug("ÁGUA FRIA") == d.slug("agua_fria") == "agua_fria"
    assert d.slug("TORRÕES") == "torroes"


def test_limpar_coordenada_suja():
    assert d.limpar_coordenada("8. 0194858") == pytest.approx(8.0194858)   # espaço no meio
    assert d.limpar_coordenada("-8,0194858") == pytest.approx(8.0194858)   # vírgula e sinal
    assert np.isnan(d.limpar_coordenada("sem coordenada"))


def test_codigo_de_observacao():
    assert d._codigo_obs("F") == "F" and d._codigo_obs("e ") == "E"
    assert d._codigo_obs("REC-2026") == "REC"
    assert np.isnan(d._codigo_obs(None)) and np.isnan(d._codigo_obs("qualquer coisa"))


def test_reconstruir_ids_so_com_contagem_igual():
    assert d.reconstruir_ids_por_ordem(3, ["A1", "A2", "A3"]) == ["A1", "A2", "A3"]
    assert d.reconstruir_ids_por_ordem(3, ["A1", "A2"]) is None    # não adivinha
    assert d.reconstruir_ids_por_ordem(3, []) is None


def test_id_com_zero_a_mais_so_e_corrigido_se_o_corrigido_existe():
    conhecidos = {"ENC2001", "ENC2002"}
    assert d._corrigir_id_extra_zero("ENC20001", conhecidos) == "ENC2001"
    assert d._corrigir_id_extra_zero("ENC2001", conhecidos) == "ENC2001"
    assert d._corrigir_id_extra_zero("ENC20099", conhecidos) == "ENC20099"   # sem par: deixa como está


def test_ids_esperados_ficam_em_ordem_numerica():
    leit = pd.DataFrame({"ds": 2, "bairro": ["ÁGUA FRIA"] * 3, "id_ovt": ["AFR2010", "AFR2002", "AFR2001"]})
    assert d.ids_esperados_por_bairro(leit)[(2, "agua_fria")] == ["AFR2001", "AFR2002", "AFR2010"]


def _aba_2025():
    """Imita o layout real: rótulo do ciclo numa linha, cabeçalho na seguinte, dados abaixo."""
    n = np.nan
    return pd.DataFrame([
        [n, "Ovitrampas 2025", n, n, n, n, n, n, n, n, n, n],
        [n, n, n, n, n, n, "CICLO 1", n, n, "CICLO 2", n, n],
        ["BAIRRO", "AGENTE", n, "QT", "ID. OVT", n, "DATA", "NÚMERO OVOS", "OBS", "DATA", "NÚMERO OVOS", "OBS"],
        ["RECIFE ANTIGO", "ANA", n, 1, "REC1001", n, "2025-01-20", 289, n, "2025-02-06", 9, n],
        [n, "ANA", n, 1, "REC1002", n, "2025-01-20", 13, n, "2025-02-06", n, "E"],
        [n, "ANA", n, 2, "REC 1003", n, n, n, "F", n, n, "F"],
        [n, "TOTAL", n, n, "Média", n, n, 100, n, n, 50, n],        # linha de resumo: deve sumir
    ])


def test_parse_aba_historica():
    r = d.parse_aba_historica(_aba_2025(), ano=2025, ds=1, aba="RECIFE ANTIGO")
    assert set(r["id_ovt"]) == {"REC1001", "REC1002", "REC1003"}       # "REC 1003" normalizado, "Média" ignorada
    assert sorted(r["ciclo"].unique()) == [1, 2]
    a = r[(r.id_ovt == "REC1001") & (r.ciclo == 1)].iloc[0]
    assert a["n_ovos"] == 289 and a["data"] == pd.Timestamp("2025-01-20") and a["ds"] == 1
    b = r[(r.id_ovt == "REC1002") & (r.ciclo == 2)].iloc[0]
    assert np.isnan(b["n_ovos"]) and b["obs_cod"] == "E"               # extraviado: sem contagem, não é zero
    c = r[(r.id_ovt == "REC1003") & (r.ciclo == 1)].iloc[0]
    assert np.isnan(c["n_ovos"]) and c["obs_cod"] == "F"


def test_parse_aba_sem_cabecalho_devolve_vazio():
    r = d.parse_aba_historica(pd.DataFrame([["a", "b"], ["c", "d"]]), 2025, 1, "x")
    assert r.empty and list(r.columns) == d.COLUNAS_LEITURA


def test_zero_ovos_e_diferente_de_sem_contagem():
    n = np.nan
    bruto = pd.DataFrame([
        [n, n, n, n, "CICLO 1", n, n],
        ["BAIRRO", "QT", "ID. OVT", n, "DATA", "NÚMERO OVOS", "OBS"],
        ["X", 1, "ABC1001", n, "2025-01-01", 0, n],
    ])
    r = d.parse_aba_historica(bruto, 2025, 1, "X")
    assert r["n_ovos"].iloc[0] == 0      # zero é uma contagem válida


def test_descobrir_anos(tmp_path):
    for nome in ["OVITRAMPAS 2018", "OVITRAMPAS 2024", "OVITRAMPAS 2026", "OUTRA PASTA"]:
        (tmp_path / nome).mkdir()
    (tmp_path / "OVITRAMPAS 2099.txt").write_text("não é pasta")
    assert d.descobrir_anos(tmp_path) == [2018, 2024, 2026]


def test_instalacao_edl_so_existe_quando_a_gevacz_manda(tmp_path):
    assert d.ler_instalacao_edl(tmp_path) is None
    (tmp_path / "edl_instalacao.csv").write_text("edl_id,data_instalacao\nDS1-1,15/03/2024\nDS1-2,\n")
    r = d.ler_instalacao_edl(tmp_path)
    assert list(r["edl_id"]) == ["DS1-1"]                                # linha sem data some
    assert r["data_instalacao"].iloc[0] == pd.Timestamp("2024-03-15")    # dia/mês/ano


# --- datas com ano digitado errado -----------------------------------------------------------------
def _ciclos(datas_por_ciclo, ano=2024):
    linhas = []
    for ciclo, datas in datas_por_ciclo.items():
        for i, d in enumerate(datas):
            linhas.append({"ano": ano, "ciclo": ciclo, "id_ovt": f"X{i}", "data": pd.Timestamp(d) if d else pd.NaT})
    return pd.DataFrame(linhas)


def test_ano_digitado_errado_e_corrigido_pelo_ciclo():
    from core.dados import corrigir_datas_por_ciclo
    # ciclo 2 foi digitado com 2025 em vez de 2024
    df = _ciclos({1: ["2024-01-15"] * 5, 2: ["2025-01-29"] * 5})
    out, r = corrigir_datas_por_ciclo(df)
    assert set(out.loc[out.ciclo == 2, "data"]) == {pd.Timestamp("2024-01-29")}
    assert out.loc[out.ciclo == 2, "data_corrigida"].all() and not out.loc[out.ciclo == 1, "data_corrigida"].any()
    assert r["datas_corrigidas"] == 5
    assert set(out.loc[out.ciclo == 2, "data_original"]) == {pd.Timestamp("2025-01-29")}   # a original fica guardada


def test_virada_de_ano_verdadeira_nao_e_mexida():
    from core.dados import corrigir_datas_por_ciclo
    df = _ciclos({1: ["2024-11-18"] * 4, 2: ["2024-12-02"] * 4, 3: ["2025-01-02"] * 4})
    out, r = corrigir_datas_por_ciclo(df)
    assert set(out.loc[out.ciclo == 3, "data"]) == {pd.Timestamp("2025-01-02")}
    assert r["datas_corrigidas"] == 0


def test_data_fora_do_ciclo_recebe_a_data_do_ciclo_e_ausente_continua_ausente():
    from core.dados import corrigir_datas_por_ciclo
    df = _ciclos({1: ["2024-01-15"] * 5, 2: ["2024-01-29"] * 4 + ["2024-08-15"], 3: ["2024-02-08"] * 4 + [None]})
    out, _ = corrigir_datas_por_ciclo(df)
    assert (out.loc[out.ciclo == 2, "data"] == pd.Timestamp("2024-01-29")).all()   # o 15/08 virou a data do ciclo
    assert out.loc[out.ciclo == 3, "data"].isna().sum() == 1                      # sem data, segue sem data


def test_ano_absurdo_vira_o_ano_do_ciclo():
    from core.dados import corrigir_datas_por_ciclo
    df = _ciclos({1: ["2024-01-15"] * 5, 2: ["2024-01-29"] * 4 + ["2037-01-29"]})
    out, _ = corrigir_datas_por_ciclo(df)
    assert (out.loc[out.ciclo == 2, "data"] == pd.Timestamp("2024-01-29")).all()
