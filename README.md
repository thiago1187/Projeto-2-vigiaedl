# VIGIAEDL

Painel em Streamlit para acompanhar as ovitrampas (OVTs) do Recife e avaliar se as Estações Disseminadoras de Larvicida (EDLs) estão ajudando a reduzir a presença do *Aedes aegypti*. É o Projeto 2 da disciplina, feito com dados da GEVACZ, da Prefeitura do Recife.

## Status

Em construção. Por enquanto o repositório tem só a estrutura inicial.

## O que o painel vai ter

Três telas, seguindo o protótipo que a equipe aprovou:

- **Visão geral:** ovos por OVT em cada ciclo, por distrito sanitário e por bairro, com mapa.
- **Ficha da OVT:** histórico de uma armadilha comparado com o bairro.
- **Histórico e eficácia:** comparação perto × longe da EDL (raio configurável, 200 m como ponto de partida) e antes × depois da instalação (data de corte configurável).

## Dados

Os dados da GEVACZ (OVTs, EDLs e leituras) são georreferenciados e **não ficam neste repositório**, que é público. As pastas `data/raw/` e `data/processed/` estão no `.gitignore`. Para rodar o painel, é preciso ter localmente a pasta de dados enviada pela GEVACZ.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

O `app.py` ainda vai ser adicionado.
