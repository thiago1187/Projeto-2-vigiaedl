# VIGIAEDL

Painel em Streamlit para acompanhar as ovitrampas (OVTs) do Recife e avaliar se as Estações Disseminadoras de Larvicida (EDLs) estão ajudando a reduzir a presença do *Aedes aegypti*. É o Projeto 2 da disciplina, feito com dados da GEVACZ, da Prefeitura do Recife.

## Status

Funcionando com os dados de 2024 a 2026. A comparação **antes × depois** já está pronta no código, mas só liga quando a GEVACZ mandar a data de instalação de cada EDL (hoje dá para testar com uma data de corte hipotética). Dados de anos anteriores (2018 em diante) entram sem mexer no código: basta colocar a pasta "OVITRAMPAS AAAA" junto das outras e rodar a preparação de novo.

## As três telas

- **Visão geral:** ovos por OVT em cada ciclo, ranking de bairros, mapa de calor distrito × ciclo, e mapa com OVTs, EDLs (com raio), distritos e Kernel Density.
- **Ficha da OVT:** histórico de uma armadilha comparado com a mediana do bairro, alerta de padrão fora do comum, últimas leituras e exportação em CSV.
- **Histórico e eficácia:** evolução por distrito sanitário, perto × longe da EDL, faixas de distância, antes × depois, texto-resumo e as limitações dos dados.

## Como rodar

```bash
pip install -r requirements.txt

# 1) Trata os dados enviados pela GEVACZ (pasta já extraída)
python scripts/preparar_dados.py --origem "<pasta INFORMAÇÕES - GEVACZ>" --destino data/processed

# 2) Sobe o painel
streamlit run app.py
```

Para ler os dados tratados de outro lugar, defina `VIGIAEDL_DADOS` com o caminho da pasta. Os testes rodam com `pytest`.

## Dados (e por que não estão aqui)

Os dados da GEVACZ são georreferenciados e **não ficam neste repositório**, que é público. As pastas `data/raw/` e `data/processed/` estão no `.gitignore`. O tratamento também descarta nome e endereço dos estabelecimentos das EDLs.

## Regras de negócio (em `core/espacial.py`)

- **Distância:** sempre em metros, no sistema SIRGAS 2000 / UTM 25S (EPSG:31985). Mapas usam EPSG:4326.
- **Perto × longe:** distância até a EDL mais próxima menor ou igual ao raio (200 m por padrão; o painel deixa escolher 100, 200, 300 ou 500 m). Exatamente no limite conta como perto.
- **Antes × depois:** corte na data de instalação (leitura em data anterior ao corte é "antes"; no dia ou depois é "depois"). Sem a data de instalação, não existe corte.
- **Unidade da comparação:** a média de ovos de cada OVT (não cada leitura), com mínimo de 3 leituras, para não contar a mesma armadilha várias vezes. Teste de Mann-Whitney, com diferença em % entre as medianas. O antes × depois usa diferença das diferenças.
- **Leitura sem contagem** (F fechada, E extraviada, R recusada, D desocupada) não entra como zero. REC = recuperada.
- **Kernel Density:** gaussiano, com suavização ajustável (padrão 300 m). Dois modos: nível de infestação (média de ovos por OVT, não depende de quantas armadilhas há) e concentração (ovos por km²).
- **Leituras atípicas:** valor 10 ou mais vezes a mediana do bairro no mesmo ciclo é sinalizado para conferência, não removido.

## O que o painel não consegue dizer (ainda)

- Perto × longe mostra **associação, não efeito**. Se as EDLs foram colocadas onde já havia mais ovos, "perto" tende a parecer pior, mesmo que a EDL ajude.
- Sem data de instalação e sem leituras de antes, não dá para separar uma coisa da outra.
- A posição de parte das OVTs do DS II foi reconstruída pela ordem do arquivo e não pôde ser conferida; por padrão o painel as deixa fora das comparações (dá para incluir na barra lateral).

## Estrutura

```
app.py                  painel (3 telas)
core/dados.py           leitura e limpeza dos arquivos da GEVACZ
core/espacial.py        distância, perto/longe, antes/depois, estatística, Kernel Density
core/analise.py         junta leituras, OVTs e EDLs
core/mapa.py, visual.py mapa e estilo
scripts/preparar_dados.py
tests/                  testes das regras de negócio
```
