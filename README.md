# VIGIAEDL

Painel em Streamlit para acompanhar as ovitrampas (OVTs) do Recife e avaliar se as Estações Disseminadoras de Larvicida (EDLs) estão ajudando a reduzir a presença do *Aedes aegypti*. É o Projeto 2 da disciplina, feito com dados da GEVACZ, da Prefeitura do Recife.

## Status

Funcionando com os dados de 2024 a 2026, sempre organizado por Distrito Sanitário (DS). O antes × depois já está pronto no código e liga sozinho quando a GEVACZ mandar a data de instalação de cada EDL; hoje dá para testar com uma data de corte hipotética. Dados de anos anteriores entram sem mexer no código (veja "Como entram anos anteriores").

## As três telas

Cada tela abre com a faixa de recorte (o que está filtrado) e um índice de seções. Cada seção começa com um texto curto de leitura, calculado dos dados, e depois vêm os gráficos.

- **Visão geral:** leitura rápida; ovos por OVT em cada ciclo; bairros mais altos; mapa de calor distrito × ciclo; tabela dos distritos lado a lado; **rankings por distrito** (top 10 de bairros e de OVTs, com seletor de DS); ovos por área (por km² e nível × cobertura de OVTs); mapa com OVTs, EDLs (com raio), distritos e Kernel Density; distribuição dos ovos e curva de concentração.
- **Ficha da OVT:** atalhos para os casos mais fora do padrão (incluindo a BVG6043 do plano); histórico da armadilha contra a mediana do bairro; últimas leituras e CSV; ranking das OVTs que mais fogem do padrão e o quanto isso mexe na média do distrito.
- **Histórico e eficácia:** evolução por distrito na mesma escala; ano a ano e sazonalidade; focos que se repetem; perto × longe (geral, por faixa de distância, por distrito com intervalo, dentro do mesmo bairro) e antes × depois; prazos do dado; cuidados com os dados.

Cada tela tem um quadro "Por que escolhemos esses gráficos" com a pergunta, o gráfico e o motivo, pensado para a apresentação.

## Decisões de visualização

- Cor tem função fixa: laranja é ovo e infestação, azul é EDL e comparação, vermelho é "fora do padrão, conferir", cinza é referência. Nada depende só da cor: os valores aparecem escritos nas células e nas barras (item de acessibilidade do backlog).
- O mapa pinta o Kernel Density em 5 faixas de cor (as mesmas do resto do painel), com legenda, contorno do hotspot e fundo em cinza. Os pontos de OVT começam desligados, porque três mil pontos em cima da mancha escondem a informação.
- Nada de eixo duplo. Quando duas medidas têm escala diferente, viram dois gráficos (ex.: ovos por km² e ovos por OVT).
- Os textos de leitura saem de funções em `core/insights.py` (com testes). Se entrar outro ano ou outro filtro, o texto muda junto.
- Intervalos de 95% por bootstrap nas diferenças perto × longe, para não se apoiar num p-valor isolado.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Os dados tratados já estão em `data/processed/`. Só é preciso rodar a preparação quando chegar dado novo da GEVACZ:

```bash
python scripts/preparar_dados.py --origem "<pasta INFORMAÇÕES - GEVACZ>" --destino data/processed
```

Para ler os dados tratados de outro lugar, defina `VIGIAEDL_DADOS` com o caminho da pasta. Os testes rodam com `pytest`.

## Dados

Os dados vêm do monitoramento de ovitrampas da GEVACZ (Prefeitura do Recife), usados com acordo da GEVACZ. O repositório guarda só os **dados tratados** em `data/processed/` (para o painel rodar online). Os arquivos brutos (planilhas, shapefiles, `.rar`) ficam fora, no `.gitignore`. O tratamento também descarta nome e endereço dos estabelecimentos das EDLs.

### Como entram anos anteriores (2018 em diante)

1. Coloque as pastas `OVITRAMPAS 2018`, `OVITRAMPAS 2019` etc. junto das atuais, dentro da pasta da GEVACZ.
2. Rode de novo `python scripts/preparar_dados.py --origem "<pasta GEVACZ>" --destino data/processed`.
3. Faça commit de `data/processed/`. O seletor de ano, o ano a ano por DS e o antes × depois passam a usar os novos anos.

### Quando a GEVACZ mandar a data de instalação das EDLs

Preencha `data/processed/edl_instalacao_modelo.csv` (colunas `edl_id` e `data_instalacao`, no formato dd/mm/aaaa), salve como `data/processed/edl_instalacao.csv` e faça commit. O painel passa a oferecer duas formas de definir antes e depois: a data da EDL de cada OVT (recomendada) ou uma data única (a menor, regra do plano). Pode preencher só parte das EDLs.

### Correções que o tratamento faz

- IDs de OVT padronizados; IDs do DS II reconstruídos pela ordem do arquivo só quando a contagem bate (e conferidos com a planilha de coordenadas).
- Pontos fora de Recife e duplicados removidos.
- **Datas com ano digitado errado** (ex.: dezembro de 2024 escrito como 2025, ou ano 0001) são corrigidas pelo ciclo. A data original fica em `data_original`; o relatório de qualidade conta quantas mudaram.

## Deploy (Streamlit Community Cloud)

Repositório público, branch `main`, arquivo principal `app.py`. Não precisa de segredos. Se a tabela de leituras crescer muito, o painel pode ficar pesado no plano gratuito.

## Regras de negócio (em `core/espacial.py`)

- **Distância:** sempre em metros, no sistema SIRGAS 2000 / UTM 25S (EPSG:31985). Mapas usam EPSG:4326.
- **Perto × longe:** distância até a EDL mais próxima menor ou igual ao raio (200 m por padrão; o painel deixa escolher 100, 200, 300 ou 500 m). Exatamente no limite conta como perto.
- **Antes × depois:** a leitura em data anterior ao corte é "antes"; no dia do corte ou depois é "depois". Com a data de instalação de cada EDL, as OVTs perto usam a data da EDL do próprio raio e as OVTs longe (comparação) usam uma data única; também dá para usar uma data única para todas (a menor, regra do plano). Sem data de instalação, não existe corte.
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
core/insights.py        cálculos e textos de leitura dos gráficos de análise
core/mapa.py, visual.py mapa e estilo
scripts/preparar_dados.py
tests/                  testes das regras de negócio
```
