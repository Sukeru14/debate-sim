# Simulação de debate eleitoral com agentes LLM — protótipo

## Pré-requisitos

1. Python 3.10+
2. Ollama instalado e rodando (`ollama serve`)
3. Modelo baixado: `ollama pull llama3.1:8b`
4. Dependências: `pip install -r requirements.txt`

## Pipeline (duas etapas separadas)

### Etapa 1 — gerar personas dos eleitores

```bash
python gerar_personas.py --n 10 --out personas/lote_01.json --seed 42
```

Gera o perfil sociodemográfico (simulado por padrão — use
`--fonte-perfil pnadc_bruto --arquivo-pnadc caminho/arquivo.txt` para
usar microdados reais da PNAD Contínua, ver seção abaixo), os traços
subjetivos (Big Five + Fundamentos Morais, amostrados estatisticamente)
e a narrativa (gerada pelo LLM), e salva tudo num JSON legível em
`personas/lote_01.json`. Abra esse arquivo para inspecionar ou editar
manualmente uma persona específica antes de rodar o debate.

### Etapa 2 — rodar a simulação de debate

```bash
python main.py --personas personas/lote_01.json --rodadas 5
```

Carrega o lote de personas já gerado e roda a simulação. Como as etapas
são separadas, o MESMO lote de eleitores pode ser reaproveitado em
várias simulações (ex: testando temas ou candidatos diferentes),
controlando essa variável entre execuções.

Opções úteis de `main.py`:
- `--rodadas N` — número de rodadas (cicla pela lista de temas se N > nº de temas)
- `--seed N` — semente para reprodutibilidade (RQ4)
- `--alternar-abertura` — alterna qual candidato abre cada rodada (em vez de A sempre abrir)
- `--db caminho.db` — onde salvar o SQLite desta execução

### Estrutura fixa de cada rodada

```
moderador formula pergunta (a partir de um tema fixo)
  -> candidato 1 responde (abertura)
  -> candidato 2 responde (abertura)
  -> candidato 1 replica (reage à fala do candidato 2)
  -> candidato 2 treplica (reage à réplica do candidato 1)
  -> todos os eleitores avaliam a rodada inteira, em paralelo
```

Os temas ficam em `TEMAS_PADRAO` dentro de `main.py` — edite essa lista
para os temas que você quer cobrir no seu debate.

### Análise exploratória

```bash
python -m analysis.explorar
```

### Gráficos

```bash
python -m analysis.gerar_graficos --db storage/debate_sim.db --out graficos/
```

Gera 5 imagens em `graficos/`:
- `correlacao_tracos.png` — correlação entre cada traço (Big Five/MFT) e número de mudanças de voto
- `ranking_mudancas_por_eleitor.png` — ranking de eleitores por nº de mudanças (bom para apontar exemplos)
- `evolucao_intencao_voto.png` — quantos eleitores estão em A/B/indeciso a cada rodada
- `confianca_por_rodada.png` — confiança média ± desvio padrão por rodada
- `mudancas_por_estrategia.png` — mudanças de voto associadas a cada estratégia argumentativa (exploratório — ver docstring da função, não é causal porque várias estratégias podem coexistir na mesma rodada)

**Todas as fórmulas e cálculos exatos usados nesses gráficos e nas tabelas
de `analysis/explorar.py` estão documentados, com notação matemática
formal, em [`analysis/FORMULAS.md`](analysis/FORMULAS.md)** — inclui
limitações de cada métrica (ex: por que a correlação é só exploratória
com N pequeno, por que o gráfico de estratégia é confundido). Use isso
como referência para a seção de metodologia do TCC.

## Traços de personalidade com correlação (alternativa à amostragem independente)

Por padrão, Big Five e Fundamentos Morais são amostrados de forma
totalmente independente entre si (`GeradorTracosSubjetivos`). Existe uma
alternativa que respeita a estrutura de correlação documentada na
literatura psicométrica:

```bash
python gerar_personas.py --n 20 --out personas/lote_01.json --fonte-tracos correlacionado
```

- **Big Five**: usa a estrutura de metatraços Stability (conscienciosidade
  + amabilidade + baixo neuroticismo) / Plasticity (extroversão +
  abertura), documentada em meta-análise (van der Linden, te Nijenhuis &
  Segers, 2010, *Journal of Research in Personality*; DeYoung, Peterson &
  Higgins, 2002; Digman, 1997).
- **Fundamentos Morais**: usa a estrutura individualizing (cuidado +
  justiça) / binding (lealdade + autoridade + pureza), de Graham, Haidt &
  Nosek (2009), replicada em amostras WEIRD e não-WEIRD e numa validação
  brasileira do MFQ (estrutura bifatorial confirmada, N=341 e N=371).
- **`--condicionar-religiosidade`** (opt-in, desligado por padrão): **sem
  efeito no momento** — religião foi retirada do perfil (ver seção
  "Migrando para microdados reais", item 5). Quando ativa, desloca
  levemente a média dos fundamentos binding conforme a religião do
  perfil, informado pela correlação real (r=0,36, p<0,01) entre o fator
  binding e religiosidade organizacional encontrada na validação
  brasileira do MFQ — reative quando religião voltar a ser modelada.

Ver docstring completa em `persona/gerador_subjetivo_correlacionado.py`
para as matrizes exatas e a limitação importante: os valores numéricos
são **ilustrativos** (direção e magnitude relativa da literatura geral),
não os coeficientes exatos de um único estudo — validar/substituir por
uma fonte específica antes de citar como final no TCC.

## Estratégia argumentativa fixa por execução (para RQ3)

Por padrão, cada candidato sorteia a estratégia argumentativa a cada
fala (comportamento histórico, mantido). Para comparar estratégias de
forma causal (sem o confundimento de várias estratégias coexistindo na
mesma rodada — ver `analysis/FORMULAS.md`, seção 9), fixe uma estratégia
por candidato para a execução inteira:

```bash
python main.py --personas personas/lote_01.json \
    --estrategia-a dados_e_logica --estrategia-b apelo_emocional
```

Rode uma execução por combinação de estratégias que quiser comparar,
sempre com o **mesmo arquivo `--personas`** (o pipeline já foi desenhado
para isso desde que a geração de persona foi separada da simulação),
variando só `--estrategia-a`/`--estrategia-b` (e `--seed`, se quiser
também dados para RQ4). Estratégias válidas: `apelo_emocional`,
`dados_e_logica`, `ataque_ao_oponente`, `apelo_a_valores_morais`,
`apelo_a_autoridade`.

## Experimento de espectro político (3 candidatos, 3 confrontos)

Em vez de um par único de candidatos fixos, `rodar_embate_espectro.py`
define 3 candidatos representando posições de direita, esquerda e centro
(cada um já com plataforma e estratégia argumentativa embutidas na
persona — não sorteada, não alternada por fala) e roda automaticamente
os 3 confrontos par-a-par possíveis entre eles, reaproveitando o mesmo
lote de eleitores em todos:

```bash
python rodar_embate_espectro.py --personas personas/lote_01.json --rodadas 5
```

Gera `storage/embate_direita_vs_esquerda.db`,
`storage/embate_direita_vs_centro.db` e
`storage/embate_esquerda_vs_centro.db`. Rode os gráficos separadamente
para cada um (`python -m analysis.gerar_graficos --db storage/embate_direita_vs_esquerda.db --out graficos/direita_esquerda/`).

**Base da escolha de estratégia por posição** (declarar como pressuposto
de modelagem no TCC, não como fato neutro): direita e esquerda usam a
MESMA categoria de estratégia (`apelo_a_valores_morais`) — o que muda é
a QUAL valor moral cada um apela (texto embutido no `estilo_retorico`),
informado por Graham, Haidt & Nosek (2009): a fundação "individualizing"
(cuidado, justiça) tende a ser relativamente mais saliente à esquerda,
enquanto as fundações "binding" (lealdade, autoridade, pureza) recebem
peso mais equilibrado — e frequentemente mais enfatizado — à direita.
O centro usa `dados_e_logica`, refletindo um posicionamento tipicamente
mais técnico/pragmático. Ver docstring completa em
`rodar_embate_espectro.py` para a justificativa e para editar os textos
das plataformas.

## Ordem de abertura (padrão: alterna a cada rodada)

Por padrão, o candidato que abre cada rodada alterna (A na rodada 1, B na
rodada 2, ...). Isso reflete o formato real de debates eleitorais no
Brasil, onde a Lei das Eleições (Lei nº 9.504/1997) determina que a
ordem de fala é definida por **sorteio**, salvo acordo em contrário —
ou seja, não há uma ordem fixa privilegiando sistematicamente um
candidato. Para desativar e deixar o candidato A sempre abrir:

```bash
python main.py --personas personas/lote_01.json --fixar-abertura
```

## Estrutura do projeto

```
debate_sim/
├── gerar_personas.py              # ETAPA 1: gera e salva personas em JSON
├── main.py                        # ETAPA 2: carrega personas, roda o debate
├── schemas.py                     # dataclasses centrais + I/O JSON de personas
├── llm/
│   └── client.py                  # abstração sobre a API do Ollama
├── persona/
│   ├── gerador_estatistico.py            # perfil sociodemo SIMULADO (distribuições aproximadas)
│   ├── gerador_estatistico_pnadc_bruto.py # perfil sociodemo de ARQUIVO BRUTO da PNAD (FTP do IBGE)
│   ├── gerador_estatistico_real.py        # perfil sociodemo via Base dos Dados/BigQuery
│   └── gerador_subjetivo.py               # Big Five + Fundamentos Morais + narrativa (LLM)
├── debate/
│   ├── agentes.py                  # AgenteCandidato, AgenteEleitor, AgenteModerador
│   └── orquestrador.py             # loop de rodadas com estrutura fixa moderador/réplica
├── storage/
│   └── db.py                       # persistência SQLite
├── personas/                       # JSONs de lotes de eleitores gerados (criar a pasta)
└── analysis/
    └── explorar.py                 # análise exploratória simples
```

## Migrando para microdados reais (PNAD/Censo)

Hoje `gerador_estatistico.py` usa distribuições aproximadas. Há dois
caminhos para usar dados reais — escolha o que fizer mais sentido pra você.

### Caminho A — arquivo bruto do FTP do IBGE (sem precisar de Google Cloud)

Esse é o caminho recomendado se você não quer lidar com cartão de
crédito/billing do Google Cloud (ver Caminho B).

1. Baixe o `.txt` de microdados trimestral básico direto do FTP do IBGE:
   `https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/<ano>/`
   (arquivo de largura fixa, ~3478 caracteres por linha no layout atual).

2. Use `persona/gerador_estatistico_pnadc_bruto.py`
   (`GeradorPerfilSociodemograficoPNADCBruto`) — faz o parsing direto do
   arquivo, sem dependências externas além do que já está no
   `requirements.txt`. Já vem com as posições de coluna confirmadas
   empiricamente contra um arquivo real (ver docstring do módulo para a
   tabela completa de posições).

   ```bash
   python gerar_personas.py --n 20 --out personas/lote_01.json \
       --fonte-perfil pnadc_bruto --arquivo-pnadc caminho/para/PNADC_MMAAAA.txt
   ```

3. **O que vem real**: idade, sexo, UF/região, raça/cor, situação do
   domicílio (urbano/rural) — amostrados respeitando o peso amostral
   real da pesquisa (V1028), igual a uma amostragem estatisticamente
   correta da população brasileira.

4. **O que continua aproximado**: escolaridade e renda. As posições de
   `VD3004` (escolaridade) e `VD4020` (renda) no arquivo bruto não foram
   confirmadas com segurança — são variáveis derivadas que ficam em uma
   região do registro com várias outras variáveis categóricas parecidas,
   e adivinhar a posição errada daria valores incorretos *sem erro
   aparente* (mais perigoso que não ter o dado). Se quiser completar
   isso com dados reais depois, busque o dicionário oficial em Excel no
   FTP do IBGE
   (`Trimestral/Microdados/Documentacao/Variaveis_PNADC_Trimestral.xls`)
   para confirmar a posição exata antes de usar — e valide o resultado
   contra estatísticas conhecidas (ex: ~20% com superior completo,
   renda média ~R$2500) antes de confiar nos valores extraídos.

5. **Religião**: RETIRADA do perfil por decisão do orientador (a PNAD
   Contínua regular não pergunta religião, e não há hoje outra fonte
   real confiável para essa variável). Todo perfil traz
   `religiao="não_modelada"`, valor que não aparece em nenhum prompt
   enviado ao modelo. Volta a ser modelada quando houver uma fonte real
   — nesse caso, basta trocar o valor constante por
   `_sortear(DISTRIBUICAO_RELIGIAO)` nos 3 geradores de perfil (ver
   comentários em `persona/gerador_estatistico.py`) e reintroduzir a
   linha correspondente em `PersonaEleitor.to_prompt_block()` e em
   `gerar_narrativa()`.

### Caminho B — Base dos Dados / BigQuery (requer Google Cloud com billing)

1. **Crie um projeto no Google Cloud com billing ativado** (Console GCP →
   criar projeto → ativar faturamento; exige cartão, mas o BigQuery dá
   1 TB de processamento gratuito por mês — não chega a custar nada
   para o volume deste projeto). O `billing_id` é o **Project ID**
   (ex: `meu-projeto-tcc-123456`), não o nome de exibição.

2. **Instale e autentique**: `pip install basedosdados --break-system-packages`
   (a primeira chamada abre o navegador para login Google).

3. **Confirme o schema antes de extrair** (evita gastar cota com query errada):
   ```bash
   python extrair_microdados.py --billing-id seu-project-id --inspecionar
   ```

4. **Extraia um recorte**:
   ```bash
   python extrair_microdados.py --billing-id seu-project-id --ano 2024 --trimestre 4 --n 5000 --out dados/pnadc_recorte.csv
   ```

5. **Plugue no sistema**: use `GeradorPerfilSociodemograficoReal` em
   `persona/gerador_estatistico_real.py` (interface igual: `amostrar()`,
   `amostrar_lote(n)`). Essa via traz os valores já traduzidos via
   dicionário (texto, não código numérico), o que inclusive permite
   também resolver escolaridade/renda com mais segurança do que no
   Caminho A, já que a query já vem com os dicionários aplicados.

## Decisões de design já tomadas (e por quê)

- **Pipeline em duas etapas** (gerar personas → rodar debate): permite
  inspecionar/editar personas manualmente e reusar o mesmo lote em
  múltiplas simulações, controlando essa variável.
- **Personas salvas em JSON legível** (`schemas.salvar_eleitores_json` /
  `carregar_eleitores_json`), para inspeção e edição manual fácil.
- **Big Five e Fundamentos Morais são amostrados estatisticamente** (não
  gerados pelo LLM); só a narrativa é gerada pela IA — evita que o
  modelo regrida a estereótipos demográficos. Ver docstring em
  `persona/gerador_subjetivo.py`.
- **Estrutura fixa de rodada** (moderador → abertura A → abertura B →
  réplica A → tréplica B), espelhando o formato real de debate eleitoral
  — aumenta validade ecológica da simulação frente a um bate-papo livre.
- **Tema da rodada é fixo (lista pré-definida)**, só a pergunta exata é
  gerada pelo LLM a partir do tema — dá controle experimental sobre o
  que foi discutido em cada rodada.
- **Estratégia argumentativa do candidato é categoria fechada**, para
  viabilizar a RQ3 sem precisar de um classificador extra depois.
- **Eleitores avaliados em paralelo (threads)** dentro de cada rodada,
  já que não interagem entre si.
- **`run_id` em todas as tabelas**, para suportar RQ4 (múltiplas execuções).

## Limitações conhecidas (documentar no TCC)

1. O perfil sociodemográfico hoje (antes de integrar microdados reais)
   usa distribuições marginais aproximadas, sorteadas de forma
   aproximadamente independente — não captura a distribuição conjunta
   real da população brasileira.
2. Big Five e Fundamentos Morais não têm correlação com o perfil
   sociodemográfico (amostrados independentemente). Correlatos
   demográficos documentados na literatura podem ser incorporados depois.
3. Llama 3.1 8B pode ter sensibilidade limitada a instruções de persona
   sutis — fazer um teste de sensibilidade (comparar respostas de
   personas com traços opostos para o mesmo trecho de debate) antes de
   tirar conclusões fortes.
4. A ordem de abertura (`alternar_abertura=False` por padrão) pode ser,
   ela mesma, uma variável de confundimento — considerar alternar ou
   controlar estatisticamente por ordem na análise.
