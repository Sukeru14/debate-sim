# Simulação de debate eleitoral com agentes LLM

Dois candidatos debatem, um mediador faz as perguntas e N eleitores
simulados assistem e atualizam a intenção de voto a cada rodada. Os
eleitores combinam perfil sociodemográfico real (PNAD Contínua/IBGE),
Big Five e Fundamentos Morais.

## Pré-requisitos

1. Python 3.10+
2. [Ollama](https://ollama.com) rodando (`ollama serve`) com o modelo: `ollama pull llama3.1:8b`
3. `pip install -r requirements.txt`

## Pipeline

### 1. Gerar os eleitores

```bash
python gerar_personas.py --n 30 --arquivo dados/PNADC_012026.txt --out personas/lote_01.json
```

- O perfil (idade, gênero, UF/região, raça/cor, urbano/rural) é sorteado do
  arquivo bruto da PNAD Contínua (FTP do IBGE), com peso amostral.
  Escolaridade e renda ainda vêm de distribuições aproximadas (ver
  docstring de `persona/gerador_estatistico_pnadc_bruto.py`).
- Alternativa: `--fonte-perfil csv --arquivo recorte.csv` com um recorte
  extraído via Base dos Dados (`extrair_microdados.py`, exige Google Cloud).
- Big Five e Fundamentos Morais são **sorteados**, não gerados pelo LLM
  (`--fonte-tracos independente|correlacionado`). O LLM só escreve a
  narrativa da persona.

O JSON gerado é legível e pode ser editado à mão.

### 2. Rodar o experimento

```bash
python main.py --personas personas/lote_01.json --rodadas 5
```

Roda os 3 confrontos com o **mesmo** lote de eleitores e salva tudo em
`storage/experimento.db`. Opções principais:

| Opção | Efeito |
|---|---|
| `--confrontos emocional:logico ...` | roda só os confrontos indicados |
| `--contrabalancear` | repete cada confronto com os rótulos 1/2 invertidos |
| `--repeticoes N` | N repetições por confronto (seed muda a cada uma) |
| `--seed`, `--temperatura`, `--modelo` | configuração do LLM |
| `--max-rodadas-historico N` | rodadas anteriores enviadas como contexto (0 = todas) |

### 3. Conferir os dados

```bash
python -m analysis.explorar --db storage/experimento.db
```

## Desenho experimental

### Tipos de candidato (`debate/candidatos.py`)

Baseados nos três modos de persuasão da *Retórica* de Aristóteles:

| Tipo | Modo | Como convence | Evita |
|---|---|---|---|
| `emocional` | pathos | histórias de pessoas comuns, empatia, esperança/indignação | números, currículo |
| `logico` | logos | diagnóstico → proposta → resultado, números e metas | apelos emocionais, currículo |
| `credibilidade` | ethos | trajetória, integridade, compromissos cumpridos | estatísticas, dramatização |

**Controle:** os três têm a mesma trajetória e a mesma plataforma, com
propostas neutras para cada tema. Só muda a forma de argumentar. Os
eleitores veem apenas "Candidato 1" e "Candidato 2", e o tipo fica
registrado só no banco.

### Estrutura de cada rodada

```
mediador pergunta (tema fixo) -> abertura do 1º -> abertura do 2º
  -> réplica do 1º -> tréplica do 2º -> eleitores avaliam (em paralelo)
```

- Quem abre alterna a cada rodada, porque a Lei 9.504/1997 define a ordem
  de fala por sorteio.
- Todos os eleitores começam indecisos (rodada 0, gravada sem chamar o LLM).
- O eleitor recebe o debate recente e a própria intenção/confiança da
  rodada anterior, e devolve `intencao_voto`, `confianca` e `justificativa`.
  O voto é validado e gravado como o id do tipo escolhido.

### Banco (`storage/db.py`)

| Tabela | Conteúdo |
|---|---|
| `runs` | uma linha por confronto/repetição: tipos, modelo, seed, temperatura |
| `eleitores` | persona de cada eleitor em cada execução |
| `falas` | pergunta do mediador e falas dos candidatos (com tipo e ordem) |
| `respostas` | intenção de voto de cada eleitor por rodada |

O mesmo `eleitor_id` aparece em várias execuções: sempre cruzar por
`(run_id, eleitor_id)`.

## Estrutura

```
gerar_personas.py          etapa 1: gera o lote de eleitores
main.py                    etapa 2: roda os confrontos
schemas.py                 dataclasses e I/O do JSON de personas
llm/client.py              cliente Ollama (texto e JSON validado)
debate/candidatos.py       tipos de candidato, trajetória e plataforma
debate/agentes.py          mediador, candidato, eleitor (prompts)
debate/orquestrador.py     loop de rodadas
persona/                   perfil (PNAD bruta ou CSV) e traços subjetivos
storage/db.py              SQLite
analysis/explorar.py       carregamento dos dados para análise
extrair_microdados.py      (opcional) recorte da PNAD via Base dos Dados
```

## Limitações conhecidas

1. Escolaridade e renda são sorteadas de forma independente do resto do
   perfil (caminho da PNAD bruta).
2. Os traços subjetivos não dependem do perfil demográfico; a matriz de
   correlação da versão correlacionada é ilustrativa.
3. O candidato `logico` usa números estimados pelo próprio modelo, e não
   dados reais.
4. Modelos ~8B podem ser pouco sensíveis a nuances de persona. Vale
   rodar um teste de sensibilidade com personas de traços opostos.
5. Mesmo com seed fixa, avaliar eleitores em paralelo pode tornar a saída
   do Ollama não determinística. Para reprodutibilidade estrita, usar
   `--workers 1`.
6. `mudou_de_posicao` na rodada 1 marca a primeira decisão (indeciso → X),
   não uma mudança de opinião.
