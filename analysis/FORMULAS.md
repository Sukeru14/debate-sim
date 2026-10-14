# Fórmulas e cálculos — camada de análise

Este documento descreve, com precisão, cada cálculo estatístico usado em
`analysis/explorar.py` e `analysis/gerar_graficos.py`. Serve como referência
metodológica para a seção de métodos do TCC — cada fórmula abaixo corresponde
exatamente ao código correspondente (não é uma aproximação).

## 1. Camada de dados

Três tabelas SQLite alimentam toda a análise:

- **`eleitores`**: uma linha por eleitor por execução (`run_id`), com o
  perfil sociodemográfico, Big Five e Fundamentos Morais serializados como
  JSON e "achatados" em colunas via `pandas.json_normalize` em
  `carregar_dataframes()`.
- **`respostas`**: uma linha por eleitor por rodada (incluindo a **rodada 0**,
  o estado inicial antes do debate — ver seção 6), com `intencao_voto`,
  `confianca` e `mudou_de_posicao`.
- **`falas`**: uma linha por fala de candidato por rodada, com a
  `estrategia` argumentativa usada nessa fala.

## 2. Notação

Para o eleitor $i$, seja $v_{i,t} \in \{A, B, \text{indeciso}\}$ sua
intenção de voto na rodada $t \in \{0, 1, \dots, T\}$, onde $t=0$ é o
estado inicial (sempre "indeciso", ver seção 6) e $T$ é o número de
rodadas de debate. Seja $c_{i,t} \in [0,1]$ a confiança relatada pelo
eleitor $i$ na rodada $t$.

## 3. Número de mudanças de intenção de voto (`n_mudancas`)

**Código-fonte**: `mudou_de_posicao` é calculado no momento em que cada
resposta é salva (`AgenteEleitor.avaliar_rodada`, em `debate/agentes.py`),
comparando a nova intenção contra o estado anterior mantido em memória
pelo agente:

$$
\delta_{i,t} = \mathbb{1}\left[v_{i,t} \neq v_{i,t-1}\right], \qquad t = 1, \dots, T
$$

Depois, `resumo_mudancas_por_eleitor()` agrega por eleitor:

$$
\text{n\_mudancas}_i = \sum_{t=1}^{T} \delta_{i,t}
$$

(`respostas.groupby("eleitor_id")["mudou_de_posicao"].sum()`)

**Observações**:
- A rodada 0 nunca entra no somatório porque `mudou_de_posicao` é sempre
  `False` por construção nela (é o ponto de partida, não uma reação).
- Isso é uma contagem de **transições**, não de "quantas vezes esteve em
  cada candidato" — um eleitor que oscila A→B→A→B em 4 rodadas tem
  `n_mudancas = 4`, o mesmo valor de um eleitor que só mudou A→B uma vez e
  ficou parado — a métrica não distingue estabilidade final de oscilação.
  Se isso importar para a análise final, considerar separar "mudanças
  totais" de "mudança líquida" ($v_{i,T} \neq v_{i,0}$).

## 4. Confiança média por eleitor (`confianca_media`)

$$
\bar c_i = \frac{1}{T}\sum_{t=1}^{T} c_{i,t}
$$

(`respostas_debate.groupby("eleitor_id")["confianca"].mean()`, onde
`respostas_debate` filtra `rodada > 0`)

A rodada 0 é **excluída** deste cálculo — sua confiança é um placeholder
fixo (`0.0`, ver seção 6), não uma confiança real avaliada pelo modelo, e
entrar na média enviesaria o valor para baixo artificialmente.

## 5. Correlação traço × mudanças (`correlacao_tracos_x_mudancas`)

**Coeficiente de correlação de Pearson**, calculado via
`pandas.DataFrame.corr()` (método padrão = `'pearson'`), entre cada traço
$X$ (um dos 5 fatores do Big Five ou uma das 5 fundações morais) e a
variável $Y = \text{n\_mudancas}$, através dos $N$ eleitores da execução:

$$
r_{X,Y} = \frac{\sum_{i=1}^{N} (X_i - \bar X)(Y_i - \bar Y)}
               {\sqrt{\sum_{i=1}^{N}(X_i-\bar X)^2}\;\sqrt{\sum_{i=1}^{N}(Y_i-\bar Y)^2}}
$$

**Limitações a declarar explicitamente no TCC**:
- É correlação **linear simples** (não parcial, não controla por outros
  traços simultaneamente nem por variáveis sociodemográficas).
- Não implica causalidade.
- Com $N$ pequeno (protótipos de 5–20 eleitores), a estimativa de $r$ tem
  variância alta — mudar a seed pode inverter o sinal. Como referência
  geral (não uma regra rígida), estimativas de correlação costumam
  estabilizar a partir de amostras na casa de several dezenas a
  ~100 observações; abaixo disso, trate $r$ como direção exploratória,
  não como evidência.
- Sensível a outliers (um único eleitor com `n_mudancas` muito alto pode
  dominar o coeficiente).

## 6. Rodada 0 (estado inicial)

Antes de qualquer rodada de debate, `OrquestradorDebate._registrar_estado_inicial()`
grava, para cada eleitor, uma linha com:

$$
v_{i,0} = \text{"indeciso"}, \qquad c_{i,0} = 0{,}0 \ (\text{placeholder}), \qquad \delta_{i,0} = \text{False}
$$

Isso **não** chama o modelo — é puramente determinístico, refletindo
`persona.intencao_voto_inicial`. Existe para que os gráficos de evolução de
voto tenham um ponto de partida real (100% indecisos) em vez de a primeira
observação registrada já ser uma reação pós-debate.

## 7. Evolução da intenção de voto por rodada

$$
N_{t,v} = \left|\{\, i : v_{i,t} = v \,\}\right|, \qquad v \in \{A, B, \text{indeciso}\}
$$

(`respostas.groupby(["rodada","intencao_voto"]).size().unstack(fill_value=0)`)

**Nota**: são **contagens absolutas**, não proporções. Se o número de
eleitores variar entre execuções (ex: comparando uma run de 10 com uma de
30 eleitores), os gráficos não são diretamente comparáveis em escala —
para comparar entre execuções de tamanhos diferentes, normalizar para
proporção ($N_{t,v} / N_{\text{total}}$) antes de plotar.

## 8. Confiança média e desvio padrão por rodada

$$
\bar c_t = \frac{1}{N}\sum_{i=1}^{N} c_{i,t}, \qquad
s_t = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N} (c_{i,t}-\bar c_t)^2}
$$

(`.groupby("rodada")["confianca"].mean()` e `.std()`)

**Nota técnica importante**: `pandas.Series.std()` usa por padrão
$\text{ddof}=1$ (desvio padrão **amostral**, com correção de Bessel,
divide por $N-1$ não por $N$) — relevante de declarar se o TCC reportar
esse desvio como estatística descritiva formal. A rodada 0 é excluída
deste cálculo pelo mesmo motivo da seção 4.

## 9. Mudanças por estratégia argumentativa (exploratório, confundido)

Este é o cálculo com a limitação mais importante de deixar explícita.

Seja $S_t$ o conjunto de estratégias distintas usadas nas (até 4) falas
de candidato da rodada $t$ (`falas.groupby("rodada")["estrategia"].apply(list)`,
depois reduzido a valores únicos por rodada). Seja

$$
M_t = \sum_{i=1}^{N} \delta_{i,t}
$$

o total de eleitores que mudaram de posição na rodada $t$ (agregado,
**não separado por qual candidato/estratégia "causou" a mudança**). Para
cada estratégia $s$:

$$
\overline{M}_s = \frac{1}{|\{t : s \in S_t\}|} \sum_{t\,:\,s \in S_t} M_t
$$

(`gerar_graficos.plot_mudancas_por_estrategia`)

**Por que isso é confundido, e por quê isso importa para a RQ3**: como o
sorteio de estratégia (no design atual, antes da mudança descrita na
seção seguinte) ocorre a cada fala individual, uma mesma rodada
tipicamente contém 2 a 4 estratégias diferentes misturadas. $M_t$ é o
total de mudanças da rodada inteira, não de uma estratégia isolada —
então $\overline{M}_s$ mede "quantas mudanças aconteceram em rodadas
onde essa estratégia apareceu pelo menos uma vez", não o efeito causal
daquela estratégia. Isso está documentado como aviso na docstring da
função e é o motivo pelo qual o design experimental de estratégia fixa
por execução (ver mudança de código separada) é necessário para uma
resposta mais limpa à RQ3.

## 10. O que ainda não existe (a implementar na análise final do TCC)

Os cálculos acima são estatística **descritiva** — adequados para
inspeção exploratória e para a apresentação, não para as conclusões
finais do TCC. Para RQ1/RQ2 formalmente, o caminho natural é um modelo de
regressão (ex: regressão logística multinomial para prever
`intencao_voto` a partir dos traços, ou um modelo de efeitos mistos com
eleitor aninhado em execução para RQ4), que ainda não está implementado
neste repositório.
