"""
Camada SUBJETIVA do gerador de persona -- versão com CORRELAÇÃO ENTRE
TRAÇOS, como alternativa a `gerador_subjetivo.GeradorTracosSubjetivos`
(que amostra cada um dos 10 traços de forma totalmente independente).

MESMA INTERFACE que a versão independente: `amostrar_big_five()` e
`amostrar_fundamentos_morais(perfil=None)`. Troca direta em
`gerar_personas.py` via `--fonte-tracos correlacionado`.

POR QUE ISSO EXISTE (contexto da pergunta "está correto ou tem forma
melhor, tipo o approach do IBGE?"):

Ao contrário do perfil sociodemográfico, NÃO existe um "microdado real"
de Big Five + Fundamentos Morais para a população brasileira que se possa
baixar e amostrar (como fizemos com a PNAD) -- não há uma fonte de dados
individual, pareada, pública. Então esta versão não é "dados reais" como
o gerador PNAD; é uma melhoria estrutural: em vez de tratar os 10 traços
como estatisticamente independentes entre si (o que a psicometria sabe
que é falso), amostra-os de uma NORMAL MULTIVARIADA cuja matriz de
correlação reflete relações empiricamente documentadas na literatura.

BASE NA LITERATURA:

1. Big Five NÃO são ortogonais entre si. A meta-análise de van der Linden,
   te Nijenhuis & Segers (2010, Journal of Research in Personality; K=212
   estudos, N=144.117) encontrou suporte para dois metatraços:
   - "Stability": Conscienciosidade + Amabilidade + (baixo) Neuroticismo
     carregam juntos (DeYoung, Peterson & Higgins, 2002; Digman, 1997).
   - "Plasticity": Extroversão + Abertura carregam juntos.
   A matriz abaixo usa essa estrutura qualitativa (sinais e magnitudes
   relativas), NÃO os valores numéricos exatos do paper -- ver seção
   "LIMITAÇÃO" abaixo.

2. Fundamentos Morais também não são ortogonais. Graham, Haidt & Nosek
   (2009, Journal of Personality and Social Psychology) mostram que os 5
   fundamentos se agrupam em dois clusters -- "individualizing" (cuidado/
   dano + justiça/reciprocidade, que correlacionam fortemente entre si) e
   "binding" (lealdade + autoridade + pureza, que também correlacionam
   fortemente entre si) -- com correlação cruzada mais fraca entre os
   clusters. Essa estrutura de 2 fatores foi replicada em amostras WEIRD
   e não-WEIRD, e também numa validação brasileira do MFQ (traduzido por
   Silvino et al.; validado com evidência de estrutura bifatorial em
   dois estudos, N=341 e N=371 adultos brasileiros).

3. Religiosidade correlaciona com os fundamentos "binding": a mesma
   validação brasileira do MFQ encontrou correlação moderada e
   significativa (r=0,36, p<0,01) entre o fator binding/coesivo e uma
   medida de religiosidade organizacional. Por isso esta classe permite
   (opcionalmente, default DESLIGADO) um leve deslocamento da média dos
   fundamentos binding para personas com religião declarada -- pequeno o
   suficiente para não dominar a variância individual (evitando o
   problema de estereótipo que já discutimos para a narrativa), mas na
   direção e ordem de grandeza documentadas.

LIMITAÇÃO A DECLARAR NO TCC: as matrizes de correlação abaixo são
ILUSTRATIVAS -- refletem a DIREÇÃO e ORDEM DE GRANDEZA relativa
qualitativa reportada na literatura (metatraços Stability/Plasticity;
clusters individualizing/binding), não os valores numéricos exatos de um
único estudo específico. Antes de usar isso como métrica final do TCC,
vale substituir por uma matriz de correlação citável de uma fonte
específica (idealmente um estudo brasileiro com a BFP -- Bateria Fatorial
de Personalidade, Nunes/Hutz -- e o MFQ brasileiro), ou reportar
explicitamente que a matriz é uma aproximação baseada na literatura
geral, com a citação correspondente.
"""

from dataclasses import dataclass, field

import numpy as np

from schemas import BigFive, FundamentosMorais, PerfilSociodemografico


ORDEM_BIG_FIVE = ["abertura", "conscienciosidade", "extroversao", "amabilidade", "neuroticismo"]

# Estrutura Stability (conscienciosidade, amabilidade, ~neuroticismo) /
# Plasticity (extroversao, abertura) -- van der Linden et al. (2010);
# DeYoung, Peterson & Higgins (2002); Digman (1997).
CORRELACAO_BIG_FIVE = np.array([
    #      abert  consc  extra  amab   neur
    [ 1.00,  0.05,  0.20,  0.10, -0.05],  # abertura
    [ 0.05,  1.00,  0.15,  0.20, -0.25],  # conscienciosidade
    [ 0.20,  0.15,  1.00,  0.15, -0.25],  # extroversao
    [ 0.10,  0.20,  0.15,  1.00, -0.15],  # amabilidade
    [-0.05, -0.25, -0.25, -0.15,  1.00],  # neuroticismo
])

ORDEM_MFT = ["cuidado_dano", "justica_reciprocidade", "lealdade_grupo", "autoridade_respeito", "pureza_santidade"]

# Estrutura individualizing (cuidado, justica) / binding (lealdade,
# autoridade, pureza) -- Graham, Haidt & Nosek (2009); replicado em
# validação brasileira do MFQ (estrutura bifatorial "individualizante"
# e "coesivo").
CORRELACAO_MFT = np.array([
    #      cuid   just   leal   auto   pure
    [1.00, 0.55, 0.15, 0.10, 0.10],  # cuidado_dano
    [0.55, 1.00, 0.15, 0.10, 0.10],  # justica_reciprocidade
    [0.15, 0.15, 1.00, 0.55, 0.50],  # lealdade_grupo
    [0.10, 0.10, 0.55, 1.00, 0.55],  # autoridade_respeito
    [0.10, 0.10, 0.50, 0.55, 1.00],  # pureza_santidade
])

# Deslocamento leve da média dos fundamentos BINDING por religiosidade,
# informado (não copiado exatamente) pelo r=0,36 encontrado na validação
# brasileira do MFQ para religiosidade organizacional. Fica DESLIGADO por
# padrão (condicionar_religiosidade=False) -- é opt-in.
DESLOCAMENTO_BINDING_POR_RELIGIAO = {
    "catolica": 0.06,
    "evangelica": 0.10,   # religiosidade organizacional tende a ser mais alta
    "espirita": 0.04,
    "outras": 0.04,
    "sem_religiao": -0.05,
}
FUNDACOES_BINDING = {"lealdade_grupo", "autoridade_respeito", "pureza_santidade"}


def _amostrar_multivariado(
    rng: np.random.Generator,
    media: float,
    desvio: float,
    matriz_correlacao: np.ndarray,
    deslocamentos: dict[str, float] | None = None,
    ordem: list[str] | None = None,
) -> np.ndarray:
    """Amostra de uma normal multivariada com a matriz de correlação dada,
    escalada pelo desvio padrão comum a todos os traços, depois recorta
    para [0,1] (mesma faixa válida usada em todo o sistema)."""
    n = matriz_correlacao.shape[0]
    medias = np.full(n, media)
    if deslocamentos and ordem:
        for idx, nome in enumerate(ordem):
            medias[idx] += deslocamentos.get(nome, 0.0)
    cov = np.outer(np.full(n, desvio), np.full(n, desvio)) * matriz_correlacao
    amostra = rng.multivariate_normal(mean=medias, cov=cov)
    return np.clip(amostra, 0.0, 1.0)


@dataclass
class GeradorTracosSubjetivosCorrelacionado:
    """
    Alternativa a GeradorTracosSubjetivos (independente) que amostra Big
    Five e Fundamentos Morais de uma normal multivariada com correlação
    entre traços informada pela literatura (ver docstring do módulo).

    `condicionar_religiosidade`: se True, desloca levemente a média dos
    fundamentos binding conforme a religião do perfil (ver
    DESLOCAMENTO_BINDING_POR_RELIGIAO). Desligado por padrão -- é uma
    decisão metodológica que vale tomar conscientemente, não como default
    silencioso.
    """
    seed: int | None = None
    desvio: float = 0.18  # mesmo desvio usado na versão independente
    condicionar_religiosidade: bool = False

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        if self.condicionar_religiosidade:
            print(
                "[GeradorTracosSubjetivosCorrelacionado] AVISO: "
                "condicionar_religiosidade=True, mas religião foi retirada do "
                "perfil sociodemográfico (decisão do orientador -- ver README). "
                "Todo perfil chega aqui com religiao='não_modelada', que não "
                "consta em DESLOCAMENTO_BINDING_POR_RELIGIAO -- o deslocamento "
                "será sempre 0.0, ou seja, esta flag está sem efeito prático "
                "agora. Reative quando religião voltar a ser amostrada."
            )

    def amostrar_big_five(self) -> BigFive:
        valores = _amostrar_multivariado(self._rng, 0.5, self.desvio, CORRELACAO_BIG_FIVE)
        return BigFive(**dict(zip(ORDEM_BIG_FIVE, valores)))

    def amostrar_fundamentos_morais(self, perfil: PerfilSociodemografico | None = None) -> FundamentosMorais:
        deslocamentos = None
        if self.condicionar_religiosidade and perfil is not None:
            desloc_religiao = DESLOCAMENTO_BINDING_POR_RELIGIAO.get(perfil.religiao, 0.0)
            deslocamentos = {nome: (desloc_religiao if nome in FUNDACOES_BINDING else 0.0) for nome in ORDEM_MFT}

        valores = _amostrar_multivariado(
            self._rng, 0.5, self.desvio, CORRELACAO_MFT,
            deslocamentos=deslocamentos, ordem=ORDEM_MFT,
        )
        return FundamentosMorais(**dict(zip(ORDEM_MFT, valores)))
