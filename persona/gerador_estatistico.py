"""
Camada ESTATÍSTICA do gerador de persona.

Responsável apenas pelo perfil sociodemográfico (idade, gênero, região,
escolaridade, renda, religião etc). Por ora, amostra de distribuições
aproximadas conhecidas da literatura/IBGE (não microdados reais).

QUANDO FOR TROCAR PARA MICRODADOS REAIS (PNAD/Censo):
A interface pública desta classe (amostrar()) não muda. Só a implementação
interna troca de "sortear de distribuições marginais hardcoded" para
"sortear uma linha real do dataframe de microdados". Ou seja: o resto do
sistema (gerador de traços subjetivos, debate, análise) não precisa saber
a diferença.

IMPORTANTE - LIMITAÇÃO ATUAL: as distribuições abaixo são amostradas de
forma aproximadamente independente (com algumas correlações simples
hardcoded), o que NÃO captura a distribuição conjunta real da população
brasileira. Isso é uma simplificação deliberada para o protótipo e deve
ser citada como limitação até que os microdados reais sejam integrados.
"""

import random
from dataclasses import dataclass

from schemas import PerfilSociodemografico


# Distribuições aproximadas - fontes: PNAD Contínua / Censo 2022 (valores
# arredondados da literatura, NÃO os microdados reais ainda).
DISTRIBUICAO_REGIAO = {
    "Sudeste": 0.42, "Nordeste": 0.27, "Sul": 0.14,
    "Norte": 0.08, "Centro-Oeste": 0.09,
}

UFS_POR_REGIAO = {
    "Sudeste": ["SP", "RJ", "MG", "ES"],
    "Nordeste": ["BA", "PE", "CE", "MA", "PB", "RN", "AL", "PI", "SE"],
    "Sul": ["RS", "PR", "SC"],
    "Norte": ["AM", "PA", "RO", "AC", "RR", "AP", "TO"],
    "Centro-Oeste": ["GO", "MT", "MS", "DF"],
}

DISTRIBUICAO_GENERO = {"feminino": 0.51, "masculino": 0.49}

DISTRIBUICAO_RACA_COR = {
    "parda": 0.45, "branca": 0.43, "preta": 0.10, "amarela": 0.01, "indigena": 0.01,
}

DISTRIBUICAO_ESCOLARIDADE = {
    "sem_instrucao_fundamental_incompleto": 0.25,
    "fundamental_completo_medio_incompleto": 0.15,
    "medio_completo_superior_incompleto": 0.32,
    "superior_completo": 0.28,
}

DISTRIBUICAO_RENDA_FAIXA = {
    "até_1": 0.28, "1-2": 0.32, "2-5": 0.24, "5-10": 0.10, "mais_de_10": 0.06,
}

DISTRIBUICAO_RELIGIAO = {
    "catolica": 0.50, "evangelica": 0.31, "sem_religiao": 0.10,
    "espirita": 0.03, "outras": 0.06,
}
# NÃO USADA ATUALMENTE -- decisão do orientador (reunião) foi retirar
# religião do perfil por enquanto, até haver dados reais confiáveis para
# essa variável (a PNAD Contínua não pergunta religião). Mantida aqui
# definida para facilitar reativar: basta voltar a usar
# `religiao=_sortear(DISTRIBUICAO_RELIGIAO)` em vez do valor constante
# "não_modelada" em amostrar() logo abaixo.

DISTRIBUICAO_DOMICILIO = {"urbano": 0.87, "rural": 0.13}


def _sortear(distribuicao: dict) -> str:
    chaves = list(distribuicao.keys())
    pesos = list(distribuicao.values())
    return random.choices(chaves, weights=pesos, k=1)[0]


def _sortear_idade() -> int:
    """Aproxima a pirâmide etária adulta brasileira (18+) com uma mistura
    simples de faixas. Substituir por amostragem real de microdados depois."""
    faixa = random.choices(
        ["18-24", "25-34", "35-44", "45-59", "60+"],
        weights=[0.15, 0.22, 0.20, 0.23, 0.20],
        k=1,
    )[0]
    limites = {"18-24": (18, 24), "25-34": (25, 34), "35-44": (35, 44),
               "45-59": (45, 59), "60+": (60, 85)}
    lo, hi = limites[faixa]
    return random.randint(lo, hi)


@dataclass
class GeradorPerfilSociodemografico:
    """
    Implementação atual: distribuições aproximadas e independentes.
    Trocar esta classe por uma que amostra linhas de um DataFrame de
    microdados (PNAD/Censo) quando os dados reais estiverem disponíveis --
    a assinatura de `amostrar()` é o contrato a preservar.
    """
    seed: int | None = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

    def amostrar(self) -> PerfilSociodemografico:
        regiao = _sortear(DISTRIBUICAO_REGIAO)
        uf = random.choice(UFS_POR_REGIAO[regiao])
        return PerfilSociodemografico(
            idade=_sortear_idade(),
            genero=_sortear(DISTRIBUICAO_GENERO),
            regiao=regiao,
            uf=uf,
            raca_cor=_sortear(DISTRIBUICAO_RACA_COR),
            escolaridade=_sortear(DISTRIBUICAO_ESCOLARIDADE),
            renda_faixa=_sortear(DISTRIBUICAO_RENDA_FAIXA),
            religiao="não_modelada",  # ver nota em DISTRIBUICAO_RELIGIAO acima
            situacao_domicilio=_sortear(DISTRIBUICAO_DOMICILIO),
        )

    def amostrar_lote(self, n: int) -> list[PerfilSociodemografico]:
        return [self.amostrar() for _ in range(n)]
