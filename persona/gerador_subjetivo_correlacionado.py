"""
Versão dos traços subjetivos com CORRELAÇÃO entre eles, amostrados de uma
normal multivariada (recortada em [0, 1]). Mesma interface de
GeradorTracosIndependente.

Não existe microdado público pareado de Big Five + Fundamentos Morais da
população brasileira; esta versão é uma melhoria estrutural sobre tratar
os 10 traços como independentes (o que a psicometria sabe que é falso):

- Big Five: metatraços Stability (conscienciosidade + amabilidade + baixo
  neuroticismo) e Plasticity (extroversão + abertura) -- Digman (1997);
  DeYoung, Peterson & Higgins (2002); meta-análise de van der Linden,
  te Nijenhuis & Segers (2010).
- Fundamentos Morais: clusters individualizing (cuidado + justiça) e
  binding (lealdade + autoridade + pureza) -- Graham, Haidt & Nosek (2009),
  estrutura bifatorial replicada na validação brasileira do MFQ.

LIMITAÇÃO: as matrizes abaixo são ILUSTRATIVAS -- seguem a direção e a
magnitude relativa da literatura, não os coeficientes de um estudo
específico. Antes da versão final, substituir por uma matriz citável
(idealmente de amostra brasileira) ou declarar a aproximação no TCC.
"""

from dataclasses import dataclass, field

import numpy as np

from schemas import BigFive, FundamentosMorais


CORRELACAO_BIG_FIVE = np.array([
    # abert  consc  extra  amab   neur
    [ 1.00,  0.05,  0.20,  0.10, -0.05],  # abertura
    [ 0.05,  1.00,  0.15,  0.20, -0.25],  # conscienciosidade
    [ 0.20,  0.15,  1.00,  0.15, -0.25],  # extroversao
    [ 0.10,  0.20,  0.15,  1.00, -0.15],  # amabilidade
    [-0.05, -0.25, -0.25, -0.15,  1.00],  # neuroticismo
])

CORRELACAO_MFT = np.array([
    # cuid  just  leal  auto  pure
    [1.00, 0.55, 0.15, 0.10, 0.10],  # cuidado_dano
    [0.55, 1.00, 0.15, 0.10, 0.10],  # justica_reciprocidade
    [0.15, 0.15, 1.00, 0.55, 0.50],  # lealdade_grupo
    [0.10, 0.10, 0.55, 1.00, 0.55],  # autoridade_respeito
    [0.10, 0.10, 0.50, 0.55, 1.00],  # pureza_santidade
])


@dataclass
class GeradorTracosCorrelacionado:
    seed: int | None = None
    media: float = 0.5
    desvio: float = 0.18
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def _amostrar(self, correlacao: np.ndarray) -> list[float]:
        n = correlacao.shape[0]
        cov = correlacao * self.desvio ** 2
        amostra = self._rng.multivariate_normal(np.full(n, self.media), cov)
        return np.clip(amostra, 0.0, 1.0).tolist()

    def amostrar_big_five(self) -> BigFive:
        return BigFive(*self._amostrar(CORRELACAO_BIG_FIVE))

    def amostrar_fundamentos_morais(self) -> FundamentosMorais:
        return FundamentosMorais(*self._amostrar(CORRELACAO_MFT))
