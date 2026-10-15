"""
Camada subjetiva da persona: Big Five, Fundamentos Morais e narrativa.

Decisão metodológica: os NÚMEROS dos traços são sorteados (normal truncada
em [0, 1]), não gerados pelo LLM. Pedir ao modelo que "invente" a
personalidade a partir do perfil demográfico tende a produzir estereótipos
e a colapsar a variância real que existe dentro de cada grupo. O LLM só
escreve a narrativa que dá voz à combinação perfil + traços.

Esta versão sorteia os 10 traços de forma independente entre si e do
perfil. Ver gerador_subjetivo_correlacionado.py para a versão com a
estrutura de correlação da literatura.
"""

from dataclasses import dataclass, field
import random
from typing import Protocol

from llm.client import LLMClient
from schemas import BigFive, FundamentosMorais, PerfilSociodemografico, PersonaEleitor


class GeradorTracos(Protocol):
    def amostrar_big_five(self) -> BigFive: ...
    def amostrar_fundamentos_morais(self) -> FundamentosMorais: ...


@dataclass
class GeradorTracosIndependente:
    seed: int | None = None
    media: float = 0.5
    desvio: float = 0.18
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def _traco(self) -> float:
        return min(1.0, max(0.0, self._rng.gauss(self.media, self.desvio)))

    def amostrar_big_five(self) -> BigFive:
        return BigFive(*(self._traco() for _ in range(5)))

    def amostrar_fundamentos_morais(self) -> FundamentosMorais:
        return FundamentosMorais(*(self._traco() for _ in range(5)))


SYSTEM_NARRATIVA = """\
Você é um assistente de pesquisa em ciência política e comportamental.
Escreva uma narrativa curta (3 a 4 frases), em português do Brasil, sobre
uma pessoa fictícia com o perfil e os traços psicológicos fornecidos.

REGRAS:
- Seja plausível e específico, mas NÃO estereotipado: respeite os traços
  fornecidos mesmo que pareçam "incomuns" para o perfil demográfico.
- NÃO invente intenção de voto, partido ou posição política explícita.
- NÃO repita os números; traduza os traços em comportamentos e atitudes.
- Foque em rotina, preocupações do dia a dia e em como a pessoa costuma
  reagir a notícias e discussões políticas.
- Responda APENAS com a narrativa."""


def gerar_narrativa(llm: LLMClient, perfil: PerfilSociodemografico,
                    big_five: BigFive, fundamentos: FundamentosMorais) -> str:
    prompt = (
        f"PERFIL: {perfil.descricao()}.\n"
        f"PERSONALIDADE (Big Five, 0=baixo, 1=alto): {big_five.descricao()}.\n"
        f"FUNDAMENTOS MORAIS (0=irrelevante, 1=muito importante): {fundamentos.descricao()}.\n\n"
        f"Escreva a narrativa."
    )
    return llm.gerar_texto(SYSTEM_NARRATIVA, prompt)


def gerar_persona(eleitor_id: str, perfil: PerfilSociodemografico,
                  gerador_tracos: GeradorTracos, llm: LLMClient) -> PersonaEleitor:
    big_five = gerador_tracos.amostrar_big_five()
    fundamentos = gerador_tracos.amostrar_fundamentos_morais()
    return PersonaEleitor(
        id=eleitor_id,
        perfil=perfil,
        big_five=big_five,
        fundamentos_morais=fundamentos,
        narrativa=gerar_narrativa(llm, perfil, big_five, fundamentos),
    )
