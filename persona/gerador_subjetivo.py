"""
Camada SUBJETIVA do gerador de persona: Big Five, Fundamentos Morais,
e a narrativa textual que humaniza a persona.

DECISÃO METODOLÓGICA IMPORTANTE (discutida e registrada aqui para o TCC):
Os NÚMEROS de Big Five e Fundamentos Morais são amostrados estatisticamente
(distribuição normal truncada em [0,1]), NÃO gerados diretamente pelo LLM.
O LLM é usado apenas para escrever a narrativa textual que da uma "voz"
à combinação perfil + traços.

Por quê: se pedíssemos ao LLM para gerar os próprios números a partir do
perfil sociodemográfico (ex: "gere o Big Five de uma pessoa de 60 anos,
nordestina, evangélica"), o risco é o modelo regredir para estereótipos
fortes e pouco variados (ex: sempre alta conscienciosidade para esse
perfil), colapsando a variância real de personalidade que existe dentro
de qualquer grupo demográfico. Ao sortear os números nós mesmos, a
variância fica sob nosso controle e é estatisticamente justificável.

LIMITAÇÃO ATUAL: os traços são amostrados de forma independente do
perfil sociodemográfico (não há correlação demográfica embutida). Existem
correlatos demográficos de Big Five/MFT na literatura (ex: abertura
correlaciona com escolaridade em alguns estudos) que podem ser
incorporados depois como uma matriz de ajuste -- documentar isso como
extensão futura, não implementar às pressas sem base na literatura.
"""

import random
from dataclasses import dataclass

from schemas import BigFive, FundamentosMorais, PerfilSociodemografico, PersonaEleitor
from llm.client import LLMClient


def _amostrar_traco(media: float = 0.5, desvio: float = 0.18) -> float:
    """Normal truncada em [0,1]. Desvio 0.18 dá variância razoável sem
    estourar os limites com frequência."""
    valor = random.gauss(media, desvio)
    return max(0.0, min(1.0, valor))


@dataclass
class GeradorTracosSubjetivos:
    seed: int | None = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

    def amostrar_big_five(self) -> BigFive:
        return BigFive(
            abertura=_amostrar_traco(),
            conscienciosidade=_amostrar_traco(),
            extroversao=_amostrar_traco(),
            amabilidade=_amostrar_traco(),
            neuroticismo=_amostrar_traco(),
        )

    def amostrar_fundamentos_morais(self, perfil: PerfilSociodemografico | None = None) -> FundamentosMorais:
        # `perfil` é aceito e ignorado aqui -- existe só para manter a
        # mesma assinatura de GeradorTracosSubjetivosCorrelacionado
        # (gerador_subjetivo_correlacionado.py), que usa o perfil para
        # condicionar levemente por religiosidade. Essa versão
        # independente nunca condiciona por demografia (ver LIMITAÇÃO
        # ATUAL na docstring do módulo).
        return FundamentosMorais(
            cuidado_dano=_amostrar_traco(),
            justica_reciprocidade=_amostrar_traco(),
            lealdade_grupo=_amostrar_traco(),
            autoridade_respeito=_amostrar_traco(),
            pureza_santidade=_amostrar_traco(),
        )


SYSTEM_PROMPT_NARRATIVA = """\
Você é um assistente de pesquisa em ciência política e comportamental.
Sua tarefa é escrever uma narrativa curta (3-4 frases) em português
do Brasil descrevendo uma pessoa fictícia, a partir de um perfil
sociodemográfico e de traços psicológicos fornecidos.

REGRAS IMPORTANTES:
- A narrativa deve ser plausível e específica, mas NÃO estereotipada.
  Pessoas com o mesmo perfil demográfico podem ter personalidades muito
  diferentes -- respeite os números de personalidade fornecidos, mesmo
  que pareçam "incomuns" para o perfil demográfico.
- NÃO invente intenção de voto ou posição política explícita.
- NÃO repita os números recebidos como texto (ex: não escreva "tem
  abertura de 0.8"). Traduza os traços em comportamento e atitude.
- Foco em: rotina, preocupações cotidianas, como essa pessoa
  costuma reagir a notícias/discussões políticas, dado seu perfil
  psicológico.
- Responda APENAS com a narrativa, sem introduções ou comentários.
"""


def gerar_narrativa(
    llm: LLMClient,
    perfil: PerfilSociodemografico,
    big_five: BigFive,
    fundamentos: FundamentosMorais,
) -> str:
    prompt = (
        f"PERFIL: {perfil.idade} anos, {perfil.genero}, mora em {perfil.regiao} ({perfil.uf}), "
        f"{perfil.raca_cor}, escolaridade {perfil.escolaridade}, renda familiar "
        f"{perfil.renda_faixa} salários mínimos, "
        f"domicílio {perfil.situacao_domicilio}.\n\n"
        f"TRAÇOS DE PERSONALIDADE (0=baixo, 1=alto):\n"
        f"- Abertura a experiências: {big_five.abertura:.2f}\n"
        f"- Conscienciosidade: {big_five.conscienciosidade:.2f}\n"
        f"- Extroversão: {big_five.extroversao:.2f}\n"
        f"- Amabilidade: {big_five.amabilidade:.2f}\n"
        f"- Neuroticismo: {big_five.neuroticismo:.2f}\n\n"
        f"FUNDAMENTOS MORAIS QUE MAIS IMPORTAM PARA ESSA PESSOA (0=irrelevante, 1=muito importante):\n"
        f"- Cuidado/dano: {fundamentos.cuidado_dano:.2f}\n"
        f"- Justiça/reciprocidade: {fundamentos.justica_reciprocidade:.2f}\n"
        f"- Lealdade ao grupo: {fundamentos.lealdade_grupo:.2f}\n"
        f"- Autoridade/respeito: {fundamentos.autoridade_respeito:.2f}\n"
        f"- Pureza/santidade: {fundamentos.pureza_santidade:.2f}\n\n"
        f"Escreva a narrativa agora."
    )
    return llm.gerar_texto(system=SYSTEM_PROMPT_NARRATIVA, prompt=prompt)


def gerar_persona_completa(
    eleitor_id: str,
    perfil: PerfilSociodemografico,
    llm: LLMClient,
    gerador_tracos: GeradorTracosSubjetivos,
) -> PersonaEleitor:
    """Monta a persona completa: estatística (perfil, já pronto) +
    subjetiva (amostrada) + narrativa (LLM)."""
    big_five = gerador_tracos.amostrar_big_five()
    fundamentos = gerador_tracos.amostrar_fundamentos_morais(perfil=perfil)
    narrativa = gerar_narrativa(llm, perfil, big_five, fundamentos)

    return PersonaEleitor(
        id=eleitor_id,
        perfil=perfil,
        big_five=big_five,
        fundamentos_morais=fundamentos,
        narrativa=narrativa,
        intencao_voto_inicial="indeciso",
    )
