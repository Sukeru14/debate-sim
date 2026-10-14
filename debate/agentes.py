"""
Agentes do debate: Candidato e Eleitor.

Candidato: gera uma fala por rodada, com uma estratégia argumentativa
explícita (necessário para RQ3 -- ver discussão no chat/metodologia).

Eleitor: lê o histórico do debate (todas as falas até agora) e devolve
uma intenção de voto estruturada em JSON, condicionada na sua persona.
NÃO interage com outros eleitores -- só observa o debate.
"""

import random

from schemas import (
    PersonaCandidato, PersonaEleitor, FalaCandidato,
    RespostaEleitor, EstrategiaArgumentativa,
)
from llm.client import LLMClient


# ---------------------------------------------------------------------------
# Moderador
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_MODERADOR = """\
Você é o(a) moderador(a) de um debate eleitoral. Sua função é apresentar
o tema da rodada como uma pergunta direta e neutra para os candidatos,
sem favorecer nenhum dos lados.

Responda APENAS com a pergunta (1-2 frases), sem introduções, sem
saudações, sem mencionar nomes de candidatos.
"""


class AgenteModerador:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def gerar_pergunta(self, tema: str, historico_texto: str) -> str:
        prompt = (
            f"Tema desta rodada: {tema}\n\n"
            f"Histórico do debate até agora:\n{historico_texto}\n\n"
            f"Formule a pergunta para os candidatos sobre este tema."
        )
        return self.llm.gerar_texto(system=SYSTEM_PROMPT_MODERADOR, prompt=prompt).strip()


# ---------------------------------------------------------------------------
# Candidato
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ABERTURA_TEMPLATE = """\
{persona_block}

Você está participando de um debate eleitoral simulado. O(a) moderador(a)
acabou de fazer uma pergunta sobre o tema "{tema}". Responda de forma
persuasiva, no seu estilo retórico, defendendo sua posição/plataforma
sobre o tema.

IMPORTANTE: nesta fala, use PREDOMINANTEMENTE a seguinte estratégia
argumentativa: {estrategia_descricao}

Responda apenas com a fala (sem indicar "Candidato:" ou metadados),
em português do Brasil, entre 2 e 4 frases.
"""

SYSTEM_PROMPT_REPLICA_TEMPLATE = """\
{persona_block}

Você está participando de um debate eleitoral simulado, no tema
"{tema}". Agora é sua vez de replicar/responder DIRETAMENTE ao que o
oponente acabou de dizer (veja o histórico). Reaja ao argumento dele(a)
de forma persuasiva, no seu estilo retórico.

IMPORTANTE: nesta fala, use PREDOMINANTEMENTE a seguinte estratégia
argumentativa: {estrategia_descricao}

Responda apenas com a fala (sem indicar "Candidato:" ou metadados),
em português do Brasil, entre 2 e 4 frases.
"""

DESCRICAO_ESTRATEGIA = {
    EstrategiaArgumentativa.APELO_EMOCIONAL: "apele a emoções, histórias pessoais ou empatia",
    EstrategiaArgumentativa.DADOS_E_LOGICA: "use dados, estatísticas e argumentação lógica",
    EstrategiaArgumentativa.ATAQUE_AO_OPONENTE: "critique diretamente a posição ou o histórico do oponente",
    EstrategiaArgumentativa.APELO_A_VALORES_MORAIS: "apele a valores morais e princípios (justiça, lealdade, tradição, cuidado com os outros etc.)",
    EstrategiaArgumentativa.APELO_A_AUTORIDADE: "cite especialistas, instituições ou autoridades reconhecidas",
}


class AgenteCandidato:
    def __init__(
        self,
        persona: PersonaCandidato,
        llm: LLMClient,
        estrategia_fixa: EstrategiaArgumentativa | None = None,
    ):
        self.persona = persona
        self.llm = llm
        # Se definida, este candidato usa SEMPRE esta estratégia em toda a
        # execução (todas as falas, todas as rodadas) -- necessário para
        # comparar estratégias de forma limpa (RQ3), ver
        # debate/orquestrador.py e a discussão de design no chat.
        self.estrategia_fixa = estrategia_fixa

    def _escolher_estrategia(self, estrategia: EstrategiaArgumentativa | None) -> EstrategiaArgumentativa:
        # Prioridade: (1) estratégia passada explicitamente nesta chamada,
        # (2) estratégia fixa do candidato para a execução inteira,
        # (3) sorteio (comportamento histórico, preservado para quem não
        # configurar nada -- não quebra o uso anterior).
        if estrategia is not None:
            return estrategia
        if self.estrategia_fixa is not None:
            return self.estrategia_fixa
        return random.choice(list(EstrategiaArgumentativa))

    def responder_pergunta_moderador(
        self,
        rodada: int,
        tema: str,
        pergunta_moderador: str,
        historico_texto: str,
        estrategia: EstrategiaArgumentativa | None = None,
    ) -> FalaCandidato:
        """Fala de abertura da rodada: responde à pergunta do moderador."""
        estrategia = self._escolher_estrategia(estrategia)
        system = SYSTEM_PROMPT_ABERTURA_TEMPLATE.format(
            persona_block=self.persona.to_prompt_block(),
            tema=tema,
            estrategia_descricao=DESCRICAO_ESTRATEGIA[estrategia],
        )
        prompt = (
            f"Histórico do debate até agora:\n{historico_texto}\n\n"
            f"Pergunta do(a) moderador(a): {pergunta_moderador}\n\n"
            f"Dê sua resposta agora."
        )
        texto = self.llm.gerar_texto(system=system, prompt=prompt)
        return FalaCandidato(
            rodada=rodada, candidato_id=self.persona.id, estrategia=estrategia.value,
            texto=texto, tipo_fala="abertura", tema=tema, pergunta_moderador=pergunta_moderador,
        )

    def replicar(
        self,
        rodada: int,
        tema: str,
        historico_texto: str,
        estrategia: EstrategiaArgumentativa | None = None,
    ) -> FalaCandidato:
        """Réplica/tréplica: responde diretamente à fala anterior do oponente."""
        estrategia = self._escolher_estrategia(estrategia)
        system = SYSTEM_PROMPT_REPLICA_TEMPLATE.format(
            persona_block=self.persona.to_prompt_block(),
            tema=tema,
            estrategia_descricao=DESCRICAO_ESTRATEGIA[estrategia],
        )
        prompt = (
            f"Histórico do debate até agora:\n{historico_texto}\n\n"
            f"Dê sua réplica agora."
        )
        texto = self.llm.gerar_texto(system=system, prompt=prompt)
        return FalaCandidato(
            rodada=rodada, candidato_id=self.persona.id, estrategia=estrategia.value,
            texto=texto, tipo_fala="replica", tema=tema,
        )


# ---------------------------------------------------------------------------
# Eleitor
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ELEITOR_TEMPLATE = """\
Você está simulando um(a) eleitor(a) com o seguinte perfil. Responda
SEMPRE como essa pessoa pensaria e reagiria, de forma consistente com
seu perfil sociodemográfico, traços de personalidade e fundamentos
morais -- mesmo que isso signifique discordar do "senso comum".

{persona_block}

Você está assistindo a um debate eleitoral entre dois candidatos (A e B).
A cada rodada, você deve avaliar o que foi dito e indicar sua intenção
de voto atual.

Responda SEMPRE em formato JSON com exatamente estes campos:
{{
  "intencao_voto": "A" ou "B" ou "indeciso",
  "confianca": número de 0 a 1 indicando o quão decidido você está,
  "justificativa": breve explicação (1-2 frases) do motivo da sua posição,
    mencionando o que no debate influenciou (ou não) sua opinião
}}

Não inclua nenhum texto fora do JSON.
"""


class AgenteEleitor:
    def __init__(self, persona: PersonaEleitor, llm: LLMClient):
        self.persona = persona
        self.llm = llm
        self.intencao_atual = persona.intencao_voto_inicial

    def avaliar_rodada(self, run_id: str, rodada: int, historico_texto: str) -> RespostaEleitor:
        system = SYSTEM_PROMPT_ELEITOR_TEMPLATE.format(
            persona_block=self.persona.to_prompt_block()
        )
        prompt = (
            f"Debate até agora (mais recente ao final):\n{historico_texto}\n\n"
            f"Sua intenção de voto na rodada anterior era: {self.intencao_atual}\n\n"
            f"Qual é sua intenção de voto agora, após a rodada {rodada}?\n\n"
            f"Responda APENAS com um JSON contendo exatamente os campos "
            f'"intencao_voto" (A, B ou indeciso), "confianca" (0 a 1) e '
            f'"justificativa" (texto curto). Nenhum texto fora do JSON.'
        )
        data = self.llm.gerar_json_validado(
            system=system,
            prompt=prompt,
            campos_obrigatorios=["intencao_voto", "confianca", "justificativa"],
        )

        resposta = RespostaEleitor(
            run_id=run_id,
            rodada=rodada,
            eleitor_id=self.persona.id,
            intencao_voto=data["intencao_voto"],
            confianca=float(data["confianca"]),
            justificativa=data["justificativa"],
            mudou_de_posicao=(data["intencao_voto"] != self.intencao_atual),
        )
        self.intencao_atual = resposta.intencao_voto
        return resposta
