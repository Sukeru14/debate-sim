"""
Agentes do debate: mediador, candidato e eleitor.

Os eleitores só observam o debate (não interagem entre si) e, ao fim de
cada rodada, devolvem a intenção de voto em JSON.
"""

from debate.candidatos import PLATAFORMA, TRAJETORIA, TipoCandidato
from llm.client import LLMClient
from schemas import INDECISO, PersonaEleitor, RespostaEleitor


# ---------------------------------------------------------------------------
# Mediador
# ---------------------------------------------------------------------------

SYSTEM_MEDIADOR = """\
Você é o(a) mediador(a) de um debate eleitoral na TV brasileira. Apresente
o tema da rodada como UMA pergunta direta, concreta e neutra, igual para
os dois candidatos, sem favorecer nenhum lado e sem repetir perguntas já
feitas no debate.

Responda APENAS com a pergunta (1 ou 2 frases), sem saudações e sem
mencionar os candidatos."""


class Mediador:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def perguntar(self, tema: str, historico: str) -> str:
        prompt = (
            f"Debate até agora:\n{historico}\n\n"
            f"Tema desta rodada: {tema}\n\n"
            f"Formule a pergunta."
        )
        return _limpar_fala(self.llm.gerar_texto(SYSTEM_MEDIADOR, prompt))


# ---------------------------------------------------------------------------
# Candidato
# ---------------------------------------------------------------------------

SYSTEM_CANDIDATO = """\
Você é o {rotulo}, em um debate eleitoral na TV brasileira contra o {rotulo_oponente}.

SUA TRAJETÓRIA: {trajetoria}

SUAS PROPOSTAS PARA O TEMA "{tema}": {propostas}

COMO VOCÊ ARGUMENTA:
{instrucoes}

REGRAS:
- Defenda as propostas acima. Você pode detalhá-las, mas não crie propostas fora delas.
- Não mencione partidos, ideologias, nem políticos ou pessoas reais.
- Fale em primeira pessoa, em português do Brasil, em 3 a 5 frases.
- Responda apenas com a sua fala, sem rótulos, aspas ou comentários."""


class AgenteCandidato:
    def __init__(self, tipo: TipoCandidato, rotulo: str, rotulo_oponente: str, llm: LLMClient):
        self.tipo = tipo
        self.rotulo = rotulo
        self.rotulo_oponente = rotulo_oponente
        self.llm = llm

    def _system(self, tema: str) -> str:
        return SYSTEM_CANDIDATO.format(
            rotulo=self.rotulo,
            rotulo_oponente=self.rotulo_oponente,
            trajetoria=TRAJETORIA,
            tema=tema,
            propostas=PLATAFORMA[tema],
            instrucoes=self.tipo.instrucoes,
        )

    def abrir(self, tema: str, pergunta: str, historico: str) -> str:
        prompt = (
            f"Debate até agora:\n{historico}\n\n"
            f"Pergunta do(a) mediador(a): {pergunta}\n\n"
            f"Responda à pergunta."
        )
        return _limpar_fala(self.llm.gerar_texto(self._system(tema), prompt))

    def replicar(self, tema: str, historico: str) -> str:
        prompt = (
            f"Debate até agora:\n{historico}\n\n"
            f"Responda diretamente ao que o {self.rotulo_oponente} acabou de dizer."
        )
        return _limpar_fala(self.llm.gerar_texto(self._system(tema), prompt))


# ---------------------------------------------------------------------------
# Eleitor
# ---------------------------------------------------------------------------

SYSTEM_ELEITOR = """\
Você vai simular um(a) eleitor(a) brasileiro(a) com o perfil abaixo. Pense
e reaja como essa pessoa reagiria, de forma coerente com o perfil, a
personalidade e os valores dela -- mesmo que isso contrarie o senso comum.

{persona}

Você está assistindo a um debate eleitoral na TV entre o Candidato 1 e o
Candidato 2. Ao fim de cada rodada, diga em quem você votaria hoje.

Responda APENAS com um JSON neste formato:
{{
  "intencao_voto": "1", "2" ou "indeciso",
  "confianca": número de 0 a 1 (o quanto você está decidido),
  "justificativa": "1 ou 2 frases, na voz dessa pessoa, sobre o que no debate pesou (ou não) na decisão"
}}"""


class AgenteEleitor:
    def __init__(self, persona: PersonaEleitor, tipos_por_rotulo: dict[str, str], llm: LLMClient):
        """`tipos_por_rotulo`: {"1": id_tipo_candidato_1, "2": id_tipo_candidato_2}."""
        self.persona = persona
        self.llm = llm
        self.tipos_por_rotulo = tipos_por_rotulo
        self.rotulos_por_tipo = {t: r for r, t in tipos_por_rotulo.items()}
        self.system = SYSTEM_ELEITOR.format(persona=persona.to_prompt_block())
        self.intencao_atual = INDECISO
        self.confianca_atual: float | None = None

    def _descrever_posicao_atual(self) -> str:
        if self.confianca_atual is None:
            return "Você ainda não tinha uma intenção de voto antes do debate."
        voto = (INDECISO if self.intencao_atual == INDECISO
                else f"Candidato {self.rotulos_por_tipo[self.intencao_atual]}")
        return f"Na rodada anterior, sua intenção de voto era: {voto} (confiança {self.confianca_atual:.2f})."

    def avaliar(self, rodada: int, historico: str) -> RespostaEleitor:
        prompt = (
            f"Debate até agora:\n{historico}\n\n"
            f"{self._descrever_posicao_atual()}\n\n"
            f"Após a rodada {rodada}, em quem você votaria? Responda só com o JSON."
        )
        data = self.llm.gerar_json(
            self.system, prompt,
            campos=["intencao_voto", "confianca", "justificativa"],
            validar=self._validar,
        )
        resposta = RespostaEleitor(
            rodada=rodada,
            eleitor_id=self.persona.id,
            intencao_voto=data["intencao_voto"],
            confianca=data["confianca"],
            justificativa=data["justificativa"],
            mudou_de_posicao=data["intencao_voto"] != self.intencao_atual,
        )
        self.intencao_atual = resposta.intencao_voto
        self.confianca_atual = resposta.confianca
        return resposta

    def _validar(self, data: dict) -> dict:
        """Converte o rótulo ("1"/"2") no id do tipo e garante confiança em [0, 1]."""
        texto = str(data["intencao_voto"]).strip().lower()
        if "indecis" in texto:
            intencao = INDECISO
        else:
            rotulos = {r for r in self.tipos_por_rotulo if r in texto}
            if len(rotulos) != 1:
                raise ValueError(f"intencao_voto inválida: {data['intencao_voto']!r}")
            intencao = self.tipos_por_rotulo[rotulos.pop()]

        confianca = float(data["confianca"])
        if 1 < confianca <= 100:  # modelo respondeu em porcentagem
            confianca /= 100
        if not 0 <= confianca <= 1:
            raise ValueError(f"confianca fora de [0, 1]: {data['confianca']!r}")

        return {"intencao_voto": intencao, "confianca": confianca,
                "justificativa": str(data["justificativa"]).strip()}


def _limpar_fala(texto: str) -> str:
    return texto.strip().strip('"“”').strip()
