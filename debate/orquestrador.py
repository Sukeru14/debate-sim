"""
Roda um confronto (uma execução) entre dois tipos de candidato.

Estrutura fixa de cada rodada:
    mediador pergunta -> abertura do 1º -> abertura do 2º
    -> réplica do 1º -> tréplica do 2º -> eleitores avaliam (em paralelo)

Quem abre alterna a cada rodada: a Lei das Eleições (Lei nº 9.504/1997)
define a ordem de fala por sorteio, então fixar sempre o mesmo candidato
criaria uma vantagem de posição que não existe no debate real.

O contexto mandado ao modelo é limitado às últimas `max_rodadas_historico`
rodadas (modelos ~8B perdem instruções de formato em prompts longos). O
debate completo é sempre salvo no banco.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

from debate.agentes import AgenteCandidato, AgenteEleitor, Mediador
from debate.candidatos import TEMAS, TipoCandidato
from llm.client import LLMClient
from schemas import INDECISO, Fala, PersonaEleitor, RespostaEleitor
from storage.db import Storage


ROTULO_1, ROTULO_2 = "Candidato 1", "Candidato 2"
NOME_FALA = {"abertura": "abertura", "replica": "réplica", "treplica": "tréplica"}


class OrquestradorDebate:
    def __init__(
        self,
        run_id: str,
        tipo_1: TipoCandidato,
        tipo_2: TipoCandidato,
        eleitores: list[PersonaEleitor],
        llm: LLMClient,
        storage: Storage,
        temas: list[str] = TEMAS,
        max_rodadas_historico: int | None = 3,
        max_workers: int = 4,
    ):
        self.run_id = run_id
        self.storage = storage
        self.temas = temas
        self.max_rodadas_historico = max_rodadas_historico
        self.max_workers = max_workers

        self.mediador = Mediador(llm)
        self.candidatos = (
            AgenteCandidato(tipo_1, ROTULO_1, ROTULO_2, llm),
            AgenteCandidato(tipo_2, ROTULO_2, ROTULO_1, llm),
        )
        tipos_por_rotulo = {"1": tipo_1.id, "2": tipo_2.id}
        self.eleitores = [AgenteEleitor(p, tipos_por_rotulo, llm) for p in eleitores]
        self.historico: list[tuple[int, str]] = []  # (rodada, linha)

    def rodar(self, n_rodadas: int) -> None:
        self._registrar_estado_inicial()
        for rodada in range(1, n_rodadas + 1):
            tema = self.temas[(rodada - 1) % len(self.temas)]
            print(f"[{self.run_id}] rodada {rodada}/{n_rodadas} -- {tema}")
            self._rodar_rodada(rodada, tema)

    def _rodar_rodada(self, rodada: int, tema: str) -> None:
        primeiro, segundo = self.candidatos if rodada % 2 == 1 else self.candidatos[::-1]

        pergunta = self.mediador.perguntar(tema, self._contexto())
        self._registrar(Fala(rodada, 0, "Mediador", None, "pergunta", tema, pergunta))

        etapas = [(primeiro, "abertura"), (segundo, "abertura"), (primeiro, "replica"), (segundo, "treplica")]
        for ordem, (candidato, tipo_fala) in enumerate(etapas, start=1):
            if tipo_fala == "abertura":
                texto = candidato.abrir(tema, pergunta, self._contexto())
            else:
                texto = candidato.replicar(tema, self._contexto())
            self._registrar(Fala(rodada, ordem, candidato.rotulo, candidato.tipo.id, tipo_fala, tema, texto))

        self._avaliar_eleitores(rodada)

    def _avaliar_eleitores(self, rodada: int) -> None:
        contexto = self._contexto()
        falhas = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futuros = {executor.submit(e.avaliar, rodada, contexto): e for e in self.eleitores}
            for futuro in as_completed(futuros):
                try:
                    self.storage.salvar_resposta(self.run_id, futuro.result())
                except Exception as e:  # um eleitor com falha não derruba a execução inteira
                    falhas += 1
                    print(f"[{self.run_id}] ERRO no {futuros[futuro].persona.id}, rodada {rodada}: {e}",
                          file=sys.stderr)
        if falhas:
            print(f"[{self.run_id}] {falhas} eleitor(es) sem resposta na rodada {rodada}", file=sys.stderr)

    def _registrar_estado_inicial(self) -> None:
        """Rodada 0: todos indecisos antes do debate (não chama o LLM)."""
        for e in self.eleitores:
            self.storage.salvar_resposta(self.run_id, RespostaEleitor(
                rodada=0, eleitor_id=e.persona.id, intencao_voto=INDECISO,
                confianca=0.0, justificativa="", mudou_de_posicao=False,
            ))

    def _registrar(self, fala: Fala) -> None:
        if fala.tipo_fala == "pergunta":
            linha = f'[Rodada {fala.rodada} | Mediador, tema "{fala.tema}"]: {fala.texto}'
        else:
            linha = f"[Rodada {fala.rodada} | {fala.orador}, {NOME_FALA[fala.tipo_fala]}]: {fala.texto}"
        self.historico.append((fala.rodada, linha))
        self.storage.salvar_fala(self.run_id, fala)

    def _contexto(self) -> str:
        if not self.historico:
            return "(o debate ainda não começou)"
        if self.max_rodadas_historico is None:
            linhas = self.historico
        else:
            rodada_minima = self.historico[-1][0] - self.max_rodadas_historico + 1
            linhas = [h for h in self.historico if h[0] >= rodada_minima]
        return "\n".join(linha for _, linha in linhas)
