"""
Orquestrador: roda o loop de rodadas do debate, com a estrutura FIXA:

    moderador pergunta (a partir de um tema) -> candidato 1 abre
    -> candidato 2 abre -> candidato 1 replica -> candidato 2 replica
    -> eleitores avaliam a rodada (em paralelo, observam tudo acima)

Cada rodada tem um tema (de uma lista pré-definida que você fornece). O
moderador transforma o tema numa pergunta natural via LLM, mas o TEMA em
si é fixo -- isso te dá controle experimental sobre o que foi discutido
em cada rodada, sem perder a naturalidade da pergunta.

A ordem de abertura (quem fala primeiro na rodada) é configurável via
`alternar_abertura` (default True -- ver justificativa abaixo): se True,
alterna a cada rodada; se False, o candidato_a sempre abre. O tipo_fala e
a ordem ficam registrados em cada FalaCandidato/linha do banco, então dá
pra controlar esse efeito na análise de qualquer forma.

POR QUE ALTERNAR É O PADRÃO: em debates eleitorais reais no Brasil, a Lei
das Eleições (Lei nº 9.504/1997) determina que a ordem de fala dos
candidatos é definida por SORTEIO (salvo acordo em contrário entre as
campanhas) -- ou seja, não há uma ordem fixa privilegiando sistematicamente
um candidato. Sempre deixar o mesmo candidato abrir introduziria uma
assimetria que não existe no fenômeno real sendo modelado (e a literatura
de persuasão documenta efeitos de ordem/primazia reais). Alternar a
abertura a cada rodada é a aproximação mais simples e determinística
disso -- não é sorteio de verdade, mas evita a assimetria sistemática.

ESTRATÉGIA ARGUMENTATIVA FIXA POR EXECUÇÃO (`estrategia_a`, `estrategia_b`):
por padrão (ambos None), cada candidato sorteia a estratégia a cada fala
individual (comportamento histórico). Se você passar uma
EstrategiaArgumentativa para `estrategia_a`/`estrategia_b`, aquele
candidato usa SEMPRE essa estratégia durante toda a execução -- isso é o
que permite comparar estratégias de forma limpa (RQ3): rode uma execução
com estrategia_a=DADOS_E_LOGICA vs estrategia_b=APELO_EMOCIONAL, depois
outra execução trocando as estratégias, reaproveitando o MESMO lote de
eleitores (já suportado pelo pipeline gerar_personas.py -> main.py
--personas), e compare os resultados entre execuções.

A paralelização dos eleitores usa threads (ThreadPoolExecutor) porque a
chamada ao Ollama é uma requisição HTTP bloqueante (I/O bound).

SOBRE TAMANHO DE CONTEXTO (`max_rodadas_no_historico`):
modelos ~8B tendem a perder instruções de formato (ex: "responda em
JSON com estes campos") conforme o prompt cresce, e o histórico
acumulado do debate cresce a cada rodada. Por padrão, o histórico
passado para CADA fala/avaliação é limitado às últimas N rodadas (não
o debate completo desde o início) -- isso reduz o risco de falha de
formato em rodadas avançadas e também o tempo de geração. O histórico
COMPLETO continua sendo salvo no banco (toda fala de toda rodada), então
nada se perde para a análise -- só o que é mandado de volta ao modelo
como contexto é truncado. Ajuste ou desative (None = histórico completo)
conforme a capacidade do seu modelo.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from schemas import PersonaCandidato, PersonaEleitor, EstrategiaArgumentativa
from llm.client import LLMClient, ConfigLLM
from debate.agentes import AgenteCandidato, AgenteEleitor, AgenteModerador
from storage.db import Storage


class OrquestradorDebate:
    def __init__(
        self,
        candidato_a: PersonaCandidato,
        candidato_b: PersonaCandidato,
        eleitores: list[PersonaEleitor],
        temas: list[str],
        config_llm: ConfigLLM,
        storage: Storage,
        descricao_run: str = "",
        max_workers_eleitores: int = 4,
        alternar_abertura: bool = True,
        max_rodadas_no_historico: int | None = 3,
        estrategia_a: EstrategiaArgumentativa | None = None,
        estrategia_b: EstrategiaArgumentativa | None = None,
    ):
        if len(temas) == 0:
            raise ValueError("É necessário fornecer ao menos um tema.")

        self.run_id = str(uuid.uuid4())[:8]
        self.storage = storage
        self.max_workers_eleitores = max_workers_eleitores
        self.temas = temas
        self.alternar_abertura = alternar_abertura
        self.max_rodadas_no_historico = max_rodadas_no_historico

        llm = LLMClient(config_llm)
        self.moderador = AgenteModerador(llm)
        self.agente_a = AgenteCandidato(candidato_a, llm, estrategia_fixa=estrategia_a)
        self.agente_b = AgenteCandidato(candidato_b, llm, estrategia_fixa=estrategia_b)
        self.agentes_eleitores = [AgenteEleitor(p, llm) for p in eleitores]

        self.storage.criar_run(self.run_id, descricao_run, config_llm.modelo, config_llm.seed)
        self.storage.salvar_candidato(self.run_id, candidato_a)
        self.storage.salvar_candidato(self.run_id, candidato_b)
        for p in eleitores:
            self.storage.salvar_eleitor(self.run_id, p)

        # cada item: (rodada, linha_de_texto) -- guardamos a rodada de
        # origem para conseguir truncar por NÚMERO DE RODADAS (não por
        # linha), já que cada rodada gera várias linhas (moderador +
        # 4 falas de candidato)
        self.historico: list[tuple[int, str]] = []

    def _historico_texto(self) -> str:
        """Histórico completo, sem truncamento -- usado para persistência
        e para quem quiser inspecionar o debate inteiro."""
        if not self.historico:
            return "(debate ainda não começou)"
        return "\n".join(linha for _, linha in self.historico)

    def _historico_texto_truncado(self) -> str:
        """Histórico passado para os agentes (candidatos/eleitores) como
        contexto -- limitado às últimas `max_rodadas_no_historico` rodadas,
        se configurado, para reduzir o risco de o modelo perder instruções
        de formato em prompts muito longos."""
        if not self.historico:
            return "(debate ainda não começou)"
        if self.max_rodadas_no_historico is None:
            return self._historico_texto()

        rodada_atual = self.historico[-1][0]
        rodada_minima = rodada_atual - self.max_rodadas_no_historico + 1
        linhas = [linha for rodada, linha in self.historico if rodada >= rodada_minima]
        return "\n".join(linhas)

    def _tema_da_rodada(self, rodada: int) -> str:
        # Cicla pela lista de temas se houver mais rodadas que temas
        return self.temas[(rodada - 1) % len(self.temas)]

    def _registrar_estado_inicial(self) -> None:
        """
        Registra a RODADA 0: a intenção de voto de cada eleitor ANTES de
        qualquer debate acontecer (vem de `persona.intencao_voto_inicial`,
        "indeciso" por padrão -- não chama o LLM, é só persistência do
        estado de partida).

        Sem isso, a primeira linha salva no banco já é a REAÇÃO à rodada 1
        (depois do moderador + 4 falas de candidato), e o gráfico de
        evolução de voto começa sem o ponto de referência "antes do
        debate" -- o que pode parecer estranho (ex: zero indecisos já na
        rodada 1) mesmo estando tecnicamente correto.
        """
        from schemas import RespostaEleitor
        for agente in self.agentes_eleitores:
            resposta = RespostaEleitor(
                run_id=self.run_id,
                rodada=0,
                eleitor_id=agente.persona.id,
                intencao_voto=agente.persona.intencao_voto_inicial,
                confianca=0.0,
                justificativa="Estado inicial, antes do início do debate.",
                mudou_de_posicao=False,
            )
            self.storage.salvar_resposta(resposta)

    def _registrar_fala(self, fala) -> None:
        self.historico.append(
            (fala.rodada, f"[Rodada {fala.rodada} - {fala.candidato_id} ({fala.tipo_fala})]: {fala.texto}")
        )
        self.storage.salvar_fala(self.run_id, fala)

    def rodar_rodada(self, rodada: int):
        tema = self._tema_da_rodada(rodada)

        # quem abre a rodada
        abre_a_primeiro = True
        if self.alternar_abertura:
            abre_a_primeiro = (rodada % 2 == 1)

        primeiro, segundo = (
            (self.agente_a, self.agente_b) if abre_a_primeiro else (self.agente_b, self.agente_a)
        )

        # 1. moderador formula a pergunta a partir do tema
        pergunta = self.moderador.gerar_pergunta(tema, self._historico_texto_truncado())
        self.historico.append(
            (rodada, f"[Rodada {rodada} - Moderador]: tema \"{tema}\" -- {pergunta}")
        )

        # 2. abertura: primeiro candidato responde à pergunta
        fala_1 = primeiro.responder_pergunta_moderador(rodada, tema, pergunta, self._historico_texto_truncado())
        self._registrar_fala(fala_1)

        # 3. abertura: segundo candidato responde à pergunta
        fala_2 = segundo.responder_pergunta_moderador(rodada, tema, pergunta, self._historico_texto_truncado())
        self._registrar_fala(fala_2)

        # 4. réplica do primeiro candidato (reage à abertura do segundo)
        fala_3 = primeiro.replicar(rodada, tema, self._historico_texto_truncado())
        self._registrar_fala(fala_3)

        # 5. tréplica do segundo candidato (reage à réplica do primeiro)
        fala_4 = segundo.replicar(rodada, tema, self._historico_texto_truncado())
        self._registrar_fala(fala_4)

        # 6. eleitores avaliam a rodada inteira, em paralelo
        historico_atual = self._historico_texto_truncado()
        with ThreadPoolExecutor(max_workers=self.max_workers_eleitores) as executor:
            futuros = {
                executor.submit(agente.avaliar_rodada, self.run_id, rodada, historico_atual): agente
                for agente in self.agentes_eleitores
            }
            for futuro in as_completed(futuros):
                resposta = futuro.result()  # propaga exceção se algum eleitor falhar
                self.storage.salvar_resposta(resposta)

    def rodar_simulacao(self, n_rodadas: int):
        self._registrar_estado_inicial()
        for rodada in range(1, n_rodadas + 1):
            tema = self._tema_da_rodada(rodada)
            print(f"[run {self.run_id}] rodada {rodada}/{n_rodadas} -- tema: {tema}")
            self.rodar_rodada(rodada)
        print(f"[run {self.run_id}] simulação concluída.")
        return self.run_id
