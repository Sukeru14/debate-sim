"""
Schemas centrais do sistema de simulação de debate eleitoral.

Define as estruturas de dados que atravessam todos os módulos:
persona do eleitor, persona do candidato, falas do debate e
respostas estruturadas do eleitor a cada rodada.

Usar dataclasses (em vez de dicts soltos) facilita:
- validação de tipo em tempo de desenvolvimento
- serialização consistente pro banco (asdict)
- documentação do "contrato" de dados entre módulos, o que ajuda
  bastante na hora de descrever a metodologia no TCC
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Big Five
# ---------------------------------------------------------------------------

@dataclass
class BigFive:
    """
    Traços de personalidade OCEAN, em escala 0-1 (normalizado).
    0-1 facilita: amostragem estatística, normalização para prompts,
    e cálculo de distância/correlação na análise depois.
    """
    abertura: float            # Openness
    conscienciosidade: float   # Conscientiousness
    extroversao: float         # Extraversion
    amabilidade: float         # Agreeableness
    neuroticismo: float        # Neuroticism

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fundamentos Morais (Moral Foundations Theory - Haidt)
# ---------------------------------------------------------------------------

@dataclass
class FundamentosMorais:
    """
    Escalas da Moral Foundations Theory, 0-1.
    Usando as 5 fundações clássicas do MFQ-30. Se depois você quiser
    adicionar 'liberdade' (6ª fundação, versão mais recente do MFQ-2),
    é só adicionar o campo aqui — o resto do sistema não muda.
    """
    cuidado_dano: float        # care/harm
    justica_reciprocidade: float  # fairness/cheating
    lealdade_grupo: float      # loyalty/betrayal
    autoridade_respeito: float  # authority/subversion
    pureza_santidade: float    # purity/degradation

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Perfil sociodemográfico
# ---------------------------------------------------------------------------

@dataclass
class PerfilSociodemografico:
    """
    Atributos amostrados estatisticamente (PNAD/Censo ou, por ora,
    distribuições aproximadas da literatura). Mantido separado dos
    traços subjetivos de propósito: esses campos têm proveniência
    estatística rastreável, os outros não.
    """
    idade: int
    genero: str
    regiao: str
    uf: str
    raca_cor: str
    escolaridade: str
    renda_faixa: str           # faixa de salários mínimos, ex: "1-2"
    religiao: str
    situacao_domicilio: str    # "urbano" / "rural"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Persona completa do eleitor
# ---------------------------------------------------------------------------

@dataclass
class PersonaEleitor:
    """
    Persona completa = camada estatística + camada subjetiva (gerada por IA)
    + uma narrativa textual curta que humaniza a persona para uso no prompt.
    """
    id: str
    perfil: PerfilSociodemografico
    big_five: BigFive
    fundamentos_morais: FundamentosMorais
    narrativa: str              # texto curto gerado pela IA, ex: "Maria, 45 anos..."
    intencao_voto_inicial: str  # "A", "B" ou "indeciso"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "PersonaEleitor":
        """Reconstrói a persona a partir de um dict (ex: lido de JSON).
        Espelha exatamente o formato produzido por to_dict()."""
        return PersonaEleitor(
            id=d["id"],
            perfil=PerfilSociodemografico(**d["perfil"]),
            big_five=BigFive(**d["big_five"]),
            fundamentos_morais=FundamentosMorais(**d["fundamentos_morais"]),
            narrativa=d["narrativa"],
            intencao_voto_inicial=d["intencao_voto_inicial"],
        )

    def to_prompt_block(self) -> str:
        """Serializa a persona em texto para uso no system prompt do eleitor."""
        p = self.perfil
        bf = self.big_five
        fm = self.fundamentos_morais
        return (
            f"PERFIL: {p.idade} anos, {p.genero}, {p.regiao} ({p.uf}), "
            f"{p.raca_cor}, escolaridade: {p.escolaridade}, "
            f"renda: {p.renda_faixa} salários mínimos, "
            f"domicílio {p.situacao_domicilio}.\n"
            f"TRAÇOS DE PERSONALIDADE (Big Five, 0=baixo, 1=alto): "
            f"abertura={bf.abertura:.2f}, conscienciosidade={bf.conscienciosidade:.2f}, "
            f"extroversão={bf.extroversao:.2f}, amabilidade={bf.amabilidade:.2f}, "
            f"neuroticismo={bf.neuroticismo:.2f}.\n"
            f"FUNDAMENTOS MORAIS (0=irrelevante, 1=muito importante): "
            f"cuidado/dano={fm.cuidado_dano:.2f}, justiça/reciprocidade={fm.justica_reciprocidade:.2f}, "
            f"lealdade ao grupo={fm.lealdade_grupo:.2f}, autoridade/respeito={fm.autoridade_respeito:.2f}, "
            f"pureza/santidade={fm.pureza_santidade:.2f}.\n"
            f"CONTEXTO PESSOAL: {self.narrativa}"
        )


# ---------------------------------------------------------------------------
# Persona do candidato
# ---------------------------------------------------------------------------

@dataclass
class PersonaCandidato:
    id: str
    nome: str
    plataforma: str             # resumo da plataforma/posições
    estilo_retorico: str        # ex: "direto e agressivo", "técnico e calmo"

    def to_prompt_block(self) -> str:
        return (
            f"Você é {self.nome}, candidato(a) em um debate eleitoral.\n"
            f"PLATAFORMA: {self.plataforma}\n"
            f"ESTILO RETÓRICO: {self.estilo_retorico}"
        )


# ---------------------------------------------------------------------------
# Estratégias argumentativas (relevante para RQ3)
# ---------------------------------------------------------------------------

class EstrategiaArgumentativa(str, Enum):
    """
    Categorias fechadas de estratégia, para facilitar a análise estatística
    (RQ3: diferentes estratégias afetam diferentes perfis de eleitor?).
    Forçar o candidato a escolher uma por fala é mais controlável do que
    deixar livre e classificar depois com outro modelo.
    """
    APELO_EMOCIONAL = "apelo_emocional"
    DADOS_E_LOGICA = "dados_e_logica"
    ATAQUE_AO_OPONENTE = "ataque_ao_oponente"
    APELO_A_VALORES_MORAIS = "apelo_a_valores_morais"
    APELO_A_AUTORIDADE = "apelo_a_autoridade"  # ex: especialistas, instituições


# ---------------------------------------------------------------------------
# Fala de um candidato numa rodada
# ---------------------------------------------------------------------------

@dataclass
class FalaCandidato:
    rodada: int
    candidato_id: str
    estrategia: str    # valor de EstrategiaArgumentativa
    texto: str
    tipo_fala: str = "abertura"   # "abertura" (responde ao moderador) ou "replica"
    tema: str = ""                 # tema da rodada (vindo do roteiro do moderador)
    pergunta_moderador: str = ""   # pergunta exata formulada pelo moderador

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Resposta estruturada do eleitor a cada rodada
# ---------------------------------------------------------------------------

@dataclass
class RespostaEleitor:
    run_id: str
    rodada: int
    eleitor_id: str
    intencao_voto: str      # "A", "B" ou "indeciso"
    confianca: float        # 0-1, o quão decidido o eleitor está
    justificativa: str      # texto curto explicando o porquê
    mudou_de_posicao: bool  # derivado: comparado à rodada anterior

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_llm_json(run_id: str, rodada: int, eleitor_id: str,
                       raw_json: str, intencao_anterior: str) -> "RespostaEleitor":
        """
        Faz parsing da saída JSON do LLM. Lança ValueError se o formato
        vier quebrado -- quem chama deve fazer retry (ver llm/client.py).
        """
        data = json.loads(raw_json)
        intencao = data["intencao_voto"]
        return RespostaEleitor(
            run_id=run_id,
            rodada=rodada,
            eleitor_id=eleitor_id,
            intencao_voto=intencao,
            confianca=float(data["confianca"]),
            justificativa=data.get("justificativa", ""),
            mudou_de_posicao=(intencao != intencao_anterior),
        )


# ---------------------------------------------------------------------------
# I/O de lotes de persona (eleitores) em JSON legível
# ---------------------------------------------------------------------------
#
# Formato pensado para ser lido/editado manualmente se necessário (ex: você
# quer ajustar à mão o traço de um eleitor específico antes de rodar o
# debate). Cada eleitor é um objeto no array "eleitores", com os campos
# na mesma ordem em que aparecem em PersonaEleitor.

def salvar_eleitores_json(eleitores: list[PersonaEleitor], caminho: str) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_eleitores": len(eleitores),
        "eleitores": [e.to_dict() for e in eleitores],
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def carregar_eleitores_json(caminho: str) -> list[PersonaEleitor]:
    with open(caminho, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [PersonaEleitor.from_dict(d) for d in payload["eleitores"]]
