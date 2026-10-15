"""
Estruturas de dados compartilhadas entre os módulos: persona do eleitor,
falas do debate e respostas dos eleitores.
"""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


INDECISO = "indeciso"


# ---------------------------------------------------------------------------
# Persona do eleitor
# ---------------------------------------------------------------------------

@dataclass
class BigFive:
    """Traços OCEAN normalizados em [0, 1]."""
    abertura: float
    conscienciosidade: float
    extroversao: float
    amabilidade: float
    neuroticismo: float

    def descricao(self) -> str:
        return (
            f"abertura={self.abertura:.2f}, conscienciosidade={self.conscienciosidade:.2f}, "
            f"extroversão={self.extroversao:.2f}, amabilidade={self.amabilidade:.2f}, "
            f"neuroticismo={self.neuroticismo:.2f}"
        )


@dataclass
class FundamentosMorais:
    """As 5 fundações do MFQ-30 (Graham, Haidt & Nosek, 2009), em [0, 1]."""
    cuidado_dano: float
    justica_reciprocidade: float
    lealdade_grupo: float
    autoridade_respeito: float
    pureza_santidade: float

    def descricao(self) -> str:
        return (
            f"cuidado/dano={self.cuidado_dano:.2f}, "
            f"justiça/reciprocidade={self.justica_reciprocidade:.2f}, "
            f"lealdade ao grupo={self.lealdade_grupo:.2f}, "
            f"autoridade/respeito={self.autoridade_respeito:.2f}, "
            f"pureza/santidade={self.pureza_santidade:.2f}"
        )


@dataclass
class PerfilSociodemografico:
    idade: int
    genero: str
    regiao: str
    uf: str
    raca_cor: str
    escolaridade: str
    renda_faixa: str
    situacao_domicilio: str  # "urbano" / "rural"

    def descricao(self) -> str:
        return (
            f"{self.idade} anos, gênero {self.genero}, mora em {self.uf} ({self.regiao}), "
            f"domicílio {self.situacao_domicilio}, raça/cor {self.raca_cor}, "
            f"escolaridade: {self.escolaridade}, renda familiar: {self.renda_faixa}"
        )


@dataclass
class PersonaEleitor:
    """Perfil (amostrado da PNAD) + traços (amostrados) + narrativa (LLM)."""
    id: str
    perfil: PerfilSociodemografico
    big_five: BigFive
    fundamentos_morais: FundamentosMorais
    narrativa: str

    def to_prompt_block(self) -> str:
        return (
            f"PERFIL: {self.perfil.descricao()}.\n"
            f"PERSONALIDADE (Big Five, 0=baixo, 1=alto): {self.big_five.descricao()}.\n"
            f"FUNDAMENTOS MORAIS (0=irrelevante, 1=muito importante): "
            f"{self.fundamentos_morais.descricao()}.\n"
            f"CONTEXTO PESSOAL: {self.narrativa}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PersonaEleitor":
        return PersonaEleitor(
            id=d["id"],
            perfil=_de_dict(PerfilSociodemografico, d["perfil"]),
            big_five=_de_dict(BigFive, d["big_five"]),
            fundamentos_morais=_de_dict(FundamentosMorais, d["fundamentos_morais"]),
            narrativa=d["narrativa"],
        )


def _de_dict(cls, dados: dict):
    """Ignora chaves que não existem mais no schema (ex: 'religiao' em lotes antigos)."""
    nomes = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dados.items() if k in nomes})


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------

@dataclass
class Fala:
    rodada: int
    ordem: int                  # posição dentro da rodada (0 = pergunta do mediador)
    orador: str                 # "Mediador", "Candidato 1" ou "Candidato 2"
    tipo_candidato: str | None  # id do TipoCandidato; None para o mediador
    tipo_fala: str              # "pergunta", "abertura", "replica" ou "treplica"
    tema: str
    texto: str


@dataclass
class RespostaEleitor:
    rodada: int
    eleitor_id: str
    intencao_voto: str      # id do tipo de candidato escolhido, ou INDECISO
    confianca: float        # 0-1
    justificativa: str
    mudou_de_posicao: bool  # comparado à rodada anterior


# ---------------------------------------------------------------------------
# I/O de lotes de eleitores (JSON legível, editável à mão)
# ---------------------------------------------------------------------------

def salvar_eleitores_json(eleitores: list[PersonaEleitor], caminho: str) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_eleitores": len(eleitores), "eleitores": [e.to_dict() for e in eleitores]}
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def carregar_eleitores_json(caminho: str) -> list[PersonaEleitor]:
    with open(caminho, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [PersonaEleitor.from_dict(d) for d in payload["eleitores"]]
