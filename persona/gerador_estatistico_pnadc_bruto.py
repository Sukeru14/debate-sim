"""
Amostra perfis sociodemográficos do arquivo de microdados BRUTO da PNAD
Contínua trimestral (largura fixa, baixado do FTP do IBGE, ex:
PNADC_012026.txt), respeitando o peso amostral V1028.

Posições confirmadas contra um arquivo real de jan/2026 (1-indexadas):

    UF     pos 6,   tam 2   (código IBGE)
    V1022  pos 33,  tam 1   (1=urbano, 2=rural)
    V1028  pos 50,  tam 15  (peso com calibração)
    V2007  pos 95,  tam 1   (1=homem, 2=mulher)
    V2009  pos 104, tam 3   (idade)
    V2010  pos 107, tam 1   (1=branca, 2=preta, 3=amarela, 4=parda, 5=indígena, 9=ignorado)

LIMITAÇÃO: escolaridade (VD3004) e renda (VD4020) não são lidas do
arquivo -- as posições não foram confirmadas com segurança, e uma posição
errada daria valores incorretos sem erro aparente. Elas são sorteadas das
distribuições marginais aproximadas abaixo, independentemente do resto do
perfil. Para usar dados reais, confirmar as posições no dicionário oficial
(Documentacao/Variaveis_PNADC_Trimestral.xls).
"""

from dataclasses import dataclass, field
import random

from schemas import PerfilSociodemografico


# Distribuições aproximadas (PNAD Contínua / Censo 2022, valores arredondados)
DISTRIBUICAO_ESCOLARIDADE = {
    "sem instrução ou fundamental incompleto": 0.25,
    "fundamental completo ou médio incompleto": 0.15,
    "médio completo ou superior incompleto": 0.32,
    "superior completo": 0.28,
}

DISTRIBUICAO_RENDA_FAIXA = {
    "até 1 salário mínimo": 0.28,
    "de 1 a 2 salários mínimos": 0.32,
    "de 2 a 5 salários mínimos": 0.24,
    "de 5 a 10 salários mínimos": 0.10,
    "mais de 10 salários mínimos": 0.06,
}

# (início, fim) 0-indexados para slicing
POS_UF, POS_V1022, POS_V1028 = (5, 7), (32, 33), (49, 64)
POS_V2007, POS_V2009, POS_V2010 = (94, 95), (103, 106), (106, 107)
TAMANHO_LINHA = 3478

SIGLA_POR_CODIGO_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE", "29": "BA",
    "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS",
    "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}

REGIAO_POR_PRIMEIRO_DIGITO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}

RACA_COR = {"1": "branca", "2": "preta", "3": "amarela", "4": "parda", "5": "indígena"}


@dataclass
class _Registro:
    uf: str
    regiao: str
    situacao_domicilio: str
    peso: float
    genero: str
    idade: int
    raca_cor: str


def _parsear_linha(linha: str) -> _Registro | None:
    """Devolve None para linhas malformadas ou com valores fora do esperado."""
    if len(linha.rstrip("\r\n")) != TAMANHO_LINHA:
        return None
    try:
        codigo_uf = linha[slice(*POS_UF)]
        uf = SIGLA_POR_CODIGO_UF.get(codigo_uf)
        situacao = {"1": "urbano", "2": "rural"}.get(linha[slice(*POS_V1022)])
        genero = {"1": "masculino", "2": "feminino"}.get(linha[slice(*POS_V2007)])
        idade = int(linha[slice(*POS_V2009)])
        if uf is None or situacao is None or genero is None or not 0 <= idade <= 130:
            return None
        return _Registro(
            uf=uf,
            regiao=REGIAO_POR_PRIMEIRO_DIGITO[codigo_uf[0]],
            situacao_domicilio=situacao,
            peso=float(linha[slice(*POS_V1028)]),
            genero=genero,
            idade=idade,
            raca_cor=RACA_COR.get(linha[slice(*POS_V2010)], "não informada"),
        )
    except ValueError:
        return None


def _sortear(rng: random.Random, distribuicao: dict[str, float]) -> str:
    return rng.choices(list(distribuicao), weights=list(distribuicao.values()))[0]


@dataclass
class GeradorPerfilPNADCBruto:
    caminho_arquivo: str
    seed: int | None = None
    idade_minima: int = 18
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._registros: list[_Registro] = []
        with open(self.caminho_arquivo, "r", encoding="latin-1") as f:
            for linha in f:
                r = _parsear_linha(linha)
                if r is not None and r.idade >= self.idade_minima:
                    self._registros.append(r)
        if not self._registros:
            raise ValueError(f"Nenhum registro válido em {self.caminho_arquivo}. "
                             f"É o arquivo bruto da PNAD Contínua trimestral?")
        print(f"[PNADC] {len(self._registros)} registros com {self.idade_minima}+ anos carregados")

    def amostrar_lote(self, n: int) -> list[PerfilSociodemografico]:
        pesos = [r.peso for r in self._registros]
        return [
            PerfilSociodemografico(
                idade=r.idade,
                genero=r.genero,
                regiao=r.regiao,
                uf=r.uf,
                raca_cor=r.raca_cor,
                escolaridade=_sortear(self._rng, DISTRIBUICAO_ESCOLARIDADE),
                renda_faixa=_sortear(self._rng, DISTRIBUICAO_RENDA_FAIXA),
                situacao_domicilio=r.situacao_domicilio,
            )
            for r in self._rng.choices(self._registros, weights=pesos, k=n)
        ]
