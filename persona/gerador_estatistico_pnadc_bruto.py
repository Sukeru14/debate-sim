"""
Lê o arquivo de microdados BRUTO da PNAD Contínua trimestral (formato de
largura fixa, baixado direto do FTP do IBGE -- ex: PNADC_012026.txt) e
amostra perfis sociodemográficos REAIS, respeitando o peso amostral.

MESMA INTERFACE que GeradorPerfilSociodemografico (gerador_estatistico.py)
e GeradorPerfilSociodemograficoReal (gerador_estatistico_real.py):
`amostrar()` e `amostrar_lote(n)`. O resto do sistema não precisa saber a
diferença.

POSIÇÕES USADAS (confirmadas empiricamente contra um arquivo real de
janeiro/2026 -- ver conversa de desenvolvimento; valores plausíveis e
consistentes com estatísticas oficiais do IBGE foram usados como critério
de validação):

    Ano       posição 1,   tamanho 4
    Trimestre posição 5,   tamanho 1
    UF        posição 6,   tamanho 2   (código IBGE de 2 dígitos)
    V1022     posição 33,  tamanho 1   (1=urbano, 2=rural)
    V1027     posição 35,  tamanho 15  (peso SEM calibração)
    V1028     posição 50,  tamanho 15  (peso COM calibração -- usar este)
    V2007     posição 95,  tamanho 1   (1=homem, 2=mulher)
    V2009     posição 104, tamanho 3   (idade em anos)
    V2010     posição 107, tamanho 1   (1=branca,2=preta,3=amarela,4=parda,5=indígena,9=ignorado)

LIMITAÇÃO DELIBERADA E DOCUMENTADA: escolaridade e renda (VD3004, VD4020)
NÃO são lidas deste arquivo -- a posição exata dessas variáveis derivadas
não foi confirmada com segurança (muitas variáveis categóricas pequenas
competem pela mesma faixa de posições no meio do registro, sem um
dicionário oficial em mãos para desambiguar). Em vez de arriscar uma
posição errada (que daria valores incorretos SEM erro aparente), essas
duas variáveis continuam vindo de `gerador_estatistico.DISTRIBUICAO_*`
(aproximação da literatura). Isso é uma limitação a registrar
explicitamente no TCC: parte do perfil é amostra real da PNAD Contínua,
parte é aproximação.

Se quiser completar escolaridade/renda com dados reais depois, o caminho
mais seguro é obter o dicionário oficial em Excel (Documentacao/
Variaveis_PNADC_Trimestral.xls, no FTP do IBGE) e confirmar a posição
exata antes de usar.
"""

from dataclasses import dataclass
import random

from schemas import PerfilSociodemografico
from persona.gerador_estatistico import (
    DISTRIBUICAO_ESCOLARIDADE,
    DISTRIBUICAO_RENDA_FAIXA,
    _sortear,
)


# Posições confirmadas (0-indexadas para slicing direto em Python:
# string[inicio:fim])
POS_UF = (5, 7)
POS_V1022 = (32, 33)
POS_V1028 = (49, 64)
POS_V2007 = (94, 95)
POS_V2009 = (103, 106)
POS_V2010 = (106, 107)

TAMANHO_LINHA_ESPERADO = 3478

MAPA_UF_SIGLA = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE", "29": "BA",
    "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS",
    "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}

MAPA_SIGLA_REGIAO = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte", "PA": "Norte", "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste", "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste", "DF": "Centro-Oeste",
}

MAPA_RACA_COR = {
    "1": "branca", "2": "preta", "3": "amarela", "4": "parda", "5": "indigena", "9": "não_informado",
}


@dataclass
class RegistroBrutoPNADC:
    """Um registro (pessoa) já parseado do arquivo bruto, antes de
    converter para o schema interno do sistema."""
    uf_sigla: str
    regiao: str
    situacao_domicilio: str  # "urbano" ou "rural"
    peso_amostral: float
    genero: str
    idade: int
    raca_cor: str


def _parsear_linha(linha: str) -> RegistroBrutoPNADC | None:
    """Extrai as 7 variáveis confirmadas de uma linha do arquivo bruto.
    Retorna None se a linha estiver malformada (tamanho inesperado) ou
    tiver valores fora do intervalo esperado -- mais seguro descartar o
    registro do que seguir com dado inválido."""
    if len(linha.rstrip("\n").rstrip("\r")) != TAMANHO_LINHA_ESPERADO:
        return None

    try:
        codigo_uf = linha[POS_UF[0]:POS_UF[1]]
        uf_sigla = MAPA_UF_SIGLA.get(codigo_uf)
        if uf_sigla is None:
            return None
        regiao = MAPA_SIGLA_REGIAO[uf_sigla]

        v1022 = linha[POS_V1022[0]:POS_V1022[1]]
        situacao_domicilio = {"1": "urbano", "2": "rural"}.get(v1022)
        if situacao_domicilio is None:
            return None

        peso_amostral = float(linha[POS_V1028[0]:POS_V1028[1]])

        v2007 = linha[POS_V2007[0]:POS_V2007[1]]
        genero = {"1": "masculino", "2": "feminino"}.get(v2007)
        if genero is None:
            return None

        idade = int(linha[POS_V2009[0]:POS_V2009[1]])
        if not (0 <= idade <= 130):
            return None

        v2010 = linha[POS_V2010[0]:POS_V2010[1]]
        raca_cor = MAPA_RACA_COR.get(v2010, "não_informado")

        return RegistroBrutoPNADC(
            uf_sigla=uf_sigla,
            regiao=regiao,
            situacao_domicilio=situacao_domicilio,
            peso_amostral=peso_amostral,
            genero=genero,
            idade=idade,
            raca_cor=raca_cor,
        )
    except (ValueError, IndexError):
        return None


@dataclass
class GeradorPerfilSociodemograficoPNADCBruto:
    """
    Amostra perfis sociodemográficos a partir do arquivo de microdados
    BRUTO da PNAD Contínua (largura fixa, baixado direto do FTP do
    IBGE), respeitando o peso amostral (V1028) -- igual ao que a query
    via Base dos Dados faria, mas sem precisar de conta no Google Cloud.

    Usa idade, sexo, UF/região, raça/cor, situação do domicílio e peso
    amostral REAIS da pesquisa. Escolaridade e renda continuam
    aproximadas (ver docstring do módulo).
    """
    caminho_arquivo: str
    seed: int | None = None
    apenas_adultos: bool = True  # filtra idade >= 18 (eleitores)

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

        registros = []
        descartados = 0
        with open(self.caminho_arquivo, "r", encoding="latin-1") as f:
            for linha in f:
                registro = _parsear_linha(linha)
                if registro is None:
                    descartados += 1
                    continue
                if self.apenas_adultos and registro.idade < 18:
                    continue
                registros.append(registro)

        if not registros:
            raise ValueError(
                f"Nenhum registro válido encontrado em {self.caminho_arquivo}. "
                f"Verifique se é o arquivo bruto correto da PNAD Contínua "
                f"trimestral (largura fixa, ~3478 caracteres por linha)."
            )

        self._registros = registros
        self._pesos = [r.peso_amostral for r in registros]
        print(
            f"[GeradorPerfilSociodemograficoPNADCBruto] {len(registros)} registros "
            f"válidos carregados ({descartados} descartados/filtrados) de "
            f"{self.caminho_arquivo}"
        )

    def _registro_para_perfil(self, registro: RegistroBrutoPNADC) -> PerfilSociodemografico:
        # escolaridade e renda continuam aproximadas (ver docstring). Religião
        # retirada do modelo por decisão do orientador -- não consta na PNAD
        # Contínua e não há, por ora, outra fonte real confiável; volta a ser
        # amostrada se/quando essa fonte existir (ver gerador_estatistico.py).
        return PerfilSociodemografico(
            idade=registro.idade,
            genero=registro.genero,
            regiao=registro.regiao,
            uf=registro.uf_sigla,
            raca_cor=registro.raca_cor,
            escolaridade=_sortear(DISTRIBUICAO_ESCOLARIDADE),
            renda_faixa=_sortear(DISTRIBUICAO_RENDA_FAIXA),
            religiao="não_modelada",
            situacao_domicilio=registro.situacao_domicilio,
        )

    def amostrar(self) -> PerfilSociodemografico:
        registro = random.choices(self._registros, weights=self._pesos, k=1)[0]
        return self._registro_para_perfil(registro)

    def amostrar_lote(self, n: int) -> list[PerfilSociodemografico]:
        registros = random.choices(self._registros, weights=self._pesos, k=n)
        return [self._registro_para_perfil(r) for r in registros]
