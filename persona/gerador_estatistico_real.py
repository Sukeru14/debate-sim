"""
Amostra perfis sociodemográficos de um CSV/Parquet de microdados da PNAD
Contínua extraído via Base dos Dados (ver extrair_microdados.py), com o
peso amostral V1028. Mesma interface de GeradorPerfilPNADCBruto.

Vantagem sobre o arquivo bruto: escolaridade (VD3004) e renda (VD4020)
vêm reais, já traduzidas pelo dicionário da Base dos Dados.

ATENÇÃO:
- Ainda não testado contra uma extração real; confira os valores únicos
  das colunas de texto (ex: df["VD3004"].unique()) antes de usar.
- VD4020 é o rendimento mensal de TODOS OS TRABALHOS da pessoa (não a
  renda familiar). Quem não trabalha vem sem valor e cai na faixa mais
  baixa -- declarar isso, ou trocar por uma variável de renda domiciliar.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from persona.gerador_estatistico_pnadc_bruto import (
    DISTRIBUICAO_ESCOLARIDADE, DISTRIBUICAO_RENDA_FAIXA, REGIAO_POR_PRIMEIRO_DIGITO, SIGLA_POR_CODIGO_UF,
)
from schemas import PerfilSociodemografico


SALARIO_MINIMO = 1518.00  # ajustar para o ano dos dados

# Nomes de coluna produzidos por extrair_microdados.py
COLUNAS = {
    "uf": "sigla_uf", "situacao_domicilio": "V1022", "peso": "V1028", "genero": "V2007",
    "idade": "V2009", "raca_cor": "V2010", "escolaridade": "VD3004", "renda": "VD4020",
}

ESCOLARIDADES = list(DISTRIBUICAO_ESCOLARIDADE)
FAIXAS_RENDA = list(DISTRIBUICAO_RENDA_FAIXA)
REGIAO_POR_SIGLA = {sigla: REGIAO_POR_PRIMEIRO_DIGITO[cod[0]] for cod, sigla in SIGLA_POR_CODIGO_UF.items()}

# Ordem importa: "superior completo" antes de "superior incompleto" etc.
PALAVRAS_ESCOLARIDADE = [
    ("superior completo", ESCOLARIDADES[3]),
    ("superior incompleto", ESCOLARIDADES[2]),
    ("médio completo", ESCOLARIDADES[2]),
    ("médio incompleto", ESCOLARIDADES[1]),
    ("fundamental completo", ESCOLARIDADES[1]),
    ("fundamental incompleto", ESCOLARIDADES[0]),
    ("sem instrução", ESCOLARIDADES[0]),
]


def _escolaridade(valor) -> str:
    texto = str(valor).strip().lower()
    for chave, categoria in PALAVRAS_ESCOLARIDADE:
        if chave in texto:
            return categoria
    raise ValueError(f"Escolaridade não reconhecida: {valor!r} -- ajuste PALAVRAS_ESCOLARIDADE")


def _renda_faixa(valor) -> str:
    if pd.isna(valor) or valor <= 0:
        return FAIXAS_RENDA[0]
    razao = valor / SALARIO_MINIMO
    for limite, faixa in zip((1, 2, 5, 10), FAIXAS_RENDA):
        if razao <= limite:
            return faixa
    return FAIXAS_RENDA[-1]


def _genero(valor) -> str:
    texto = str(valor).strip().lower()
    if texto in ("1", "homem") or "masculin" in texto:
        return "masculino"
    if texto in ("2", "mulher") or "feminin" in texto:
        return "feminino"
    return "não informado"


def _raca_cor(valor) -> str:
    texto = str(valor).strip().lower()
    codigos = {"1": "branca", "2": "preta", "3": "amarela", "4": "parda", "5": "indígena"}
    textos = {"branca", "preta", "amarela", "parda", "indígena"}
    if texto == "indigena":
        return "indígena"
    return codigos.get(texto) or (texto if texto in textos else "não informada")


def _situacao_domicilio(valor) -> str:
    texto = str(valor).strip().lower()
    return "rural" if texto in ("2", "rural") else "urbano"


@dataclass
class GeradorPerfilMicrodadosCSV:
    caminho_arquivo: str
    seed: int | None = None
    idade_minima: int = 18

    def __post_init__(self):
        path = Path(self.caminho_arquivo)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        faltando = [c for c in COLUNAS.values() if c not in df.columns]
        if faltando:
            raise ValueError(f"Colunas ausentes em {path}: {faltando}")
        self.df = df[df[COLUNAS["idade"]] >= self.idade_minima].reset_index(drop=True)

    def amostrar_lote(self, n: int) -> list[PerfilSociodemografico]:
        linhas = self.df.sample(n=n, weights=self.df[COLUNAS["peso"]], replace=True, random_state=self.seed)
        return [self._para_perfil(linha) for _, linha in linhas.iterrows()]

    @staticmethod
    def _para_perfil(linha: pd.Series) -> PerfilSociodemografico:
        uf = str(linha[COLUNAS["uf"]]).strip()
        return PerfilSociodemografico(
            idade=int(linha[COLUNAS["idade"]]),
            genero=_genero(linha[COLUNAS["genero"]]),
            regiao=REGIAO_POR_SIGLA.get(uf, "desconhecida"),
            uf=uf,
            raca_cor=_raca_cor(linha[COLUNAS["raca_cor"]]),
            escolaridade=_escolaridade(linha[COLUNAS["escolaridade"]]),
            renda_faixa=_renda_faixa(linha[COLUNAS["renda"]]),
            situacao_domicilio=_situacao_domicilio(linha[COLUNAS["situacao_domicilio"]]),
        )
