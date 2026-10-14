"""
Versão do gerador de perfil sociodemográfico que amostra de MICRODADOS
REAIS (ex: PNAD Contínua, via Base dos Dados ou arquivo tratado do IBGE),
em vez de distribuições aproximadas.

IMPORTANTE: esta classe tem a MESMA interface pública que
`GeradorPerfilSociodemografico` (em gerador_estatistico.py):
`amostrar()` e `amostrar_lote(n)`. Isso significa que, no resto do
sistema (gerador_subjetivo.py, main.py, gerar_personas.py), basta trocar
qual classe é instanciada -- nenhum outro código muda.

COMO PREPARAR O ARQUIVO DE MICRODADOS (passo a passo):

1. Baixe um recorte da PNAD Contínua trimestral via Base dos Dados
   (https://basedosdados.org/dataset/br-ibge-pnadc) ou direto do FTP do
   IBGE (https://ftp.ibge.gov.br/.../Microdados/).
2. Mantenha só as colunas que você precisa (ver MAPEAMENTO_COLUNAS
   abaixo) e exporte como .csv ou .parquet.
3. Ajuste MAPEAMENTO_COLUNAS para os nomes de coluna do seu arquivo.
4. Aplique o peso amostral da pesquisa (coluna de peso, geralmente
   'V1028' na PNAD Contínua) na amostragem -- ver `amostrar()` abaixo.
   Isso é importante: sem o peso, você amostra a distribuição do
   DESENHO AMOSTRAL da pesquisa, não da população real.

LIMITAÇÃO ATUAL DESTE ARQUIVO: o código abaixo é um esqueleto funcional,
mas ainda não foi testado contra um arquivo real -- os nomes de coluna em
MAPEAMENTO_COLUNAS são um ponto de partida baseado no dicionário público
da PNAD Contínua e PROVAVELMENTE vão precisar de ajuste fino contra o
arquivo real que você baixar.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from schemas import PerfilSociodemografico


# Ajuste estes nomes para bater com as colunas do seu arquivo de microdados.
# Os nomes à esquerda são os usados internamente pelo sistema; os da
# direita são os nomes de coluna típicos da PNAD Contínua (códigos do
# dicionário oficial). CONFIRME contra o dicionário do arquivo que você
# baixar -- pode variar entre trimestres/anos.
MAPEAMENTO_COLUNAS = {
    "idade": "VD3324",          # ou "V2009" dependendo da versão do dataset
    "genero": "VD3302",          # ou "V2007" (1=homem, 2=mulher no IBGE -> mapear)
    "uf": "UF",
    "raca_cor": "V2010",
    "escolaridade": "VD3004",    # nível de instrução mais alto concluído
    "renda_total_domiciliar": "VD4020",
    "religiao": None,            # PNAD Contínua não tem religião por padrão;
                                  # ver observação abaixo
    "situacao_domicilio": "V1022",  # 1=urbano, 2=rural
    "peso_amostral": "V1028",
}

REGIAO_POR_UF = {
    "SP": "Sudeste", "RJ": "Sudeste", "MG": "Sudeste", "ES": "Sudeste",
    "RS": "Sul", "PR": "Sul", "SC": "Sul",
    "BA": "Nordeste", "PE": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "RN": "Nordeste", "AL": "Nordeste", "PI": "Nordeste", "SE": "Nordeste",
    "AM": "Norte", "PA": "Norte", "RO": "Norte", "AC": "Norte",
    "RR": "Norte", "AP": "Norte", "TO": "Norte",
    "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste", "DF": "Centro-Oeste",
}


@dataclass
class GeradorPerfilSociodemograficoReal:
    """
    Amostra perfis sociodemográficos de um DataFrame de microdados reais,
    respeitando a distribuição conjunta observada (em vez de sortear cada
    atributo de forma independente). Usa o peso amostral da pesquisa, se
    disponível, para que a amostragem reflita a população -- não o
    desenho amostral bruto.

    OBSERVAÇÃO SOBRE RELIGIÃO: a PNAD Contínua regular não pergunta
    religião (isso aparece em pesquisas específicas, como o Censo
    Demográfico ou suplementos esporádicos). Se você quiser manter
    religião na persona, há duas opções: (a) usar dados do Censo 2022
    para essa variável especificamente, amostrando-a de forma
    semi-independente (condicionada a região, que tem distribuição
    religiosa bem diferente entre si no Brasil), ou (b) manter como
    estatística aproximada simulada enquanto o resto vem de microdados
    reais -- registre isso como limitação explícita se for o caminho.
    """
    caminho_arquivo: str  # .csv ou .parquet com os microdados já tratados
    seed: int | None = None
    usar_peso_amostral: bool = True

    def __post_init__(self):
        path = Path(self.caminho_arquivo)
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo de microdados não encontrado: {self.caminho_arquivo}. "
                f"Baixe um recorte da PNAD Contínua (ver docstring deste módulo) "
                f"e ajuste o caminho."
            )
        if path.suffix == ".parquet":
            self.df = pd.read_parquet(path)
        else:
            self.df = pd.read_csv(path)

        col_peso = MAPEAMENTO_COLUNAS["peso_amostral"]
        if self.usar_peso_amostral and col_peso not in self.df.columns:
            raise ValueError(
                f"Coluna de peso amostral '{col_peso}' não encontrada no arquivo. "
                f"Ajuste MAPEAMENTO_COLUNAS ou passe usar_peso_amostral=False "
                f"(não recomendado -- amostra ficaria sem correção de desenho amostral)."
            )

    def _linha_para_perfil(self, linha: pd.Series) -> PerfilSociodemografico:
        """Converte uma linha do dataframe de microdados no schema interno.
        AJUSTAR os mapeamentos de categoria (ex: códigos de escolaridade)
        conforme o dicionário real da PNAD Contínua."""
        uf = str(linha[MAPEAMENTO_COLUNAS["uf"]]).strip()
        regiao = REGIAO_POR_UF.get(uf, "desconhecida")

        return PerfilSociodemografico(
            idade=int(linha[MAPEAMENTO_COLUNAS["idade"]]),
            genero=self._mapear_genero(linha[MAPEAMENTO_COLUNAS["genero"]]),
            regiao=regiao,
            uf=uf,
            raca_cor=self._mapear_raca_cor(linha[MAPEAMENTO_COLUNAS["raca_cor"]]),
            escolaridade=self._mapear_escolaridade(linha[MAPEAMENTO_COLUNAS["escolaridade"]]),
            renda_faixa=self._mapear_renda_faixa(linha[MAPEAMENTO_COLUNAS["renda_total_domiciliar"]]),
            religiao="não_modelada",  # retirada do modelo por decisão do orientador -- ver README
            situacao_domicilio=self._mapear_situacao_domicilio(linha[MAPEAMENTO_COLUNAS["situacao_domicilio"]]),
        )

    @staticmethod
    def _mapear_genero(valor) -> str:
        """Aceita tanto código numérico bruto (1/2) quanto texto já
        traduzido via dicionário (ex: query de extrair_microdados.py
        traz 'Homem'/'Mulher' diretamente)."""
        if isinstance(valor, str):
            valor_lower = valor.strip().lower()
            if "homem" in valor_lower or "masculin" in valor_lower:
                return "masculino"
            if "mulher" in valor_lower or "feminin" in valor_lower:
                return "feminino"
            return "não_informado"
        # código numérico: 1=homem, 2=mulher (confirmar contra dicionário real)
        return {1: "masculino", 2: "feminino"}.get(int(valor), "não_informado")

    @staticmethod
    def _mapear_raca_cor(valor) -> str:
        """Aceita código numérico (V2010) ou texto já traduzido."""
        if isinstance(valor, str):
            valor_lower = valor.strip().lower()
            mapa_texto = {
                "branca": "branca", "preta": "preta", "amarela": "amarela",
                "parda": "parda", "indígena": "indigena", "indigena": "indigena",
            }
            return mapa_texto.get(valor_lower, "não_informado")
        # códigos V2010 do IBGE: 1=branca, 2=preta, 3=amarela, 4=parda, 5=indígena
        mapa = {1: "branca", 2: "preta", 3: "amarela", 4: "parda", 5: "indigena"}
        return mapa.get(int(valor), "não_informado")

    @staticmethod
    def _mapear_escolaridade(valor) -> str:
        """
        Mapeia para as mesmas 4 faixas usadas em gerador_estatistico.py
        (ver DISTRIBUICAO_ESCOLARIDADE), para manter compatibilidade entre
        as duas implementações.

        Se `valor` vier como texto (caso da query de extrair_microdados.py,
        que já traduz VD3004 via dicionário), faz correspondência por
        palavras-chave. As categorias exatas de texto da VD3004 PODEM
        variar -- confirme contra a saída de `inspecionar_schema()` ou
        contra os valores únicos da coluna no seu CSV
        (`df['VD3004'].unique()`) e ajuste os termos abaixo se necessário.
        """
        if isinstance(valor, str):
            texto = valor.strip().lower()
            if "superior completo" in texto:
                return "superior_completo"
            if "médio completo" in texto or "medio completo" in texto:
                return "medio_completo_superior_incompleto"
            if "fundamental completo" in texto:
                return "fundamental_completo_medio_incompleto"
            if "sem instrução" in texto or "sem instrucao" in texto or "fundamental incompleto" in texto:
                return "sem_instrucao_fundamental_incompleto"
            # categoria de texto não reconhecida -- não falha silenciosamente,
            # avisa para você revisar e ajustar os termos acima
            raise ValueError(
                f"Categoria de escolaridade não reconhecida: {valor!r}. "
                f"Ajuste os termos em _mapear_escolaridade() contra os "
                f"valores reais da coluna VD3004 no seu arquivo."
            )
        raise NotImplementedError(
            "Mapeamento de código numérico de escolaridade ainda não "
            "implementado -- a query de extrair_microdados.py já traz "
            "VD3004 como texto traduzido; se seu arquivo tem código "
            "numérico bruto, ajuste este método contra o dicionário real."
        )

    @staticmethod
    def _mapear_renda_faixa(valor_renda: float) -> str:
        SALARIO_MINIMO = 1518.00  # ajustar para o valor vigente no ano dos dados
        if pd.isna(valor_renda) or valor_renda <= 0:
            return "até_1"
        razao = valor_renda / SALARIO_MINIMO
        if razao <= 1:
            return "até_1"
        elif razao <= 2:
            return "1-2"
        elif razao <= 5:
            return "2-5"
        elif razao <= 10:
            return "5-10"
        return "mais_de_10"

    @staticmethod
    def _mapear_situacao_domicilio(codigo) -> str:
        return {1: "urbano", 2: "rural"}.get(int(codigo), "urbano")

    def amostrar(self) -> PerfilSociodemografico:
        if self.usar_peso_amostral:
            col_peso = MAPEAMENTO_COLUNAS["peso_amostral"]
            linha = self.df.sample(n=1, weights=self.df[col_peso], random_state=self.seed).iloc[0]
        else:
            linha = self.df.sample(n=1, random_state=self.seed).iloc[0]
        return self._linha_para_perfil(linha)

    def amostrar_lote(self, n: int) -> list[PerfilSociodemografico]:
        col_peso = MAPEAMENTO_COLUNAS["peso_amostral"]
        if self.usar_peso_amostral:
            linhas = self.df.sample(n=n, weights=self.df[col_peso], random_state=self.seed, replace=True)
        else:
            linhas = self.df.sample(n=n, random_state=self.seed, replace=True)
        return [self._linha_para_perfil(linha) for _, linha in linhas.iterrows()]
