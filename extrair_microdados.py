"""
Extrai um recorte da PNAD Contínua via Base dos Dados (BigQuery) e salva
em CSV local, no formato que `gerador_estatistico_real.py` espera.

PRÉ-REQUISITOS:
1. Conta no Google Cloud + projeto criado com billing (faturamento)
   ativado. O billing_id abaixo é o PROJECT ID (não o nome de exibição),
   algo como "meu-projeto-tcc-123456" -- confira no Console do GCP.
2. pip install basedosdados --break-system-packages
3. Na primeira execução, o pacote vai abrir o navegador para você
   autenticar com sua conta Google.

IMPORTANTE SOBRE A QUERY:
- Adicionei filtro de `ano`/`trimestre` -- SEM ISSO a consulta varre TODA
  a série histórica da PNAD Contínua desde 2012, o que é desnecessário
  para este projeto e consome cota do BigQuery rápido. AJUSTE ANO_REFERENCIA
  e TRIMESTRE_REFERENCIA para o período mais recente disponível.
- Os nomes de coluna para período (`ano`, `trimestre`) são minha melhor
  estimativa a partir da convenção do IBGE -- a Base dos Dados costuma
  preservar esses nomes, mas CONFIRME contra o schema real antes de
  rodar (ver função `inspecionar_schema` abaixo, que roda uma query
  bem mais barata só para listar colunas).
- Adicionei amostragem (TABLESAMPLE) para não trazer milhões de linhas
  -- você não precisa disso para gerar algumas dezenas/centenas de
  personas.

Uso:
    python extrair_microdados.py --billing-id meu-projeto-123456 --inspecionar
    python extrair_microdados.py --billing-id meu-projeto-123456 --n 5000 --out dados/pnadc_recorte.csv
"""

import argparse
import sys

try:
    import basedosdados as bd
except ImportError:
    print(
        "Pacote 'basedosdados' não instalado. Rode:\n"
        "    pip install basedosdados --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


def inspecionar_schema(billing_id: str):
    """
    Roda uma query bem mais barata (LIMIT 0, ou metadados) só para você
    confirmar os nomes reais de coluna antes de rodar a extração completa.
    Rode isso PRIMEIRO -- evita gastar cota com uma query errada.
    """
    query = """
    SELECT column_name, data_type
    FROM `basedosdados.br_ibge_pnadc.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'microdados'
    ORDER BY ordinal_position
    """
    print("Consultando schema da tabela microdados...")
    df = bd.read_sql(query=query, billing_project_id=billing_id)
    print(df.to_string(index=False))
    print(
        "\nProcure aqui pelas colunas de período (ex: 'ano', 'trimestre' ou "
        "'Ano', 'Trimestre') e confirme contra ANO_REFERENCIA/TRIMESTRE_REFERENCIA "
        "antes de rodar a extração completa."
    )


def extrair_recorte(billing_id: str, ano: int, trimestre: int, n_amostra: int, out_path: str):
    """
    Extrai um recorte da PNAD Contínua filtrado por ano/trimestre, já com
    os dicionários de valor aplicados (mesma lógica da query que você
    recebeu do site da Base dos Dados), e salva em CSV.

    AJUSTE os nomes das colunas de filtro (`ano = `, `trimestre = `)
    conforme o que você confirmar em inspecionar_schema().
    """
    query = f"""
    WITH
    dicionario_V1022 AS (
        SELECT chave AS chave_V1022, valor AS descricao_V1022
        FROM `basedosdados.br_ibge_pnadc.dicionario`
        WHERE nome_coluna = 'V1022' AND id_tabela = 'microdados'
    ),
    dicionario_V2007 AS (
        SELECT chave AS chave_V2007, valor AS descricao_V2007
        FROM `basedosdados.br_ibge_pnadc.dicionario`
        WHERE nome_coluna = 'V2007' AND id_tabela = 'microdados'
    ),
    dicionario_V2010 AS (
        SELECT chave AS chave_V2010, valor AS descricao_V2010
        FROM `basedosdados.br_ibge_pnadc.dicionario`
        WHERE nome_coluna = 'V2010' AND id_tabela = 'microdados'
    ),
    dicionario_VD3004 AS (
        SELECT chave AS chave_VD3004, valor AS descricao_VD3004
        FROM `basedosdados.br_ibge_pnadc.dicionario`
        WHERE nome_coluna = 'VD3004' AND id_tabela = 'microdados'
    )
    SELECT
        dados.sigla_uf AS sigla_uf,
        diretorio_sigla_uf.nome AS sigla_uf_nome,
        descricao_V1022 AS V1022,
        dados.V1028 AS V1028,
        descricao_V2007 AS V2007,
        dados.V2009 AS V2009,
        descricao_V2010 AS V2010,
        descricao_VD3004 AS VD3004,
        dados.VD4020 AS VD4020
    FROM `basedosdados.br_ibge_pnadc.microdados` AS dados
    LEFT JOIN (SELECT DISTINCT sigla, nome FROM `basedosdados.br_bd_diretorios_brasil.uf`) AS diretorio_sigla_uf
        ON dados.sigla_uf = diretorio_sigla_uf.sigla
    LEFT JOIN dicionario_V1022 ON dados.V1022 = chave_V1022
    LEFT JOIN dicionario_V2007 ON dados.V2007 = chave_V2007
    LEFT JOIN dicionario_V2010 ON dados.V2010 = chave_V2010
    LEFT JOIN dicionario_VD3004 ON dados.VD3004 = chave_VD3004
    WHERE
        dados.ano = {ano}
        AND dados.trimestre = {trimestre}
        -- AJUSTAR: confirme os nomes 'ano'/'trimestre' via inspecionar_schema()
    ORDER BY RAND()
    LIMIT {n_amostra}
    """
    print(f"Extraindo recorte: ano={ano}, trimestre={trimestre}, n={n_amostra}...")
    print("(isso pode levar alguns minutos na primeira vez)")
    df = bd.read_sql(query=query, billing_project_id=billing_id)

    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} linhas salvas em {out_path}")
    print("\nPróximo passo: ajustar MAPEAMENTO_COLUNAS em "
          "persona/gerador_estatistico_real.py para bater com estas colunas "
          "(note que esta query já traz os valores traduzidos via dicionário, "
          "não os códigos numéricos -- isso SIMPLIFICA os métodos _mapear_* "
          "dessa classe, que hoje esperam códigos numéricos).")


def main():
    parser = argparse.ArgumentParser(description="Extrai recorte da PNAD Contínua via Base dos Dados")
    parser.add_argument("--billing-id", type=str, required=True, help="Project ID do GCP com billing ativado")
    parser.add_argument("--inspecionar", action="store_true",
                         help="só lista as colunas disponíveis, não extrai dados")
    parser.add_argument("--ano", type=int, default=2024, help="ano de referência")
    parser.add_argument("--trimestre", type=int, default=4, help="trimestre de referência (1-4)")
    parser.add_argument("--n", type=int, default=5000, help="número de linhas a amostrar")
    parser.add_argument("--out", type=str, default="dados/pnadc_recorte.csv")
    args = parser.parse_args()

    if args.inspecionar:
        inspecionar_schema(args.billing_id)
        return

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    extrair_recorte(args.billing_id, args.ano, args.trimestre, args.n, args.out)


if __name__ == "__main__":
    main()
