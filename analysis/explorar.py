"""
Carrega um banco da simulação em DataFrames. A análise propriamente dita
será escrita quando as RQs estiverem definidas.

    python -m analysis.explorar --db storage/experimento.db

IMPORTANTE: o mesmo eleitor_id aparece em várias execuções (o lote é
reaproveitado), então todo cruzamento entre tabelas deve usar
(run_id, eleitor_id), nunca só eleitor_id.
"""

import argparse
import json
import sqlite3

import pandas as pd


def carregar_dataframes(db_path: str) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        tabelas = {t: pd.read_sql(f"SELECT * FROM {t}", conn)
                   for t in ("runs", "eleitores", "falas", "respostas")}

    brutos = tabelas["eleitores"]
    tabelas["eleitores"] = pd.concat(
        [brutos[["run_id", "eleitor_id", "narrativa"]]]
        + [pd.json_normalize(brutos[col].apply(json.loads))
           for col in ("perfil_json", "big_five_json", "fundamentos_morais_json")],
        axis=1,
    )
    tabelas["respostas"]["mudou_de_posicao"] = tabelas["respostas"]["mudou_de_posicao"].astype(bool)
    return tabelas


def painel(tabelas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Uma linha por (execução, eleitor, rodada), com confronto, perfil e traços."""
    return (
        tabelas["respostas"]
        .merge(tabelas["runs"][["run_id", "tipo_candidato_1", "tipo_candidato_2", "seed"]], on="run_id")
        .merge(tabelas["eleitores"], on=["run_id", "eleitor_id"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="storage/experimento.db")
    args = parser.parse_args()

    tabelas = carregar_dataframes(args.db)
    dados = painel(tabelas)
    ultima = dados[dados["rodada"] == dados.groupby("run_id")["rodada"].transform("max")]

    print("Intenção de voto ao fim de cada execução:")
    resumo = ultima.groupby(["run_id", "tipo_candidato_1", "tipo_candidato_2", "intencao_voto"]).size()
    print(resumo.unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
